import os, time, platform, argparse, random
import torch, mlflow, mlflow.pytorch
import numpy as np
from collections import Counter
from datasets import concatenate_datasets, load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from sklearn.metrics import f1_score as skf1, accuracy_score

os.environ["AWS_ACCESS_KEY_ID"]      = "datanauts-key"
os.environ["AWS_SECRET_ACCESS_KEY"]  = "datanauts-secret"
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://129.114.27.190:9000"

MLFLOW_URI = "http://129.114.27.190:8000"
EXPERIMENT = "deadline-detection-classifier"
BASE_MODEL = "roberta-base"

EVENT2ID = {"payment_due": 0, "expiration": 1, "effective": 2, "agreement": 3}
ID2EVENT = {v: k for k, v in EVENT2ID.items()}

CONFIGS = {
    "baseline": {
        "epochs": 0, "learning_rate": 0, "batch_size": 16, "max_seq_length": 128, "fp16": False,
    },
    "roberta_clf_v1": {
        "epochs": 3, "learning_rate": 2e-5, "batch_size": 16, "max_seq_length": 128, "fp16": True,
    },
    "roberta_clf_v2": {
        "epochs": 3, "learning_rate": 5e-5, "batch_size": 16, "max_seq_length": 128, "fp16": True,
    },
    "roberta_clf_v3": {
        "epochs": 5, "learning_rate": 2e-5, "batch_size": 16, "max_seq_length": 128, "fp16": True,
    },
}

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_classifier_data(seed=42):
    ds       = load_dataset("tanvitakavane/datanauts_project_cuad-deadline-ner")
    all_data = concatenate_datasets([ds["train"], ds["test"]])
    all_data = all_data.filter(lambda x: "B-DATE" in x["ner_labels"])

    print(f"Total samples with dates: {len(all_data)}")
    print("Class distribution:", dict(Counter(all_data["event_type"])))

    def prepare(example):
        example["label"] = EVENT2ID[example["event_type"]]
        return example

    all_data = all_data.map(prepare)
    all_data = all_data.select_columns(["sentence", "label"])
    all_data = all_data.shuffle(seed=seed)

    n         = len(all_data)
    train_end = int(n * 0.8)
    val_end   = int(n * 0.9)

    train = all_data.select(range(0, train_end))
    val   = all_data.select(range(train_end, val_end))
    test  = all_data.select(range(val_end, n))

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test

def tokenize_data(train_ds, val_ds, test_ds, tokenizer, max_length):
    def tokenize(examples):
        return tokenizer(examples["sentence"], truncation=True,
                         max_length=max_length, padding="max_length")

    tok_train = train_ds.map(tokenize, batched=True)
    tok_val   = val_ds.map(tokenize,   batched=True)
    tok_test  = test_ds.map(tokenize,  batched=True)

    for ds in [tok_train, tok_val, tok_test]:
        ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return tok_train, tok_val, tok_test

def compute_clf_metrics(p):
    preds  = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       skf1(labels, preds, average="weighted", zero_division=0),
    }

def run_baseline(test_ds):
    labels_all     = [test_ds[i]["label"] for i in range(len(test_ds))]
    majority       = Counter(labels_all).most_common(1)[0][0]
    preds_majority = [majority] * len(labels_all)
    f1             = skf1(labels_all, preds_majority, average="weighted", zero_division=0)
    acc            = accuracy_score(labels_all, preds_majority)
    print(f"Majority class baseline — class: {ID2EVENT[majority]} | F1: {f1:.4f} | Acc: {acc:.4f}")
    return {"eval_f1": f1, "eval_accuracy": acc, "eval_loss": 0}

def train_classifier(model_name, cfg, tok_train, tok_val, tok_test, tokenizer, train_ds):
    model  = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(EVENT2ID), id2label=ID2EVENT, label2id=EVENT2ID,
    )
    t_args = TrainingArguments(
        output_dir=f"/tmp/clf-{model_name}",
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        weight_decay=0.01,
        warmup_steps=50,
        fp16=cfg["fp16"] and torch.cuda.is_available(),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=20,
        report_to="none",
    )
    trainer = Trainer(
        model=model, args=t_args,
        train_dataset=tok_train, eval_dataset=tok_val,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_clf_metrics,
    )
    print(f"Training {model_name} | lr={cfg['learning_rate']} | epochs={cfg['epochs']}...")
    t0           = time.time()
    train_result = trainer.train()
    train_time   = time.time() - t0
    print("Evaluating on test set...")
    predictions  = trainer.predict(tok_test)
    raw          = predictions.metrics
    test_result  = {
        "eval_f1":       raw.get("test_f1", 0),
        "eval_accuracy": raw.get("test_accuracy", 0),
        "eval_loss":     raw.get("test_loss", 0),
    }
    print(f"Done! Time: {train_time:.1f}s | Test F1: {test_result['eval_f1']:.4f} | Accuracy: {test_result['eval_accuracy']:.4f}")
    return trainer.model, train_result, test_result, train_time

def log_to_mlflow(model_name, cfg, model, train_result, test_result, train_time, train_ds, val_ds, test_ds):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({
            "model_name":     model_name,
            "base_model":     BASE_MODEL,
            "task":           "sequence_classification",
            "num_labels":     len(EVENT2ID),
            "epochs":         cfg["epochs"],
            "batch_size":     cfg["batch_size"],
            "learning_rate":  cfg["learning_rate"],
            "max_seq_length": cfg["max_seq_length"],
            "fp16":           cfg["fp16"],
            "train_size":     len(train_ds),
            "val_size":       len(val_ds),
            "test_size":      len(test_ds),
            "gpu":            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "gpu_vram_gb":    round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else 0,
            "pytorch":        torch.__version__,
            "platform":       platform.platform(),
        })
        mlflow.log_metrics({
            "total_train_time_sec": train_time,
            "time_per_epoch_sec":   train_time / max(cfg["epochs"], 1),
            "samples_per_sec":      len(train_ds) * max(cfg["epochs"], 1) / max(train_time, 1),
            "train_loss":           train_result.training_loss if train_result else 0,
            "test_f1":              test_result.get("eval_f1", 0),
            "test_accuracy":        test_result.get("eval_accuracy", 0),
            "test_loss":            test_result.get("eval_loss", 0),
        })
        if model is not None:
            pass  # model artifact skipped - metrics logged above
        print(f"Logged {model_name} → {MLFLOW_URI}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(CONFIGS.keys()),
                        help="Which classifier variant to run")
    args = parser.parse_args()

    set_seeds(42)
    cfg                          = CONFIGS[args.model]
    train_ds, val_ds, test_ds    = load_classifier_data(seed=42)
    tokenizer                    = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok_train, tok_val, tok_test = tokenize_data(train_ds, val_ds, test_ds, tokenizer, cfg["max_seq_length"])

    if args.model == "baseline":
        t0          = time.time()
        test_result = run_baseline(test_ds)
        elapsed     = time.time() - t0
        log_to_mlflow("clf_baseline_majority",
                      {**cfg, "epochs": 0}, None, None,
                      test_result, elapsed, train_ds, val_ds, test_ds)
        return

    model, train_result, test_result, train_time = train_classifier(
        args.model, cfg, tok_train, tok_val, tok_test, tokenizer, train_ds
    )
    log_to_mlflow(args.model, cfg, model, train_result, test_result, train_time, train_ds, val_ds, test_ds)

if __name__ == "__main__":
    main()
