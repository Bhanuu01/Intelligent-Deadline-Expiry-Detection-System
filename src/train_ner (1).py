import os, time, platform, argparse, random
import torch, mlflow, mlflow.pytorch
import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from seqeval.metrics import f1_score, precision_score, recall_score

os.environ["AWS_ACCESS_KEY_ID"]      = "datanauts-key"
os.environ["AWS_SECRET_ACCESS_KEY"]  = "datanauts-secret"
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://129.114.27.190:9000"

MLFLOW_URI   = "http://129.114.27.190:8000"
EXPERIMENT   = "deadline-detection-ner"
LABEL_LIST   = ["O", "B-DATE", "I-DATE"]
LABEL2ID     = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL     = {i: l for i, l in enumerate(LABEL_LIST)}
MONTHS       = {
    "january","february","march","april","may","june",
    "july","august","september","october","november","december",
    "jan","feb","mar","apr","jun","jul","aug","sep","oct","nov","dec"
}

CONFIGS = {
    "baseline": {
        "base_model": None, "epochs": 0, "learning_rate": 0,
        "batch_size": 16, "max_seq_length": 128, "fp16": False,
    },
    "bert_finetune_v1": {
        "base_model": "dslim/bert-base-NER", "epochs": 3, "learning_rate": 2e-5,
        "batch_size": 16, "max_seq_length": 128, "fp16": True,
    },
    "bert_finetune_v2": {
        "base_model": "dslim/bert-base-NER", "epochs": 3, "learning_rate": 5e-5,
        "batch_size": 16, "max_seq_length": 128, "fp16": True,
    },
    "bert_finetune_v3": {
        "base_model": "dslim/bert-base-NER", "epochs": 5, "learning_rate": 2e-5,
        "batch_size": 16, "max_seq_length": 128, "fp16": True,
    },
    "bert_base_cased": {
        "base_model": "bert-base-cased", "epochs": 3, "learning_rate": 2e-5,
        "batch_size": 16, "max_seq_length": 128, "fp16": True,
    },
}

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_and_split_data(seed=42):
    ds       = load_dataset("tanvitakavane/datanauts_project_cuad-deadline-ner")
    all_data = concatenate_datasets([ds["train"], ds["test"]])
    all_data = all_data.filter(lambda x: "B-DATE" in x["ner_labels"])

    def convert_labels(example):
        example["ner_tags"] = [LABEL2ID.get(l, 0) for l in example["ner_labels"]]
        return example

    all_data    = all_data.map(convert_labels)
    event_types = list(set(all_data["event_type"]))
    random.seed(seed)
    train_list, val_list, test_list = [], [], []

    for etype in event_types:
        subset  = all_data.filter(lambda x: x["event_type"] == etype)
        n       = len(subset)
        indices = list(range(n))
        random.shuffle(indices)
        t_end   = int(n * 0.8)
        v_end   = int(n * 0.9)
        train_list.append(subset.select(indices[:t_end]))
        val_list.append(subset.select(indices[t_end:v_end]))
        test_list.append(subset.select(indices[v_end:]))

    train_ds = concatenate_datasets(train_list).select_columns(["tokens", "ner_tags"]).shuffle(seed=seed)
    val_ds   = concatenate_datasets(val_list).select_columns(["tokens", "ner_tags"]).shuffle(seed=seed)
    test_ds  = concatenate_datasets(test_list).select_columns(["tokens", "ner_tags"]).shuffle(seed=seed)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds

def tokenize_dataset(train_ds, val_ds, test_ds, tokenizer, max_length):
    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples["tokens"], truncation=True,
            max_length=max_length, is_split_into_words=True,
        )
        aligned = []
        for i, labels in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            prev, ids = None, []
            for wid in word_ids:
                if wid is None:      ids.append(-100)
                elif wid != prev:    ids.append(labels[wid])
                else:                ids.append(-100)
                prev = wid
            aligned.append(ids)
        tokenized["labels"] = aligned
        return tokenized

    tok_train = train_ds.map(tokenize_and_align, batched=True, remove_columns=train_ds.column_names)
    tok_val   = val_ds.map(tokenize_and_align,   batched=True, remove_columns=val_ds.column_names)
    tok_test  = test_ds.map(tokenize_and_align,  batched=True, remove_columns=test_ds.column_names)
    return tok_train, tok_val, tok_test

def compute_metrics(p):
    preds, labels = p
    preds       = np.argmax(preds, axis=2)
    true_preds  = [[ID2LABEL[p] for p, l in zip(pred, lab) if l != -100] for pred, lab in zip(preds, labels)]
    true_labels = [[ID2LABEL[l] for l in lab if l != -100] for lab in labels]
    return {
        "f1":        f1_score(true_labels, true_preds),
        "precision": precision_score(true_labels, true_preds),
        "recall":    recall_score(true_labels, true_preds),
    }

def run_baseline(test_ds):
    import re
    true_seqs, pred_seqs = [], []
    for sample in test_ds:
        tokens   = sample["tokens"]
        tags     = sample["ner_tags"]
        true_seq = [ID2LABEL[t] for t in tags]
        pred_seq = []
        i = 0
        while i < len(tokens):
            tok_clean = tokens[i].rstrip(".,").lower()
            if tok_clean in MONTHS:
                pred_seq.append("B-DATE")
                i += 1
                while i < len(tokens) and re.match(r"^\d{1,4}[,.]?$", tokens[i]):
                    pred_seq.append("I-DATE")
                    i += 1
            else:
                pred_seq.append("O")
                i += 1
        true_seqs.append(true_seq)
        pred_seqs.append(pred_seq)
    return {
        "entity_f1":        f1_score(true_seqs, pred_seqs),
        "entity_precision": precision_score(true_seqs, pred_seqs),
        "entity_recall":    recall_score(true_seqs, pred_seqs),
    }

def train_model(model_name, cfg, tok_train, tok_val, tok_test, tokenizer, train_ds):
    model  = AutoModelForTokenClassification.from_pretrained(
        cfg["base_model"], num_labels=len(LABEL_LIST),
        id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True
    )
    t_args = TrainingArguments(
        output_dir=f"/tmp/deadline-{model_name}",
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
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )
    print(f"Training {model_name} | lr={cfg['learning_rate']} | epochs={cfg['epochs']}...")
    t0           = time.time()
    train_result = trainer.train()
    train_time   = time.time() - t0
    print("Evaluating on test set...")
    predictions  = trainer.predict(tok_test)
    raw          = predictions.metrics
    test_result  = {
        "eval_f1":        raw.get("test_f1", 0),
        "eval_precision": raw.get("test_precision", 0),
        "eval_recall":    raw.get("test_recall", 0),
        "eval_loss":      raw.get("test_loss", 0),
    }
    print(f"Done! Time: {train_time:.1f}s | Test F1: {test_result['eval_f1']:.4f}")
    return trainer.model, train_result, test_result, train_time

def log_to_mlflow(model_name, cfg, model, train_result, test_result, train_time, train_ds, val_ds, test_ds):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({
            "model_name":     model_name,
            "base_model":     cfg.get("base_model", "regex"),
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
            "test_f1":              test_result.get("eval_f1", test_result.get("entity_f1", 0)),
            "test_precision":       test_result.get("eval_precision", test_result.get("entity_precision", 0)),
            "test_recall":          test_result.get("eval_recall", test_result.get("entity_recall", 0)),
            "test_loss":            test_result.get("eval_loss", 0),
        })
        if model is not None:
            pass  # model artifact skipped - metrics logged above
        print(f"Logged {model_name} → {MLFLOW_URI}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(CONFIGS.keys()),
                        help="Which model variant to run")
    args = parser.parse_args()

    set_seeds(42)
    cfg        = CONFIGS[args.model]
    train_ds, val_ds, test_ds = load_and_split_data(seed=42)

    if args.model == "baseline":
        t0      = time.time()
        metrics = run_baseline(test_ds)
        elapsed = time.time() - t0
        print(f"Baseline F1: {metrics['entity_f1']:.4f} | Time: {elapsed:.2f}s")
        log_to_mlflow(
            "baseline", {"base_model": "regex", "epochs": 0, "batch_size": 16,
                         "learning_rate": 0, "max_seq_length": 128, "fp16": False},
            None,
            None,
            {**metrics, "eval_f1": metrics["entity_f1"],
             "eval_precision": metrics["entity_precision"],
             "eval_recall": metrics["entity_recall"]},
            elapsed, train_ds, val_ds, test_ds
        )
        return

    tokenizer               = AutoTokenizer.from_pretrained(cfg["base_model"])
    tok_train, tok_val, tok_test = tokenize_dataset(train_ds, val_ds, test_ds, tokenizer, cfg["max_seq_length"])
    model, train_result, test_result, train_time = train_model(
        args.model, cfg, tok_train, tok_val, tok_test, tokenizer, train_ds
    )
    log_to_mlflow(args.model, cfg, model, train_result, test_result, train_time, train_ds, val_ds, test_ds)

if __name__ == "__main__":
    main()
