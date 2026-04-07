import os, time, random
import numpy as np
import torch
import mlflow
import ray
from ray import tune
from ray.train import RunConfig
from ray.tune import TuneConfig
from ray.tune.schedulers import ASHAScheduler
from datasets import Dataset, Features, Sequence, Value, concatenate_datasets, load_dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
)
from seqeval.metrics import f1_score, precision_score, recall_score

os.environ["AWS_ACCESS_KEY_ID"]      = "datanauts-key"
os.environ["AWS_SECRET_ACCESS_KEY"]  = "datanauts-secret"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://129.114.27.190:9000"
os.environ["GIT_PYTHON_REFRESH"]     = "quiet"

MLFLOW_URI = "http://129.114.27.190:8000"
EXPERIMENT = "deadline-detection-ray-tune"
BASE_MODEL = "dslim/bert-base-NER"
LABEL_LIST = ["O", "B-DATE", "I-DATE"]
LABEL2ID   = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL   = {i: l for i, l in enumerate(LABEL_LIST)}

# Force both datasets to use identical feature schema
FEATS = Features({"tokens": Sequence(Value("string")), "ner_tags": Sequence(Value("int64"))})

def generate_synthetic_invoices(n=300, seed=42):
    from faker import Faker
    fake = Faker()
    random.seed(seed)
    templates = [
        "Payment due by {date}.", "Invoice due on {date}.",
        "Amount owed must be paid by {date}.", "Please remit payment before {date}.",
        "Due date: {date}.", "Net 30 terms apply. Payment due {date}.",
        "Balance of ${amount} due by {date}.",
    ]
    samples = []
    for _ in range(n):
        date_str    = fake.date_between(start_date="+1d", end_date="+90d").strftime("%B %d, %Y")
        tmpl        = random.choice(templates)
        text        = tmpl.format(date=date_str, amount=random.randint(100, 9999))
        tokens      = text.split()
        date_tokens = date_str.split()
        labels, i   = [], 0
        while i < len(tokens):
            clean = tokens[i].rstrip(".,")
            if clean == date_tokens[0]:
                labels.append("B-DATE")
                for j in range(1, len(date_tokens)):
                    i += 1
                    labels.append("I-DATE")
            else:
                labels.append("O")
            i += 1
        samples.append({"tokens": tokens, "ner_tags": [LABEL2ID[l] for l in labels]})
    return Dataset.from_list(samples, features=FEATS)

def load_cuad_dates(max_samples=500, seed=42):
    ds = load_dataset("tanvitakavane/datanauts_project_cuad-deadline-ner", trust_remote_code=True)
    all_data = concatenate_datasets([ds["train"], ds["test"]])
    all_data = all_data.filter(lambda x: x["event_type"] == "expiration")
    all_data = all_data.filter(lambda x: "B-DATE" in x["ner_labels"])
    def convert(example):
        example["ner_tags"] = [LABEL2ID.get(l, 0) for l in example["ner_labels"]]
        return example
    all_data = all_data.map(convert)
    all_data = all_data.select_columns(["tokens", "ner_tags"])
    all_data = all_data.cast(FEATS)
    if max_samples and len(all_data) > max_samples:
        all_data = all_data.select(range(max_samples))
    return all_data.shuffle(seed=seed)

def build_splits(seed=42):
    combined  = concatenate_datasets([generate_synthetic_invoices(300, seed), load_cuad_dates(500, seed)]).shuffle(seed=seed)
    n         = len(combined)
    train_end = int(n * 0.80)
    val_end   = int(n * 0.90)
    return combined.select(range(0, train_end)), combined.select(range(train_end, val_end)), combined.select(range(val_end, n))

def tokenize_splits(train_ds, val_ds, test_ds, tokenizer, max_length=128):
    def align(examples):
        tok = tokenizer(examples["tokens"], truncation=True, max_length=max_length, is_split_into_words=True)
        aligned = []
        for i, labels in enumerate(examples["ner_tags"]):
            word_ids = tok.word_ids(batch_index=i)
            prev, ids = None, []
            for wid in word_ids:
                if wid is None: ids.append(-100)
                elif wid != prev: ids.append(labels[wid])
                else: ids.append(-100)
                prev = wid
            aligned.append(ids)
        tok["labels"] = aligned
        return tok
    cols = train_ds.column_names
    return (train_ds.map(align, batched=True, remove_columns=cols),
            val_ds.map(align,   batched=True, remove_columns=cols),
            test_ds.map(align,  batched=True, remove_columns=cols))

def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=2)
    true_preds  = [[ID2LABEL[p] for p, l in zip(pred, lab) if l != -100] for pred, lab in zip(preds, labels)]
    true_labels = [[ID2LABEL[l] for l in lab if l != -100] for lab in labels]
    return {"f1": f1_score(true_labels, true_preds), "precision": precision_score(true_labels, true_preds), "recall": recall_score(true_labels, true_preds)}

def train_trial(config):
    lr           = config["learning_rate"]
    batch_size   = config["batch_size"]
    warmup_ratio = config["warmup_ratio"]
    epochs       = config["epochs"]
    run_name     = f"tune_lr{lr:.1e}_bs{batch_size}_wr{warmup_ratio:.2f}_ep{epochs}"

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"learning_rate": lr, "batch_size": batch_size,
                           "warmup_ratio": warmup_ratio, "epochs": epochs,
                           "base_model": BASE_MODEL, "search_method": "Ray Tune ASHA",
                           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"})

        train_ds, val_ds, test_ds    = build_splits(seed=42)
        tokenizer                    = AutoTokenizer.from_pretrained(BASE_MODEL)
        tok_train, tok_val, tok_test = tokenize_splits(train_ds, val_ds, test_ds, tokenizer)

        model = AutoModelForTokenClassification.from_pretrained(
            BASE_MODEL, num_labels=len(LABEL_LIST), id2label=ID2LABEL,
            label2id=LABEL2ID, ignore_mismatched_sizes=True)

        t_args = TrainingArguments(
            output_dir=f"/tmp/ray_trial_{run_name}",
            num_train_epochs=epochs, per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size, learning_rate=lr,
            warmup_ratio=warmup_ratio, weight_decay=0.01,
            fp16=torch.cuda.is_available(), evaluation_strategy="epoch",
            save_strategy="epoch", load_best_model_at_end=True,
            metric_for_best_model="f1", logging_steps=20, report_to="none",
        )

        trainer = Trainer(model=model, args=t_args, train_dataset=tok_train,
                          eval_dataset=tok_val, tokenizer=tokenizer,
                          data_collator=DataCollatorForTokenClassification(tokenizer),
                          compute_metrics=compute_metrics)

        t0           = time.time()
        train_result = trainer.train()
        train_time   = time.time() - t0
        test_result  = trainer.evaluate(tok_test)
        test_f1      = test_result.get("eval_f1", 0)

        mlflow.log_metrics({"test_f1": test_f1,
                            "test_precision": test_result.get("eval_precision", 0),
                            "test_recall":    test_result.get("eval_recall", 0),
                            "total_train_time_sec": train_time,
                            "time_per_epoch_sec":   train_time / epochs,
                            "train_loss":           train_result.training_loss})

    from ray import train as ray_train; ray_train.report({"f1": test_f1})

def main():
    ray.init(ignore_reinit_error=True)

    search_space = {
        "learning_rate": tune.loguniform(1e-5, 1e-4),
        "batch_size":    tune.choice([8, 16]),
        "warmup_ratio":  tune.uniform(0.05, 0.20),
        "epochs":        tune.choice([3, 5]),
    }

    scheduler = ASHAScheduler(max_t=5, grace_period=1, reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_resources(train_trial, resources={"GPU": 0.5, "CPU": 4}),
        param_space=search_space,
        tune_config=TuneConfig(metric="f1", mode="max", num_samples=8,
                               scheduler=scheduler, max_concurrent_trials=2),
        run_config=RunConfig(name="deadline_ner_ray_tune", storage_path="/tmp/ray_results"),
    )

    print("\n=== Ray Tune ASHA: 8 trials on BERT NER, 2 concurrent ===\n")
    results = tuner.fit()

    best = results.get_best_result(metric="f1", mode="max")
    print(f"\n=== BEST TRIAL ===")
    print(f"  F1:            {best.metrics['f1']:.4f}")
    print(f"  learning_rate: {best.config['learning_rate']:.2e}")
    print(f"  batch_size:    {best.config['batch_size']}")
    print(f"  warmup_ratio:  {best.config['warmup_ratio']:.3f}")
    print(f"  epochs:        {best.config['epochs']}")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="RAY_TUNE_BEST_SUMMARY"):
        mlflow.log_params({**best.config, "search_method": "Ray Tune ASHA", "total_trials": len(results)})
        mlflow.log_metrics({"best_f1": best.metrics["f1"]})

    print(f"\nAll trials → {MLFLOW_URI} | Experiment: {EXPERIMENT}\n")
    ray.shutdown()

if __name__ == "__main__":
    main()
