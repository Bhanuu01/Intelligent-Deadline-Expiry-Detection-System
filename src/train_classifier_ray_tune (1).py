import os, time, random
import numpy as np
import torch
import mlflow
import ray
from ray import tune
from ray.train import RunConfig
from ray.tune import TuneConfig
from ray.tune.schedulers import ASHAScheduler
from datasets import concatenate_datasets, load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
)
from sklearn.metrics import f1_score as skf1, accuracy_score

os.environ["AWS_ACCESS_KEY_ID"]      = "datanauts-key"
os.environ["AWS_SECRET_ACCESS_KEY"]  = "datanauts-secret"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://129.114.27.190:9000"
os.environ["GIT_PYTHON_REFRESH"]     = "quiet"

MLFLOW_URI = "http://129.114.27.190:8000"
EXPERIMENT = "deadline-classifier-ray-tune"
BASE_MODEL = "roberta-base"

EVENT2ID = {"payment_due": 0, "expiration": 1, "effective": 2, "agreement": 3}
ID2EVENT = {v: k for k, v in EVENT2ID.items()}

# Create experiment once before Ray starts to avoid concurrent race condition
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)

def load_classifier_data(seed=42):
    ds       = load_dataset("tanvitakavane/datanauts_project_cuad-deadline-ner")
    all_data = concatenate_datasets([ds["train"], ds["test"]])
    all_data = all_data.filter(lambda x: "B-DATE" in x["ner_labels"])
    def prepare(example):
        example["label"] = EVENT2ID[example["event_type"]]
        return example
    all_data = all_data.map(prepare)
    all_data = all_data.select_columns(["sentence", "label"])
    all_data = all_data.shuffle(seed=seed)
    n         = len(all_data)
    train_end = int(n * 0.8)
    val_end   = int(n * 0.9)
    return (all_data.select(range(0, train_end)),
            all_data.select(range(train_end, val_end)),
            all_data.select(range(val_end, n)))

def compute_clf_metrics(p):
    preds  = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    return {
        "f1":       skf1(labels, preds, average="weighted", zero_division=0),
        "accuracy": accuracy_score(labels, preds),
    }

def train_trial(config):
    lr           = config["learning_rate"]
    batch_size   = config["batch_size"]
    warmup_ratio = config["warmup_ratio"]
    epochs       = config["epochs"]
    run_name     = f"clf_lr{lr:.1e}_bs{batch_size}_wr{warmup_ratio:.2f}_ep{epochs}"

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "learning_rate": lr, "batch_size": batch_size,
            "warmup_ratio": warmup_ratio, "epochs": epochs,
            "base_model": BASE_MODEL, "search_method": "Ray Tune ASHA",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "task": "sequence_classification", "num_labels": len(EVENT2ID),
        })

        train_ds, val_ds, test_ds = load_classifier_data(seed=42)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        def tokenize(examples):
            return tokenizer(examples["sentence"], truncation=True,
                             max_length=128, padding="max_length")

        tok_train = train_ds.map(tokenize, batched=True)
        tok_val   = val_ds.map(tokenize,   batched=True)
        tok_test  = test_ds.map(tokenize,  batched=True)

        for ds in [tok_train, tok_val, tok_test]:
            ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=len(EVENT2ID),
            id2label=ID2EVENT, label2id=EVENT2ID,
        )

        t_args = TrainingArguments(
            output_dir=f"/tmp/ray_clf_{run_name}",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            warmup_ratio=warmup_ratio,
            weight_decay=0.01,
            fp16=torch.cuda.is_available(),
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

        t0           = time.time()
        train_result = trainer.train()
        train_time   = time.time() - t0

        predictions = trainer.predict(tok_test)
        raw         = predictions.metrics
        test_f1     = raw.get("test_f1", 0)
        test_acc    = raw.get("test_accuracy", 0)

        mlflow.log_metrics({
            "test_f1":              test_f1,
            "test_accuracy":        test_acc,
            "test_loss":            raw.get("test_loss", 0),
            "total_train_time_sec": train_time,
            "time_per_epoch_sec":   train_time / epochs,
            "train_loss":           train_result.training_loss,
        })

    from ray import train as ray_train
    ray_train.report({"f1": test_f1})

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
        tune_config=TuneConfig(
            metric="f1", mode="max", num_samples=8,
            scheduler=scheduler, max_concurrent_trials=2,
        ),
        run_config=RunConfig(
            name="deadline_clf_ray_tune",
            storage_path="/tmp/ray_results",
        ),
    )
    print("\n=== Ray Tune ASHA: 8 trials on RoBERTa Classifier, 2 concurrent ===\n")
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
    with mlflow.start_run(run_name="RAY_TUNE_CLF_BEST_SUMMARY"):
        mlflow.log_params({**best.config, "search_method": "Ray Tune ASHA", "total_trials": len(results)})
        mlflow.log_metrics({"best_f1": best.metrics["f1"]})
    print(f"\nAll trials → {MLFLOW_URI} | Experiment: {EXPERIMENT}\n")
    ray.shutdown()

if __name__ == "__main__":
    main()
