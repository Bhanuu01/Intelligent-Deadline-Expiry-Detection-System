# Intelligent Deadline Expiry Detection System

This repository captures an earlier end-to-end version of the contract deadline extraction project that later grew into the fuller Datanauts workflow.

The core idea was to combine document classification and named-entity extraction so that contract clauses could be filtered first and then parsed for the dates or renewal terms that actually mattered.

## What this repo includes

- training scripts for the contract classifier
- training scripts for the NER model
- Ray Tune variants for hyperparameter search
- MLflow infrastructure for experiment tracking
- MinIO and PostgreSQL setup for artifact and metadata storage
- Kubernetes manifests for training and tuning jobs

## Main files

- `src/train_classifier (1).py`
  Training flow for the classification stage.
- `src/train_classifier_ray_tune (1).py`
  Ray Tune version for classifier search.
- `src/train_ner (1).py`
  Training flow for the NER stage.
- `src/train_ner_ray_tune (1).py`
  Ray Tune version for NER search.
- `docker-compose-mlflow.yaml`
  Local experiment-tracking stack with MLflow, MinIO, and PostgreSQL.
- `k8s-training.yaml`, `k8s-ray-tune.yaml`
  Kubernetes manifests for running training and tuning workloads.

## Why this repo matters

This project is a good example of how I was thinking about ML systems beyond a single model notebook. The useful part here was not only training models, but setting up the surrounding experiment and infrastructure flow so the work could be tracked, repeated, and moved into a larger pipeline.

## Stack

- Python
- MLflow
- Ray Tune
- Docker
- Kubernetes
- MinIO
- PostgreSQL

## Related project

The more complete public-facing version of this work is documented here:

- [Deadline Detection System](https://bhanuu01.github.io/projects/deadline-detection-system/)
