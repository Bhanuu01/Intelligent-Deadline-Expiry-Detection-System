FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN pip install --no-cache-dir \
    transformers==4.40.0 \
    datasets==2.19.0 \
    mlflow==2.13.0 \
    seqeval \
    boto3 \
    psycopg2-binary \
    scikit-learn \
    faker \
    reportlab \
    huggingface-hub \
    accelerate

COPY src/ /app/src/

ENV PYTHONPATH=/app
ENV GIT_PYTHON_REFRESH=quiet

# Two training scripts:
# NER model:        python src/train_ner.py --model [baseline|bert_finetune_v1|bert_finetune_v2|bert_finetune_v3|bert_base_cased]
# Classifier model: python src/train_classifier.py --model [baseline|roberta_clf_v1|roberta_clf_v2|roberta_clf_v3]

CMD ["python", "src/train_ner.py", "--model", "bert_finetune_v1"]
