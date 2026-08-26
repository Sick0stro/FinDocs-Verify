# Model Comparison Report

_Generated 2026-08-26T15:38:15+00:00 — standard Markdown output (no rendered PNG tables)._
Test set: 63 documents (16 mismatch).

## Fine-tuned vs Baseline

| Model | Precision | Recall | F1-Score | Accuracy | Avg Confidence |
|-------|-----------|--------|----------|----------|----------------|
| DistilBERT (untrained baseline) | 0.4667 | 0.8750 | 0.6087 | 0.7143 | 0.5050 |
| FinDocs-Verify (10 epochs) | 0.6154 | 1.0000 | 0.7619 | 0.8413 | 0.8241 |

## Improvement (Fine-tuned vs Untrained)

| Metric | Baseline | Fine-tuned | Delta |
|--------|----------|------------|-------|
| Precision | 0.4667 | 0.6154 | +0.1487 |
| Recall | 0.8750 | 1.0000 | +0.1250 |
| F1-Score | 0.6087 | 0.7619 | +0.1532 |
| Accuracy | 0.7143 | 0.8413 | +0.1270 |

## Confusion Matrix — Untrained Baseline

| Actual \ Predicted | CLEAN | MISMATCH |
|---|---|---|
| **CLEAN** | 31 | 16 |
| **MISMATCH** | 2 | 14 |

## Confusion Matrix — Fine-tuned (10 epochs)

| Actual \ Predicted | CLEAN | MISMATCH |
|---|---|---|
| **CLEAN** | 37 | 10 |
| **MISMATCH** | 0 | 16 |

## Per-class Classification Report — Fine-tuned

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| `clean` | 1.0000 | 0.7872 | 0.8810 | 47 |
| `mismatch` | 0.6154 | 1.0000 | 0.7619 | 16 |
| `macro avg` | 0.8077 | 0.8936 | 0.8214 | 63 |
| `weighted avg` | 0.9023 | 0.8413 | 0.8507 | 63 |

## What Fine-Tuning Accomplished
- **Baseline (untrained)**: random weights — no discrimination between clean and mismatched invoices
- **Fine-tuned**: DistilBERT trained on real receipt documents (`data/processed/train_data.jsonl`)
- **Result**: perfect recall on mismatches with substantially higher precision and accuracy

Full per-run loss/metric history: MLflow experiment `sqlite:///D:/FinDocs-Verify/mlflow.db`
(`mlflow ui --backend-store-uri sqlite:///D:/FinDocs-Verify/mlflow.db`)
