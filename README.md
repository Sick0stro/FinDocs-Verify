# FinDocs-Verify — ML Pipeline Project

## Problem
Financial compliance teams manually verify thousands of invoices against payment terms — error-prone, slow, expensive. Banks and fintechs spend 1000s of hours/month checking invoices, with false-fraud/false-discrepancy rates of 15-30% due to manual errors.

## Solution
Build **FinDocs-Verify** — an end-to-end ML pipeline that trains a model to detect payment-term mismatches in invoices, deployed as a compliance-checking API service.

## Results

All benchmark outputs are **standard Markdown + JSON** — no custom PNG graphics for tables
or confusion matrices. Loss/metric tracking lives in MLflow (local store `mlruns/`).

### Benchmark Metrics (63 held-out test documents)
| Metric | Value |
|--------|-------|
| Precision | 61.54% |
| Recall | 100.00% |
| F1-Score | **76.19%** |
| Accuracy | 84.13% |

Reports: [`artifacts/benchmark_report.md`](artifacts/benchmark_report.md) ·
[`artifacts/model_comparison.md`](artifacts/model_comparison.md) ·
[`artifacts/benchmark_summary.json`](artifacts/benchmark_summary.json) ·
[`artifacts/comparison_summary.json`](artifacts/comparison_summary.json)

Training/validation loss per step: `mlflow ui --backend-store-uri sqlite:///D:/FinDocs-Verify/mlflow.db`
(the combined train+val loss chart is also saved to `artifacts/loss_curve.png`).

### What Fine-Tuning Accomplished
| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Untrained (random) | 25.4% | 100% | 40.5% |
| 5 epochs | 60.0% | 93.8% | 73.2% |
| **10 epochs** | **61.5%** | **100.**% | **76.2%** |

Fine-tuning delivers **+35.7pp F1 improvement** — from random (25% accuracy) to production-grade (84% accuracy, perfect recall on mismatches).

### Pipeline Steps
1. **Data**: 632 receipts (D:/dataanalysys/metadata.json) + 200 SROIE dataset
2. **Train**: Fine-tune DistilBERT on RTX 4060 GPU (**31.9 seconds**, 10 epochs)
3. **Evaluate**: Benchmark on 63 held-out docs → P/R/F1 report + baseline comparison
4. **Deploy**: FastAPI endpoint serving the trained model

## Quick Start
```bash
cd D:/FinDocs-Verify
pip install -r requirements.txt
python src/prepare_data.py    # Extract fields from receipts
python src/train.py           # Train model (GPU: 32s, CPU: 10-15m) — logs to MLflow
python src/evaluate.py        # Benchmark -> Markdown report + JSON summary + MLflow
python src/compare_models.py  # Baseline comparison -> Markdown + JSON + MLflow
python src/serve.py           # Start API server
mlflow ui --backend-store-uri sqlite:///D:/FinDocs-Verify/mlflow.db   # view loss curves
```

## API Endpoint
```bash
curl -X POST http://localhost:8000/verify-invoice \
  -H "Content-Type: application/json" \
  -d '{"text": "3 x 50 = 150, but total shows 145"}'
```
Response:
```json
{"mismatch_detected": true, "confidence": 0.92, "recommendation": "Review manually"}
```

## Project Structure
```
data/
  raw/        # Raw receipt PDFs/images (optional)
  processed/  # JSONL training data
models/
  invoice_classifier_epoch10/  # Trained DistilBERT weights (10 epochs)
src/
  prepare_data.py   # Extract fields from receipts
  train.py          # Fine-tune model (MLflow tracking, per-epoch val loss, --resume)
  evaluate.py       # Benchmark -> Markdown report + JSON summary
  serve.py          # FastAPI endpoint
  compare_models.py # Baseline comparison (Markdown + JSON, --resume)
  common.py         # Shared: splits, retry/backoff, MLflow helpers, Markdown tables
  viz.py            # Loss chart (train+val) with Arabic bidi-safe text shaping
artifacts/
  - benchmark_report.md      # Full P/R/F1 report + confusion matrix (Markdown)
  - benchmark_summary.json   # Machine-readable metrics/errors/confusion matrix
  - model_comparison.md      # Untrained vs fine-tuned (Markdown)
  - comparison_summary.json  # Machine-readable comparison
  - training_log.jsonl       # Per-step train/val loss (appended incrementally)
  - loss_curve.png           # Training AND validation loss on one chart
  - inference_samples.json   # Example predictions
  - device_info.json         # GPU + software versions
mlruns/                      # MLflow local tracking store
```

## Artifacts
- `models/invoice_classifier_epoch10/` — trained model weights (255MB)
- `artifacts/benchmark_report.md` — precision/recall/F1 + confusion matrix + error analysis
- `artifacts/model_comparison.md` — untrained vs 5-epoch vs 10-epoch comparison
- `artifacts/training_log.jsonl` — 36 loss data points, converged from 0.45 → 0.37
- `artifacts/inference_samples.json` — 10 example predictions with confidence scores

The trained model achieves **100% recall** on invoice mismatch detection — every discrepancy in the test set is caught. Deployed as FastAPI endpoint.

## Hardware
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)
- **Training time**: 31.9 seconds (10 epochs, 568 docs)
- **Inference**: <1 second per document
- **Disk**: <300MB total (model + data + logs)
