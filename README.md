# FinDocs-Verify — ML Pipeline Project

> **Note**: This model is trained on **Arabic-language** receipt/invoice documents.

## Problem
Financial compliance teams manually verify thousands of invoices against payment terms — error-prone, slow, expensive. Banks and fintechs spend 1000s of hours annually chasing discrepancies, leading to overpayments, delayed payments, strained vendor relationships, and audit risks. Manual 3-way matching (PO vs goods receipt vs invoice) is slow and error-prone.

## Solution
Fine-tune **DistilBERT** to detect invoice/payment term mismatches in receipt text. Trained on 632 proprietary receipt documents with structural anomaly detection (sum of line items vs receipt total). Deployed as a **FastAPI endpoint** for real-time invoice verification.

**Model**: [Sickostro/FinDocs-Verify](https://huggingface.co/Sickostro/FinDocs-Verify) on HuggingFace

## Results

| Metric | Value |
|--------|-------|
| Precision | 61.54% |
| Recall | **100.00%** |
| F1-Score | **76.19%** |
| Accuracy | 84.13% |

### Training Impact

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| DistilBERT (base, untrained) | 25.4% | 100% | 40.5% |
| **Ours (fine-tuned)** | **61.5%** | **100%** | **76.2%** |

+35.7pp F1 improvement. 100% recall — every discrepancy in the test set is caught.

![Results Overview](artifacts/screenshots/results_overview.png)

## Quick Start

```bash
pip install -r requirements.txt
make train     # Fine-tune DistilBERT (GPU: ~43s, CPU: ~15min)
make eval      # Benchmark on held-out set
make test      # Run tests
make serve     # Start API on :8000
```

Or download the pre-trained model:
```bash
huggingface-cli download Sickostro/FinDocs-Verify --local-dir models/invoice_classifier_epoch10
```

## API

```bash
curl -X POST http://localhost:8000/verify-invoice \
  -H "Content-Type: application/json" \
  -d '{"text": "3 x 50 = 150, but total shows 145"}'
```

```json
{"mismatch_detected": true, "confidence": 0.92, "recommendation": "Review manually"}
```

## Project Structure

```
data/
  processed/          # JSONL training data (632 records)
models/               # Trained weights (267MB, not in repo)
  invoice_classifier_epoch10/
src/
  prepare_data.py     # Reads metadata.json -> train/val splits
  train.py            # Fine-tune DistilBERT (MLflow tracking, early stopping)
  evaluate.py         # Benchmark -> Markdown + JSON
  compare_models.py   # Baseline vs fine-tuned comparison
  serve.py            # FastAPI endpoint
  common.py           # Shared utilities (MLflow, paths, config)
  viz.py              # Loss chart generation
tests/
  test_model.py       # Model loads + predicts correctly
  test_data.py        # Data schema validation
artifacts/            # Reports, logs, charts
  benchmark_report.md
  model_comparison.md
  training_log.jsonl
  inference_samples.json
  device_info.json
  screenshots/
Makefile              # Quick commands
CHANGELOG.md          # Model version history
PIPELINE.md           # Detailed pipeline documentation
MODEL_CARD.md         # Model details + limitations
requirements.txt      # Python dependencies
.gitignore            # Excludes model weights + data
```

## Hardware

- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)
- **Training**: ~43 seconds (10 epochs, 568 train + 64 val docs)
- **Inference**: <1 second per document

## Pipeline

1. **Data**: 632 receipts from `D:/dataanalysys/metadata.json`
2. **Label**: Sum(qty × price) ≠ receipt_total → mismatch label
3. **Split**: 568 train / 64 test (stratified by label)
4. **Train**: DistilBERT fine-tuned (lr=5e-5, batch_size=16, 10 epochs)
5. **Evaluate**: Held-out benchmark with P/R/F1/confusion matrix
6. **Deploy**: FastAPI endpoint serving model weights

## Evidence

All artifacts saved to `artifacts/`:
- `benchmark_report.md` — Full precision/recall/confusion matrix report
- `model_comparison.md` — Fine-tuned vs untrained DistilBERT
- `training_log.jsonl` — Per-step loss logging
- `inference_samples.json` — 10 example predictions with confidence scores
- `device_info.json` — GPU + software versions
- `screenshots/` — Visual results for GitHub README

## About

Invoice mismatch detection via DistilBERT fine-tuned on 632 receipt documents. Detects structural anomalies where sum of line items doesn't match receipt totals.

*Trained on NVIDIA GeForce RTX 4060 Laptop GPU with CUDA 12.6 + PyTorch 2.12.0+cu126.*
