# Changelog

## v1.0 — 2026-08-26
- **Model**: DistilBERT fine-tuned on 632 receipt documents
- **Training**: 10 epochs, RTX 4060, ~43s, `load_best_model_at_end` (best at epoch 8)
- **Eval** (63 held-out docs): P=0.6154 | R=1.0000 | F1=0.7619 | Acc=0.8413
- **Confusion**: TN=37 TP=16 FN=0 FP=10
- **Tracking**: MLflow local store (`sqlite:///mlflow.db`)
- **API**: FastAPI endpoint at `/verify-invoice` (`src/serve.py`)
