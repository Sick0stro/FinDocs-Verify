---
language: en
tags:
- invoice-mismatch
- distilbert
- text-classification
- financial-compliance
metrics:
- precision
- recall
- f1
- accuracy
pipeline_tag: text-classification
library_name: transformers
model_name: distilbert-base-uncased
finetuned_from: distilbert-base-uncased
---

# FinDocs-Verify

Fine-tuned DistilBERT for detecting payment-term mismatches in invoices.

## Training Data

Proprietary dataset of receipt documents. Binary classification:
- **0** = Clean invoice (no mismatch)
- **1** = Mismatch detected (quantity × unit price ≠ total, etc.)

## Training

- **Base model**: `distilbert-base-uncased`
- **Epochs**: 10 (best at epoch 8 by eval_loss)
- **Batch size**: 16
- **Learning rate**: 5e-5
- **GPU**: NVIDIA RTX 4060 (8GB VRAM)
- **Training time**: ~43 seconds
- **Early stopping**: patience=2 on eval_loss

## Results (63 held-out test documents)

| Metric | Value |
|--------|-------|
| Precision | 61.54% |
| Recall | 100.00% |
| F1-Score | 76.19% |
| Accuracy | 84.13% |

Confusion matrix: TN=37, TP=16, FN=0, FP=10

100% recall — every discrepancy in the test set is caught.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained("Sickostro/FinDocs-Verify")
tokenizer = AutoTokenizer.from_pretrained("Sickostro/FinDocs-Verify")

text = "3 x 50 = 150 but total shows 145"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    pred = int(probs.argmax())

print({"mismatch": bool(pred), "confidence": float(max(probs))})
```

## API

```bash
pip install fastapi uvicorn
python src/serve.py

curl -X POST http://localhost:8000/verify-invoice \
  -H "Content-Type: application/json" \
  -d '{"text": "3 x 50 = 150, but total shows 145"}'
```

Response:
```json
{"mismatch_detected": true, "confidence": 0.92, "recommendation": "Review manually"}
```

## Limitations

- Trained on a small proprietary dataset
- Precision is moderate (61.5%) — some false positives on clean invoices
- DistilBERT has limited context window (512 tokens)

## License

MIT
