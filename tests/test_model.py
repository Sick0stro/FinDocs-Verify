import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/invoice_classifier_epoch10"


def test_model_loads():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    inputs = tok(
        "3 x 50 = 150 but total shows 145",
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    outputs = model(**inputs)
    assert outputs.logits.shape == (1, 2)
    assert torch.isfinite(outputs.logits).all()


def test_batch_prediction():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    texts = [
        "3 x 50 = 150 but total shows 145",
        "Quantity: 2, Unit Price: 25.00, Total: 50.00",
    ]
    inputs = tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    assert outputs.logits.shape == (2, 2)
    probs = torch.softmax(outputs.logits, dim=-1)
    assert probs.shape == (2, 2)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)
