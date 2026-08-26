#!/usr/bin/env python3
"""FastAPI endpoint serving the trained invoice mismatch classifier."""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="InvoiceCheck API")

MODEL_PATH = "models/invoice_classifier_epoch10"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


class InvoiceRequest(BaseModel):
    text: str  # Extracted text fields from receipt


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/verify-invoice")
async def verify_invoice(req: InvoiceRequest):
    inputs = tokenizer(
        req.text, truncation=True, padding=True,
        max_length=256, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).numpy()[0]
    mismatch = int(probs.argmax())
    confidence = float(max(probs))
    return {
        "mismatch_detected": bool(mismatch),
        "confidence": round(confidence, 4),
        "recommendation": "Review manually" if mismatch else "Pass",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
