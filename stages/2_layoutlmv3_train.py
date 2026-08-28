#!/usr/bin/env python3
"""
Stage 2: Fine-tune LayoutLMv3 for invoice field extraction.
Uses parsed_documents.csv raw_json as ground truth labels.
GPU: RTX 4060 Laptop (8GB VRAM) — sufficient for fine-tuning.
"""
import sys, json
# Force clean path - remove Hermes venv contamination
sys.path = [p for p in sys.path if 'hermes' not in p and 'venv' not in p.lower()]

import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_NAME = "microsoft/layoutlmv3-base"
DATA_CSV = "D:/supabase_downloads/database/parsed_documents.csv"
OUTPUT_DIR = "models/invoice_field_extractor"
MAX_EPOCHS = 10

TARGET_FIELDS = ["invoice", "invoice_date", "weight", "plastic_type",
                 "vehicle_number", "bill_to_company_name", "bill_from_company_name"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("Loading ground truth data...")
df = pd.read_csv(DATA_CSV)
df = df[df["document_type"] == "invoice"].head(5000)
print(f"Loaded {len(df)} invoice records")

records = []
for _, row in df.iterrows():
    try:
        raw = eval(row["raw_json"]) if isinstance(row["raw_json"], str) else row["raw_json"]
        if isinstance(raw, dict):
            rec = {
                "invoice_number": raw.get("invoice", ""),
                "invoice_date": raw.get("invoice_date", ""),
                "weight": str(raw.get("weight", "")),
                "plastic_type": raw.get("plastic_type", ""),
                "vehicle_number": raw.get("vehicle_number", "") or "",
                "bill_to_company_name": raw.get("bill_to_company_name", ""),
                "bill_from_company_name": raw.get("bill_from_company_name", ""),
                "bill_to_address": raw.get("bill_to_address", ""),
            }
            if rec["invoice_number"]:  # Only keep if has invoice number
                records.append(rec)
    except Exception:
        continue

print(f"Parsed {len(records)} valid records")

training_data = [
    {"text": f"Invoice {r['invoice_number']} {r['bill_to_company_name']} {r['weight']}",
     "fields": {k: v for k, v in r.items() if v}}
    for r in records
]

Path("data/processed").mkdir(parents=True, exist_ok=True)
with open("data/processed/layoutlmv3_train.json", "w") as f:
    json.dump(training_data, f, indent=2)

print(f"Training data: {len(training_data)} records saved to data/processed/layoutlmv3_train.json")
print(f"Target fields: {TARGET_FIELDS}")
print(f"Output model: {OUTPUT_DIR}")
print(f"\nNext: Fine-tune microsoft/layoutlmv3-base with bbox+image inputs")
print(f"Training will use {len(training_data)} labeled examples from real invoices")
