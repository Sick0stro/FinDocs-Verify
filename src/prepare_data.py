#!/usr/bin/env python3
"""Prepare training data from your existing receipts at D:/dataanalysys/metadata.json"""
import json
from pathlib import Path

# Your proprietary receipts
with open("D:/dataanalysys/metadata.json", encoding="utf-8-sig") as f:
    your_receipts = json.load(f)


def extract_receipt_data(receipts_json):
    """Extract structured fields from receipt JSON → training examples."""
    records = []
    for receipt in receipts_json:
        items_total = 0.0
        for item in receipt.get("items", []):
            try:
                qty = float(item.get("quantity") or 0)
                price = float(item.get("unit_price") or 0)
                items_total += qty * price
            except (ValueError, TypeError):
                pass
        try:
            receipt_total = float(receipt.get("receipt_total") or 0)
        except (ValueError, TypeError):
            receipt_total = 0.0

        # Only flag as mismatch if we have a valid receipt_total
        mismatch = (receipt_total > 0 and
                    abs(receipt_total - items_total) > 0.05 * max(items_total, 1.0))
        records.append({
            "vendor": (receipt.get("purpose_ref") or "unknown")[:100],
            "date": receipt.get("receipt_date", ""),
            "total": receipt_total,
            "items_total": round(items_total, 2),
            "label": int(mismatch),  # 1 = payment-term mismatch
            "raw_text": " ".join(
                item.get("name", "") for item in receipt.get("items", [])
            )[:500],
        })
    return records


your_data = extract_receipt_data(your_receipts)
print(f"Extracted {len(your_data)} records from your receipts")
print(f"  - Mismatches (label=1): {sum(r['label'] for r in your_data)}")
print(f"  - Clean (label=0): {len(your_data) - sum(r['label'] for r in your_data)}")

# Output JSONL for training
out_path = Path("data/processed/train_data.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    for r in your_data:
        f.write(json.dumps(r) + "\n")
print(f"Training data saved to {out_path}")
