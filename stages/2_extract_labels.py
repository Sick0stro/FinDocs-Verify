#!/usr/bin/env python3
"""
Stage 2: Extract labeled fields from parsed_documents.csv
Creates train/validation splits with structured labels for LayoutLMv3 fine-tuning.
"""
import sys, json, ast
# Force clean sys.path - remove Hermes venv contamination
sys.path = [p for p in sys.path if 'hermes' not in p and 'venv' not in p.lower()]

import pandas as pd
from pathlib import Path

DATA_CSV = "D:/supabase_downloads/database/parsed_documents.csv"
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fields we want to extract from raw_json
TARGET_FIELDS = [
    "invoice", "invoice_date", "weight", "plastic_type",
    "vehicle_number", "bill_to_company_name", "bill_from_company_name",
    "bill_to_address", "weight_unit_of_mesurement"
]

def parse_raw_json(raw_json_str):
    """Parse raw_json string from CSV into structured dict using ast.literal_eval."""
    try:
        # Use ast.literal_eval for safe parsing of Python dict format
        raw = ast.literal_eval(raw_json_str) if isinstance(raw_json_str, str) else raw_json_str
        if isinstance(raw, dict):
            result = {}
            for k in TARGET_FIELDS:
                if k in raw:
                    v = raw[k]
                    if v is None or v == "None":
                        result[k] = ""
                    else:
                        result[k] = str(v)
            return result
    except (ValueError, SyntaxError) as e:
        return {}

def main():
    print("Loading ground truth data...")
    df = pd.read_csv(DATA_CSV)
    print(f"Total records: {len(df)}")
    
    # Filter only invoices with valid raw_json
    df = df[df["document_type"] == "invoice"]
    df = df.dropna(subset=["raw_json"])
    df = df[df["raw_json"].str.len() > 50]
    print(f"Invoice records with raw_json: {len(df)}")
    
    # Parse all records
    records = []
    for _, row in df.iterrows():
        fields = parse_raw_json(row["raw_json"])
        if fields.get("invoice"):  # Only keep if has invoice number
            record = {
                "id": str(row["id"]),
                "file_url": row.get("file_url", ""),
                "fields": fields,
                "has_invoice_no": bool(fields.get("invoice")),
                "has_weight": bool(fields.get("weight")),
                "has_vehicle_no": bool(fields.get("vehicle_number")),
                "has_company": bool(fields.get("bill_to_company_name")),
            }
            records.append(record)
    
    print(f"Valid records with invoice number: {len(records)}")
    
    if len(records) == 0:
        print("ERROR: No records with invoice numbers found!")
        # Debug: check raw_json parsing
        sample = df.iloc[0]
        print(f"\nDebug raw_json: {str(sample['raw_json'])[:200]}")
        parsed = parse_raw_json(sample["raw_json"])
        print(f"Debug parse result: {parsed}")
        return
    
    # Stats on field coverage
    inv_count = sum(1 for r in records if r["has_invoice_no"])
    weight_count = sum(1 for r in records if r["has_weight"])
    vehicle_count = sum(1 for r in records if r["has_vehicle_no"])
    company_count = sum(1 for r in records if r["has_company"])
    
    print(f"\n=== Field Coverage ===")
    print(f"Invoice numbers: {inv_count}/{len(records)} ({inv_count*100//len(records)}%)")
    print(f"Weights: {weight_count}/{len(records)} ({weight_count*100//len(records)}%)")
    print(f"Vehicle numbers: {vehicle_count}/{len(records)} ({vehicle_count*100//len(records)}%)")
    print(f"Company names: {company_count}/{len(records)} ({company_count*100//len(records)}%)")
    
    # Split: 90% train, 10% validation
    split_idx = int(len(records) * 0.9)
    train_records = records[:split_idx]
    val_records = records[split_idx:]
    
    train_path = OUTPUT_DIR / "layoutlmv3_train.json"
    val_path = OUTPUT_DIR / "layoutlmv3_val.json"
    
    with open(train_path, "w") as f:
        json.dump(train_records, f, indent=2, ensure_ascii=False)
    with open(val_path, "w") as f:
        json.dump(val_records, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Saved ===")
    print(f"Train: {len(train_records)} records -> {train_path}")
    print(f"Validation: {len(val_records)} records -> {val_path}")
    print(f"\nSample record:")
    print(json.dumps(train_records[0], indent=2, ensure_ascii=False)[:500])
    
    # Cross-reference with converted PNGs
    png_dir = Path("data/raw_pages")
    if png_dir.exists():
        png_files = {f.stem: f.name for f in png_dir.glob("*.png")}
        print(f"\n=== Cross-reference with PNGs ===")
        print(f"Converted PNGs so far: {len(png_files)}")
        
        # Build mapping: try both id and file_url stem
        matched_train = sum(1 for r in train_records 
                          if r["id"] in png_files 
                          or Path(r["file_url"]).stem in png_files)
        matched_val = sum(1 for r in val_records 
                        if r["id"] in png_files 
                        or Path(r["file_url"]).stem in png_files)
        total_matched = matched_train + matched_val
        print(f"Train records with PNG: {matched_train}/{len(train_records)}")
        print(f"Val records with PNG: {matched_val}/{len(val_records)}")
        print(f"Total matched: {total_matched}/{len(records)}")

if __name__ == "__main__":
    main()
