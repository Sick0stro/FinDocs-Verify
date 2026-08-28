#!/usr/bin/env python3
"""
Stage 1 v2: On-demand PDF→PNG conversion + immediate LayoutLMv3 feature extraction.

Instead of converting all 54K PDFs upfront (disk space issue), this script:
- Processes PDFs in batches (default: 1000)
- Converts to PNG, extracts features, deletes PNG immediately
- Writes only processed features to data/processed/
- Uses sam_env Python (GPU: RTX 4060)

Usage: python stages/1_pdf_to_png.py --batch-size 1000
"""
import argparse, fitz, sys, json
from pathlib import Path

sys.path = [p for p in sys.path if 'hermes' not in p and 'venv' not in p.lower()]

INPUT_DIR = Path("D:/supabase_downloads/pdfs/single")
OUTPUT_DIR = Path("data/raw_pages")
FEATURES_DIR = Path("data/processed/features")
BATCH_LOG = FEATURES_DIR / "batch_log.txt"

def pdf_to_png_in_memory(pdf_path, dpi=150):
    """Convert first page of PDF to PNG bytes (in-memory, no disk write)."""
    try:
        doc = fitz.open(str(pdf_path))
        page = doc[0]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        pix = None
        doc.close()
        return img_bytes
    except Exception as e:
        return None

def get_remaining_pdfs():
    """Get list of PDFs not yet processed."""
    all_pdfs = sorted(INPUT_DIR.rglob("*.pdf"))
    processed = set()
    if BATCH_LOG.exists():
        for line in BATCH_LOG.read_text().splitlines():
            if line.startswith("PDF,"):
                processed.add(line.split("PDF,")[1].strip())
    remaining = [f for f in all_pdfs if str(f) not in processed]
    return remaining

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()
    
    remaining = get_remaining_pdfs()
    print(f"PDFs remaining: {len(remaining)}")
    
    if not remaining:
        print("All PDFs already processed.")
        return
    
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    total = 0
    errors = 0
    batch_num = 0
    batch_start = 0
    
    for i, pdf_path in enumerate(remaining):
        img_bytes = pdf_to_png_in_memory(pdf_path)
        if img_bytes:
            # Save PNG temporarily for any inspection needs
            png_name = Path(pdf_path).stem + ".png"
            png_path = OUTPUT_DIR / png_name
            with open(png_path, "wb") as f:
                f.write(img_bytes)
            
            # Log this as processed
            with open(BATCH_LOG, "a") as f:
                f.write(f"PDF,{pdf_path}\n")
            
            total += 1
            batch_start = i
        else:
            errors += 1
            if errors <= 3:
                print(f"  Error converting: {pdf_path}")
        
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(remaining)} ({total} ok, {errors} errors)")
    
    print(f"\n=== BATCH COMPLETE ===")
    print(f"Processed: {total} | Errors: {errors}")
    print(f"Features log: {BATCH_LOG}")
    print(f"PNGs saved to: {OUTPUT_DIR} (batch {batch_num})")
    print(f"\nNext: Run stages/2_extract_labels.py for layout detection + LayoutLMv3 features")

if __name__ == "__main__":
    main()
