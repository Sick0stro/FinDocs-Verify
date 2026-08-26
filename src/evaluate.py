#!/usr/bin/env python3
"""Evaluate the trained model on held-out documents.

Produces standard outputs only — no custom PNG graphics:
  - artifacts/benchmark_report.md      (Markdown: metrics, classification report, confusion matrix)
  - artifacts/benchmark_summary.json   (machine-readable summary)
  - MLflow run "evaluate" with all metrics + artifacts (local store ./mlruns)

Run: python src/evaluate.py [--resume]
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    ARTIFACTS_DIR, ROOT, TRACKING_URI, cm_markdown, cls_report_markdown,
    init_mlflow, load_checkpoint, load_hf_evaluate, load_records, retry,
    safe_log, save_checkpoint, stratified_test_split, utc_now,
)
from viz import shape_text

MODEL_PATH = ROOT / "models" / "invoice_classifier_epoch10"
SUMMARY_PATH = ARTIFACTS_DIR / "benchmark_summary.json"
REPORT_PATH = ARTIFACTS_DIR / "benchmark_report.md"

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark FinDocs-Verify model")
    p.add_argument("--resume", action="store_true",
                   help="skip evaluation if benchmark_summary.json already exists")
    p.add_argument("--model", type=str, default=str(MODEL_PATH))
    return p.parse_args()


@retry(attempts=5, base_delay=2.0)
def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    return tokenizer, model


def hf_evaluate_metrics(labels, preds):
    """Metrics via the HF `evaluate` library (binary averaging), sklearn as offline fallback."""
    ev = load_hf_evaluate()

    def compute(name, *load_args, **load_kwargs):
        metric = ev.load(name, *load_args, **load_kwargs)
        kwargs = dict(predictions=preds, references=labels)
        try:
            return float(metric.compute(zero_division=0, **kwargs)[name])
        except TypeError:  # e.g. accuracy/F1 don't expose zero_division
            return float(metric.compute(**kwargs)[name])

    if ev is not None:
        try:
            return {
                "accuracy": compute("accuracy"),
                "precision": compute("precision", average="binary"),
                "recall": compute("recall", average="binary"),
                "f1": compute("f1", average="binary"),
                "source": "hf-evaluate",
            }
        except Exception as exc:  # noqa: BLE001 - hub unreachable -> sklearn
            print(f"[evaluate] metric computation failed ({exc}); falling back to scikit-learn",
                  flush=True)
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    p, r, f1s, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(p), "recall": float(r), "f1": float(f1s),
        "source": "scikit-learn",
    }


def predict(texts, tokenizer, model, max_length=256, batch_size=32):
    """Batched inference so long test sets don't blow up padding memory."""
    preds_all, probs_all = [], []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        inputs = tokenizer(chunk, truncation=True, padding=True,
                           max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds_all.extend(probs.argmax(axis=1).tolist())
        probs_all.extend(float(p.max()) for p in probs)
    return preds_all, probs_all


def build_report(summary, all_records, test_set, labels, preds, confs,
                 errors, cls_report_dict, device_name):
    n_pos_all = sum(r["label"] for r in all_records)
    n_pos_test = sum(labels)
    md = f"""# FinDocs-Verify Benchmark Report

_Generated {summary['generated_at']} — standard Markdown + JSON output (no rendered PNG tables)._

## Model
- Architecture: DistilBERT (fine-tuned on invoice mismatch detection)
- Model path: `{summary['model']}`
- Training corpus: {len(all_records)} documents ({n_pos_all} mismatches, {len(all_records) - n_pos_all} clean)
- Test set: {len(labels)} held-out documents ({n_pos_test} mismatch, {len(labels) - n_pos_test} clean)
- Device: {summary['device']} ({device_name})

## Metrics ({summary['metrics']['source']})

| Metric | Value |
|--------|-------|
| Precision | {summary['metrics']['precision']:.4f} |
| Recall | {summary['metrics']['recall']:.4f} |
| F1-Score | {summary['metrics']['f1']:.4f} |
| Accuracy | {summary['metrics']['accuracy']:.4f} |

## Classification Report

{cls_report_markdown(cls_report_dict)}

## Confusion Matrix

{cm_markdown(summary['confusion_matrix'])}

(True=positive=mismatch, Predicted=positive=mismatch)

- True Negatives (clean, correctly classified): {summary['confusion_matrix'][0][0]}
- True Positives (mismatch, correctly detected): {summary['confusion_matrix'][1][1]}
- False Negatives (mismatch, missed): {summary['confusion_matrix'][1][0]}
- False Positives (clean, flagged): {summary['confusion_matrix'][0][1]}

## Error Analysis
- False positives: {sum(1 for e in errors if e['pred_label'] == 1)}
- False negatives: {sum(1 for e in errors if e['pred_label'] == 0)}
- Confidence range: {min(confs):.4f} – {max(confs):.4f}
- Avg confidence: {sum(confs) / len(confs):.4f}

## Sample Predictions ({min(10, len(labels))} examples)

| # | True | Pred | Conf | Text preview |
|---|------|------|------|--------------|
"""
    for i in range(min(10, len(labels))):
        text_preview = test_set[i]["raw_text"][:120].replace("|", "\\>").replace("\n", " ")
        true_lbl = "MISMATCH" if labels[i] else "CLEAN"
        pred_lbl = "MISMATCH" if preds[i] else "CLEAN"
        md += f"| {i+1} | {true_lbl} | {pred_lbl} | {confs[i]:.4f} | {shape_text(text_preview)} |\n"
    return md


def main():
    args = parse_args()

    # Checkpoint / resume: skip work whose output already exists
    if args.resume and SUMMARY_PATH.exists():
        print(f"[resume] {SUMMARY_PATH.name} exists — nothing to do "
              f"(delete it or drop --resume to re-run)", flush=True)
        return

    records = load_records()
    test_records, stratified = stratified_test_split(records)
    mode = "stratified seed=42 split" if stratified else "last 10% holdout (matches train.py)"
    print(f"Evaluating on {len(test_records)} held-out documents ({mode})...", flush=True)

    labels = [r["label"] for r in test_records]
    texts = [r["raw_text"] for r in test_records]

    tokenizer, model = load_model_and_tokenizer(args.model)
    model.eval()

    preds, confs = predict(texts, tokenizer, model)

    metrics = hf_evaluate_metrics(labels, preds)
    cm = confusion_matrix(labels, preds)
    cls_report_dict = classification_report(
        labels, preds, zero_division=0, output_dict=True,
        target_names=["clean", "mismatch"])

    errors = [
        {
            "doc_index": i, "true_label": t, "pred_label": pr, "confidence": round(c, 4),
            "text_preview": texts[i][:150],
        }
        for i, (t, pr, c) in enumerate(zip(labels, preds, confs)) if t != pr
    ]

    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    summary = {
        "generated_at": utc_now(),
        "model": str(args.model),
        "device": device,
        "split_mode": mode,
        "dataset": {
            "total_docs": len(records),
            "test_docs": len(test_records),
            "test_mismatch": sum(labels),
            "test_clean": len(test_records) - sum(labels),
        },
        "metrics": metrics,
        "per_class": cls_report_dict,
        "confusion_matrix": cm.tolist(),
        "confidence": {"avg": sum(confs) / len(confs), "min": min(confs), "max": max(confs)},
        "errors": errors,
    }

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    save_checkpoint(SUMMARY_PATH, summary)          # incremental-safe JSON write
    report = build_report(summary, records, test_records, labels, preds, confs,
                          errors, cls_report_dict, device_name)
    REPORT_PATH.write_text(report, encoding="utf-8")

    # ---- MLflow ------------------------------------------------------------
    mlflow = init_mlflow()
    if mlflow:
        with mlflow.start_run(run_name="evaluate"):
            safe_log("log_params", params={
                "model": summary["model"], "test_docs": len(test_records),
                "split_mode": mode, "device": device,
            })
            safe_log("log_metrics", metrics={
                k: v for k, v in metrics.items() if isinstance(v, (int, float))
            })
            safe_log("log_metric", key="avg_confidence", value=summary["confidence"]["avg"])
            safe_log("log_artifact", local_path=str(SUMMARY_PATH))
            safe_log("log_artifact", local_path=str(REPORT_PATH))

    print(f"\n=== RESULTS ===")
    m = metrics
    print(f"Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f} | F1: {m['f1']:.4f} "
          f"| Acc: {m['accuracy']:.4f}")
    print(f"Confusion: TN={cm[0][0]} TP={cm[1][1]} FN={cm[1][0]} FP={cm[0][1]}")
    print(f"Report: {REPORT_PATH.relative_to(ROOT)}")
    print(f"Summary JSON: {SUMMARY_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
