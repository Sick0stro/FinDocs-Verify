#!/usr/bin/env python3
"""Train invoice mismatch classifier on your receipts.
Uses GPU if available (RTX 4060), else CPU.
Run: python src/train.py [--resume] [--epochs N]

Tracking: MLflow (local store ./mlruns) logs every train/val metric per step;
artifacts/training_log.jsonl is appended incrementally so nothing is lost on crash.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback,
)

sys.path.insert(0, str(Path(__file__).parent))
from common import (ARTIFACTS_DIR, MODELS_DIR, ROOT, SEED, TRACKING_URI,
                    init_mlflow, safe_log)
from viz import extract_loss_points, plot_loss_curve

MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = ROOT / "data" / "processed" / "train_data.jsonl"
DEFAULT_OUTPUT = MODELS_DIR / "invoice_classifier_epoch10"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"   # resume checkpoints (latest kept only)
TRAINING_LOG = ARTIFACTS_DIR / "training_log.jsonl"

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune DistilBERT for invoice mismatch detection")
    p.add_argument("--resume", action="store_true",
                   help="resume from the last checkpoint under models/checkpoints if one exists")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--out", type=str, default=str(DEFAULT_OUTPUT),
                   help="where to save the final model weights")
    return p.parse_args()


def load_data(path):
    records = [json.loads(l) for l in open(path, encoding="utf-8")]
    return Dataset.from_dict({
        "text": [r["raw_text"] for r in records],
        "label": [r["label"] for r in records],
    })


class TrackAndSaveCallback(TrainerCallback):
    """Logs every training/eval event to MLflow AND appends it to the JSONL log immediately."""

    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _record(self, split, logs, step):
        entry = {"split": split, "step": step, **logs}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        metrics = {f"{split}/{k}": v for k, v in logs.items()
                   if isinstance(v, (int, float)) and k not in ("epoch", "step")
                   and not k.startswith("eval_")}
        safe_log("log_metrics", metrics=metrics, step=step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self._record("train", dict(logs), state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self._record("val", {k: v for k, v in metrics.items() if k != "eval_runtime"},
                         state.global_step)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {
        "accuracy": float((preds == labels).mean()),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


def main():
    args = parse_args()
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print("Loading data...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_data(DATA_PATH)
    print(f"Total samples: {len(dataset)}", flush=True)

    # Split: 90% train, 10% held-out
    split = dataset.train_test_split(test_size=0.1, seed=SEED)
    dataset = DatasetDict({"train": split["train"], "test": split["test"]})
    print(f"Train: {len(dataset['train'])} | Test: {len(dataset['test'])}", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    def preprocess(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

    dataset = dataset.map(preprocess)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Per-epoch evaluation -> validation loss curve alongside training loss.
    # Checkpoints are kept (latest only) so --resume can pick up after a crash.
    ta_kwargs = dict(
        output_dir=str(CHECKPOINT_DIR),
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        fp16=(device == "cuda"),
        dataloader_num_workers=0,
        seed=SEED,
    )
    try:
        training_args = TrainingArguments(**ta_kwargs)
    except TypeError:  # transformers <4.41 used 'evaluation_strategy'
        ta_kwargs["evaluation_strategy"] = ta_kwargs.pop("eval_strategy")
        training_args = TrainingArguments(**ta_kwargs)

    tracker_cb = TrackAndSaveCallback(TRAINING_LOG)
    early_stop_cb = EarlyStoppingCallback(early_stopping_patience=2)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
        callbacks=[tracker_cb, early_stop_cb],
    )

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    with open(ARTIFACTS_DIR / "device_info.json", "w") as f:
        json.dump({
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "torch_version": torch.__version__,
            "model": MODEL_NAME,
            "seed": SEED,
        }, f, indent=2)

    # ---- MLflow run -------------------------------------------------------
    mlflow = init_mlflow()
    run_ctx = (mlflow.start_run(run_name=f"train-distilbert-{args.epochs}ep")
               if mlflow else None)
    if mlflow:
        safe_log("log_params", params={
            "model": MODEL_NAME, "epochs": args.epochs, "learning_rate": args.learning_rate,
            "batch_size": 16, "max_length": 256, "seed": SEED,
            "train_samples": len(dataset["train"]), "eval_samples": len(dataset["test"]),
            "device": device, "resumed": bool(args.resume),
        })

    resume_from = args.resume  # None -> fresh; True -> Trainer finds last checkpoint
    if args.resume and not list(CHECKPOINT_DIR.glob("checkpoint-*")):
        print("[resume] no checkpoints found — starting fresh", flush=True)
        resume_from = None

    output_dir = Path(args.out)
    print("Starting training...", flush=True)
    trainer.train(resume_from_checkpoint=resume_from)

    # ---- Save model -------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}/", flush=True)

    # ---- Loss chart: training + validation on the same axes ---------------
    train_pts, val_pts = extract_loss_points(trainer.state.log_history)
    loss_png = plot_loss_curve(train_pts, val_pts, ARTIFACTS_DIR / "loss_curve.png",
                               title=f"DistilBERT fine-tune — train vs val loss ({args.epochs} epochs)")
    print(f"Loss curve (train + val): {loss_png}", flush=True)

    # ---- Final summary ----------------------------------------------------
    final_train = [l for l in trainer.state.log_history if "train_loss" in l]
    final_val = [l for l in trainer.state.log_history if "eval_loss" in l]
    summary = {}
    if final_train:
        t = final_train[-1]
        summary.update({"final_train_loss": t["train_loss"], "train_runtime_s": t["train_runtime"]})
    if final_val:
        summary["final_val_loss"] = final_val[-1]["eval_loss"]
        summary["final_val_f1"] = final_val[-1].get("eval_f1")
    if summary:
        safe_log("log_metrics", metrics={k: v for k, v in summary.items()})
        print(" | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                         for k, v in summary.items()), flush=True)

    if mlflow and run_ctx:
        for artifact in ["training_log.jsonl", "loss_curve.png", "device_info.json"]:
            safe_log("log_artifact", local_path=str(ARTIFACTS_DIR / artifact))
        (ARTIFACTS_DIR / "mlflow_run_id.txt").write_text(run_ctx.info.run_id)
        safe_log("end_run", status="FINISHED")
        print(f"MLflow run logged ({run_ctx.info.run_id}) -> view with: mlflow ui "
              f"--backend-store-uri {TRACKING_URI}", flush=True)


if __name__ == "__main__":
    main()
