#!/usr/bin/env python3
"""Shared utilities: paths, data splits, retry/backoff, MLflow tracking, markdown tables."""
import json
import random
import sys
import time
import functools
from datetime import datetime, timezone
from pathlib import Path

SRC_DIR = Path(__file__).parent
ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "processed" / "train_data.jsonl"

SEED = 42


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Retry with exponential backoff (rate limits, connection errors, 5xx)
# ---------------------------------------------------------------------------

RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)


def retry(attempts=5, base_delay=2.0, max_delay=60.0):
    """Decorator: retry on transient network/server errors with exponential backoff."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except RETRYABLE_EXCEPTIONS as exc:
                    last_exc = exc
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    print(f"    [retry] {fn.__name__} failed ({exc}); "
                          f"attempt {attempt}/{attempts}, sleeping {delay:.0f}s", flush=True)
                    time.sleep(delay)
            raise last_exc
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Data loading / test split (shared by train, evaluate, compare_models)
# ---------------------------------------------------------------------------

def load_records(path=None):
    path = Path(path) if path else DATA_PATH
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return records


def stratified_test_split(records, frac=0.1, seed=SEED):
    """Last-frac holdout; falls back to stratified sampling if positives are missing there."""
    test_records = records[int(len(records) * (1 - frac)):]
    n_pos = sum(r["label"] for r in test_records)
    if n_pos > 0:
        return test_records, False

    mismatches = [r for r in records if r["label"] == 1]
    clean = [r for r in records if r["label"] == 0]
    rng = random.Random(seed)
    rng.shuffle(mismatches)
    rng.shuffle(clean)
    n_test_m = max(1, len(mismatches) // 10)
    n_test_c = max(1, len(clean) // 10)
    test_records = mismatches[:n_test_m] + clean[:n_test_c]
    rng.shuffle(test_records)
    return test_records, True


# ---------------------------------------------------------------------------
# Checkpoint / resume helpers
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def save_checkpoint(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# MLflow tracking (local SQLite store, optional-safe wrappers)
# ---------------------------------------------------------------------------

# MLflow >=3.15 requires a database backend (file store raises unless opted out).
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
ARTIFACT_LOCATION = (ROOT / "mlruns").as_uri()   # model/report artifacts live here
EXPERIMENT_NAME = "FinDocs-Verify"


def init_mlflow():
    """Point MLflow at the local sqlite store and select the project experiment.

    Returns the mlflow module, or None if mlflow is not installed (tracking
    degrades gracefully to on-disk artifacts only).
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        print("[mlflow] not installed — skipping tracker (pip install mlflow)", flush=True)
        return None
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
        client.create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACT_LOCATION)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow


def safe_log(fn_name, *args, **kwargs):
    """Call an mlflow.<log_*>() function without letting tracking failures kill the run."""
    import mlflow
    try:
        getattr(mlflow, fn_name)(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - tracking must never break training
        print(f"[mlflow] {fn_name} failed: {exc}", flush=True)


def load_hf_evaluate():
    """Import the Hugging Face `evaluate` library (the PyPI package).

    This repo ships its own src/evaluate.py, which shadows the installed
    library when scripts are run directly (`python src/evaluate.py`), because
    Python puts the script's directory first on sys.path. We temporarily strip
    the script dir so the real library wins resolution.
    """
    import importlib

    here = str(SRC_DIR.resolve())
    original = list(sys.path)
    try:
        filtered = []
        for p in original:
            if not p.strip():
                continue
            try:
                if str(Path(p).resolve()) == here:
                    continue
            except OSError:
                pass
            filtered.append(p)
        sys.path[:] = filtered
        sys.modules.pop("evaluate", None)
        mod = importlib.import_module("evaluate")
        if not hasattr(mod, "load"):
            raise ImportError(f"resolved module '{mod.__name__}' has no .load()")
        return mod
    except Exception as exc:  # noqa: BLE001 - caller falls back to scikit-learn
        print(f"[evaluate] HF evaluate unavailable ({exc})", flush=True)
        return None
    finally:
        sys.path[:] = original


# ---------------------------------------------------------------------------
# Markdown rendering helpers (no PNGs for tables/matrices anymore)
# ---------------------------------------------------------------------------

LABEL_NAMES = ["CLEAN", "MISMATCH"]


def cm_markdown(cm):
    """Render a 2x2 confusion matrix as a standard Markdown table."""
    header = "| Actual \\ Predicted | " + " | ".join(LABEL_NAMES) + " |"
    sep = "|" + "---|" * (len(LABEL_NAMES) + 1)
    rows = []
    for i, actual in enumerate(LABEL_NAMES):
        cells = " | ".join(str(int(v)) for v in cm[i])
        rows.append(f"| **{actual}** | {cells} |")
    return "\n".join([header, sep] + rows)


def cls_report_markdown(report_dict):
    """Render sklearn classification_report(output_dict=True) as a Markdown table."""
    lines = [
        "| Class | Precision | Recall | F1-Score | Support |",
        "|-------|-----------|--------|----------|---------|",
    ]
    for key, vals in report_dict.items():
        if not isinstance(vals, dict) or "precision" not in vals:
            continue
        support = int(vals.get("support", 0))
        lines.append(
            f"| `{key}` | {vals['precision']:.4f} | {vals['recall']:.4f} "
            f"| {vals['f1-score']:.4f} | {support} |"
        )
    return "\n".join(lines)
