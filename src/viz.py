#!/usr/bin/env python3
"""Plotting utilities: Arabic bidi shaping + training/validation loss chart.

Any text drawn on a plot goes through shape_text(), which applies proper
Arabic reshaping (isolated/joined glyph forms) and bidi reordering so Arabic
labels render correctly in matplotlib. An Arabic-capable font is selected
automatically (matplotlib's default DejaVu Sans has no Arabic glyphs).
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

try:
    import arabic_reshaper
    from bidi.algorithm import get_display

    _BIDI_AVAILABLE = True
except ImportError:
    _BIDI_AVAILABLE = False

# Unicode ranges covering the Arabic block, supplements and presentation forms
_ARABIC_RE = re.compile(
    "[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)

# Windows ships Tahoma/Arial/Segoe UI with full Arabic coverage
_PREFERRED_FONTS = [
    "Noto Naskh Arabic", "Amiri", "Tahoma", "Arial", "Segoe UI",
]


def shape_text(text):
    """Apply Arabic reshaping + bidi reordering when text contains Arabic."""
    if not isinstance(text, str) or not _ARABIC_RE.search(text):
        return text
    if not _BIDI_AVAILABLE:
        print("[viz] arabic-reshaper/python-bidi missing — rendering raw text", flush=True)
        return text
    return get_display(arabic_reshaper.reshape(text))


def configure_arabic_font():
    """Pick the first installed font that covers Arabic so shaped text isn't tofu boxes."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in _PREFERRED_FONTS:
        if name in installed:
            plt.rcParams["font.family"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_loss_curve(train_points, val_points, out_path,
                    title="Training & Validation Loss"):
    """Plot training and validation loss on the SAME chart.

    train_points / val_points: iterable of (epoch: float, loss: float).
    """
    out_path = Path(out_path)
    configure_arabic_font()
    fig, ax = plt.subplots(figsize=(9, 5.5))

    if train_points:
        xs, ys = zip(*sorted(train_points))
        ax.plot(xs, ys, marker="o", ms=3.5, lw=1.6, color="#1f77b4",
                label=shape_text("Training loss"))
    if val_points:
        xs, ys = zip(*sorted(val_points))
        ax.plot(xs, ys, marker="s", ms=5, lw=1.8, ls="--", color="#d62728",
                label=shape_text("Validation loss"))

    ax.set_title(shape_text(title))
    ax.set_xlabel(shape_text("Epoch"))
    ax.set_ylabel(shape_text("Cross-entropy loss"))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def extract_loss_points(log_history):
    """Pull (epoch, loss) pairs for train and validation splits from a HF log history.

    Training logs carry 'loss'; evaluation logs carry 'eval_loss'.
    """
    train_pts, val_pts = [], []
    for entry in log_history:
        epoch = entry.get("epoch")
        if epoch is None:
            continue
        if "loss" in entry:
            train_pts.append((float(epoch), float(entry["loss"])))
        if "eval_loss" in entry:
            val_pts.append((float(epoch), float(entry["eval_loss"])))
    return train_pts, val_pts
