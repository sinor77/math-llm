"""
utils.py
========
Shared utilities: logging, checkpointing, reproducibility, metrics helpers,
and pretty-printing tools.
"""

import os
import sys
import json
import math
import time
import glob
import random
import logging
import hashlib
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str = "math_llm", level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger that writes timestamped messages to stdout.

    Usage
    -----
    log = get_logger()
    log.info("Training started")
    log.warning("Loss is NaN!")
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger   # already configured

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


log = get_logger()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for reproducibility.

    Sets seeds for Python's random module, NumPy, PyTorch CPU and GPU.
    Also enables deterministic CUDA operations where possible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic CUDA ops (may be slower)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    log.info(f"Random seed set to {seed}")


# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------

def get_device(preferred: str = "cuda") -> torch.device:
    """
    Return the best available device.

    preferred : "cuda" | "mps" | "cpu"
    Falls back gracefully if the preferred device is unavailable.
    """
    if preferred == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        name   = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"Using GPU: {name}  ({mem_gb:.1f} GB)")
    elif preferred == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        log.info("Using CPU — training will be slow; consider a GPU runtime")
    return device


# ---------------------------------------------------------------------------
# Learning-rate scheduler
# ---------------------------------------------------------------------------

def cosine_lr_with_warmup(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """
    Cosine learning-rate schedule with linear warm-up.

    During warm-up (step < warmup_steps):
        lr = max_lr * (step / warmup_steps)

    After warm-up:
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))

    where progress = (step - warmup_steps) / (max_steps - warmup_steps)

    This is the schedule used in many recent LLMs (GPT-3, LLaMA, etc.).
    """
    if step < warmup_steps:
        return max_lr * (step / max(warmup_steps, 1))
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    val_loss: float,
    cfg_model_dict: Dict,
    cfg_train_dict: Dict,
    tokenizer_path: str,
    checkpoint_dir: str = "checkpoints",
    keep_last_n: int = 3,
    is_best: bool = False,
    tag: str = "",
) -> str:
    """
    Save a training checkpoint.

    Saves:
        checkpoint_dir/step_{step:07d}{tag}.pt   ← per-step checkpoint
        checkpoint_dir/best_model.pt              ← overwritten when is_best=True

    Returns
    -------
    Path of the saved checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    filename = os.path.join(checkpoint_dir, f"step_{step:07d}{tag}.pt")
    payload = {
        "step":            step,
        "val_loss":        val_loss,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cfg_model":       cfg_model_dict,
        "cfg_train":       cfg_train_dict,
        "tokenizer_path":  tokenizer_path,
        "timestamp":       datetime.utcnow().isoformat(),
    }
    torch.save(payload, filename)
    log.info(f"Checkpoint saved → {filename}  (val_loss={val_loss:.4f})")

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pt")
        shutil.copy2(filename, best_path)
        log.info(f"  ↳ New best model saved → {best_path}")

    # Keep only the last N step checkpoints (not best_model.pt)
    _prune_checkpoints(checkpoint_dir, keep_last_n)
    return filename


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Load a checkpoint from disk.

    Parameters
    ----------
    path      : path to the .pt file
    model     : model to load weights into (must match saved architecture)
    optimizer : if provided, optimizer state is also restored
    device    : device to map tensors to (default: same device as model)

    Returns
    -------
    The full checkpoint dict (contains step, val_loss, cfg dicts, …)
    """
    map_loc = device or next(model.parameters()).device
    ckpt = torch.load(path, map_location=map_loc, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    log.info(f"Checkpoint loaded ← {path}  "
             f"(step={ckpt.get('step', '?')}  val_loss={ckpt.get('val_loss', '?'):.4f})")
    return ckpt


def _prune_checkpoints(checkpoint_dir: str, keep_last_n: int) -> None:
    """Delete old step checkpoints, keeping only the most recent N."""
    pattern = os.path.join(checkpoint_dir, "step_*.pt")
    checkpoints = sorted(glob.glob(pattern))
    if len(checkpoints) > keep_last_n:
        for old in checkpoints[:-keep_last_n]:
            os.remove(old)
            log.debug(f"Removed old checkpoint: {old}")


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def token_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Compute per-token accuracy, ignoring positions where label == -100.

    Parameters
    ----------
    logits : (B, T, V) — raw model outputs before softmax
    labels : (B, T)   — target token IDs (-100 for masked positions)

    Returns
    -------
    Accuracy as a float in [0, 1]
    """
    # Predicted token is the argmax of logits
    preds = logits.argmax(dim=-1)   # (B, T)

    # Only evaluate on non-masked positions
    mask  = labels != -100          # (B, T) bool
    if mask.sum() == 0:
        return 0.0

    correct = (preds == labels) & mask
    return correct.sum().item() / mask.sum().item()


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Return a dict with total, trainable, and non-trainable parameter counts."""
    total      = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen     = total - trainable
    return {"total": total, "trainable": trainable, "frozen": frozen}


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def save_results(results: Dict, path: str) -> None:
    """Save evaluation results dict to a JSON file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_serialisable)
    log.info(f"Results saved → {path}")


def _json_serialisable(obj: Any) -> Any:
    """JSON serialiser for objects that aren't JSON serialisable by default."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_table(rows: List[Dict], columns: Optional[List[str]] = None) -> None:
    """
    Print a list of dicts as a formatted table.

    Example:
        print_table([
            {"category": "add_or_sub", "accuracy": 0.95, "n": 500},
            {"category": "mul_or_div", "accuracy": 0.88, "n": 500},
        ])
    """
    if not rows:
        print("(empty table)")
        return

    cols = columns or list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}

    # Header
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep    = "  ".join("-" * widths[c] for c in cols)
    print(header)
    print(sep)
    for row in rows:
        line = "  ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols)
        print(line)


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as 'Xh Ym Zs'."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Experiment log
# ---------------------------------------------------------------------------

class ExperimentLog:
    """
    Lightweight CSV experiment tracker.

    Each call to log_experiment() appends a row to a CSV file.
    Results can be loaded and compared later.
    """

    def __init__(self, path: str = "results/experiment_log.csv"):
        self.path = path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    def log(self, record: Dict) -> None:
        """Append a result record to the CSV."""
        record["timestamp"] = datetime.utcnow().isoformat()
        file_exists = os.path.isfile(self.path)
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            import csv
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)
        log.info(f"Experiment logged → {self.path}")

    def load(self) -> List[Dict]:
        """Load all experiment records from the CSV."""
        if not os.path.isfile(self.path):
            return []
        import csv
        with open(self.path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    set_seed(42)
    dev = get_device()
    print(f"Device: {dev}")

    # LR schedule check
    lrs = [cosine_lr_with_warmup(s, 100, 1000, 3e-4, 3e-5) for s in range(0, 1100, 100)]
    print(f"LR schedule (every 100 steps): {[f'{x:.2e}' for x in lrs]}")

    # Token accuracy check
    logits = torch.randn(2, 10, 50)
    labels = torch.randint(0, 50, (2, 10))
    labels[0, :3] = -100
    acc = token_accuracy(logits, labels)
    print(f"Token accuracy (random baseline ~2%): {acc:.3f}")

    # Table printing
    print_table([
        {"category": "add_or_sub", "accuracy": "0.95", "n": "500"},
        {"category": "mul_or_div", "accuracy": "0.88", "n": "500"},
    ])
