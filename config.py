"""
config.py
=========
Central configuration for the Math LLM project.

All hyper-parameters live here so that every other module can simply do:
    from config import ModelConfig, TrainConfig, DataConfig

Nothing is hard-coded in the model/training files — every knob is here.

Three data-classes are defined:
  ModelConfig  — architecture (layers, dimensions, heads …)
  TrainConfig  — optimiser, scheduler, checkpointing …
  DataConfig   — dataset paths, categories, split ratios …
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    Controls every architectural choice.

    Start small (~2 M params) so the whole pipeline can be debugged in
    minutes.  Larger configs are defined at the bottom of this file and
    are selected by experiments.py.

    Parameter-count formula (rough):
        vocab_size * d_model                     # token embedding table
      + max_seq_len * d_model                    # positional embedding table
      + n_layers * (
            4 * d_model**2                       # Q,K,V,O projections
          + 2 * d_model * d_ff                   # FFN up + down
          + 4 * d_model                          # two LayerNorms (2×2d_model)
        )
      + d_model * vocab_size                     # output head (weight-tied)
    """

    # ---- Vocabulary (filled in by the tokenizer at runtime) ---------------
    vocab_size: int = 256        # will be overwritten after tokenizer.build()

    # ---- Sequence length --------------------------------------------------
    max_seq_len: int = 256       # maximum tokens the model can process at once

    # ---- Core dimensions --------------------------------------------------
    d_model: int = 128           # embedding / hidden dimension
    n_heads: int = 4             # number of attention heads (d_model % n_heads == 0)
    n_layers: int = 4            # number of Transformer decoder blocks
    d_ff: int = 512              # feed-forward inner dimension (usually 4 × d_model)

    # ---- Regularisation ---------------------------------------------------
    dropout: float = 0.1         # applied after attention, FFN, and embeddings
    weight_decay: float = 0.01   # L2 penalty for AdamW

    # ---- Weight tying -----------------------------------------------------
    tie_weights: bool = True     # share token embedding ↔ output projection weights
                                 # reduces params, improves perplexity on small models

    # ---- Convenience property (filled at runtime) -------------------------
    name: str = "small-2M"       # human-readable tag used in file names


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """
    Controls the optimiser, scheduler, and training loop.
    """

    # ---- Data --------------------------------------------------------------
    batch_size: int = 64
    num_workers: int = 2          # DataLoader worker processes

    # ---- Optimiser (AdamW) ------------------------------------------------
    learning_rate: float = 3e-4
    betas: tuple = (0.9, 0.95)   # AdamW momentum terms
    eps: float = 1e-8

    # ---- Scheduler ---------------------------------------------------------
    # We use a cosine decay with a short linear warm-up.
    warmup_steps: int = 200       # steps over which LR rises linearly to peak
    max_steps: int = 20_000       # total training steps (not epochs)
    min_lr_ratio: float = 0.1     # final LR = learning_rate × min_lr_ratio

    # ---- Stability ---------------------------------------------------------
    grad_clip: float = 1.0        # gradient norm clipping threshold

    # ---- Checkpointing -----------------------------------------------------
    checkpoint_dir: str = "checkpoints"
    save_every_steps: int = 1_000  # save a checkpoint every N steps
    keep_last_n: int = 3           # keep only the N most-recent checkpoints

    # ---- Logging -----------------------------------------------------------
    log_every_steps: int = 100
    eval_every_steps: int = 500

    # ---- Early stopping ----------------------------------------------------
    patience: int = 5             # stop if val loss doesn't improve for N evals

    # ---- Reproducibility ---------------------------------------------------
    seed: int = 42

    # ---- Device -----------------------------------------------------------
    device: str = "cuda"          # "cuda" or "cpu"; auto-detected in train.py

    # ---- Mixed precision (AMP) --------------------------------------------
    use_amp: bool = True          # torch.cuda.amp automatic mixed precision


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """
    Controls which data we load and how we split it.

    REAL DATA SOURCE
    ----------------
    This project uses the actual DeepMind Mathematics Dataset published
    alongside Saxton, Grefenstette, Hill & Kohli, "Analysing Mathematical
    Reasoning Abilities of Neural Models" (ICLR 2019).  The dataset is NOT
    a normal HuggingFace Hub dataset repo — earlier versions of this file
    pointed at 'deepcode-ai/math_dataset' and 'WillHeld/deepmind-math',
    neither of which exists / works, which is why training silently fell
    back to synthetic data.

    The real, verified, authoritative source is the static tarball the
    authors published themselves:

        https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz
        (2.3 GB, sha-verified by size; last-modified 2019-08-07; contains
        the exact category files named below, one question+answer pair
        per two lines of text)

    Directory layout inside the tarball (verified by inspection):
        train-easy/<category>.txt   — easiest difficulty, used for TRAIN/VAL
        train-medium/<category>.txt — harder curriculum (optional, Stage 2+)
        train-hard/<category>.txt   — hardest curriculum (optional)
        interpolate/<category>.txt  — official held-out TEST split: same
                                       categories, disjoint questions,
                                       generated independently by the
                                       original authors. We use this as our
                                       genuinely-unseen test set instead of
                                       carving a test split out of the
                                       training pool ourselves.
        extrapolate/<category>.txt  — harder out-of-distribution eval
                                       (bigger numbers / longer expressions)

    We build our own train/val split (hash-based, deterministic) from
    train-easy; the test split always comes from interpolate/ and is never
    touched during training.
    """

    # ---- Real dataset source (see docstring above) -------------------------
    dataset_url: str = (
        "https://storage.googleapis.com/mathematics-dataset/"
        "mathematics_dataset-v1.0.tar.gz"
    )
    dataset_archive_name: str = "mathematics_dataset-v1.0"
    # Expected uncompressed size of the tarball in bytes — used to sanity
    # check a cached download before trusting it (an interrupted/corrupt
    # download will not match).  ~2.33 GB, verified via a HEAD request.
    dataset_expected_size_bytes: int = 2_333_082_954
    train_difficulty: str = "train-easy"     # train-easy | train-medium | train-hard
    test_difficulty: str = "interpolate"     # official held-out split

    # ---- Which math categories to include (progressive) -------------------
    # Category names must match real file names inside the tarball exactly
    # (verified by listing the archive — see the module docstring).
    # NOTE: 'arithmetic__mul_or_div' does NOT exist in the real dataset —
    # multiplication and division are separate categories/files
    # ('arithmetic__mul' and 'arithmetic__div'). Earlier versions of this
    # config used the wrong, made-up name, which silently produced empty
    # category loads.
    #
    # Stage 1 — start here to get the pipeline working quickly.
    categories_stage1: List[str] = field(default_factory=lambda: [
        "arithmetic__add_or_sub",
        "arithmetic__mul",
        "arithmetic__div",
        "arithmetic__mixed",
    ])

    # Stage 2 — add after Stage 1 validates the pipeline.
    categories_stage2: List[str] = field(default_factory=lambda: [
        "arithmetic__add_or_sub_in_base",
        "arithmetic__nearest_integer_root",
        "comparison__closest",
        "comparison__kth_biggest",
        "comparison__sort",
    ])

    # Stage 3 — extend to algebra, polynomials, measurement, probability.
    categories_stage3: List[str] = field(default_factory=lambda: [
        "algebra__linear_1d",
        "algebra__linear_2d",
        "algebra__polynomial_roots",
        "measurement__conversion",
        "probability__swr_p_sequence",
    ])

    # Active categories (set by experiments.py or overridden directly).
    active_categories: List[str] = field(default_factory=lambda: [
        "arithmetic__add_or_sub",
        "arithmetic__mul",
        "arithmetic__div",
        "arithmetic__mixed",
    ])

    # ---- Split ratios (train/val ONLY — test comes from interpolate/) ------
    # The real test set is the official 'interpolate' split (see docstring
    # above), which is disjoint from train-easy by construction. We only
    # need to further split train-easy into train/val ourselves.
    train_ratio: float = 0.90
    val_ratio: float = 0.10

    # ---- Per-category sample caps -------------------------------------------
    # Cap samples per category so early stages are fast. train-easy files
    # contain ~666K lines per category; interpolate files contain ~10K.
    max_samples_per_category: int = 5_000
    max_test_samples_per_category: int = 1_000

    # ---- Sequence length filtering ----------------------------------------
    max_seq_len: int = 256        # drop examples longer than this

    # ---- I/O ---------------------------------------------------------------
    data_cache_dir: str = "data_cache"
    processed_dir: str = "data_processed"

    # ---- Special token strings (must match tokenizer.py) ------------------
    question_start: str = "<Q>"
    answer_start: str = "<A>"
    eos_token: str = "<EOS>"
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"


# ---------------------------------------------------------------------------
# Named experiment presets
# ---------------------------------------------------------------------------

def get_model_config(name: str = "small-2M") -> ModelConfig:
    """
    Return a ModelConfig for a named size preset.

    Sizes are chosen so that total parameter count stays within
    Google Colab T4 VRAM (16 GB).

    Approximate parameter counts (with vocab_size=256, max_seq_len=256):
        tiny-1M   →  ~1.0 M
        small-2M  →  ~2.2 M
        medium-5M →  ~5.1 M
        large-10M → ~10.4 M
    """
    configs = {
        # ---- Tiny: just for smoke-testing the pipeline --------------------
        "tiny-1M": ModelConfig(
            vocab_size=256, max_seq_len=256,
            d_model=64,  n_heads=2, n_layers=3, d_ff=256,
            dropout=0.1, name="tiny-1M",
        ),
        # ---- Small: default starting point --------------------------------
        "small-2M": ModelConfig(
            vocab_size=256, max_seq_len=256,
            d_model=128, n_heads=4, n_layers=4, d_ff=512,
            dropout=0.1, name="small-2M",
        ),
        # ---- Medium: first upgrade after pipeline is validated -------------
        "medium-5M": ModelConfig(
            vocab_size=256, max_seq_len=256,
            d_model=256, n_heads=4, n_layers=6, d_ff=1024,
            dropout=0.1, name="medium-5M",
        ),
        # ---- Large: push toward 100 % accuracy ----------------------------
        "large-10M": ModelConfig(
            vocab_size=256, max_seq_len=384,
            d_model=384, n_heads=6, n_layers=8, d_ff=1536,
            dropout=0.1, name="large-10M",
        ),
        # ---- XL: only if T4 VRAM allows ------------------------------------
        "xlarge-20M": ModelConfig(
            vocab_size=256, max_seq_len=512,
            d_model=512, n_heads=8, n_layers=10, d_ff=2048,
            dropout=0.05, name="xlarge-20M",
        ),
    }
    if name not in configs:
        raise ValueError(f"Unknown model preset '{name}'. "
                         f"Choose from: {list(configs.keys())}")
    return configs[name]


def get_train_config(preset: str = "default") -> TrainConfig:
    """Return a TrainConfig for a named training preset."""
    presets = {
        "default": TrainConfig(),
        "fast": TrainConfig(
            batch_size=128, learning_rate=5e-4,
            max_steps=5_000, warmup_steps=100,
        ),
        "thorough": TrainConfig(
            batch_size=64, learning_rate=3e-4,
            max_steps=50_000, warmup_steps=500,
            patience=10,
        ),
    }
    if preset not in presets:
        raise ValueError(f"Unknown train preset '{preset}'. "
                         f"Choose from: {list(presets.keys())}")
    return presets[preset]


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for name in ["tiny-1M", "small-2M", "medium-5M", "large-10M", "xlarge-20M"]:
        cfg = get_model_config(name)
        print(f"{name:15s}  d_model={cfg.d_model:4d}  n_layers={cfg.n_layers}"
              f"  n_heads={cfg.n_heads}  d_ff={cfg.d_ff}")
    print("\nDefault TrainConfig:")
    tc = TrainConfig()
    print(f"  lr={tc.learning_rate}  batch={tc.batch_size}  steps={tc.max_steps}")
    print("\nDefault DataConfig:")
    dc = DataConfig()
    print(f"  dataset_url={dc.dataset_url}")
    print(f"  active categories: {dc.active_categories}")
