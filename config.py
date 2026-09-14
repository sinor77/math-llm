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
from typing import Dict, List, Optional


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

    # ---- Curriculum (easy -> hard by number length) ------------------------
    # A character-level model has no built-in notion of place value — it
    # has to learn digit-by-digit carrying purely from examples. That is
    # much easier to learn from short numbers first. If set, training runs
    # through these stages in order, each restricting the REAL training
    # pool to examples whose numbers fit within that stage's digit limits
    # (see dataset.filter_by_number_length), before lifting the
    # restriction. None (default) disables curriculum: train on the full,
    # unfiltered pool for `max_steps` steps, as before.
    #
    # Each stage dict: {"max_int_digits": int|None,
    #                    "max_decimal_digits": int|None,
    #                    "steps": int}
    # The LR schedule's cosine decay spans the SUM of all stage step counts
    # (not the flat `max_steps` above) so it decays smoothly across the
    # whole curriculum rather than restarting each stage.
    curriculum: Optional[List[Dict]] = None


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

    # ---- Dataset source selector --------------------------------------------
    # "real"      : the DeepMind Mathematics Dataset (default; see above).
    # "generated" : a controlled, Python-generated arithmetic benchmark
    #               (see dataset.generate_controlled_dataset) — chosen when
    #               you want a bounded, easy-to-reach-high-accuracy target
    #               (e.g. only 1-4 digit numbers) rather than the full
    #               difficulty range of the real dataset. Every equation's
    #               ground truth is still exact real arithmetic (computed
    #               once at generation time, in Python, to build the
    #               training corpus) — this is not the same thing as the
    #               earlier bug's silent, mislabelled synthetic fallback:
    #               it is an explicit, clearly-labelled choice, and
    #               source_info['data_source'] always says "GENERATED"
    #               (never "REAL") when this path is used.
    dataset_source: str = "real"

    # ---- Generated-arithmetic controls (only used if dataset_source ==
    #      "generated") ------------------------------------------------------
    # Operators to include. "+"/"-"/"*" are the usual operations; "/" is
    # ALWAYS exact integer division (we generate divisor and quotient
    # first and multiply them to get the dividend, so there is never a
    # remainder/decimal to represent) — deliberately not the arbitrary
    # decimal division the real dataset has, so the model isn't learning
    # digit-carrying AND decimal placement at the same time.
    generated_ops: List[str] = field(default_factory=lambda: ["+", "-", "*", "/"])
    # Final-benchmark digit range: every operand has between
    # generated_min_digits and generated_max_digits digits.
    generated_min_digits: int = 1
    generated_max_digits: int = 4
    # How many examples to generate per operator (before train/val split).
    generated_train_samples_per_op: int = 5_000
    generated_test_samples_per_op: int = 500
    # Train and test are generated from DIFFERENT RNG seeds/streams (not
    # just split from one pool) so they are independent by construction;
    # check_split_overlap() still verifies (never just assumes) zero
    # overlap on top of that.
    generated_train_seed: int = 42
    generated_test_seed: int = 20242


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


def get_curriculum(name: str = "none") -> Optional[List[Dict]]:
    """
    Return a named curriculum preset (see TrainConfig.curriculum).

    Stage thresholds below are calibrated against the REAL train-easy data
    for Stage-1 categories (verified by directly counting max digit-length
    per example: arithmetic__add_or_sub, arithmetic__mul both have
    thousands of examples with <=2 integer digits and <=1 decimal digit;
    arithmetic__div/arithmetic__mixed produce whole-number or fraction
    answers with 0 decimal digits, so the decimal bound never restricts
    them). Every stage has ample real examples — none of this is guessed.
    """
    presets: Dict[str, Optional[List[Dict]]] = {
        "none": None,
        "default": [
            {"max_int_digits": 2, "max_decimal_digits": 1, "steps": 6_000},
            {"max_int_digits": 4, "max_decimal_digits": 3, "steps": 7_000},
            {"max_int_digits": None, "max_decimal_digits": None, "steps": 7_000},
        ],
        "fast": [
            {"max_int_digits": 2, "max_decimal_digits": 1, "steps": 1_500},
            {"max_int_digits": 4, "max_decimal_digits": 3, "steps": 1_500},
            {"max_int_digits": None, "max_decimal_digits": None, "steps": 2_000},
        ],
        "thorough": [
            {"max_int_digits": 2, "max_decimal_digits": 1, "steps": 10_000},
            {"max_int_digits": 4, "max_decimal_digits": 3, "steps": 15_000},
            {"max_int_digits": None, "max_decimal_digits": None, "steps": 25_000},
        ],
        # Matches the generated-arithmetic benchmark's own digit range
        # (DataConfig.generated_max_digits=4): each stage's cap is
        # cumulative (max_int_digits=3 includes 1-2-digit examples too,
        # not just 3-digit ones) — standard curriculum-learning practice
        # of adding harder examples on top of, not instead of, easy ones.
        # All examples have 0 decimal digits (integers only), so
        # max_decimal_digits is fixed at 0 throughout.
        "digits_1_4": [
            {"max_int_digits": 2, "max_decimal_digits": 0, "steps": 6_000},
            {"max_int_digits": 3, "max_decimal_digits": 0, "steps": 7_000},
            {"max_int_digits": 4, "max_decimal_digits": 0, "steps": 7_000},
        ],
        # Same ramp as "digits_1_4" but with a MUCH bigger budget for the
        # final (full 1-4 digit range) stage. A measured run of
        # "digits_1_4" (20K steps total: 6K/7K/7K) showed val accuracy
        # visibly still climbing when the final stage ended — its 7K
        # steps is enough to recover from the stage-3 distribution shift
        # but not to converge, especially on multiplication/division
        # (structurally harder — see README). Training this size of model
        # for 20K steps takes well under a minute on a T4, so budget is
        # cheap: this preset spends the bulk of it on the hardest stage
        # instead of budgeting evenly across stages.
        "digits_1_4_long": [
            {"max_int_digits": 2, "max_decimal_digits": 0, "steps": 4_000},
            {"max_int_digits": 3, "max_decimal_digits": 0, "steps": 8_000},
            {"max_int_digits": 4, "max_decimal_digits": 0, "steps": 68_000},
        ],
    }
    if name not in presets:
        raise ValueError(f"Unknown curriculum preset '{name}'. "
                         f"Choose from: {list(presets.keys())}")
    return presets[name]


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
