"""
experiments.py
==============
Automated experiment runner for the Math LLM.

PURPOSE
-------
Rather than manually trying different hyper-parameter combinations and
keeping notes in a text file, this module:

  1. Defines a set of sensible experiment configurations.
  2. Trains one model per configuration.
  3. Evaluates each on the held-out validation set.
  4. Records all results in a CSV file.
  5. Selects the best configuration by validation exact-answer accuracy.
  6. Runs a final evaluation on the test set with the best model.

CONTROLLED EXPERIMENTATION
--------------------------
We do NOT run a large random search.  Instead we use a staged approach:

  Stage A — baseline
    Establish a working pipeline with the smallest model.

  Stage B — model capacity
    Try larger models (5M, 10M params) with the same training setup.

  Stage C — training recipe
    For the best model size, try different LRs and batch sizes.

  Stage D — data coverage
    Add more categories (Stage 2 and 3 data) with the best model+recipe.

Each stage builds on the best result from the previous stage.
This is much more efficient than a full grid search.

READING THE RESULTS
--------------------
After running, see:
    results/experiment_log.csv    — one row per experiment
    checkpoints/                  — checkpoints for every run
    results/best_experiment.json  — the winner

TARGETING 100%
--------------
After finding the best configuration, the system re-trains with the full
budget and evaluates on the true test set.  If accuracy < 100%, it prints
an analysis of the failures and suggests next steps.
"""

import os
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch

from config import (
    ModelConfig, TrainConfig, DataConfig,
    get_model_config, get_train_config,
)
from tokenizer import MathTokenizer
from dataset import load_raw_data, split_data
from model import MathLLM
from train import train
from evaluate import evaluate_exact_answer, evaluate_token_accuracy, full_evaluation
from generate import generate_answer
from utils import (
    get_logger, set_seed, get_device,
    save_results, print_table, format_duration, ExperimentLog,
)

log = get_logger()


# ---------------------------------------------------------------------------
# Experiment specification
# ---------------------------------------------------------------------------

@dataclass
class ExperimentSpec:
    """
    A single experiment: named configuration with model + training + data settings.
    """
    name:        str
    model_name:  str          # preset name (tiny-1M, small-2M, …)
    train_preset: str         # default / fast / thorough
    data_stage:  int          # 1, 2, or 3
    max_steps:   Optional[int]  = None
    learning_rate: Optional[float] = None
    batch_size:  Optional[int]  = None
    dropout:     Optional[float] = None
    description: str = ""


# ---------------------------------------------------------------------------
# Experiment grid (staged, controlled)
# ---------------------------------------------------------------------------

EXPERIMENT_GRID: List[ExperimentSpec] = [

    # ── Stage A: baseline ──────────────────────────────────────────────────
    ExperimentSpec(
        name="baseline_small",
        model_name="small-2M",
        train_preset="default",
        data_stage=1,
        max_steps=10_000,
        description="Baseline: 2M-param model, Stage-1 data, default training",
    ),

    # ── Stage B: model capacity ────────────────────────────────────────────
    ExperimentSpec(
        name="medium_5M_s1",
        model_name="medium-5M",
        train_preset="default",
        data_stage=1,
        max_steps=10_000,
        description="5M-param model, Stage-1 data — does bigger help?",
    ),
    ExperimentSpec(
        name="large_10M_s1",
        model_name="large-10M",
        train_preset="default",
        data_stage=1,
        max_steps=10_000,
        description="10M-param model, Stage-1 data",
    ),

    # ── Stage C: training recipe ───────────────────────────────────────────
    ExperimentSpec(
        name="small_high_lr",
        model_name="small-2M",
        train_preset="default",
        data_stage=1,
        max_steps=10_000,
        learning_rate=1e-3,
        description="Higher LR — faster convergence?",
    ),
    ExperimentSpec(
        name="small_low_lr",
        model_name="small-2M",
        train_preset="default",
        data_stage=1,
        max_steps=10_000,
        learning_rate=1e-4,
        description="Lower LR — more stable?",
    ),
    ExperimentSpec(
        name="small_large_batch",
        model_name="small-2M",
        train_preset="default",
        data_stage=1,
        max_steps=10_000,
        batch_size=256,
        description="Larger batch — smoother gradients?",
    ),

    # ── Stage D: data coverage ─────────────────────────────────────────────
    ExperimentSpec(
        name="medium_5M_s2",
        model_name="medium-5M",
        train_preset="default",
        data_stage=2,
        max_steps=15_000,
        description="5M model + Stage-2 data (add comparison categories)",
    ),
    ExperimentSpec(
        name="medium_5M_s3",
        model_name="medium-5M",
        train_preset="default",
        data_stage=3,
        max_steps=20_000,
        description="5M model + all stages (algebra, probability, measurement)",
    ),

    # ── Stage E: extended training for best config ─────────────────────────
    ExperimentSpec(
        name="best_extended",
        model_name="medium-5M",
        train_preset="thorough",
        data_stage=3,
        max_steps=50_000,
        description="Extended training: best architecture, all data, full budget",
    ),
]


# ---------------------------------------------------------------------------
# Build configs from an ExperimentSpec
# ---------------------------------------------------------------------------

def build_configs_from_spec(
    spec: ExperimentSpec,
) -> Tuple[ModelConfig, TrainConfig, DataConfig]:
    """
    Convert an ExperimentSpec into (ModelConfig, TrainConfig, DataConfig).
    Applies any per-experiment overrides on top of presets.
    """
    model_cfg = get_model_config(spec.model_name)
    train_cfg = get_train_config(spec.train_preset)
    data_cfg  = DataConfig()

    # Overrides
    if spec.max_steps   is not None: train_cfg.max_steps     = spec.max_steps
    if spec.learning_rate is not None: train_cfg.learning_rate = spec.learning_rate
    if spec.batch_size  is not None: train_cfg.batch_size    = spec.batch_size
    if spec.dropout     is not None: model_cfg.dropout       = spec.dropout

    # Data stage
    if spec.data_stage == 1:
        data_cfg.active_categories = data_cfg.categories_stage1
    elif spec.data_stage == 2:
        data_cfg.active_categories = (data_cfg.categories_stage1
                                      + data_cfg.categories_stage2)
    else:
        data_cfg.active_categories = (data_cfg.categories_stage1
                                      + data_cfg.categories_stage2
                                      + data_cfg.categories_stage3)

    # Give each experiment its own checkpoint subdirectory
    train_cfg.checkpoint_dir = os.path.join("checkpoints", spec.name)

    return model_cfg, train_cfg, data_cfg


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    spec:    ExperimentSpec,
    exp_log: ExperimentLog,
    device:  str = "cuda",
    dry_run: bool = False,
) -> Dict:
    """
    Train and evaluate one experiment configuration.

    Parameters
    ----------
    spec    : ExperimentSpec to run
    exp_log : ExperimentLog for appending results
    device  : "cuda" or "cpu"
    dry_run : if True, skip actual training (just build model and report params)

    Returns
    -------
    Result dict with accuracy metrics.
    """
    log.info(f"\n{'='*60}")
    log.info(f"EXPERIMENT: {spec.name}")
    log.info(f"  {spec.description}")
    log.info(f"{'='*60}")

    model_cfg, train_cfg, data_cfg = build_configs_from_spec(spec)
    train_cfg.device = device

    t_start = time.time()

    if dry_run:
        # Just build the model and report param count
        model_cfg.vocab_size = 200  # approximate
        model = MathLLM(model_cfg)
        result = {
            "name":       spec.name,
            "params":     model.num_parameters(),
            "dry_run":    True,
        }
        exp_log.log(result)
        return result

    # ── Train ────────────────────────────────────────────────────────────
    history = train(model_cfg, train_cfg, data_cfg)

    train_duration = time.time() - t_start

    # ── Evaluate on validation set ────────────────────────────────────────
    best_ckpt = os.path.join(train_cfg.checkpoint_dir, "best_model.pt")
    if not os.path.isfile(best_ckpt):
        # Fallback: final model
        best_ckpt = os.path.join(train_cfg.checkpoint_dir, "final_model.pt")

    if not os.path.isfile(best_ckpt):
        log.warning(f"No checkpoint found at {best_ckpt}, skipping eval")
        result = {"name": spec.name, "error": "no checkpoint"}
        exp_log.log(result)
        return result

    # Load model for evaluation
    dev     = get_device(device)
    raw     = torch.load(best_ckpt, map_location=dev, weights_only=False)
    m_cfg   = ModelConfig(**raw["cfg_model"])
    model   = MathLLM(m_cfg).to(dev)
    model.load_state_dict(raw["model_state"])
    model.eval()

    tok_path  = raw.get("tokenizer_path", os.path.join("checkpoints", "tokenizer.json"))
    tokenizer = MathTokenizer.load(tok_path)

    # Rebuild val items
    by_category = load_raw_data(data_cfg)
    _, val_items, _ = split_data(by_category, data_cfg)

    gen_results = evaluate_exact_answer(
        model=model,
        tokenizer=tokenizer,
        items=val_items[:500],   # evaluate on up to 500 val examples for speed
        device=dev,
        max_new_tokens=64,
    )

    exact_acc = gen_results["exact_answer_accuracy"]
    val_loss  = min(history.get("val_loss", [float("inf")]))

    # ── Record result ─────────────────────────────────────────────────────
    result = {
        "name":               spec.name,
        "model":              spec.model_name,
        "data_stage":         spec.data_stage,
        "max_steps":          train_cfg.max_steps,
        "learning_rate":      train_cfg.learning_rate,
        "batch_size":         train_cfg.batch_size,
        "val_exact_acc":      round(exact_acc, 4),
        "val_loss_min":       round(val_loss, 4),
        "train_duration_s":   round(train_duration, 1),
        "n_params":           model.num_parameters(),
        "checkpoint":         best_ckpt,
        "description":        spec.description,
    }
    exp_log.log(result)

    log.info(f"\nResult for {spec.name}:")
    log.info(f"  val_exact_acc = {exact_acc:.4f}")
    log.info(f"  val_loss_min  = {val_loss:.4f}")
    log.info(f"  duration      = {format_duration(train_duration)}")

    return result


# ---------------------------------------------------------------------------
# Run all experiments
# ---------------------------------------------------------------------------

def run_all_experiments(
    experiments: Optional[List[ExperimentSpec]] = None,
    device: str = "cuda",
    stop_at_100: bool = True,
    dry_run: bool = False,
) -> Dict:
    """
    Run all experiments in sequence, track results, and identify the best.

    Parameters
    ----------
    experiments : list of ExperimentSpec (default: EXPERIMENT_GRID)
    device      : compute device
    stop_at_100 : stop early if 100% exact-answer accuracy is reached
    dry_run     : skip training (just build models and report params)

    Returns
    -------
    Summary dict with best experiment info and all results.
    """
    if experiments is None:
        experiments = EXPERIMENT_GRID

    os.makedirs("results", exist_ok=True)
    exp_log = ExperimentLog("results/experiment_log.csv")

    all_results  = []
    best_result  = None
    best_acc     = -1.0

    for spec in experiments:
        result = run_experiment(spec, exp_log, device=device, dry_run=dry_run)
        all_results.append(result)

        acc = result.get("val_exact_acc", 0.0)
        if acc > best_acc:
            best_acc    = acc
            best_result = result

        # Early-stop if we hit 100%
        if stop_at_100 and acc >= 1.0:
            log.info("100% validation accuracy reached — stopping experiment search.")
            break

    # ── Final test-set evaluation with the best model ────────────────────
    if best_result and not dry_run and best_result.get("checkpoint"):
        log.info(f"\nRunning FINAL TEST-SET evaluation with best model: "
                 f"{best_result['name']}")
        # Rebuild data config for best experiment
        spec = next(s for s in experiments if s.name == best_result["name"])
        _, _, data_cfg = build_configs_from_spec(spec)
        test_results = full_evaluation(
            checkpoint_path=best_result["checkpoint"],
            data_cfg=data_cfg,
            device_str=device,
        )
        best_result["test_exact_acc"] = test_results["gen_results"]["exact_answer_accuracy"]
        best_result["test_results"]   = test_results["gen_results"]

    # ── Save summary ─────────────────────────────────────────────────────
    summary = {
        "best_experiment": best_result,
        "all_results":     all_results,
        "best_val_acc":    best_acc,
    }
    save_results(summary, "results/best_experiment.json")

    # ── Print leaderboard ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT LEADERBOARD")
    print("=" * 70)
    sorted_results = sorted(
        [r for r in all_results if "val_exact_acc" in r],
        key=lambda r: r["val_exact_acc"], reverse=True,
    )
    print_table(
        sorted_results,
        columns=["name", "val_exact_acc", "val_loss_min", "n_params", "train_duration_s"],
    )
    print()

    if best_result:
        print(f"Best experiment : {best_result['name']}")
        print(f"Val exact acc   : {best_acc:.4f}")
        test_acc = best_result.get("test_exact_acc")
        if test_acc is not None:
            print(f"Test exact acc  : {test_acc:.4f}")
            _report_accuracy(test_acc, best_result)

    return summary


# ---------------------------------------------------------------------------
# Accuracy analysis
# ---------------------------------------------------------------------------

def _report_accuracy(test_acc: float, best_result: Dict) -> None:
    """
    Report whether 100% was reached and, if not, analyse what to improve.
    """
    if test_acc >= 1.0:
        print("\n" + "★" * 60)
        print("100% EXACT-ANSWER ACCURACY ACHIEVED ON UNSEEN TEST SET!")
        print("★" * 60)
        print(f"  Model         : {best_result.get('model')}")
        print(f"  Test examples : {best_result.get('test_results', {}).get('total_examples')}")
        print(f"  All examples are genuinely unseen (hash-partitioned split).")
        print("★" * 60)
    else:
        print(f"\n⚠  Test accuracy: {test_acc:.4f}  ({test_acc*100:.1f}%)")
        failures = best_result.get("test_results", {}).get("failures", [])
        print(f"   Failures      : {len(failures)}")

        if failures:
            print("\n   Sample failures:")
            for f in failures[:5]:
                print(f"     Q    : {f['question'][:70]}")
                print(f"     GOLD : {f['gold_answer']}")
                print(f"     PRED : {f['pred_answer']}")
                print()

        print("\n   IMPROVEMENT SUGGESTIONS:")
        if test_acc < 0.5:
            print("   → Accuracy < 50%: likely under-trained or model too small.")
            print("     Try: medium-5M model, more training steps (20K+), lower LR.")
        elif test_acc < 0.80:
            print("   → Accuracy 50–80%: model converges but struggles on harder cases.")
            print("     Try: larger model, more data (Stage 2/3), dropout reduction.")
        elif test_acc < 0.95:
            print("   → Accuracy 80–95%: good progress. Fine-tune on failure cases.")
            print("     Try: extended training (thorough preset), beam search decoding.")
        else:
            print("   → Accuracy > 95%: near-perfect. Hard examples may need")
            print("     curriculum learning or per-category data augmentation.")


# ---------------------------------------------------------------------------
# Failure analysis
# ---------------------------------------------------------------------------

def analyse_failures(results_path: str = "results/eval_results.json") -> None:
    """
    Load evaluation results and print a detailed failure analysis.
    """
    import json
    with open(results_path) as f:
        data = json.load(f)

    failures = data.get("gen_results", {}).get("failures", [])
    if not failures:
        print("No failures found! 🎉")
        return

    # Group by category
    by_cat: Dict[str, List] = {}
    for fail in failures:
        cat = fail.get("category", "unknown")
        by_cat.setdefault(cat, []).append(fail)

    print(f"\nTotal failures: {len(failures)}")
    print("\nBy category:")
    for cat, items in sorted(by_cat.items(), key=lambda x: -len(x[1])):
        print(f"  {cat}: {len(items)} failures")

    print("\nSample failures per category:")
    for cat, items in sorted(by_cat.items(), key=lambda x: -len(x[1])):
        print(f"\n  [{cat}]")
        for item in items[:3]:
            print(f"    Q   : {item['question'][:70]}")
            print(f"    GOLD: {item['gold_answer']}")
            print(f"    PRED: {item['pred_answer']}")


# ---------------------------------------------------------------------------
# Quick-start: single stage-1 baseline (for Colab / first run)
# ---------------------------------------------------------------------------

def quick_start(device: str = "cuda") -> Dict:
    """
    Run a single baseline experiment with small-2M on Stage-1 data.
    Good entry point for a first run on Colab.

    Returns
    -------
    result dict
    """
    exp_log = ExperimentLog("results/experiment_log.csv")
    spec    = ExperimentSpec(
        name="quick_start",
        model_name="small-2M",
        train_preset="default",
        data_stage=1,
        max_steps=10_000,
        description="Quick-start baseline",
    )
    return run_experiment(spec, exp_log, device=device)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Run Math LLM experiments")
    p.add_argument("--device",    default="cuda")
    p.add_argument("--quick",     action="store_true",
                   help="Run only the quick-start baseline")
    p.add_argument("--dry-run",   action="store_true",
                   help="Build models and log params without training")
    p.add_argument("--stages",    default="ABC",
                   help="Which experiment stages to run (e.g. AB or ABCDE)")
    p.add_argument("--analyse",   action="store_true",
                   help="Analyse failures in results/eval_results.json")
    args = p.parse_args()

    if args.analyse:
        analyse_failures()
    elif args.quick:
        quick_start(device=args.device)
    else:
        # Filter experiments by stage prefix if requested
        stage_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        stage_indices = [stage_map[c] for c in args.stages if c in stage_map]
        # Stage boundaries in EXPERIMENT_GRID
        stage_boundaries = [0, 1, 3, 6, 8, 9]   # start indices of stages A-E
        exps = []
        for si in stage_indices:
            lo = stage_boundaries[si]
            hi = stage_boundaries[si + 1] if si + 1 < len(stage_boundaries) else len(EXPERIMENT_GRID)
            exps.extend(EXPERIMENT_GRID[lo:hi])

        run_all_experiments(experiments=exps, device=args.device,
                            dry_run=args.dry_run)
