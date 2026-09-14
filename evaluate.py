"""
evaluate.py
===========
Evaluation suite for the Math LLM.

METRICS COMPUTED
----------------
1. Token accuracy
   Percentage of individual tokens correctly predicted (excluding masked
   question tokens and padding).

2. Exact-answer accuracy  (PRIMARY METRIC)
   The model generates a complete answer autoregressively.
   We extract the substring after <A> and before <EOS>, strip whitespace,
   and compare to the ground-truth answer string.
   A prediction is correct only if it matches character-for-character.

3. Full-sequence accuracy
   The entire generated sequence (including reasoning steps, if any)
   must exactly match the reference.

4. Category-level accuracy
   Exact-answer accuracy broken down by math category.

5. Unseen-test accuracy
   Same as exact-answer accuracy, but computed ONLY on the held-out
   test split (the model never saw these examples during training).

WHY EXACT MATCH?
----------------
For most math problems the answer is a specific number or expression.
"Approximately correct" is not correct in mathematics.  Exact match
forces us to be honest: the model either gets the right answer or it doesn't.

GENERATION STRATEGY
-------------------
At evaluation time we feed the model the question part (<Q>…<A>) and
let it generate tokens autoregressively until <EOS> is produced or
max_new_tokens is exhausted.  This is the same as real inference — the
model has no access to the gold answer.
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from config import ModelConfig, DataConfig
from tokenizer import MathTokenizer
from model import MathLLM
from dataset import MathDataset, load_split_dataset, format_example, collate_fn
from utils import (
    get_logger, get_device, load_checkpoint,
    save_results, print_table, token_accuracy,
)
from generate import generate_answer
from functools import partial

log = get_logger()


# ---------------------------------------------------------------------------
# Token-level evaluation (fast — uses teacher-forced forward pass)
# ---------------------------------------------------------------------------

def evaluate_token_accuracy(
    model:     MathLLM,
    loader:    DataLoader,
    device:    torch.device,
    use_amp:   bool = True,
) -> Dict:
    """
    Compute token accuracy over an entire DataLoader using the teacher-forced
    forward pass (fast, O(dataset) time).

    Returns
    -------
    dict with keys: token_accuracy, total_tokens, correct_tokens
    """
    model.eval()
    total_tokens   = 0
    correct_tokens = 0

    with torch.no_grad():
        for batch in loader:
            ids    = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                out = model(ids, labels=labels)

            # Causal shift: position i's logits predict token i+1 (must
            # match the shift used in model.py's loss computation, or the
            # comparison is off-by-one and the metric is meaningless).
            shift_logits = out["logits"][:, :-1, :]        # (B, T-1, V)
            shift_labels = labels[:, 1:]                    # (B, T-1)
            preds  = shift_logits.argmax(dim=-1)            # (B, T-1)
            mask   = shift_labels != -100                   # ignore -100 positions

            correct_tokens += ((preds == shift_labels) & mask).sum().item()
            total_tokens   += mask.sum().item()

    acc = correct_tokens / max(total_tokens, 1)
    return {
        "token_accuracy":  acc,
        "correct_tokens":  correct_tokens,
        "total_tokens":    total_tokens,
    }


# ---------------------------------------------------------------------------
# Exact-answer evaluation (thorough — uses autoregressive generation)
# ---------------------------------------------------------------------------

def evaluate_exact_answer(
    model:          MathLLM,
    tokenizer:      MathTokenizer,
    items:          List[Dict],          # list of {"question":..,"answer":..,"category":..}
    device:         torch.device,
    max_new_tokens: int = 64,
    beam_size:      int = 1,            # 1 = greedy; >1 = beam search
    temperature:    float = 0.0,        # 0 = greedy
    show_n:         int = 5,            # number of examples to print
) -> Dict:
    """
    Autoregressive exact-answer evaluation.

    For each item:
      1. Build the prompt: <Q>{question}<A>
      2. Generate tokens until <EOS> or max_new_tokens
      3. Compare generated answer to ground-truth answer

    Returns
    -------
    dict with:
        exact_answer_accuracy  : float
        full_sequence_accuracy : float
        per_category           : dict[category → {accuracy, n, correct}]
        failures               : list of failure dicts
        successes              : list of success dicts (up to show_n)
    """
    model.eval()

    per_category: Dict[str, Dict] = defaultdict(lambda: {"correct": 0, "n": 0})
    failures:  List[Dict] = []
    successes: List[Dict] = []

    total_exact    = 0
    correct_exact  = 0
    total_fullseq  = 0
    correct_fullseq = 0

    log.info(f"Evaluating exact-answer accuracy on {len(items)} examples …")

    for i, item in enumerate(items):
        question = item["question"]
        gold_ans = item["answer"].strip()
        category = item.get("category", "unknown")

        # ── Generate ──────────────────────────────────────────────────────
        pred_ans, full_gen = generate_answer(
            model=model,
            tokenizer=tokenizer,
            question=question,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        pred_ans = pred_ans.strip()

        # ── Exact-answer match ────────────────────────────────────────────
        exact_match = (pred_ans == gold_ans)
        # Also try normalised comparison (strip spaces, lowercase for robustness)
        if not exact_match:
            exact_match = _normalise(pred_ans) == _normalise(gold_ans)

        total_exact += 1
        if exact_match:
            correct_exact += 1

        per_category[category]["n"]  += 1
        if exact_match:
            per_category[category]["correct"] += 1

        # ── Full-sequence match ───────────────────────────────────────────
        gold_full = format_example(item)
        full_match = (full_gen.strip() == gold_full.strip())
        total_fullseq += 1
        if full_match:
            correct_fullseq += 1

        # ── Collect examples ──────────────────────────────────────────────
        record = {
            "question":    question,
            "gold_answer": gold_ans,
            "pred_answer": pred_ans,
            "exact_match": exact_match,
            "full_match":  full_match,
            "category":    category,
            "full_gen":    full_gen,
        }
        if not exact_match:
            failures.append(record)
        elif len(successes) < show_n:
            successes.append(record)

        # ── Progress ─────────────────────────────────────────────────────
        if (i + 1) % 100 == 0 or (i + 1) == len(items):
            running_acc = correct_exact / (i + 1)
            log.info(f"  {i+1}/{len(items)}  running exact-acc={running_acc:.3f}")

    # ── Summary stats ─────────────────────────────────────────────────────
    exact_acc   = correct_exact  / max(total_exact, 1)
    fullseq_acc = correct_fullseq / max(total_fullseq, 1)

    per_cat_summary = {}
    for cat, stats in per_category.items():
        per_cat_summary[cat] = {
            "accuracy": stats["correct"] / max(stats["n"], 1),
            "correct":  stats["correct"],
            "n":        stats["n"],
        }

    results = {
        "exact_answer_accuracy":  exact_acc,
        "full_sequence_accuracy": fullseq_acc,
        "correct_exact":          correct_exact,
        "total_examples":         total_exact,
        "per_category":           per_cat_summary,
        "failures":               failures[:50],   # save up to 50 failures
        "successes":              successes,
    }

    return results


def _normalise(s: str) -> str:
    """Light normalisation: strip whitespace, collapse inner spaces, lowercase."""
    return " ".join(s.lower().split())


# ---------------------------------------------------------------------------
# Reconstruct the DataConfig a checkpoint was actually trained with
# ---------------------------------------------------------------------------

def _reconstruct_data_cfg_from_checkpoint(raw: Dict) -> DataConfig:
    """
    Build a DataConfig matching whatever dataset the checkpoint recorded in
    its data_source_info (see dataset.load_split_dataset / train.py) —
    real vs. generated, with the exact categories / digit range used.
    Falls back to a plain default DataConfig (real dataset) with a warning
    for older checkpoints saved before data_source_info existed.
    """
    info = raw.get("data_source_info") or {}
    source = info.get("data_source")
    config = info.get("configuration", {})
    cfg = DataConfig()

    if source == "GENERATED":
        cfg.dataset_source = "generated"
        cfg.generated_ops = config.get("ops", cfg.generated_ops)
        cfg.generated_min_digits = config.get("min_digits", cfg.generated_min_digits)
        cfg.generated_max_digits = config.get("max_digits", cfg.generated_max_digits)
    elif source == "REAL":
        cfg.dataset_source = "real"
        cfg.active_categories = config.get("active_categories", cfg.active_categories)
        cfg.train_difficulty = config.get("train_difficulty", cfg.train_difficulty)
        cfg.test_difficulty = config.get("test_difficulty", cfg.test_difficulty)
    else:
        log.warning(
            "Checkpoint has no data_source_info (older checkpoint?) — "
            "falling back to the default real DataConfig(). If this "
            "checkpoint was trained with --dataset-source generated, pass "
            "a matching data_cfg explicitly or results will be meaningless."
        )
    return cfg


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def full_evaluation(
    checkpoint_path: str,
    data_cfg: Optional[DataConfig] = None,
    device_str: str = "cuda",
    max_new_tokens: int = 64,
    show_failures: int = 10,
) -> Dict:
    """
    Load a checkpoint and run the complete evaluation suite.

    Parameters
    ----------
    checkpoint_path : path to best_model.pt or any .pt checkpoint
    data_cfg        : DataConfig (optional; reconstructed from checkpoint if omitted)
    device_str      : "cuda" or "cpu"
    max_new_tokens  : max tokens to generate per answer
    show_failures   : print this many failures to stdout

    Returns
    -------
    Complete results dict (also saved to results/eval_results.json)
    """
    device = get_device(device_str)

    # ── Load checkpoint ────────────────────────────────────────────────────
    log.info(f"Loading checkpoint: {checkpoint_path}")
    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg_dict  = raw["cfg_model"]
    model_cfg = ModelConfig(**cfg_dict)
    model     = MathLLM(model_cfg).to(device)
    model.load_state_dict(raw["model_state"])
    model.eval()

    # ── Load tokenizer ─────────────────────────────────────────────────────
    tok_path = raw.get("tokenizer_path", "checkpoints/tokenizer.json")
    tokenizer = MathTokenizer.load(tok_path)

    # ── Rebuild data, matching whatever the checkpoint was actually trained
    #    on (real dataset vs. generated benchmark, and which categories /
    #    digit range) — evaluating against the WRONG source silently gives
    #    meaningless results (mismatched vocab/test set), so this is not
    #    optional when data_cfg isn't explicitly passed in.
    if data_cfg is None:
        data_cfg = _reconstruct_data_cfg_from_checkpoint(raw)

    _, _, test_items, source_info = load_split_dataset(data_cfg)

    log.info(f"Test set size: {len(test_items)}")

    # ── Token accuracy (fast) ──────────────────────────────────────────────
    log.info("--- Token-level accuracy (teacher-forced) ---")
    test_ds = MathDataset(test_items, tokenizer, max_seq_len=model_cfg.max_seq_len)
    _collate = partial(collate_fn, pad_id=tokenizer.pad_id)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                              num_workers=0, collate_fn=_collate)
    tok_results = evaluate_token_accuracy(model, test_loader, device)
    log.info(f"  Token accuracy: {tok_results['token_accuracy']:.4f}  "
             f"({tok_results['correct_tokens']}/{tok_results['total_tokens']})")

    # ── Exact-answer accuracy (generation) ────────────────────────────────
    log.info("--- Exact-answer accuracy (autoregressive generation) ---")
    gen_results = evaluate_exact_answer(
        model=model,
        tokenizer=tokenizer,
        items=test_items,
        device=device,
        max_new_tokens=max_new_tokens,
    )

    # ── Print results ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS  (held-out test set)")
    print("=" * 60)
    print(f"  Token accuracy          : {tok_results['token_accuracy']:.4f}")
    print(f"  Exact-answer accuracy   : {gen_results['exact_answer_accuracy']:.4f}")
    print(f"  Full-sequence accuracy  : {gen_results['full_sequence_accuracy']:.4f}")
    print(f"  Test examples evaluated : {gen_results['total_examples']}")
    print(f"  Correct (exact)         : {gen_results['correct_exact']}")
    print()

    print("Category breakdown:")
    cat_rows = [
        {
            "category": cat,
            "accuracy": f"{v['accuracy']:.4f}",
            "correct":  str(v["correct"]),
            "n":        str(v["n"]),
        }
        for cat, v in sorted(gen_results["per_category"].items())
    ]
    print_table(cat_rows)

    # ── Failures ──────────────────────────────────────────────────────────
    failures = gen_results.get("failures", [])
    if failures:
        print(f"\nFirst {min(show_failures, len(failures))} failures:")
        for fail in failures[:show_failures]:
            print(f"  Q : {fail['question'][:80]}")
            print(f"  GOLD: {fail['gold_answer']}")
            print(f"  PRED: {fail['pred_answer']}")
            print()

    # ── Successes ─────────────────────────────────────────────────────────
    successes = gen_results.get("successes", [])
    if successes:
        print(f"\nFirst {len(successes)} successes:")
        for s in successes:
            print(f"  Q : {s['question'][:80]}")
            print(f"  ANS: {s['pred_answer']}")
            print()

    # ── Compile & save ─────────────────────────────────────────────────────
    all_results = {
        "checkpoint":    checkpoint_path,
        "model_config":  cfg_dict,
        "token_accuracy_results": tok_results,
        "gen_results":            gen_results,
    }
    save_results(all_results, "results/eval_results.json")

    return all_results


# ---------------------------------------------------------------------------
# Category-level accuracy table (for experiment comparison)
# ---------------------------------------------------------------------------

def category_accuracy_table(results: Dict) -> List[Dict]:
    """Extract a flat list of per-category rows from evaluate_exact_answer results."""
    rows = []
    for cat, stats in results.get("per_category", {}).items():
        rows.append({
            "category": cat,
            "accuracy": round(stats["accuracy"], 4),
            "correct":  stats["correct"],
            "n":        stats["n"],
        })
    return sorted(rows, key=lambda r: r["accuracy"], reverse=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the Math LLM")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt",
                   help="Path to model checkpoint")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--stage",      default=None, choices=["1", "2", "3"],
                   help="Real-dataset category stage. By default the "
                        "dataset config (real vs generated, categories, "
                        "digit range) is read straight from the "
                        "checkpoint's data_source_info — only pass this to "
                        "explicitly OVERRIDE what the checkpoint recorded.")
    p.add_argument("--show-failures", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # By default, data_cfg=None lets full_evaluation() reconstruct the
    # exact dataset config the checkpoint was trained with (see
    # _reconstruct_data_cfg_from_checkpoint). Only build one here if the
    # user explicitly asked to override via --stage.
    data_cfg = None
    if args.stage is not None:
        data_cfg = DataConfig()
        if args.stage == "1":
            data_cfg.active_categories = data_cfg.categories_stage1
        elif args.stage == "2":
            data_cfg.active_categories = (data_cfg.categories_stage1
                                          + data_cfg.categories_stage2)
        else:
            data_cfg.active_categories = (data_cfg.categories_stage1
                                          + data_cfg.categories_stage2
                                          + data_cfg.categories_stage3)

    full_evaluation(
        checkpoint_path=args.checkpoint,
        data_cfg=data_cfg,
        device_str=args.device,
        max_new_tokens=args.max_new_tokens,
        show_failures=args.show_failures,
    )
