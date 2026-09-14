"""
tiny_overfit.py
===============
MANDATORY sanity check (Phase 8 of the debugging plan): before any long
training run, prove the pipeline can memorize a small, controlled set of
REAL examples. If it cannot, the bug is in the pipeline (tokenizer,
label masking, target shifting, model forward pass, attention, causal
mask, loss, optimiser, generation) — not something a bigger model or a
longer run would fix.

This script:
  1. Loads ~60 REAL examples (from the actual DeepMind Mathematics
     Dataset, same source as full training — NOT synthetic).
  2. Trains a tiny model on ONLY these examples until loss is near zero.
  3. Reports initial loss, final loss, teacher-forced training accuracy.
  4. Separately tests autoregressive GENERATION on the memorized examples
     (Phase 10) — this distinguishes a training failure from a
     generation-only failure.

Usage:
    python tiny_overfit.py
"""

import random
import torch

from config import DataConfig, get_model_config
from dataset import (
    load_real_dataset, format_example, MathDataset, collate_fn,
)
from tokenizer import MathTokenizer
from model import MathLLM
from generate import generate_answer
from utils import set_seed, get_device, token_accuracy


def main():
    set_seed(42)
    device = get_device("cpu")   # tiny-overfit runs fine on CPU

    # ── 1. Load a small REAL subset ─────────────────────────────────────────
    data_cfg = DataConfig()
    data_cfg.active_categories = ["arithmetic__add_or_sub"]
    data_cfg.max_samples_per_category = 60
    data_cfg.max_test_samples_per_category = 10   # unused here, kept small

    print("=" * 60)
    print("PHASE 8 — TINY OVERFIT TEST (mandatory)")
    print("=" * 60)
    print("Loading ~60 REAL examples (arithmetic__add_or_sub, train-easy) …")
    train_pool, _test_pool = load_real_dataset(data_cfg)
    items = train_pool["arithmetic__add_or_sub"]
    random.Random(0).shuffle(items)
    items = items[:60]

    print(f"\nLoaded {len(items)} REAL examples. Sample:")
    for it in items[:5]:
        print(f"  Q: {it['question']!r}  ->  A: {it['answer']!r}")

    # ── 2. Tokenizer + dataset ───────────────────────────────────────────────
    tokenizer = MathTokenizer()
    tokenizer.build([format_example(it) for it in items])

    model_cfg = get_model_config("tiny-1M")
    model_cfg.vocab_size = tokenizer.vocab_size
    model_cfg.dropout = 0.0   # no regularisation — we WANT to memorize

    ds = MathDataset(items, tokenizer, max_seq_len=model_cfg.max_seq_len)
    assert len(ds) == len(items), "Some examples were dropped by MathDataset!"

    from functools import partial
    collate = partial(collate_fn, pad_id=tokenizer.pad_id)
    batch = collate([ds[i] for i in range(len(ds))])
    ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    # ── 3. Model + optimiser ──────────────────────────────────────────────────
    model = MathLLM(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    n_steps = 800
    model.train()
    losses = []
    accs = []
    for step in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(ids, labels=labels)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        accs.append(token_accuracy(out["logits"].detach(), labels))
        if step % 100 == 0 or step == n_steps - 1:
            print(f"  step {step:4d}  loss={loss.item():.4f}  "
                  f"tok_acc={accs[-1]:.3f}")

    initial_loss = losses[0]
    final_loss = losses[-1]
    final_tok_acc = accs[-1]

    # ── 4. Teacher-forced exact-sequence accuracy ────────────────────────────
    model.eval()
    with torch.no_grad():
        out = model(ids, labels=labels)
        preds = out["logits"].argmax(dim=-1)   # (B, T) predicts position i -> token i+1
    n_exact_teacher_forced = 0
    for i in range(len(items)):
        mask = labels[i, 1:] != -100
        shifted_preds = preds[i, :-1]
        shifted_labels = labels[i, 1:]
        if mask.sum() > 0 and torch.equal(shifted_preds[mask], shifted_labels[mask]):
            n_exact_teacher_forced += 1

    # ── 5. Phase 10 — test GENERATION separately from training ──────────────
    print("\n" + "=" * 60)
    print("PHASE 10 — GENERATION TEST on memorized examples")
    print("=" * 60)
    n_gen_correct = 0
    gen_records = []
    for it in items:
        pred_ans, full_gen = generate_answer(
            model, tokenizer, it["question"], device,
            max_new_tokens=32, temperature=0.0,
        )
        gold = it["answer"].strip()
        correct = pred_ans.strip() == gold
        n_gen_correct += int(correct)
        gen_records.append((it["question"], gold, pred_ans, full_gen, correct))

    print("\nSample generations (first 10):")
    for q, gold, pred, full_gen, correct in gen_records[:10]:
        print(f"  Prompt              : <Q>{q}<A>")
        print(f"  Decoded output      : {full_gen!r}")
        print(f"  Extracted answer    : {pred!r}")
        print(f"  Expected answer     : {gold!r}")
        print(f"  Correct             : {correct}")
        print()

    gen_acc = n_gen_correct / len(items)
    teacher_forced_acc = n_exact_teacher_forced / len(items)

    # ── 6. Final report ───────────────────────────────────────────────────────
    print("=" * 60)
    print("TINY OVERFIT TEST — RESULTS")
    print("=" * 60)
    print(f"  Examples                          : {len(items)} (REAL data)")
    print(f"  Training steps                    : {n_steps}")
    print(f"  Initial loss                      : {initial_loss:.4f}")
    print(f"  Final loss                        : {final_loss:.4f}")
    print(f"  Final teacher-forced token acc    : {final_tok_acc:.4f}")
    print(f"  Teacher-forced exact-sequence acc : {teacher_forced_acc:.4f} "
          f"({n_exact_teacher_forced}/{len(items)})")
    print(f"  Autoregressive generation acc     : {gen_acc:.4f} "
          f"({n_gen_correct}/{len(items)})")

    memorized = teacher_forced_acc >= 0.95
    generated = gen_acc >= 0.90
    print()
    if memorized and generated:
        print("RESULT: PASS — the model memorized the tiny dataset and "
              "generation reproduces it correctly.")
        print("        The pipeline (tokenizer -> targets -> loss -> model -> "
              "generation) is verified end-to-end.")
    elif memorized and not generated:
        print("RESULT: FAIL — training memorized the data (teacher-forced) "
              "but GENERATION does not reproduce it.")
        print("        This isolates the bug to generate.py / inference, "
              "NOT the training pipeline.")
    else:
        print("RESULT: FAIL — the model could not memorize even this tiny "
              "REAL dataset.")
        print("        Do NOT proceed to full training. The bug is in "
              "tokenizer/dataset/label-masking/model/loss.")

    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "final_tok_acc": final_tok_acc,
        "teacher_forced_exact_acc": teacher_forced_acc,
        "generation_acc": gen_acc,
        "passed": memorized and generated,
    }


if __name__ == "__main__":
    result = main()
    raise SystemExit(0 if result["passed"] else 1)
