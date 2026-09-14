"""
train.py
========
Complete training loop for the Math LLM.

WHAT HAPPENS DURING TRAINING?
------------------------------
Training is a repeated cycle of four steps:

  1. FORWARD PASS
     Feed a batch of token sequences into the model.
     The model outputs logits (unnormalised scores) for every token position.

  2. LOSS COMPUTATION
     Compare the model's predictions against the true next-tokens using
     cross-entropy loss.  The loss is a single number: higher means worse.
     We only compute loss over the ANSWER part of each sequence (the
     question tokens are masked with -100).

  3. BACKWARD PASS (backpropagation)
     Automatically compute the gradient of the loss with respect to every
     learnable parameter.  Gradients tell us which direction to move each
     parameter to reduce the loss.

  4. OPTIMISER STEP
     Adjust every parameter a small amount in the direction that reduces
     the loss (AdamW optimiser).  The learning rate controls how large
     that step is.

  After enough cycles, the model's parameters converge to values that
  produce low loss — i.e. the model has learned to predict the next token
  in a mathematical answer sequence.

TRAINING FEATURES IMPLEMENTED
------------------------------
  • Cosine LR schedule with linear warm-up
  • Gradient clipping (prevents exploding gradients)
  • Mixed-precision training (torch.amp) for GPU speed
  • Validation loss evaluated periodically
  • Checkpoint saving (best model + periodic)
  • Early stopping (optional)
  • Comprehensive logging
  • Resume from checkpoint
"""

import os
import time
import argparse
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── project imports ──────────────────────────────────────────────────────────
from functools import partial

from torch.utils.data import DataLoader

from config import ModelConfig, TrainConfig, DataConfig, get_model_config, get_train_config, get_curriculum
from tokenizer import MathTokenizer
from dataset import (
    RealDatasetLoadError,
    load_split_dataset, format_example, filter_by_number_length,
    MathDataset, collate_fn,
)
from model import MathLLM
from utils import (
    get_logger, set_seed, get_device, cosine_lr_with_warmup,
    save_checkpoint, load_checkpoint, token_accuracy,
    save_results, format_duration,
)

log = get_logger()


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train(
    model_cfg:  ModelConfig,
    train_cfg:  TrainConfig,
    data_cfg:   DataConfig,
    resume_from: Optional[str] = None,
) -> Dict:
    """
    Full training run.

    Parameters
    ----------
    model_cfg   : architecture config
    train_cfg   : optimiser / schedule config
    data_cfg    : dataset config
    resume_from : path to a checkpoint to resume from (optional)

    Returns
    -------
    dict with training history and final metrics
    """
    # ── Reproducibility & device ────────────────────────────────────────────
    set_seed(train_cfg.seed)
    device = get_device(train_cfg.device)

    # ── Data ────────────────────────────────────────────────────────────────
    # Loaded once as raw item lists (not DataLoaders yet) so a curriculum
    # can re-filter train/val by number length between stages without
    # re-downloading/re-extracting the real dataset.
    log.info("Loading real dataset …")
    train_items, val_items, test_items, source_info = load_split_dataset(data_cfg)

    # Tokenizer is built once on the FULL, unfiltered pool (train+val+test)
    # so the vocabulary is complete regardless of which curriculum stage is
    # active — number-length filtering only changes which examples are
    # used, never which characters can appear (digits/operators are the
    # same whether a number is short or long).
    tokenizer = MathTokenizer()
    all_texts = [format_example(it) for it in (train_items + val_items + test_items)]
    tokenizer.build(all_texts)
    model_cfg.vocab_size = tokenizer.vocab_size

    # Save tokenizer right away so it's available for generation/eval.
    # IMPORTANT: this must live inside train_cfg.checkpoint_dir, not a
    # hardcoded "checkpoints/" — otherwise every run (e.g. each experiment
    # in experiments.py, which gives each a distinct checkpoint_dir) would
    # overwrite the SAME shared tokenizer file, silently corrupting the
    # tokenizer_path recorded in earlier checkpoints (they'd still "load"
    # successfully but decode with the wrong, later vocab mapping).
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    tok_path = os.path.join(train_cfg.checkpoint_dir, "tokenizer.json")
    tokenizer.save(tok_path)

    # ── Model ───────────────────────────────────────────────────────────────
    log.info("Building model …")
    model = MathLLM(model_cfg).to(device)

    # ── Optimiser ───────────────────────────────────────────────────────────
    # AdamW separates weight decay from gradient updates (unlike Adam).
    # We only apply weight decay to weight matrices, NOT to biases or
    # LayerNorm parameters (standard practice).
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:          # weight matrices
            decay_params.append(p)
        else:                    # biases, LayerNorm weights
            no_decay_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": model_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=train_cfg.learning_rate,
        betas=train_cfg.betas,
        eps=train_cfg.eps,
    )

    # ── Mixed-precision scaler ───────────────────────────────────────────────
    use_amp = train_cfg.use_amp and device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Resume from checkpoint ───────────────────────────────────────────────
    start_step = 0
    best_val_loss = float("inf")
    if resume_from:
        ckpt = load_checkpoint(resume_from, model, optimizer, device)
        start_step    = ckpt.get("step", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        log.info(f"Resuming from step {start_step}")

    # ── Training state ───────────────────────────────────────────────────────
    history = {
        "train_loss":   [],
        "val_loss":     [],
        "val_tok_acc":  [],
        "lr":           [],
        "steps":        [],
    }
    patience_counter = 0
    step = start_step
    epoch = 0

    # Compute min LR for the schedule
    min_lr = train_cfg.learning_rate * train_cfg.min_lr_ratio

    # ── Curriculum stages ────────────────────────────────────────────────────
    # Without a curriculum, this is a single "stage" spanning the whole
    # unfiltered pool for train_cfg.max_steps — identical to the old
    # behaviour. With one, we train through progressively less-restricted
    # slices of the REAL data (see dataset.filter_by_number_length), never
    # synthetic/altered examples — just a different subset at each stage.
    if train_cfg.curriculum:
        stages = train_cfg.curriculum
    else:
        stages = [{"max_int_digits": None, "max_decimal_digits": None,
                   "steps": train_cfg.max_steps}]
    total_steps = sum(s["steps"] for s in stages)

    # ── Helper: set learning rate for this step ──────────────────────────────
    # The cosine decay spans total_steps (the FULL curriculum), not any one
    # stage's step count, so LR decays smoothly across stage transitions
    # instead of restarting.
    def update_lr(step: int) -> float:
        lr = cosine_lr_with_warmup(
            step,
            warmup_steps=train_cfg.warmup_steps,
            max_steps=total_steps,
            max_lr=train_cfg.learning_rate,
            min_lr=min_lr,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr

    # ── Helper: run validation against whichever val_loader is currently
    #    active (reassigned per curriculum stage below) ──────────────────────
    def run_validation(loader: DataLoader) -> Tuple[float, float]:
        model.eval()
        total_loss = 0.0
        total_acc  = 0.0
        n_batches  = 0
        with torch.no_grad():
            for batch in loader:
                ids    = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(ids, labels=labels)
                if out["loss"] is not None:
                    total_loss += out["loss"].item()
                    total_acc  += token_accuracy(out["logits"], labels)
                    n_batches  += 1
        model.train()
        if n_batches == 0:
            return float("inf"), 0.0
        return total_loss / n_batches, total_acc / n_batches

    def _make_loader(items, shuffle: bool) -> DataLoader:
        ds = MathDataset(items, tokenizer, max_seq_len=model_cfg.max_seq_len)
        _collate = partial(collate_fn, pad_id=tokenizer.pad_id)
        return DataLoader(
            ds, batch_size=train_cfg.batch_size if shuffle else 128,
            shuffle=shuffle, num_workers=0, collate_fn=_collate, pin_memory=True,
        )

    # ── Training loop ────────────────────────────────────────────────────────
    log.info(f"Starting training: {total_steps} total steps"
             f"{' across ' + str(len(stages)) + ' curriculum stages' if train_cfg.curriculum else ''}")
    log.info(f"  Batch size    : {train_cfg.batch_size}")
    log.info(f"  Peak LR       : {train_cfg.learning_rate}")
    log.info(f"  Warmup steps  : {train_cfg.warmup_steps}")
    log.info(f"  Device        : {device}")
    log.info(f"  Mixed precision: {use_amp}")

    model.train()
    t_start = time.time()
    running_loss = 0.0
    running_acc  = 0.0
    running_n    = 0

    cumulative_steps = 0
    for stage_idx, stage in enumerate(stages):
        stage_start_step = cumulative_steps
        cumulative_steps += stage["steps"]
        stage_end_step = cumulative_steps

        if step >= stage_end_step:
            continue   # already completed this stage in a previous (resumed) run

        max_int_d = stage.get("max_int_digits")
        max_dec_d = stage.get("max_decimal_digits")
        stage_train_items = filter_by_number_length(train_items, max_int_d, max_dec_d)
        stage_val_items   = filter_by_number_length(val_items,   max_int_d, max_dec_d)

        if not stage_train_items:
            raise RuntimeError(
                f"Curriculum stage {stage_idx + 1}/{len(stages)} "
                f"(max_int_digits={max_int_d}, max_decimal_digits={max_dec_d}) "
                f"matched ZERO real training examples. Loosen the stage's "
                f"limits or check DataConfig.active_categories."
            )

        if train_cfg.curriculum:
            log.info(
                f"[Curriculum] Stage {stage_idx + 1}/{len(stages)}: "
                f"max_int_digits={max_int_d}  max_decimal_digits={max_dec_d}  "
                f"train={len(stage_train_items)}  val={len(stage_val_items)}  "
                f"steps={stage['steps']} (global {stage_start_step}→{stage_end_step})"
            )

        train_loader = _make_loader(stage_train_items, shuffle=True)
        val_loader   = _make_loader(stage_val_items,   shuffle=False)

        # Best-val-loss/patience are tracked PER STAGE: a harder stage's
        # intrinsically higher loss is not a regression against an easier
        # stage's best, and comparing them would trigger spurious early
        # stopping right at every stage transition.
        best_val_loss    = float("inf")
        patience_counter = 0

        while step < stage_end_step:
            epoch += 1
            for batch in train_loader:
                if step >= stage_end_step:
                    break

                # ── Move data to device ─────────────────────────────────
                ids    = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                # ── Update learning rate ────────────────────────────────
                current_lr = update_lr(step)

                # ── Forward pass ────────────────────────────────────────
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out  = model(ids, labels=labels)
                    loss = out["loss"]

                if loss is None or torch.isnan(loss):
                    log.warning(f"Step {step}: loss is None or NaN — skipping batch")
                    step += 1
                    continue

                # ── Backward pass ────────────────────────────────────────
                scaler.scale(loss).backward()

                # Gradient clipping prevents parameter updates that are too large.
                # We unscale first so the clip threshold is in the correct units.
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

                scaler.step(optimizer)
                scaler.update()

                # ── Accumulate running stats ─────────────────────────────
                running_loss += loss.item()
                running_acc  += token_accuracy(out["logits"].detach(), labels)
                running_n    += 1

                # ── Periodic logging ─────────────────────────────────────
                if step % train_cfg.log_every_steps == 0 and running_n > 0:
                    avg_loss = running_loss / running_n
                    avg_acc  = running_acc  / running_n
                    elapsed  = time.time() - t_start
                    steps_remaining = total_steps - step
                    eta = elapsed / max(step - start_step, 1) * steps_remaining
                    log.info(
                        f"Step {step:6d}/{total_steps}  "
                        f"loss={avg_loss:.4f}  tok_acc={avg_acc:.3f}  "
                        f"lr={current_lr:.2e}  "
                        f"elapsed={format_duration(elapsed)}  "
                        f"ETA={format_duration(eta)}"
                    )
                    running_loss = 0.0
                    running_acc  = 0.0
                    running_n    = 0

                # ── Periodic validation ───────────────────────────────────
                if step % train_cfg.eval_every_steps == 0:
                    val_loss, val_acc = run_validation(val_loader)
                    log.info(f"  [Val] step={step}  loss={val_loss:.4f}  "
                             f"tok_acc={val_acc:.3f}")

                    history["steps"].append(step)
                    history["val_loss"].append(val_loss)
                    history["val_tok_acc"].append(val_acc)
                    history["lr"].append(current_lr)

                    # ── Best model / early stopping (per stage) ──────────
                    is_best = val_loss < best_val_loss
                    if is_best:
                        best_val_loss    = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        log.info(f"  [EarlyStopping] patience {patience_counter}/{train_cfg.patience}")
                        if patience_counter >= train_cfg.patience:
                            log.info("  Early stopping triggered for this stage.")
                            break

                    # ── Checkpoint ─────────────────────────────────────────
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        val_loss=val_loss,
                        cfg_model_dict=model_cfg.__dict__,
                        cfg_train_dict=train_cfg.__dict__,
                        tokenizer_path=tok_path,
                        checkpoint_dir=train_cfg.checkpoint_dir,
                        keep_last_n=train_cfg.keep_last_n,
                        is_best=is_best,
                        data_source_info=source_info,
                    )

                step += 1
            else:
                continue
            break   # patience break above also exits the stage's epoch loop

    # ── End of training ──────────────────────────────────────────────────────
    log.info(f"Training complete. Best val loss (final stage): {best_val_loss:.4f}")
    _save_final_and_history(
        model, optimizer, step, best_val_loss,
        model_cfg, train_cfg, tok_path, history, source_info,
    )
    return history


# ---------------------------------------------------------------------------
# Helper: save final model + history
# ---------------------------------------------------------------------------

def _save_final_and_history(
    model, optimizer, step, best_val_loss,
    model_cfg, train_cfg, tok_path, history, source_info=None,
):
    """Save final_model.pt and training_history.json."""
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

    # Final model
    final_path = os.path.join(train_cfg.checkpoint_dir, "final_model.pt")
    torch.save({
        "step":        step,
        "val_loss":    best_val_loss,
        "model_state": model.state_dict(),
        "cfg_model":   model_cfg.__dict__,
        "cfg_train":   train_cfg.__dict__,
        "tokenizer_path": tok_path,
        "data_source_info": source_info or {},
    }, final_path)
    log.info(f"Final model saved → {final_path}")

    # Training history
    history_path = os.path.join("results", "training_history.json")
    os.makedirs("results", exist_ok=True)
    save_results(history, history_path)


# ---------------------------------------------------------------------------
# Plot training curve (optional — skipped if matplotlib not available)
# ---------------------------------------------------------------------------

def plot_training_curve(history: Dict, save_path: str = "results/training_curve.png") -> None:
    """Plot and save the training/validation loss curve."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available — skipping training curve plot")
        return

    if not history.get("steps"):
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["steps"], history["val_loss"], marker="o", label="Val Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Validation Loss")
    axes[0].grid(True)
    axes[0].legend()

    if history.get("val_tok_acc"):
        axes[1].plot(history["steps"], history["val_tok_acc"],
                     marker="o", color="green", label="Val Token Acc")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Token Accuracy")
        axes[1].set_title("Validation Token Accuracy")
        axes[1].grid(True)
        axes[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"Training curve saved → {save_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train the Math LLM")
    p.add_argument("--model",    default="small-2M",
                   choices=["tiny-1M", "small-2M", "medium-5M", "large-10M", "xlarge-20M"],
                   help="Model size preset")
    p.add_argument("--train",    default="default",
                   choices=["default", "fast", "thorough"],
                   help="Training preset")
    p.add_argument("--stage",    default="1", choices=["1", "2", "3"],
                   help="Data stage (1=basic arithmetic, 2=+comparison, 3=+algebra)")
    p.add_argument("--resume",   default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override max_steps from preset")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override batch_size from preset")
    p.add_argument("--lr", type=float, default=None,
                   help="Override learning rate")
    p.add_argument("--curriculum", default="none",
                   choices=["none", "default", "fast", "thorough"],
                   help="Train easy-to-hard by number length instead of a "
                        "flat step budget (see config.get_curriculum). "
                        "Overrides --max-steps with the preset's own total.")
    p.add_argument("--device", default="cuda",
                   help="Device: cuda / cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model_cfg = get_model_config(args.model)
    train_cfg = get_train_config(args.train)
    data_cfg  = DataConfig()

    # Apply CLI overrides
    if args.max_steps:  train_cfg.max_steps  = args.max_steps
    if args.batch_size: train_cfg.batch_size = args.batch_size
    if args.lr:         train_cfg.learning_rate = args.lr
    train_cfg.device = args.device
    train_cfg.curriculum = get_curriculum(args.curriculum)

    # Select data stage
    if args.stage == "1":
        data_cfg.active_categories = data_cfg.categories_stage1
    elif args.stage == "2":
        data_cfg.active_categories = (data_cfg.categories_stage1
                                      + data_cfg.categories_stage2)
    else:
        data_cfg.active_categories = (data_cfg.categories_stage1
                                      + data_cfg.categories_stage2
                                      + data_cfg.categories_stage3)

    log.info(f"=== Math LLM Training ===")
    log.info(f"Model   : {args.model}")
    log.info(f"Preset  : {args.train}")
    log.info(f"Stage   : {args.stage}")
    log.info(f"Cats    : {data_cfg.active_categories}")

    try:
        history = train(model_cfg, train_cfg, data_cfg, resume_from=args.resume)
    except RealDatasetLoadError as exc:
        # NEVER fall back to synthetic data — stop the run and surface the
        # real underlying failure so it can be diagnosed and fixed.
        print("\n" + "!" * 60)
        print("REAL DATASET LOAD FAILED")
        print("Reason:")
        print(str(exc))
        print("!" * 60)
        raise SystemExit(1)

    plot_training_curve(history)

    log.info("Done.")
