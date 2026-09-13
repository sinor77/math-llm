"""
dataset.py
==========
Data loading, preprocessing, and PyTorch Dataset/DataLoader creation.

PIPELINE OVERVIEW
-----------------
1. Download the DeepMind Mathematics Dataset from HuggingFace Hub.
   Primary source  : deepcode-ai/math_dataset  (per-category config)
   Secondary source: WillHeld/deepmind-math    (flat combined dataset)
   Fallback        : synthetic arithmetic data (offline/CI use)
2. Filter to the selected categories (from DataConfig.active_categories).
3. Format each example as:
       <Q>{question}<A>{answer}<EOS>
4. Tokenize with MathTokenizer.
5. Split into train / val / test (strict: test set is held-out ONLY).
6. Wrap in PyTorch Dataset objects.
7. Return DataLoader objects ready for training.

DATA LEAKAGE PREVENTION
------------------------
The test split is created BEFORE any training so the model never sees
those examples.  The split is done by hashing the question string to a
deterministic bucket — identical questions always end up in the same
split regardless of dataset download order.

REAL DATASET SCHEMA
--------------------
deepcode-ai/math_dataset:
    question : str   (up to 160 chars)
    answer   : str   (up to 30 chars)
    — no category column; category is injected from the config name —

WillHeld/deepmind-math:
    question : str
    answer   : str
    category : str   (e.g. "algebra__linear_1d")

WHY WE FORMAT AS <Q>…<A>…<EOS>
--------------------------------
This is a sequence-to-sequence task framed as next-token prediction.
The model sees the full sequence during training and learns to predict
every token given all previous tokens (causal language modelling).
At inference time we feed <Q>…<A> and let the model generate the rest.
"""

import hashlib
import os
import random
import json
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from config import DataConfig, ModelConfig
from tokenizer import MathTokenizer


# ---------------------------------------------------------------------------
# Raw data loading
# ---------------------------------------------------------------------------

def load_raw_data(cfg: DataConfig) -> Dict[str, List[Dict]]:
    """
    Download / load the HuggingFace dataset and return a dict mapping
    category name → list of {"question": str, "answer": str, "category": str}.

    Two real HuggingFace datasets are supported (tried in order):

      1. deepcode-ai/math_dataset
         The canonical Hub mirror of the DeepMind Mathematics Dataset.
         Each category is a separate config name (e.g. 'arithmetic__add_or_sub').
         Schema: question (str), answer (str)  — NO category column.
         We inject the category name from the config string.

      2. WillHeld/deepmind-math
         A flat, combined version with all categories merged.
         Schema: question (str), answer (str), category (str).
         One 'train' split containing all categories.

    Falls back to a synthetic dataset if both Hub loads fail
    (useful for offline debugging and CI without network access).
    """
    try:
        from datasets import load_dataset
        # ── Try deepcode-ai/math_dataset (per-category config) ────────────
        print(f"[Dataset] Trying 'deepcode-ai/math_dataset' (per-category) …")
        return _load_deepcode_dataset(cfg)
    except Exception as exc1:
        print(f"[Dataset] deepcode-ai load failed: {exc1}")
        try:
            # ── Try WillHeld/deepmind-math (flat, single load) ────────────
            print(f"[Dataset] Trying 'WillHeld/deepmind-math' (flat) …")
            from datasets import load_dataset
            raw = load_dataset("WillHeld/deepmind-math", trust_remote_code=True)
            return _parse_willheld_dataset(raw, cfg)
        except Exception as exc2:
            print(f"[Dataset] WillHeld load failed: {exc2}")
            print(f"[Dataset] Both HuggingFace sources unavailable. "
                  f"Using synthetic arithmetic data.")
            return _generate_synthetic_data(cfg)


def _load_deepcode_dataset(cfg: DataConfig) -> Dict[str, List[Dict]]:
    """
    Load deepcode-ai/math_dataset one category at a time.

    Each active category is a separate dataset config.  The schema has
    only 'question' and 'answer' columns — we inject 'category' from
    the config name.

    Both 'train' and 'test' splits exist (~2M and ~10K per category).
    We load only 'train' here; our own hash-split creates the test set.
    """
    from datasets import load_dataset

    by_category: Dict[str, List[Dict]] = {cat: [] for cat in cfg.active_categories}
    rng = random.Random(42)

    for cat in cfg.active_categories:
        try:
            ds = load_dataset(
                "deepcode-ai/math_dataset",
                cat,
                split="train",
                trust_remote_code=True,
            )
            items = []
            for ex in ds:
                q = str(ex.get("question", "")).strip()
                a = str(ex.get("answer",   "")).strip()
                if q and a:
                    items.append({"question": q, "answer": a, "category": cat})

            # Deterministic shuffle then cap
            rng.shuffle(items)
            by_category[cat] = items[: cfg.max_samples_per_category]
            print(f"  {cat}: {len(by_category[cat])} examples "
                  f"(from {len(items)} total)")
        except Exception as exc:
            print(f"  WARNING: could not load category '{cat}': {exc}")
            by_category[cat] = []

    return by_category


def _parse_willheld_dataset(raw, cfg: DataConfig) -> Dict[str, List[Dict]]:
    """
    Parse WillHeld/deepmind-math into our internal format.

    Schema: question (str), answer (str), category (str).
    All categories are in the 'train' split.
    We merge all splits (usually just 'train') and re-split ourselves.
    """
    by_category: Dict[str, List[Dict]] = {cat: [] for cat in cfg.active_categories}
    rng = random.Random(42)

    for split_name in raw.keys():
        for example in raw[split_name]:
            # 'category' is the correct column name in WillHeld/deepmind-math
            cat = example.get("category", "unknown")
            if cat not in cfg.active_categories:
                continue
            q = str(example.get("question", "")).strip()
            a = str(example.get("answer",   "")).strip()
            if q and a:
                by_category[cat].append({"question": q, "answer": a, "category": cat})

    # Cap per category and report
    for cat in cfg.active_categories:
        items = by_category[cat]
        if len(items) > cfg.max_samples_per_category:
            rng.shuffle(items)
            by_category[cat] = items[: cfg.max_samples_per_category]
        print(f"  {cat}: {len(by_category[cat])} examples")


def _generate_synthetic_data(cfg: DataConfig) -> Dict[str, List[Dict]]:
    """
    Generate simple arithmetic examples programmatically.

    Used as a fallback when the HuggingFace dataset is unavailable.
    Covers: add/sub, mul/div, mixed, and basic linear algebra.
    """
    import operator as op
    rng = random.Random(42)
    data: Dict[str, List[Dict]] = {}

    def _add_sub(n=5000):
        items = []
        for _ in range(n):
            a, b = rng.randint(-999, 999), rng.randint(-999, 999)
            if rng.random() < 0.5:
                items.append({"question": f"What is {a} + {b}?",
                               "answer": str(a + b),
                               "category": "arithmetic__add_or_sub"})
            else:
                items.append({"question": f"What is {a} - {b}?",
                               "answer": str(a - b),
                               "category": "arithmetic__add_or_sub"})
        return items

    def _mul_div(n=5000):
        items = []
        for _ in range(n):
            a = rng.randint(1, 99)
            b = rng.randint(1, 99)
            if rng.random() < 0.5:
                items.append({"question": f"What is {a} * {b}?",
                               "answer": str(a * b),
                               "category": "arithmetic__mul_or_div"})
            else:
                # integer division only
                items.append({"question": f"What is {a * b} / {b}?",
                               "answer": str(a),
                               "category": "arithmetic__mul_or_div"})
        return items

    def _mixed(n=5000):
        items = []
        for _ in range(n):
            a = rng.randint(1, 50)
            b = rng.randint(1, 50)
            c = rng.randint(1, 50)
            items.append({"question": f"What is {a} + {b} * {c}?",
                           "answer": str(a + b * c),
                           "category": "arithmetic__mixed"})
        return items

    def _linear_1d(n=5000):
        items = []
        for _ in range(n):
            # ax + b = c  →  x = (c - b) / a
            a = rng.randint(1, 12)
            b = rng.randint(-20, 20)
            x = rng.randint(-20, 20)
            c = a * x + b
            items.append({
                "question": f"Solve {a}x + ({b}) = {c}",
                "answer": f"x = {x}",
                "category": "algebra__linear_1d",
            })
        return items

    def _comparison(n=5000):
        items = []
        for _ in range(n):
            nums = [rng.randint(-100, 100) for _ in range(rng.randint(2, 5))]
            items.append({
                "question": f"What is the largest of {nums}?",
                "answer": str(max(nums)),
                "category": "comparison__sort",
            })
        return items

    generators = {
        "arithmetic__add_or_sub": _add_sub,
        "arithmetic__mul_or_div": _mul_div,
        "arithmetic__mixed":     _mixed,
        "algebra__linear_1d":    _linear_1d,
        "comparison__sort":      _comparison,
    }

    for cat in cfg.active_categories:
        gen = generators.get(cat)
        if gen is None:
            print(f"  [Synthetic] No generator for '{cat}', skipping.")
            data[cat] = []
            continue
        items = gen(cfg.max_samples_per_category)
        data[cat] = items
        print(f"  [Synthetic] {cat}: {len(items)} examples")

    return data


# ---------------------------------------------------------------------------
# Train / Val / Test split
# ---------------------------------------------------------------------------

def split_data(
    by_category: Dict[str, List[Dict]],
    cfg: DataConfig,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split data into train / val / test using a deterministic hash.

    For each example we hash its question string modulo 100 to get a
    bucket number:
        0  – 79  → train
        80 – 89  → val
        90 – 99  → test

    This guarantees that the same question ALWAYS ends up in the same
    split even if the dataset is downloaded in a different order on
    different machines.
    """
    train, val, test = [], [], []
    train_cutoff = int(cfg.train_ratio * 100)   # e.g. 80
    val_cutoff   = train_cutoff + int(cfg.val_ratio * 100)  # e.g. 90

    for cat, items in by_category.items():
        for item in items:
            bucket = _hash_bucket(item["question"])
            if bucket < train_cutoff:
                train.append(item)
            elif bucket < val_cutoff:
                val.append(item)
            else:
                test.append(item)

    print(f"[Dataset] Split:  train={len(train)}  val={len(val)}  test={len(test)}")
    return train, val, test


def _hash_bucket(text: str) -> int:
    """Hash text to an integer in [0, 99]."""
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest, 16) % 100


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_example(
    item: Dict,
    q_token: str = "<Q>",
    a_token: str = "<A>",
    eos_token: str = "<EOS>",
) -> str:
    """
    Format a raw example dict into the string the model is trained on.

    Example:
        {"question": "What is 3+4?", "answer": "7"}
        →  "<Q>What is 3+4?<A>7<EOS>"
    """
    return f"{q_token}{item['question']}{a_token}{item['answer']}{eos_token}"


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class MathDataset(Dataset):
    """
    PyTorch Dataset for language-model training.

    Each item returns:
        input_ids  : LongTensor of shape (seq_len,)
                     — the full tokenised sequence <Q>…<A>…<EOS>
        labels     : LongTensor of shape (seq_len,)
                     — same as input_ids, but with question tokens masked (-100)
                       so the loss is only computed over the answer tokens

    Causal language modelling:
        input_ids[:-1]   fed as model input  (all but last token)
        labels[1:]       used as prediction targets (shifted by 1)

    The -100 mask over question tokens means the model is only penalised
    for its answer predictions, not for "predicting" the question it already
    has as context.
    """

    def __init__(
        self,
        items: List[Dict],
        tokenizer: MathTokenizer,
        max_seq_len: int = 256,
        mask_question: bool = True,
    ):
        self.tokenizer    = tokenizer
        self.max_seq_len  = max_seq_len
        self.mask_question = mask_question
        self.examples: List[Dict] = []

        a_id = tokenizer.token_to_id.get("<A>", None)

        skipped = 0
        for item in items:
            text = format_example(item)
            ids  = tokenizer.encode(text, add_eos=False, max_length=max_seq_len)

            if len(ids) < 4:            # too short to be useful
                skipped += 1
                continue

            # Build label tensor: -100 for question tokens
            labels = ids.copy()
            if mask_question and a_id is not None:
                # Find position of <A> token; mask everything up to and including it
                a_pos = next((i for i, x in enumerate(ids) if x == a_id), None)
                if a_pos is not None:
                    for j in range(a_pos + 1):  # mask <Q>…<A> inclusive
                        labels[j] = -100

            self.examples.append({
                "input_ids": ids,
                "labels":    labels,
                "category":  item.get("category", "unknown"),
                "question":  item["question"],
                "answer":    item["answer"],
            })

        if skipped:
            print(f"[Dataset] Skipped {skipped} examples (too short or malformed)")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "labels":    torch.tensor(ex["labels"],    dtype=torch.long),
            "category":  ex["category"],
            "question":  ex["question"],
            "answer":    ex["answer"],
        }


# ---------------------------------------------------------------------------
# Collate function  (converts a list of samples into a padded batch)
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict], pad_id: int = 0) -> Dict:
    """
    Pad sequences in a batch to equal length.

    All sequences are right-padded to the length of the longest sequence
    in the batch.  The padding token ID for input_ids is pad_id (0 = <PAD>).
    The padding value for labels is -100 (ignored by cross-entropy loss).
    """
    max_len = max(len(b["input_ids"]) for b in batch)

    input_ids = []
    labels    = []
    categories = []
    questions  = []
    answers    = []

    for b in batch:
        seq_len = len(b["input_ids"])
        pad_len = max_len - seq_len

        input_ids.append(
            torch.cat([b["input_ids"],
                       torch.full((pad_len,), pad_id, dtype=torch.long)])
        )
        labels.append(
            torch.cat([b["labels"],
                       torch.full((pad_len,), -100, dtype=torch.long)])
        )
        categories.append(b["category"])
        questions.append(b["question"])
        answers.append(b["answer"])

    return {
        "input_ids":  torch.stack(input_ids),   # (B, L)
        "labels":     torch.stack(labels),       # (B, L)
        "categories": categories,
        "questions":  questions,
        "answers":    answers,
    }


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------

def build_dataloaders(
    cfg_data:   DataConfig,
    cfg_model:  ModelConfig,
    tokenizer:  Optional[MathTokenizer] = None,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, DataLoader, MathTokenizer]:
    """
    Full pipeline: load → split → tokenize → DataLoader.

    Returns
    -------
    train_loader, val_loader, test_loader, tokenizer
    """
    # 1 — Load raw data
    by_category = load_raw_data(cfg_data)

    # 2 — Split
    train_items, val_items, test_items = split_data(by_category, cfg_data)

    # 3 — Build or reuse tokenizer
    if tokenizer is None:
        tokenizer = MathTokenizer()
        all_texts = [format_example(item) for items in by_category.values()
                     for item in items]
        tokenizer.build(all_texts)

    # Update vocab size in model config
    cfg_model.vocab_size = tokenizer.vocab_size

    # 4 — PyTorch Datasets
    max_len = cfg_model.max_seq_len
    train_ds = MathDataset(train_items, tokenizer, max_seq_len=max_len)
    val_ds   = MathDataset(val_items,   tokenizer, max_seq_len=max_len)
    test_ds  = MathDataset(test_items,  tokenizer, max_seq_len=max_len)

    print(f"[Dataset] Sizes after tokenisation: "
          f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # 5 — DataLoaders
    from functools import partial
    _collate = partial(collate_fn, pad_id=tokenizer.pad_id)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=_collate, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=128, shuffle=False,
        num_workers=0, collate_fn=_collate, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=128, shuffle=False,
        num_workers=0, collate_fn=_collate, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, tokenizer


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from config import DataConfig, ModelConfig, get_model_config

    dcfg = DataConfig()
    mcfg = get_model_config("small-2M")

    train_loader, val_loader, test_loader, tok = build_dataloaders(dcfg, mcfg)

    batch = next(iter(train_loader))
    print("\nSample batch:")
    print(f"  input_ids shape : {batch['input_ids'].shape}")
    print(f"  labels shape    : {batch['labels'].shape}")
    print(f"  categories      : {batch['categories'][:3]}")
    print(f"  questions       : {batch['questions'][:2]}")
    print(f"  answers         : {batch['answers'][:2]}")
    print(f"\nDecoded first example:")
    print(f"  {tok.decode(batch['input_ids'][0].tolist())}")
