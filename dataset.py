"""
dataset.py
==========
Data loading, preprocessing, and PyTorch Dataset/DataLoader creation.

PIPELINE OVERVIEW
-----------------
1. Download the REAL DeepMind Mathematics Dataset (Saxton et al. 2019)
   from its official, authoritative source:
       https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz
   This is a static, versioned (v1.0) tarball published by the dataset's
   authors — not a HuggingFace Hub dataset repo. (Earlier versions of this
   file pointed at 'deepcode-ai/math_dataset' and 'WillHeld/deepmind-math',
   two HuggingFace repo names that do not work — the first does not exist,
   and the second's loader had a bug that made it always raise — which is
   why every run silently fell back to synthetic data. Both are gone now.)
2. Extract only the active-category files we need directly from the
   tarball (no need to unpack the whole 2.3 GB archive to disk).
3. Filter to the selected categories (from DataConfig.active_categories).
4. Format each example as:
       <Q>{question}<A>{answer}<EOS>
5. Tokenize with MathTokenizer.
6. Split: train/val come from the 'train-easy' difficulty tier (our own
   deterministic hash split); test comes from the official 'interpolate'
   tier, which the dataset's own authors generated independently and
   separately from train-easy. Test is therefore genuinely unseen by
   construction, not just by a hash bucket.
7. Wrap in PyTorch Dataset objects.
8. Return DataLoader objects ready for training.

FAILURE POLICY
--------------
If the real dataset cannot be downloaded or parsed, we raise
RealDatasetLoadError with the full underlying exception and STOP. We never
silently substitute synthetic data. A synthetic data generator is still
provided at the bottom of this file for offline unit-testing of the
tokenizer/dataset/label-masking logic (see audit.py) — it is never called
automatically by load_real_dataset()/build_dataloaders().

DATA LEAKAGE PREVENTION
------------------------
- The test split is the official 'interpolate' tier: different,
  independently-generated questions from train-easy. It is never touched
  by train/val splitting or by training.
- train/val (both drawn from train-easy) are split by hashing the question
  string to a deterministic bucket, so re-running the pipeline never
  reshuffles an example into a different split.
- check_split_overlap() actually verifies (does not merely assume) that
  there is zero exact and zero normalised overlap between splits.

REAL DATASET FORMAT (verified by direct inspection of the tarball)
--------------------------------------------------------------------
Each `<difficulty>/<category>.txt` file is plain text with QUESTION and
ANSWER alternating one per line:
    What is -5 - 110911?
    -110916
    What is -0.188 + -0.814?
    -1.002
    ...

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
import tarfile
import urllib.request
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from config import DataConfig, ModelConfig
from tokenizer import MathTokenizer


class RealDatasetLoadError(RuntimeError):
    """Raised when the real dataset cannot be downloaded or parsed.

    Callers MUST NOT catch this and silently substitute synthetic data —
    that is exactly the bug this project needed fixed. Let it propagate
    and stop the run.
    """
    pass


# ---------------------------------------------------------------------------
# Tarball download / cache
# ---------------------------------------------------------------------------

def _tarball_cache_path(cfg: DataConfig) -> str:
    os.makedirs(cfg.data_cache_dir, exist_ok=True)
    return os.path.join(cfg.data_cache_dir, f"{cfg.dataset_archive_name}.tar.gz")


def _ensure_tarball(cfg: DataConfig) -> str:
    """
    Return a local path to the dataset tarball, downloading it if needed.

    A cached file is only trusted if its size matches the known-good size
    published by the authors (an interrupted download will not match, and
    will be re-downloaded rather than silently used).
    """
    path = _tarball_cache_path(cfg)

    if os.path.isfile(path):
        size = os.path.getsize(path)
        if size == cfg.dataset_expected_size_bytes:
            print(f"[Dataset] Using cached tarball: {path} "
                  f"({size / 1e9:.2f} GB)")
            return path
        else:
            print(f"[Dataset] Cached tarball at {path} has unexpected size "
                  f"({size:,} bytes, expected {cfg.dataset_expected_size_bytes:,}) "
                  f"— re-downloading.")
            os.remove(path)

    print(f"[Dataset] Downloading real dataset from:\n  {cfg.dataset_url}")
    print(f"[Dataset] This is a one-time ~{cfg.dataset_expected_size_bytes / 1e9:.1f} GB "
          f"download (cached afterwards at {path}).")

    tmp_path = path + ".partial"
    req = urllib.request.Request(cfg.dataset_url, headers={"User-Agent": "math-llm/1.0"})
    downloaded = 0
    chunk_size = 1 << 20  # 1 MB
    last_report_mb = 0
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp_path, "wb") as f:
        total = int(resp.headers.get("Content-Length", cfg.dataset_expected_size_bytes))
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            mb = downloaded // (50 * 1 << 20)
            if mb != last_report_mb:
                last_report_mb = mb
                pct = 100.0 * downloaded / max(total, 1)
                print(f"  … {downloaded / 1e9:.2f} / {total / 1e9:.2f} GB ({pct:.0f}%)")

    final_size = os.path.getsize(tmp_path)
    if final_size != cfg.dataset_expected_size_bytes:
        os.remove(tmp_path)
        raise RealDatasetLoadError(
            f"Downloaded tarball size mismatch: got {final_size:,} bytes, "
            f"expected {cfg.dataset_expected_size_bytes:,} bytes. "
            f"The download was likely interrupted or the source changed. "
            f"Refusing to use a partial/corrupt file."
        )
    os.rename(tmp_path, path)
    print(f"[Dataset] Download complete → {path}")
    return path


# ---------------------------------------------------------------------------
# Extracting specific category files from the tarball
# ---------------------------------------------------------------------------

def _parse_category_text(raw_text: str, member_name: str) -> List[Tuple[str, str]]:
    """
    Parse a '<difficulty>/<category>.txt' file's contents into a list of
    (question, answer) pairs.

    The real format alternates one question per line, one answer per line.
    """
    lines = raw_text.strip("\n").split("\n")
    if len(lines) % 2 != 0:
        raise RealDatasetLoadError(
            f"Malformed data file '{member_name}': expected an even number "
            f"of lines (question/answer pairs), got {len(lines)}."
        )
    pairs = []
    for i in range(0, len(lines), 2):
        q, a = lines[i].strip(), lines[i + 1].strip()
        if q and a:
            pairs.append((q, a))
    return pairs


def _extract_category_files(
    tarball_path: str,
    wanted: Dict[str, str],   # member_name -> label (for error messages)
) -> Dict[str, str]:
    """
    Single sequential pass over the (locally cached) tarball, pulling out
    the raw text of every member whose name is in `wanted`.

    Returns dict: member_name -> decoded text content.
    Raises RealDatasetLoadError if any requested member is missing.
    """
    found: Dict[str, str] = {}
    remaining = set(wanted.keys())

    try:
        with tarfile.open(tarball_path, mode="r:gz") as tf:
            for member in tf:
                if not remaining:
                    break
                if member.name in remaining:
                    fobj = tf.extractfile(member)
                    if fobj is None:
                        raise RealDatasetLoadError(
                            f"Archive member '{member.name}' is not a regular file."
                        )
                    found[member.name] = fobj.read().decode("utf-8")
                    remaining.discard(member.name)
    except tarfile.TarError as exc:
        raise RealDatasetLoadError(
            f"Failed to read tarball '{tarball_path}': {exc!r}"
        ) from exc

    if remaining:
        missing = ", ".join(sorted(remaining))
        raise RealDatasetLoadError(
            f"The following files were not found inside the dataset archive: "
            f"{missing}. Check that the category names in DataConfig match "
            f"the real file names (see config.py docstring)."
        )
    return found


# ---------------------------------------------------------------------------
# Top-level real-data loader
# ---------------------------------------------------------------------------

def load_real_dataset(
    cfg: DataConfig,
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
    """
    Download (or reuse a cached copy of) the real DeepMind Mathematics
    Dataset and return per-category example pools for the train and test
    tiers.

    Returns
    -------
    (train_pool, test_pool)
        train_pool : category -> list of {"question","answer","category"}
                     drawn from cfg.train_difficulty (default 'train-easy'),
                     deterministically shuffled and capped at
                     cfg.max_samples_per_category.
        test_pool  : category -> list of {"question","answer","category"}
                     drawn from cfg.test_difficulty (default 'interpolate'),
                     the dataset authors' own held-out split — never
                     touched during training. Capped at
                     cfg.max_test_samples_per_category.

    Raises
    ------
    RealDatasetLoadError on ANY failure (network, missing file, malformed
    content). This function never silently substitutes synthetic data.
    """
    try:
        tarball_path = _ensure_tarball(cfg)

        prefix = cfg.dataset_archive_name
        wanted: Dict[str, str] = {}
        for cat in cfg.active_categories:
            wanted[f"{prefix}/{cfg.train_difficulty}/{cat}.txt"] = f"train:{cat}"
            wanted[f"{prefix}/{cfg.test_difficulty}/{cat}.txt"] = f"test:{cat}"

        print(f"[Dataset] Extracting {len(cfg.active_categories)} categories "
              f"from '{cfg.train_difficulty}' and '{cfg.test_difficulty}' …")
        raw_texts = _extract_category_files(tarball_path, wanted)

        rng = random.Random(42)
        train_pool: Dict[str, List[Dict]] = {}
        test_pool: Dict[str, List[Dict]] = {}

        for cat in cfg.active_categories:
            train_text = raw_texts[f"{prefix}/{cfg.train_difficulty}/{cat}.txt"]
            test_text = raw_texts[f"{prefix}/{cfg.test_difficulty}/{cat}.txt"]

            train_pairs = _parse_category_text(train_text, cat)
            test_pairs = _parse_category_text(test_text, cat)

            rng.shuffle(train_pairs)
            train_pairs = train_pairs[: cfg.max_samples_per_category]
            test_pairs = test_pairs[: cfg.max_test_samples_per_category]

            train_pool[cat] = [
                {"question": q, "answer": a, "category": cat} for q, a in train_pairs
            ]
            test_pool[cat] = [
                {"question": q, "answer": a, "category": cat} for q, a in test_pairs
            ]
            print(f"  {cat:35s}  train={len(train_pool[cat]):5d}  "
                  f"test={len(test_pool[cat]):5d}")

        return train_pool, test_pool

    except RealDatasetLoadError:
        raise
    except Exception as exc:
        # Anything else (network error, permissions, decode error, ...) —
        # surface it loudly instead of ever falling back to synthetic data.
        raise RealDatasetLoadError(
            f"REAL DATASET LOAD FAILED\nReason:\n{exc!r}"
        ) from exc


# ---------------------------------------------------------------------------
# Train / Val split  (test comes from the interpolate tier, see above)
# ---------------------------------------------------------------------------

def _hash_bucket(text: str) -> int:
    """Hash text to an integer in [0, 99]."""
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest, 16) % 100


def split_train_val(
    train_pool: Dict[str, List[Dict]],
    cfg: DataConfig,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split the train-easy pool into train/val using a deterministic hash of
    the question string, so re-running the pipeline never reshuffles an
    example into a different split.
    """
    train, val = [], []
    train_cutoff = int(cfg.train_ratio * 100)   # e.g. 90

    for cat, items in train_pool.items():
        for item in items:
            bucket = _hash_bucket(item["question"])
            if bucket < train_cutoff:
                train.append(item)
            else:
                val.append(item)

    print(f"[Dataset] Train/Val split (from '{cfg.train_difficulty}'): "
          f"train={len(train)}  val={len(val)}")
    return train, val


def flatten_pool(pool: Dict[str, List[Dict]]) -> List[Dict]:
    """Flatten a category -> items dict into a single flat list."""
    out = []
    for items in pool.values():
        out.extend(items)
    return out


def _normalise_question(s: str) -> str:
    return " ".join(s.lower().split())


def check_split_overlap(
    train: List[Dict], val: List[Dict], test: List[Dict],
) -> Dict[str, int]:
    """
    Actually verify (never merely assume) there is no leakage between
    splits. Checks both exact-string and whitespace/case-normalised
    overlap of the question field.

    Returns a dict of overlap counts; also prints a human-readable report.
    """
    train_q = {it["question"] for it in train}
    val_q = {it["question"] for it in val}
    test_q = {it["question"] for it in test}

    train_norm = {_normalise_question(q) for q in train_q}
    val_norm = {_normalise_question(q) for q in val_q}
    test_norm = {_normalise_question(q) for q in test_q}

    report = {
        "train_test_exact": len(train_q & test_q),
        "train_val_exact": len(train_q & val_q),
        "val_test_exact": len(val_q & test_q),
        "train_test_normalised": len(train_norm & test_norm),
        "train_val_normalised": len(train_norm & val_norm),
        "val_test_normalised": len(val_norm & test_norm),
    }

    print("[Dataset] Overlap check:")
    print(f"  Train/Test exact overlap      : {report['train_test_exact']}")
    print(f"  Train/Test normalised overlap : {report['train_test_normalised']}")
    print(f"  Train/Val  exact overlap      : {report['train_val_exact']}")
    print(f"  Train/Val  normalised overlap : {report['train_val_normalised']}")
    print(f"  Val/Test   exact overlap      : {report['val_test_exact']}")
    print(f"  Val/Test   normalised overlap : {report['val_test_normalised']}")

    if any(report.values()):
        print("  WARNING: non-zero overlap detected between splits!")
    else:
        print("  OK — zero overlap between all splits.")

    return report


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

def load_split_dataset(
    cfg_data: DataConfig,
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
    """
    Full real-data pipeline: download → extract → split → verify overlap.

    Returns
    -------
    (train_items, val_items, test_items, source_info)
        source_info is a dict describing exactly what was loaded, suitable
        for printing / saving into checkpoints.
    """
    train_pool, test_pool = load_real_dataset(cfg_data)
    train_items, val_items = split_train_val(train_pool, cfg_data)
    test_items = flatten_pool(test_pool)

    overlap = check_split_overlap(train_items, val_items, test_items)

    source_info = {
        "data_source": "REAL",
        "dataset": "DeepMind Mathematics Dataset (Saxton et al. 2019)",
        "dataset_url": cfg_data.dataset_url,
        "configuration": {
            "active_categories": list(cfg_data.active_categories),
            "train_difficulty": cfg_data.train_difficulty,
            "test_difficulty": cfg_data.test_difficulty,
        },
        "train_examples": len(train_items),
        "val_examples": len(val_items),
        "test_examples": len(test_items),
        "overlap": overlap,
    }

    print("\n" + "=" * 60)
    print("Data source: REAL")
    print(f"Dataset      : {source_info['dataset']}")
    print(f"Configuration: {cfg_data.active_categories}")
    print(f"Train examples     : {len(train_items)}")
    print(f"Validation examples: {len(val_items)}")
    print(f"Test examples      : {len(test_items)}  (official '{cfg_data.test_difficulty}' split)")
    print("=" * 60 + "\n")

    return train_items, val_items, test_items, source_info


def build_dataloaders_from_items(
    train_items: List[Dict],
    val_items:   List[Dict],
    test_items:  List[Dict],
    cfg_model:   ModelConfig,
    tokenizer:   Optional[MathTokenizer] = None,
    batch_size:  int = 64,
) -> Tuple[DataLoader, DataLoader, DataLoader, MathTokenizer]:
    """
    Build tokenizer + PyTorch DataLoaders from already-loaded item lists.

    Split out from build_dataloaders() so callers that already have
    train/val/test items (e.g. the Colab notebook, which prints and
    verifies the real data before building loaders) don't need to
    re-download/re-extract the dataset a second time.
    """
    # Build or reuse tokenizer — built on ALL data (train+val+test) so no
    # split's characters are unseen by the tokenizer. The tokenizer only
    # stores character→id mappings; it carries no information about which
    # split an example belongs to, so this does not leak labels/answers.
    if tokenizer is None:
        tokenizer = MathTokenizer()
        all_texts = [format_example(item) for item in (train_items + val_items + test_items)]
        tokenizer.build(all_texts)

    # Update vocab size in model config
    cfg_model.vocab_size = tokenizer.vocab_size

    # PyTorch Datasets
    max_len = cfg_model.max_seq_len
    train_ds = MathDataset(train_items, tokenizer, max_seq_len=max_len)
    val_ds   = MathDataset(val_items,   tokenizer, max_seq_len=max_len)
    test_ds  = MathDataset(test_items,  tokenizer, max_seq_len=max_len)

    print(f"[Dataset] Sizes after tokenisation: "
          f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # DataLoaders
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


def build_dataloaders(
    cfg_data:   DataConfig,
    cfg_model:  ModelConfig,
    tokenizer:  Optional[MathTokenizer] = None,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, DataLoader, MathTokenizer, Dict]:
    """
    Full pipeline: load real data → split → tokenize → DataLoader.

    Raises RealDatasetLoadError if the real dataset cannot be loaded — this
    function never falls back to synthetic data.

    Returns
    -------
    train_loader, val_loader, test_loader, tokenizer, source_info
    """
    train_items, val_items, test_items, source_info = load_split_dataset(cfg_data)
    train_loader, val_loader, test_loader, tokenizer = build_dataloaders_from_items(
        train_items, val_items, test_items, cfg_model,
        tokenizer=tokenizer, batch_size=batch_size,
    )
    return train_loader, val_loader, test_loader, tokenizer, source_info


# ---------------------------------------------------------------------------
# Synthetic data generator — OFFLINE UNIT-TESTING ONLY.
#
# This is NEVER called by load_real_dataset() / build_dataloaders() / train().
# It exists solely so tokenizer/dataset/label-masking logic can be unit
# tested (see audit.py) without a network connection. Any script that uses
# it MUST clearly print that synthetic data is in use.
# ---------------------------------------------------------------------------

def generate_synthetic_debug_data(cfg: DataConfig, n_per_category: int = 200) -> Dict[str, List[Dict]]:
    """
    Generate simple arithmetic examples programmatically, for offline
    testing of the pipeline plumbing ONLY. Not real data — do not train a
    reportable model on this.
    """
    rng = random.Random(1234)
    data: Dict[str, List[Dict]] = {}

    def _add_sub(n):
        items = []
        for _ in range(n):
            a, b = rng.randint(-999, 999), rng.randint(-999, 999)
            op = "+" if rng.random() < 0.5 else "-"
            ans = a + b if op == "+" else a - b
            items.append({"question": f"What is {a} {op} {b}?",
                          "answer": str(ans), "category": "arithmetic__add_or_sub"})
        return items

    def _mul(n):
        items = []
        for _ in range(n):
            a, b = rng.randint(1, 99), rng.randint(1, 99)
            items.append({"question": f"What is {a} * {b}?",
                          "answer": str(a * b), "category": "arithmetic__mul"})
        return items

    def _div(n):
        items = []
        for _ in range(n):
            a, b = rng.randint(1, 99), rng.randint(1, 12)
            items.append({"question": f"What is {a * b} / {b}?",
                          "answer": str(a), "category": "arithmetic__div"})
        return items

    def _mixed(n):
        items = []
        for _ in range(n):
            a, b, c = rng.randint(1, 50), rng.randint(1, 50), rng.randint(1, 50)
            items.append({"question": f"What is {a} + {b} * {c}?",
                          "answer": str(a + b * c), "category": "arithmetic__mixed"})
        return items

    generators = {
        "arithmetic__add_or_sub": _add_sub,
        "arithmetic__mul": _mul,
        "arithmetic__div": _div,
        "arithmetic__mixed": _mixed,
    }

    print("!" * 60)
    print("SYNTHETIC DEBUG DATA IN USE — NOT REAL DATA. NOT FOR REPORTING.")
    print("!" * 60)

    for cat in cfg.active_categories:
        gen = generators.get(cat)
        if gen is None:
            print(f"  [Synthetic] No generator for '{cat}', skipping.")
            data[cat] = []
            continue
        data[cat] = gen(n_per_category)
        print(f"  [Synthetic] {cat}: {len(data[cat])} examples")

    return data


# ---------------------------------------------------------------------------
# Quick self-test (requires network access to download/cache the real
# dataset on first run)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from config import DataConfig, ModelConfig, get_model_config

    dcfg = DataConfig()
    mcfg = get_model_config("small-2M")

    train_loader, val_loader, test_loader, tok, _source_info = build_dataloaders(dcfg, mcfg)

    batch = next(iter(train_loader))
    print("\nSample batch:")
    print(f"  input_ids shape : {batch['input_ids'].shape}")
    print(f"  labels shape    : {batch['labels'].shape}")
    print(f"  categories      : {batch['categories'][:3]}")
    print(f"  questions       : {batch['questions'][:2]}")
    print(f"  answers         : {batch['answers'][:2]}")
    print(f"\nDecoded first example:")
    print(f"  {tok.decode(batch['input_ids'][0].tolist())}")
