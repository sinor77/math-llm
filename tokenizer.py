"""
tokenizer.py
============
A simple, transparent, character-level tokenizer designed for mathematical text.

WHY CHARACTER-LEVEL?
--------------------
Mathematical expressions contain digits, operators, variables, and special
symbols in very compact combinations.  A character-level tokenizer:
  • never produces an <UNK> token for any printable ASCII character
  • is fully deterministic and inspectable
  • requires NO external vocabulary files or pre-training
  • handles novel number combinations the model has never seen (1234567
    is just the sequence of characters '1','2','3','4','5','6','7')

HOW IT WORKS
------------
1. We scan every character that appears in the dataset (build phase).
2. We assign a unique integer ID to each character.
3. Special tokens get the lowest IDs (<PAD>=0, <UNK>=1, <EOS>=2, etc.).
4. encode(text)  → list[int]
5. decode(ids)   → str

SPECIAL TOKENS
--------------
<PAD>  : padding — used to fill shorter sequences in a batch to equal length
<UNK>  : unknown — emitted when a character is not in the vocabulary
<EOS>  : end-of-sequence — marks where the answer finishes
<Q>    : question delimiter — prepended to every question
<A>    : answer delimiter — separates question from answer
"""

import json
import os
from typing import List, Dict, Optional


class MathTokenizer:
    """
    Character-level tokenizer for mathematical text.

    Usage
    -----
    tok = MathTokenizer()
    tok.build(list_of_all_text_strings)   # scan corpus, build vocab
    tok.save("tok_config.json")            # persist to disk

    tok2 = MathTokenizer.load("tok_config.json")  # reload later
    ids  = tok2.encode("3x + 7 = 22<EOS>")
    text = tok2.decode(ids)
    """

    # -----------------------------------------------------------------------
    # Special tokens — order here defines their IDs (0, 1, 2, …)
    # -----------------------------------------------------------------------
    SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<EOS>", "<Q>", "<A>", "<BOS>"]

    def __init__(self):
        # Maps token-string → integer ID
        self.token_to_id: Dict[str, int] = {}
        # Maps integer ID → token-string
        self.id_to_token: Dict[int, str] = {}
        self._built = False

    # -----------------------------------------------------------------------
    # Build vocabulary
    # -----------------------------------------------------------------------

    def build(self, texts: List[str]) -> "MathTokenizer":
        """
        Scan a list of strings and build a character-level vocabulary.

        Parameters
        ----------
        texts : list of str
            Every training/validation/test string should be included so
            the vocabulary is complete.

        Returns
        -------
        self (for chaining)
        """
        # Step 1 — reserve IDs for special tokens
        self.token_to_id = {}
        self.id_to_token = {}
        for idx, tok in enumerate(self.SPECIAL_TOKENS):
            self.token_to_id[tok] = idx
            self.id_to_token[idx] = tok

        # Step 2 — collect every unique character in the corpus
        seen_chars = set()
        for text in texts:
            seen_chars.update(text)

        # Step 3 — sort for reproducibility, then assign consecutive IDs
        next_id = len(self.SPECIAL_TOKENS)
        for ch in sorted(seen_chars):
            if ch not in self.token_to_id:   # don't overwrite specials
                self.token_to_id[ch] = next_id
                self.id_to_token[next_id] = ch
                next_id += 1

        self._built = True
        print(f"[Tokenizer] Vocabulary built: {len(self.token_to_id)} tokens "
              f"({len(self.SPECIAL_TOKENS)} special + "
              f"{len(self.token_to_id) - len(self.SPECIAL_TOKENS)} characters)")
        return self

    # -----------------------------------------------------------------------
    # Encoding  (text → token IDs)
    # -----------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_eos: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """
        Convert a string into a list of integer token IDs.

        Parameters
        ----------
        text       : input string (may contain special token strings like <Q>)
        add_eos    : if True, append the <EOS> token at the end
        max_length : if set, truncate to this many tokens

        Returns
        -------
        List[int] of token IDs
        """
        assert self._built, "Call build() before encode()"

        # Expand special tokens embedded in the string:
        # e.g. "<Q>3 + 4<A>7<EOS>" must be split so that <Q>, <A>, <EOS>
        # are treated as single tokens, not as sequences of characters.
        tokens = self._split_with_specials(text)

        ids = []
        unk_id = self.token_to_id["<UNK>"]
        for tok in tokens:
            ids.append(self.token_to_id.get(tok, unk_id))

        if add_eos:
            ids.append(self.token_to_id["<EOS>"])

        if max_length is not None:
            ids = ids[:max_length]

        return ids

    def _split_with_specials(self, text: str) -> List[str]:
        """
        Split text into a list of tokens where special token strings
        (e.g. '<Q>', '<A>', '<EOS>') are kept as single units and the
        remaining characters are each their own token.
        """
        # Build a sorted list of special tokens (longest first to avoid
        # partial matches, e.g. '<BOS>' before '<B>').
        specials = sorted(self.SPECIAL_TOKENS, key=len, reverse=True)

        result = []
        i = 0
        while i < len(text):
            matched = False
            for sp in specials:
                if text[i:i + len(sp)] == sp:
                    result.append(sp)
                    i += len(sp)
                    matched = True
                    break
            if not matched:
                result.append(text[i])
                i += 1
        return result

    # -----------------------------------------------------------------------
    # Decoding  (token IDs → text)
    # -----------------------------------------------------------------------

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = False,
    ) -> str:
        """
        Convert a list of integer token IDs back into a string.

        Parameters
        ----------
        ids                 : sequence of token IDs
        skip_special_tokens : if True, omit <PAD>, <EOS>, <Q>, <A>, <BOS>

        Returns
        -------
        str
        """
        assert self._built, "Call build() before decode()"
        special_set = set(self.SPECIAL_TOKENS)
        parts = []
        for idx in ids:
            tok = self.id_to_token.get(idx, "<UNK>")
            if skip_special_tokens and tok in special_set:
                continue
            parts.append(tok)
        return "".join(parts)

    # -----------------------------------------------------------------------
    # Batch helpers
    # -----------------------------------------------------------------------

    def batch_encode(
        self,
        texts: List[str],
        add_eos: bool = True,
        max_length: Optional[int] = None,
        pad: bool = True,
    ) -> Dict:
        """
        Encode a list of strings into a padded 2-D integer array.

        Returns a dict with keys:
            "input_ids"      : List[List[int]]  shape (B, L)
            "attention_mask" : List[List[int]]  1 = real token, 0 = pad
        """
        encoded = [self.encode(t, add_eos=add_eos, max_length=max_length)
                   for t in texts]
        if pad:
            max_len = max(len(e) for e in encoded)
            pad_id  = self.token_to_id["<PAD>"]
            masks   = []
            padded  = []
            for e in encoded:
                length = len(e)
                padded.append(e + [pad_id] * (max_len - length))
                masks.append([1] * length + [0] * (max_len - length))
            return {"input_ids": padded, "attention_mask": masks}
        return {"input_ids": encoded}

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save vocabulary to a JSON file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "token_to_id": self.token_to_id,
            "special_tokens": self.SPECIAL_TOKENS,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[Tokenizer] Saved to {path}  ({len(self.token_to_id)} tokens)")

    @classmethod
    def load(cls, path: str) -> "MathTokenizer":
        """Load vocabulary from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.token_to_id = {k: int(v) for k, v in data["token_to_id"].items()}
        tok.id_to_token  = {int(v): k for k, v in data["token_to_id"].items()}
        tok._built = True
        print(f"[Tokenizer] Loaded from {path}  ({len(tok.token_to_id)} tokens)")
        return tok

    # -----------------------------------------------------------------------
    # Convenience properties
    # -----------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<EOS>"]

    @property
    def q_id(self) -> int:
        return self.token_to_id["<Q>"]

    @property
    def a_id(self) -> int:
        return self.token_to_id["<A>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<BOS>"]

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        status = "built" if self._built else "empty"
        return f"MathTokenizer(vocab_size={self.vocab_size}, status={status})"


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tok = MathTokenizer()
    corpus = [
        "<Q>What is 3 + 4?<A>7<EOS>",
        "<Q>Solve 2x = 10<A>x = 5<EOS>",
        "<Q>12 * 34 = ?<A>408<EOS>",
        "<Q>sqrt(9) = ?<A>3<EOS>",
        "<Q>d/dx x^2 = ?<A>2x<EOS>",
    ]
    tok.build(corpus)
    print(tok)

    test = "<Q>3 + 4<A>7<EOS>"
    ids  = tok.encode(test)
    back = tok.decode(ids)
    assert back == test, f"Round-trip failed: {back!r} != {test!r}"
    print(f"Round-trip OK: {test!r}  →  {ids}  →  {back!r}")

    # Padding test
    batch = tok.batch_encode(["<Q>1+1<A>2<EOS>", "<Q>100+200<A>300<EOS>"])
    for row, mask in zip(batch["input_ids"], batch["attention_mask"]):
        print(f"  ids={row}  mask={mask}")
