"""
generate.py
===========
Autoregressive text generation for the Math LLM.

HOW AUTOREGRESSIVE GENERATION WORKS
-------------------------------------
The model is a next-token predictor.  Given a sequence of tokens
[t0, t1, …, tN], it outputs a probability distribution over the
vocabulary for the NEXT token tN+1.

To generate a full answer we repeat:
  1. Feed the current sequence into the model.
  2. Sample (or take the argmax of) the output distribution for the LAST position.
  3. Append that token to the sequence.
  4. Repeat until <EOS> is produced or we hit max_new_tokens.

This is called "autoregressive" because each new token depends on all
previously generated tokens.

GENERATION STRATEGIES
----------------------
1. Greedy decoding (temperature=0.0)
   Always pick the highest-probability token.
   Fast and deterministic.  Can get stuck in repetitive loops.

2. Temperature sampling (temperature > 0)
   Divide logits by temperature before softmax.
   • temperature < 1.0 → sharper distribution → more confident/deterministic
   • temperature > 1.0 → flatter distribution → more random/creative
   temperature=1.0 is "pure" sampling.

3. Top-k sampling
   Before sampling, keep only the top-k highest probability tokens
   and set the rest to -inf.  Prevents very unlikely tokens.

4. Beam search
   Maintain a beam of B candidate sequences.  At each step expand each
   candidate, keep the top-B by cumulative log-probability.
   More accurate but O(B) times slower than greedy.
"""

import argparse
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F

from config import ModelConfig
from tokenizer import MathTokenizer
from model import MathLLM
from utils import get_logger, get_device, load_checkpoint

log = get_logger()


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_answer(
    model:          MathLLM,
    tokenizer:      MathTokenizer,
    question:       str,
    device:         torch.device,
    max_new_tokens: int = 64,
    temperature:    float = 0.0,    # 0 = greedy
    top_k:          int = 0,        # 0 = disabled
    top_p:          float = 1.0,    # 1.0 = disabled (nucleus sampling)
    reverse_answer: bool = False,
) -> Tuple[str, str]:
    """
    Generate an answer for a single question.

    Parameters
    ----------
    model          : trained MathLLM (eval mode)
    tokenizer      : fitted MathTokenizer
    question       : plain question string (e.g. "What is 3 + 4?")
    device         : torch device
    max_new_tokens : maximum answer tokens to generate
    temperature    : sampling temperature (0 = greedy)
    top_k          : top-k filtering (0 = off)
    top_p          : nucleus sampling threshold (1.0 = off)
    reverse_answer : set True iff the model was trained with
        dataset.format_example(..., reverse_answer=True) (i.e.
        DataConfig.generated_reverse_answer) — the model then generates
        the answer least-significant-digit-first, so the extracted text
        must be reversed back to normal reading order before it's
        returned. Get this from the checkpoint's data_source_info rather
        than guessing (see evaluate._reconstruct_data_cfg_from_checkpoint).

    Returns
    -------
    (predicted_answer, full_generation_string)
        predicted_answer     : text between <A> and <EOS>, already in
                                normal reading order regardless of
                                reverse_answer
        full_generation_string : complete generated sequence including
                                <Q>…<A>…<EOS> — NOTE: if reverse_answer is
                                True, the answer portion of this raw
                                string is still in the reversed form the
                                model actually produced (useful for
                                debugging what the model literally
                                generated); only the returned
                                predicted_answer is un-reversed.
    """
    model.eval()

    # ── Build prompt: <Q>{question}<A> ──────────────────────────────────────
    prompt = f"<Q>{question}<A>"
    prompt_ids = tokenizer.encode(prompt, add_eos=False)

    # Make sure the prompt itself isn't longer than the model's context
    max_prompt = model.cfg.max_seq_len - max_new_tokens - 1
    if len(prompt_ids) > max_prompt:
        prompt_ids = prompt_ids[:max_prompt]

    # Convert to tensor: shape (1, prompt_len)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    eos_id = tokenizer.eos_id
    generated_ids = []

    for _ in range(max_new_tokens):
        # ── Trim context if it exceeds max_seq_len ─────────────────────────
        context_ids = ids[:, -model.cfg.max_seq_len:]

        # ── Forward pass ─────────────────────────────────────────────────
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            out = model(context_ids)

        # Logits for the LAST position: (1, vocab_size)
        logits = out["logits"][:, -1, :]

        # ── Apply temperature ─────────────────────────────────────────────
        if temperature == 0.0:
            # Greedy: just take argmax
            next_id = logits.argmax(dim=-1)  # (1,)
        else:
            logits = logits / temperature

            # ── Top-k filtering ───────────────────────────────────────────
            if top_k > 0:
                top_k_actual = min(top_k, logits.size(-1))
                threshold = logits.topk(top_k_actual, dim=-1).values[:, -1, None]
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            # ── Top-p (nucleus) filtering ─────────────────────────────────
            if top_p < 1.0:
                sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                # Remove tokens with cumulative probability above top_p
                remove = cum_probs - sorted_logits.softmax(dim=-1) > top_p
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                # Scatter back to original order
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            # ── Sample ────────────────────────────────────────────────────
            probs   = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (1,)

        # ── Append token ──────────────────────────────────────────────────
        generated_ids.append(next_id.item())
        ids = torch.cat([ids, next_id.unsqueeze(1)], dim=1)

        # ── Stop at EOS ───────────────────────────────────────────────────
        if next_id.item() == eos_id:
            break

    # ── Decode ───────────────────────────────────────────────────────────────
    # Full sequence = prompt + generated tokens
    full_ids   = prompt_ids + generated_ids
    full_text  = tokenizer.decode(full_ids, skip_special_tokens=False)

    # Extract just the answer: text between <A> and <EOS>
    answer = _extract_answer(full_text)
    if reverse_answer:
        answer = answer[::-1]

    return answer, full_text


def _extract_answer(text: str) -> str:
    """
    Extract the answer string from a formatted generation.

    Input example:  "<Q>What is 3+4?<A>7<EOS>"
    Output:         "7"
    """
    a_token   = "<A>"
    eos_token = "<EOS>"

    a_pos = text.find(a_token)
    if a_pos == -1:
        return text.strip()   # fallback: return full text

    answer_start = a_pos + len(a_token)

    eos_pos = text.find(eos_token, answer_start)
    if eos_pos == -1:
        return text[answer_start:].strip()

    return text[answer_start:eos_pos].strip()


# ---------------------------------------------------------------------------
# Beam search (optional, more accurate)
# ---------------------------------------------------------------------------

@torch.no_grad()
def beam_search(
    model:          MathLLM,
    tokenizer:      MathTokenizer,
    question:       str,
    device:         torch.device,
    beam_size:      int = 4,
    max_new_tokens: int = 64,
    length_penalty: float = 1.0,
    reverse_answer: bool = False,
) -> Tuple[str, str]:
    """
    Beam search decoding.

    Maintains `beam_size` candidate sequences and expands the one with
    the highest cumulative log-probability at each step.

    Parameters
    ----------
    beam_size      : number of beams
    length_penalty : > 1 favours longer sequences, < 1 shorter
    reverse_answer : see generate_answer() — undoes least-significant-
        digit-first training format if the model was trained that way.

    Returns
    -------
    (best_answer, best_full_generation)
    """
    model.eval()

    prompt     = f"<Q>{question}<A>"
    prompt_ids = tokenizer.encode(prompt, add_eos=False)
    eos_id     = tokenizer.eos_id

    # Beam = list of (score, list_of_ids, is_finished)
    beams = [(0.0, list(prompt_ids), False)]

    for step in range(max_new_tokens):
        if all(b[2] for b in beams):   # all beams finished
            break

        candidates = []

        for score, beam_ids, finished in beams:
            if finished:
                candidates.append((score, beam_ids, True))
                continue

            # Forward pass on this beam
            ctx = torch.tensor([beam_ids[-model.cfg.max_seq_len:]],
                                dtype=torch.long, device=device)
            out    = model(ctx)
            logits = out["logits"][0, -1, :]                  # (V,)
            log_probs = F.log_softmax(logits, dim=-1)

            # Expand: take top-beam_size tokens
            top_log_probs, top_ids = log_probs.topk(beam_size)

            for lp, tid in zip(top_log_probs.tolist(), top_ids.tolist()):
                new_score = score + lp
                new_ids   = beam_ids + [tid]
                is_done   = (tid == eos_id)
                # Length-normalised score for ranking
                norm_score = new_score / (len(new_ids) ** length_penalty)
                candidates.append((norm_score, new_ids, is_done))

        # Keep the top beam_size candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_size]

    # Return the highest-scoring beam
    best_score, best_ids, _ = beams[0]
    full_text = tokenizer.decode(best_ids, skip_special_tokens=False)
    answer    = _extract_answer(full_text)
    if reverse_answer:
        answer = answer[::-1]
    return answer, full_text


# ---------------------------------------------------------------------------
# Batch generation (for evaluation speed)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_batch(
    model:          MathLLM,
    tokenizer:      MathTokenizer,
    questions:      List[str],
    device:         torch.device,
    max_new_tokens: int = 64,
    temperature:    float = 0.0,
    reverse_answer: bool = False,
) -> List[Tuple[str, str]]:
    """
    Generate answers for a list of questions sequentially.
    (True batched generation is more complex; this is a simple loop wrapper.)

    Returns
    -------
    List of (answer, full_generation) tuples
    """
    return [
        generate_answer(model, tokenizer, q, device,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        reverse_answer=reverse_answer)
        for q in questions
    ]


# ---------------------------------------------------------------------------
# Interactive generation demo
# ---------------------------------------------------------------------------

def interactive_demo(
    model:          MathLLM,
    tokenizer:      MathTokenizer,
    device:         torch.device,
    max_new_tokens: int = 64,
    temperature:    float = 0.0,
    reverse_answer: bool = False,
) -> None:
    """
    Run an interactive loop: user types a math question, model answers.
    Type 'quit' or press Ctrl-C to exit.

    reverse_answer : see generate_answer() — pass True iff this model was
        trained with DataConfig.generated_reverse_answer=True, so the
        displayed answer is un-reversed back to normal reading order.
    """
    print("\nSmall Math LLM")
    print("Type a mathematical question.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not question:
            continue

        # The displayed answer comes straight from the model's own
        # generated tokens (extract_answer just slices <A>...<EOS> out of
        # what the model produced) — no calculator, no external API, no
        # hard-coded lookup is involved anywhere in this path.
        answer, _full = generate_answer(
            model, tokenizer, question, device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            reverse_answer=reverse_answer,
        )
        print(f"\nModel: {answer}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate answers with the Math LLM")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--question",   default=None,
                   help="Single question to answer (non-interactive mode)")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k",      type=int, default=0)
    p.add_argument("--beam-size",  type=int, default=1)
    p.add_argument("--interactive", action="store_true",
                   help="Start interactive question-answering loop")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = get_device(args.device)

    # ── Load checkpoint ────────────────────────────────────────────────────
    log.info(f"Loading: {args.checkpoint}")
    raw = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model_cfg = ModelConfig(**raw["cfg_model"])
    model = MathLLM(model_cfg).to(device)
    model.load_state_dict(raw["model_state"])
    model.eval()

    tok_path  = raw.get("tokenizer_path", "checkpoints/tokenizer.json")
    tokenizer = MathTokenizer.load(tok_path)

    # Read whether this checkpoint was trained with reversed-digit answers
    # straight from what it recorded at training time — never guessed.
    _data_info = raw.get("data_source_info", {}) or {}
    reverse_answer = bool(_data_info.get("configuration", {}).get("reverse_answer", False))
    if reverse_answer:
        log.info("Checkpoint trained with reverse_answer=True — un-reversing generated answers.")

    # ── Generate ───────────────────────────────────────────────────────────
    if args.question:
        if args.beam_size > 1:
            ans, full = beam_search(model, tokenizer, args.question, device,
                                    beam_size=args.beam_size,
                                    max_new_tokens=args.max_new_tokens,
                                    reverse_answer=reverse_answer)
        else:
            ans, full = generate_answer(model, tokenizer, args.question, device,
                                        max_new_tokens=args.max_new_tokens,
                                        temperature=args.temperature,
                                        top_k=args.top_k,
                                        reverse_answer=reverse_answer)
        print(f"Question : {args.question}")
        print(f"Answer   : {ans}")
        print(f"Full gen : {full}")

    elif args.interactive:
        interactive_demo(model, tokenizer, device,
                         max_new_tokens=args.max_new_tokens,
                         temperature=args.temperature,
                         reverse_answer=reverse_answer)

    else:
        # Demo questions
        demo_questions = [
            "What is 12 + 45?",
            "What is 99 - 37?",
            "What is 7 * 8?",
            "What is 144 / 12?",
            "What is 23 + 14 * 2?",
        ]
        print("\nDemo predictions:")
        for q in demo_questions:
            ans, _ = generate_answer(model, tokenizer, q, device,
                                     max_new_tokens=args.max_new_tokens,
                                     reverse_answer=reverse_answer)
            print(f"  Q: {q}")
            print(f"  A: {ans}")
            print()
