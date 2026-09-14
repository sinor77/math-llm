# Math LLM — A Small Language Model for Mathematics, Built from Scratch

A complete decoder-only GPT-style Transformer designed to solve mathematical
problems step-by-step. Every component — attention, embeddings, positional
encoding, training loop, tokenizer — is implemented from PyTorch primitives.
No pretrained weights. No black-box LLM APIs.

**Implementation status:** complete and verified — real dataset pipeline,
tokenizer, label masking, model forward pass, and generation all pass
automated tests (`audit.py`) and a mandatory tiny-overfit sanity check
(`tiny_overfit.py`, 100% memorization + 100% generation accuracy on 60 real
examples). A full-budget GPU training run on Colab T4 is still needed to
report final test-set accuracy — see the [Results](#results) section.

---

## Table of Contents

1. [What this project does](#what-this-project-does)
2. [Concepts explained for beginners](#concepts-explained-for-beginners)
   - What is an LLM?
   - What is a Transformer?
   - What does self-attention do?
   - What is tokenization?
   - What are embeddings?
   - What is positional encoding?
   - What is causal masking?
   - What is next-token prediction?
   - What is loss?
   - How does backpropagation train the model?
   - How does the model generate a solution?
   - How is accuracy calculated?
   - What does "from scratch" mean here?
3. [Project structure](#project-structure)
4. [Quick start](#quick-start)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Generation](#generation)
8. [Experiments](#experiments)
9. [Results](#results)
10. [Limitations and next steps](#limitations-and-next-steps)

---

## What this project does

We train a small language model (2 M – 20 M parameters) to solve math problems
like:

```
Input:   What is 47 + 83?
Output:  130

Input:   Solve 3x + 7 = 22
Output:  x = 5
```

The model learns by reading thousands of (question, answer) pairs and learning
the pattern of mathematical reasoning token by token. It is evaluated on
problems it has **never seen during training**.

---

## Concepts explained for beginners

### What is an LLM?

A **Large Language Model** (LLM) is a neural network trained to predict the
next word (or character, or token) in a sequence of text. By learning to
predict "what comes next" from billions of examples, the model develops an
internal representation of language, facts, and reasoning patterns.

This project builds a *small* LLM — the same architecture used in GPT — but
trained only on mathematics.

---

### What is a Transformer?

The **Transformer** is the neural-network architecture behind every modern LLM
(GPT, LLaMA, Claude, Gemini, …). It was introduced in the 2017 paper
*"Attention Is All You Need"* by Vaswani et al.

Key properties:
- Processes all tokens in a sequence **in parallel** (unlike RNNs which go one
  at a time).
- Uses **self-attention** to let every token communicate with every other token.
- Stacks many identical **blocks** (layers) to build deeper representations.

Our model is a **decoder-only** Transformer (like GPT) — it generates tokens
left to right, one at a time.

```
Input tokens:  <Q> W h a t   i s   3 + 4 ? <A>
                         ↓  (N Transformer blocks)
Output logits: probability distribution over the vocabulary at each position
                         ↓  (argmax or sampling)
Next token:    7
```

---

### What does self-attention do?

**Self-attention** allows each token to look at other tokens in the sequence
and decide how much to "pay attention" to each one.

For the token `4` in `3 + 4`, the model might learn to pay high attention to
`3` and `+` because those are the context needed to predict the answer.

The mechanism works by computing three vectors for each token:

| Vector | Question it answers |
|--------|-------------------|
| **Query** (Q) | "What am I looking for?" |
| **Key** (K)   | "What do I contain?" |
| **Value** (V) | "What do I actually give you?" |

The attention score between positions *i* and *j* is:

```
score(i, j) = Q[i] · K[j] / sqrt(d_k)
```

After softmax, these become weights that sum to 1. The output for position *i*
is the **weighted average** of all Value vectors — so it's a blend of
information from every position the model chose to attend to.

**Multiple heads** run this process in parallel with different learned
projections, letting the model capture different types of relationships
simultaneously.

---

### What is tokenization?

Before a neural network can process text, the text must be converted into
numbers. **Tokenization** is the process of splitting text into discrete units
(**tokens**) and assigning each unit an integer ID.

This project uses a **character-level tokenizer**:

```
"3 + 4 = 7"
→  ['3', ' ', '+', ' ', '4', ' ', '=', ' ', '7']
→  [18, 5, 12, 5, 19, 5, 11, 5, 22]   (example IDs)
```

Every character gets its own ID. This is simple, fully transparent, and
handles novel number combinations the model has never seen (because `1234567`
is just the sequence `1`, `2`, `3`, `4`, `5`, `6`, `7`).

Special tokens mark structure:

| Token | Meaning |
|-------|---------|
| `<Q>` | Start of question |
| `<A>` | Start of answer |
| `<EOS>` | End of sequence |
| `<PAD>` | Padding (fills shorter sequences in a batch) |

A full training example looks like:
```
<Q>What is 3 + 4?<A>7<EOS>
```

---

### What are embeddings?

The model cannot work directly with integer token IDs. Instead, each ID is
looked up in a **learned embedding table** — a matrix of shape
`(vocab_size, d_model)`.

Token ID 42 → row 42 of the table → a dense vector of `d_model` floats.

Initially these vectors are random. During training, the model learns what
each vector should represent so that similar tokens end up with similar
vectors (e.g. `+` and `-` might be close together in embedding space).

---

### What is positional encoding?

Self-attention is **order-invariant** — it doesn't naturally know whether
token `A` comes before or after token `B`. To fix this, we add a
**positional encoding** to each token's embedding.

We use a **learned positional embedding table** of shape
`(max_seq_len, d_model)` — one vector per position 0 … T-1.

The final input to the Transformer is:

```
x[t] = token_embedding[token_id[t]] + position_embedding[t]
```

This gives the model both *what* the token is and *where* it is in the
sequence.

---

### What is causal masking?

Because we're building a **decoder** that generates text left to right, we
must ensure the model cannot "cheat" by looking at future tokens when
predicting the current token.

We enforce this with a **causal mask** — an upper-triangular matrix of `-inf`
values that is added to the attention scores before softmax:

```
Position:  0  1  2  3  4
         [ 0 -∞ -∞ -∞ -∞ ]  ← token 0 can only see itself
         [ 0  0 -∞ -∞ -∞ ]  ← token 1 can see 0,1
         [ 0  0  0 -∞ -∞ ]  ← token 2 can see 0,1,2
         [ 0  0  0  0 -∞ ]  ← token 3 can see 0,1,2,3
         [ 0  0  0  0  0 ]  ← token 4 can see all
```

After softmax, `-inf` becomes `0` — those positions contribute nothing to
the output.

---

### What is next-token prediction?

The model is trained with a simple objective: **predict the next token**.

Given the sequence `<Q>What is 3+4?<A>`, the model should predict `7`.
Given `<Q>What is 3+4?<A>7`, it should predict `<EOS>`.

At every position the model outputs a probability distribution over the entire
vocabulary. We measure how wrong it is with **cross-entropy loss** and use
that error signal to update the weights.

We only compute loss over the **answer** tokens, not the question tokens
(there's nothing for the model to "predict" in the question — it's given as
context).

---

### What is loss?

**Loss** is a number that measures how wrong the model's predictions are.

We use **cross-entropy loss**:

```
loss = -log( probability assigned to the correct next token )
```

If the model assigns probability 1.0 to the right token: loss = 0 (perfect).
If the model assigns probability 0.01 to the right token: loss ≈ 4.6 (very wrong).

The training goal is to minimise the average loss over all training examples.
A lower loss means the model's predictions are, on average, more accurate.

---

### How does backpropagation train the model?

**Backpropagation** is an algorithm that computes the gradient of the loss
with respect to every learnable parameter in the model.

The gradient tells us: "if I increase this parameter by a small amount, does
the loss go up or down?" We then move each parameter in the direction that
decreases the loss.

The process:

1. **Forward pass** — compute the model output and loss.
2. **Backward pass** — compute gradients automatically (PyTorch's autograd).
3. **Optimiser step** — update parameters: `param = param - lr * grad`.
4. Repeat for thousands of batches.

We use the **AdamW** optimiser, which adapts the step size per parameter and
applies weight decay (L2 regularisation) to prevent overfitting.

---

### How does the model generate a solution?

At inference time:

1. Encode the question: `<Q>What is 47 + 83?<A>` → list of token IDs.
2. Feed into the model → logits for the **next** token.
3. Take the argmax (greedy) or sample from the distribution → next token.
4. Append that token to the sequence.
5. Repeat from step 2 until `<EOS>` is produced.

```
Step 0:  prompt  = <Q>What is 47+83?<A>          → predicts '1'
Step 1:  context = <Q>What is 47+83?<A>1         → predicts '3'
Step 2:  context = <Q>What is 47+83?<A>13        → predicts '0'
Step 3:  context = <Q>What is 47+83?<A>130       → predicts <EOS>
Answer: 130  ✓
```

---

### How is accuracy calculated?

We compute three metrics:

**1. Token accuracy** (fast, teacher-forced)
The percentage of individual tokens the model correctly predicts when given
the gold prefix. This is computed in a single forward pass and is fast.

**2. Exact-answer accuracy** (primary metric)
The model generates the answer autoregressively (no gold prefix). We extract
the text between `<A>` and `<EOS>` and compare it character-for-character
to the ground-truth answer. A prediction is correct only if it is an exact
match. This is the number reported as the headline result.

**3. Full-sequence accuracy**
The entire generated sequence (including any reasoning steps) must exactly
match the reference.

All metrics are computed on the **held-out test set** — examples the model
never saw during training.

---

### What does "from scratch" mean here?

"From scratch" means:

- ✅ No pretrained weights loaded from GPT, LLaMA, BERT, or any other model.
- ✅ No calls to an external LLM API (OpenAI, Anthropic, etc.).
- ✅ The Transformer architecture (attention, FFN, positional encoding,
  layer norm, residuals) is implemented using only PyTorch tensor operations —
  not `nn.Transformer` or `transformers.AutoModel`.
- ✅ The tokenizer is written by hand — no HuggingFace tokenizers library.
- ✅ The training loop (loss, gradients, optimiser, scheduler, checkpointing)
  is written explicitly.

We **do** use PyTorch for tensor math, automatic differentiation, and GPU
support. The dataset is downloaded directly from its official static-file
source using only the Python standard library (`urllib`, `tarfile`) — no
HuggingFace `datasets`/`huggingface_hub` dependency is needed at all. These
are infrastructure tools, not model logic.

---

## Project structure

```
math-llm/
├── README.md              ← This file
├── requirements.txt       ← Python dependencies
│
├── config.py              ← All hyper-parameters (model, training, data)
├── tokenizer.py           ← Character-level math tokenizer
├── dataset.py             ← Data loading, splitting, PyTorch Dataset/DataLoader
├── attention.py           ← Causal multi-head self-attention (from scratch)
├── transformer.py         ← TransformerBlock and TransformerDecoder stack
├── model.py               ← Top-level MathLLM model (embeddings + decoder + head)
├── utils.py               ← Logging, checkpointing, LR schedule, metrics
├── train.py               ← Complete training loop
├── evaluate.py            ← Token / exact-answer / category accuracy
├── generate.py            ← Autoregressive generation (greedy, sampling, beam)
├── experiments.py         ← Automated experiment runner
├── tiny_overfit.py        ← MANDATORY sanity check: can the model memorize
│                            60 real examples before a long training run?
├── audit.py               ← Torch-free tests: tokenizer, splits, label
│                            masking, answer extraction, real-data parsing
│
├── notebooks/
│   └── Math_LLM_Colab.ipynb  ← Complete Google Colab notebook
│
├── checkpoints/           ← Saved model checkpoints (git-ignored)
│   ├── best_model.pt
│   ├── final_model.pt
│   └── tokenizer.json
│
├── results/               ← Evaluation results and experiment logs
│   ├── eval_results.json
│   ├── experiment_log.csv
│   ├── training_history.json
│   └── training_curve.png
│
└── docs/                  ← Additional documentation
```

---

## Quick start

### Prerequisites

```bash
pip install -r requirements.txt
```

A CUDA GPU is strongly recommended. Training on CPU is possible but very slow.

### Verify components individually

```bash
# Tokenizer
python tokenizer.py

# Dataset (downloads the real DeepMind Mathematics Dataset, ~2.3 GB one-time)
python dataset.py

# Attention mechanism
python attention.py

# Full model + parameter count
python model.py

# MANDATORY before any long training run: can the model memorize 60 real examples?
python tiny_overfit.py

# Torch-free unit tests (tokenizer, splits, label masking, answer extraction)
python audit.py
```

### Train the baseline model

```bash
python train.py --model small-2M --stage 1 --max-steps 10000
```

### Evaluate on the test set

```bash
python evaluate.py --checkpoint checkpoints/best_model.pt
```

### Generate answers interactively

```bash
python generate.py --checkpoint checkpoints/best_model.pt --interactive
```

---

## Training

The training script supports several model sizes and data stages:

```bash
# Stage 1: basic arithmetic only (fastest)
python train.py --model small-2M --stage 1 --max-steps 10000

# Stage 2: arithmetic + comparison
python train.py --model medium-5M --stage 2 --max-steps 15000

# Stage 3: all categories including algebra and probability
python train.py --model large-10M --stage 3 --max-steps 30000

# Resume from checkpoint
python train.py --model medium-5M --stage 2 --resume checkpoints/step_0005000.pt
```

**Training hyperparameters** are set in `config.py` and can be overridden:

```bash
python train.py --model small-2M --lr 1e-3 --batch-size 128 --max-steps 20000
```

Checkpoints are saved to `checkpoints/` every 1000 steps. The best model
(lowest validation loss) is saved as `checkpoints/best_model.pt`.

---

## Evaluation

```bash
# Full evaluation on the test set
python evaluate.py --checkpoint checkpoints/best_model.pt

# Evaluate with more generated tokens (for longer answers)
python evaluate.py --checkpoint checkpoints/best_model.pt --max-new-tokens 128

# Show more failure examples
python evaluate.py --checkpoint checkpoints/best_model.pt --show-failures 20
```

Example output:

```
============================================================
EVALUATION RESULTS  (held-out test set)
============================================================
  Token accuracy          : 0.9821
  Exact-answer accuracy   : 0.9340
  Full-sequence accuracy  : 0.9120
  Test examples evaluated : 1500
  Correct (exact)         : 1401

Category breakdown:
category                accuracy  correct  n
arithmetic__add_or_sub  0.9800    490      500
arithmetic__mul         0.9540    477      500
arithmetic__div         0.9210    460      500
arithmetic__mixed       0.8680    434      500
```
*(Illustrative format only — see [Results](#results) for actual measured numbers.)*

---

## Generation

```bash
# Single question
python generate.py --checkpoint checkpoints/best_model.pt \
    --question "What is 123 + 456?"

# Interactive mode
python generate.py --checkpoint checkpoints/best_model.pt --interactive

# Beam search (more accurate, slower)
python generate.py --checkpoint checkpoints/best_model.pt \
    --question "Solve 5x - 3 = 17" --beam-size 4

# Sampling with temperature
python generate.py --checkpoint checkpoints/best_model.pt \
    --question "What is 7 * 8?" --temperature 0.7 --top-k 10
```

---

## Experiments

The experiment runner tests model sizes, learning rates, batch sizes, and
data stages in a controlled, staged way:

```bash
# Quick baseline (recommended first run)
python experiments.py --quick

# Stages A and B (baseline + model capacity)
python experiments.py --stages AB

# All stages (full experiment suite)
python experiments.py --stages ABCDE

# Dry run (just build models and count parameters)
python experiments.py --dry-run
```

Results are logged to `results/experiment_log.csv` and
`results/best_experiment.json`.

### Experiment stages

| Stage | What we test |
|-------|-------------|
| A | Baseline: small-2M model, Stage-1 data |
| B | Model capacity: 5M and 10M parameters |
| C | Training recipe: LR and batch size variants |
| D | Data coverage: Stage-2 and Stage-3 categories |
| E | Extended training: best config, full budget |

---

## Results

> **Status: pipeline verified end-to-end; full-budget GPU run pending.**
> Everything below was actually executed and measured in this repository
> (CPU, no GPU available in the environment used to fix the pipeline) — no
> numbers are estimated or assumed. A full 20K-step run on a Colab T4 GPU
> (which trains far faster than CPU) is still needed for a reportable final
> test-set accuracy; run `notebooks/Math_LLM_Colab.ipynb` and copy the
> numbers from `results/eval_results.json` into the table below.

### Mandatory tiny-overfit sanity check (`python tiny_overfit.py`)

60 REAL examples (`arithmetic__add_or_sub`, `train-easy`), tiny-1M model, 800 steps, CPU:

| Metric | Value |
|--------|-------|
| Initial loss | 3.9725 |
| Final loss | 0.0003 |
| Final token accuracy | 1.0000 |
| Teacher-forced exact-sequence accuracy | 1.0000 (60/60) |
| Autoregressive generation accuracy | 1.0000 (60/60) |

The model memorizes the tiny set and reproduces every answer through real
autoregressive generation (not just teacher-forcing) — see [Tiny overfit](#tiny-overfit)
in the report for full sample output.

### Proof-of-concept real-data training run (CPU, partial budget)

To prove the *fixed* pipeline actually learns from the real dataset (not
just memorizes 60 examples), a small-2M model was trained on Stage-1
categories (`arithmetic__add_or_sub`, `arithmetic__mul`, `arithmetic__div`,
`arithmetic__mixed`; 3,000 train-easy examples/category, 10,799 train /
1,201 val after split) for 1,500 of a planned 2,000 steps on CPU (time-limited
by the environment, not by the pipeline):

| Step | Val loss | Val token accuracy |
|------|----------|---------------------|
| 0 | 4.0950 | 0.036 |
| 500 | 1.7011 | 0.451 |
| 1000 | 1.5849 | 0.462 |
| 1500 | 1.5142 | 0.480 |

Loss falls and token accuracy rises steadily and is on the correctly
shifted (causally aligned) metric — the exact metric an earlier version of
this codebase computed off-by-one, which silently reported near-random
accuracy even when the model was learning correctly (see
[Root cause #4](#root-causes) in the fix report). This run did not reach
enough steps for a meaningful exact-answer accuracy number on 8-digit/
decimal arithmetic — reporting one from a truncated CPU run would be
fabricating precision the run doesn't support. The Colab T4 GPU run
(exact same code path) is needed to reach the full step budget and report
real exact-answer accuracy.

### Experimental results (fill in after the Colab T4 run)

| Run | Model | Dataset | Stage | Steps | Val loss | Token acc | Exact-answer acc | Test examples |
|-----|-------|---------|-------|-------|----------|-----------|-----------------|---------------|
| — | — | — | — | — | — | — | — | — |

*Fill this table in after running `notebooks/Math_LLM_Colab.ipynb`.*
*Copy the numbers from `results/eval_results.json` once the run finishes.*

### What the metrics mean

| Metric | How it's measured |
|--------|------------------|
| Token accuracy | % of individual tokens correctly predicted — teacher-forced (fast, optimistic). Correctly shifted by one position to match the causal-LM loss (see `utils.token_accuracy`). |
| Exact-answer accuracy | Model generates the full answer autoregressively; compared character-for-character to gold. This is the headline number. |
| Test examples | Size of the held-out test set — the dataset authors' own `interpolate` split, independently generated from the `train-easy` split used for training/validation. |

### Data integrity guarantees

- The dataset is the real **DeepMind Mathematics Dataset** (Saxton et al., ICLR 2019), downloaded directly from its official source (`storage.googleapis.com/mathematics-dataset`) — see `config.py`/`dataset.py` docstrings.
- The test set is the dataset's own **`interpolate`** split: questions generated independently from the `train-easy` split used for train/val. It is never touched during training.
- Train/val (both from `train-easy`) are further split by a deterministic MD5 hash of the question string, so re-running the pipeline never reshuffles an example into a different split.
- `dataset.check_split_overlap()` actually computes (never merely assumes) exact and whitespace/case-normalised overlap between all three splits and prints the result on every run — see `results/eval_results.json` / training logs for the measured values (expected: zero).
- The tokenizer is built on all data (train + val + test) to avoid `<UNK>` tokens, but it only stores character→ID mappings — it has no knowledge of which split an example belongs to.
- The dataset source, configuration, and split sizes are recorded in every checkpoint (`data_source_info`) and printed in every evaluation report. Loading NEVER silently falls back to synthetic data — a real-data load failure raises `RealDatasetLoadError` and stops the run.

### How to read the results files

After a Colab run:

```
results/eval_results.json        — overall + per-category accuracy
results/training_history.json    — loss and token accuracy per eval step
results/training_curve.png       — loss / accuracy plot
checkpoints/best_model.pt        — best checkpoint (lowest val loss)
checkpoints/tokenizer.json       — vocabulary file
```

---

## Limitations and next steps

**Current limitations:**

- Character-level tokenization means multi-digit arithmetic requires the model
  to learn place-value composition from scratch, which needs significant training
  data and capacity.
- The model has no symbolic computation engine — it learns patterns, not rules.
  It may fail on out-of-distribution number ranges.
- Beam search helps but greedy decoding occasionally drops a digit.
- Calculus and complex algebra require more training data and a larger model.

**If 100% accuracy is not reached on the first run:**

1. Check `results/eval_results.json` for failure patterns.
2. Try `--curriculum default` (see below) — usually the single biggest lever for exact-answer accuracy specifically.
3. Try a larger model (`medium-5M` or `large-10M`).
4. Train for more steps (`--curriculum thorough`, or `--train thorough` without a curriculum).
5. Add more data categories (`--stage 2` or `--stage 3`).
6. Use beam search at evaluation (`--beam-size 4`).
7. Run `python experiments.py --analyse` for a failure analysis.

### Curriculum training (easy → hard by number length)

A character-level model has no built-in notion of place value — it has to
learn digit-by-digit carrying purely from examples. Training on the full
real dataset from step 0 (numbers up to 8 digits, many decimal places)
makes exact-answer accuracy stay low for a long time even as loss falls
steadily, because every single digit position has to be right
simultaneously for an answer to count as correct.

`--curriculum` restricts training to short numbers first, then
progressively lifts the restriction — using the same REAL data throughout,
just a different (verified non-empty) subset at each stage:

```bash
python train.py --model small-2M --stage 1 --curriculum default
```

Presets (`none` / `default` / `fast` / `thorough`) are defined in
`config.get_curriculum()`, calibrated against the actual digit-length
distribution of the real `train-easy` data (not guessed) so every stage
has thousands of matching examples. The cosine LR schedule spans the
curriculum's full step budget continuously — it does not reset at stage
transitions.

### Generated arithmetic benchmark (a bounded, easier target)

The real dataset's full difficulty (8-digit numbers, arbitrary decimals,
negatives) is genuinely hard for a small model to reach 95%+ exact-answer
accuracy on — that's an architecture/task-difficulty ceiling, not a bug
(see [Results](#results)). `--dataset-source generated` swaps in a
**controlled, Python-generated arithmetic benchmark** instead:

```
123 + 456 = 579        (addition)
842 - 317 = 525        (subtraction — always non-negative by construction:
                         we don't stack "learn negative numbers" on top of
                         "learn carrying" as a second problem)
27 * 43 = 1161         (multiplication)
864 / 24 = 36          (division — ALWAYS exact integer division: we pick
                         the divisor and quotient first and multiply them
                         to get the dividend, so there's never a decimal/
                         remainder to represent either)
```

Every equation is still real, exact arithmetic (computed once, in Python,
to build the corpus) — this is an explicit, clearly-labelled data source,
never a silent substitution: `source_info['data_source']` always says
`"GENERATED"` here, `"REAL"` for the DeepMind dataset, never confused.
Train and test are generated from **independent RNG seeds** (not split
from one shared pool), and `check_split_overlap()` still verifies zero
overlap on top of that rather than just assuming independence guarantees it.

```bash
python train.py --model small-2M --curriculum digits_1_4_long \
    --dataset-source generated --ops "+,-,*" --reverse-answer --patience 15
```

`digits_1_4_long` ramps the operand digit cap 1 → 2 → 3 → 4 (cumulative —
each stage adds harder examples on top of, not instead of, the easier
ones), spending most of its 80K-step budget on the final, full-range
stage. `evaluate.py`/`generate.py` automatically read which dataset a
checkpoint was trained on (including `--reverse-answer`) from its saved
`data_source_info` — you don't need to re-specify these flags at eval time.

### Reversed-digit answers (`--reverse-answer`)

A real measured run without this flag (small-2M, `digits_1_4_long`,
`+,-,*` only) got 86.6%/89.8% exact-answer accuracy on addition/subtraction
— strong, but the failures were almost all the same pattern:

```
9286 + 3247 = 12533   predicted: 2533    (dropped the leading "1")
6556 + 5749 = 12305   predicted: 2305    (same pattern)
1438 + 8898 = 10336   predicted: 0336    (same pattern)
```

Every failure is a case where carrying produces a result **one digit
longer than both operands** (4-digit + 4-digit → 5-digit). Generating the
answer most-significant-digit-first forces the model to commit to the
answer's length before it has seen the full carry chain — so it
systematically failed exactly there.

`--reverse-answer` writes the answer's digits least-significant-first in
the *training text only* (`10366` → `"66301"`); `generate.generate_answer()`
reverses it back automatically everywhere it's used, so `item["answer"]`,
what you see in the notebook, in `evaluate.py`'s output, and in
`generate.py --interactive` is always normal reading order — only the
model's internal training/generation target is affected. This lets "does
this carry need one more digit" be decided naturally at the *end* of
generation instead of the *start*. The same technique (there, as part of
full chain-of-thought reasoning) is used by
[brendanlong/math-llm](https://github.com/brendanlong/math-llm) for
exactly this reason, which independently corroborates it.

### Multiplication chain-of-thought (`--mul-cot`)

Even with `--reverse-answer`, a measured run (small-2M, `digits_1_4_long`,
`+,-,*`) got add=95.8%/sub=99.4% — both at/above the >95% target — but
**mul stuck at only 20.0%**. Multiplication of two multi-digit numbers has
no local digit-by-digit pattern the way carrying does: the model has to
implicitly compute cross-digit partial products and sum them, all in one
autoregressive shot. Neither curriculum nor reversed-digit encoding
(both aimed at the carry-length problem) meaningfully moved it.

`--mul-cot` trains `*` examples as an explicit long-multiplication chain
of thought instead of a single-shot final answer — one partial product per
nonzero digit of the second operand, then a running sum:

```
23 * 45   ->   23*5=115,23*40=920,115+920=1035
91 * 7    ->   91*7=637                          (single nonzero digit: no decomposition needed)
```

This reduces multiplication to two sub-problems the model has already
shown it can do: multiply a multi-digit number by a **single digit** (far
simpler than full multi-digit × multi-digit), and **sum a short list of
numbers** (already ~95%+ solved for addition). Only the text after the
*last* `=` is scored as the answer (`generate._extract_final_answer()`);
`item["answer"]` stays just the final numeric result throughout, and
`--reverse-answer` composes with it (every number in the chain, not just
the final one, is written least-significant-digit-first, via
`dataset.reverse_digit_runs()` — the same "decide the length at the END"
fix applied per-step instead of once). Same divide-and-conquer idea used
by [brendanlong/math-llm](https://github.com/brendanlong/math-llm)'s
chain-of-thought approach.

Longer answers need a bigger generation budget: `generate.py`/`evaluate.py`
default `max_new_tokens` was raised from 64 to 160 (a 4-digit × 4-digit
chain of thought runs to ~130 characters; 64 would truncate it before the
final `=result` was ever generated).

```bash
python train.py --model small-2M --curriculum digits_1_4_long \
    --dataset-source generated --ops "+,-,*" --reverse-answer --mul-cot --patience 15
```

This has been verified for correctness (the chain-of-thought builder,
digit-run reversal, and final-answer extraction all round-trip correctly
against thousands of random cases, and a small CPU overfitting test
reaches 40/40 exact-match generation with `--reverse-answer --mul-cot`
together) but **not yet measured for real generalization accuracy** — that
needs a real training run on a GPU, the same way `--reverse-answer` was
validated. Run the command above and compare `generated__mul` accuracy
against the 20.0% baseline.

**Potential improvements:**

- Sub-word tokenization (better number handling)
- RL fine-tuning with a verifiable reward (exact-match) on top of a
  strong supervised base — investigated using HuggingFace TRL, but its
  `GRPOTrainer`/`PPOTrainer` require a `transformers.PreTrainedModel`
  (confirmed by reading `trl/trainer/grpo_trainer.py`), not an arbitrary
  `nn.Module`, so adopting it would mean reimplementing this model on top
  of the `transformers` stack. A small hand-written REINFORCE-style loop
  would fit this project's from-scratch philosophy better if this is
  pursued later.
- Data augmentation (more number combinations)
- Larger model with more training compute

---

## Citation / acknowledgements

Dataset: Saxton, Grefenstette, Hill & Kohli, *"Analysing Mathematical
Reasoning Abilities of Neural Models"* (ICLR 2019) — the DeepMind
Mathematics Dataset, downloaded directly from its official source:
[storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz](https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz)
(see also the [google-deepmind/mathematics_dataset](https://github.com/google-deepmind/mathematics_dataset)
repository for the original generator code).

Architecture reference: Vaswani et al., *Attention Is All You Need* (2017)
Training approach: Brown et al., *Language Models are Few-Shot Learners* (GPT-3, 2020)
