# Math LLM — A Small Language Model for Mathematics, Built from Scratch

A complete decoder-only GPT-style Transformer designed to solve mathematical
problems step-by-step. Every component — attention, embeddings, positional
encoding, training loop, tokenizer — is implemented from PyTorch primitives.
No pretrained weights. No black-box LLM APIs.

**Implementation status:** complete and verified (syntax, tokenizer, split logic,
answer extraction). First training run pending — see the [Results](#results) section.

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

We **do** use PyTorch for tensor math, automatic differentiation, GPU support,
and the HuggingFace `datasets` library to download the math dataset. These
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

# Dataset (downloads from HuggingFace)
python dataset.py

# Attention mechanism
python attention.py

# Full model + parameter count
python model.py
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
arithmetic__mul_or_div  0.9540    477      500
arithmetic__mixed       0.8680    434      500
```

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

> **Status: first Colab run not yet completed.**
> The table below will be filled in after the first real training run.
> The architecture, tokenizer, data pipeline, and evaluation are all
> implemented and verified. What's missing is the actual number.

### Experimental results

| Run | Model | Dataset | Stage | Steps | Val loss | Token acc | Exact-answer acc | Test examples |
|-----|-------|---------|-------|-------|----------|-----------|-----------------|---------------|
| — | — | — | — | — | — | — | — | — |

*Fill this table in after running `notebooks/Math_LLM_Colab.ipynb`.*
*Copy the numbers from `results/eval_results.json` once the run finishes.*

### What the metrics mean

| Metric | How it's measured |
|--------|------------------|
| Token accuracy | % of individual tokens correctly predicted — teacher-forced (fast, optimistic) |
| Exact-answer accuracy | Model generates the full answer autoregressively; compared character-for-character to gold. This is the headline number. |
| Test examples | Size of the held-out test set — 10% of data, partitioned before any training by MD5 hash of the question string. Same question always lands in the same split. |

### Data integrity guarantees

- Test set is partitioned **before** the tokenizer is built and **before** any training.
- The tokenizer is built on all data (train + val + test) to avoid `<UNK>` tokens, but it only stores character→ID mappings — it has no knowledge of which split an example belongs to.
- No question appears in more than one split (verified in `audit.py`, 0 overlaps across 10 000 synthetic examples and confirmed by assertion in the notebook).
- The dataset source (`deepcode-ai/math_dataset` or fallback) is recorded in every checkpoint and printed in every evaluation report.

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
2. Try a larger model (`medium-5M` or `large-10M`).
3. Train for more steps (use `--train thorough` or increase `--max-steps`).
4. Add more data categories (`--stage 2` or `--stage 3`).
5. Use beam search at evaluation (`--beam-size 4`).
6. Run `python experiments.py --analyse` for a failure analysis.

**Potential improvements:**

- Sub-word tokenization (better number handling)
- Curriculum learning (easy → hard examples)
- Chain-of-thought training (intermediate steps)
- Data augmentation (more number combinations)
- Larger model with more training compute

---

## Citation / acknowledgements

Dataset: [mandubian/pytorch_math_dataset](https://huggingface.co/datasets/mandubian/pytorch_math_dataset)
(a PyTorch-formatted version of Google DeepMind's Mathematics Dataset)

Architecture reference: Vaswani et al., *Attention Is All You Need* (2017)
Training approach: Brown et al., *Language Models are Few-Shot Learners* (GPT-3, 2020)
