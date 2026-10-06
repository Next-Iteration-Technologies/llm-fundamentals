# Local LLM: build a toy language model from scratch

A tiny GPT-style model that you train on your own laptop, using children's stories. Each step is a short Python file that you can read top to bottom:

```
tokenizer -> embeddings -> attention -> layers -> training -> generation
```

The model has about 63,000 parameters. Real models have billions. The ideas are the same.

## Setup

You need `uv` (see the [day-1 setup](../README.md#setup)). Then, inside this folder:

```bash
cd llm-fundamentals/local-llm
uv sync
```

`uv sync` creates a `.venv` here and installs PyTorch into it. Nothing is installed system-wide, and the day-1 rounds in the parent folder don't get PyTorch.

> **Intel Mac?** PyTorch stopped publishing Intel-Mac builds after version 2.2.2, so `pyproject.toml` pins that version on Intel Macs. Apple Silicon, Linux and Windows get a current PyTorch. Python 3.11 or 3.12 is needed; `uv` fetches it if you don't have it.

## The steps

| Step | File | What it does | Run it |
|------|------|--------------|--------|
| 1 | `bpe_tokenizer.py` | Learns byte-level BPE merges from `stories.txt` and writes `tokenizer.json` | `uv run bpe_tokenizer.py stories.txt` |
| 2 | `attention.py` | Embeddings and **one** attention head, every line spelled out | `uv run attention.py` |
| 3 | `model.py` | The full transformer: multi-head attention, feed-forward, layers | `uv run model.py` |
| 4 | `train.py` | Trains the model on the stories and writes `model.pt` | `uv run train.py` |
| 5 | `generate.py` | Shows next-token probabilities and lets the model keep writing | `uv run generate.py "Once upon a"` |

Run them in order the first time. `tokenizer.json` is already in the repo, so you can skip step 1 and go straight to step 2.

### 1. Tokenizer

Turns text into a list of numbers. It starts with the 256 byte values and adds 300 merged tokens, giving a vocabulary of 556. The tokenizer has its own page, which explains every line of its output: [bpe_tokenizer_wiki.md](bpe_tokenizer_wiki.md).

### 2. Attention

Every token gets a vector: its token row plus its position row. The head then asks three questions of each vector:

- **query**: the question a token asks
- **key**: the badge a token wears
- **value**: what a token says when someone listens to it

Each query is scored against every key. Softmax turns the scores into shares of attention. Each token's output is a weighted blend of the values.

The demo prints a 6 × 6 table of attention weights. Each row adds up to 1, and **everything above the diagonal is 0**: that is the mask, and it means no token can look at the tokens after it. The numbers themselves are random, because nothing has been trained yet.

### 3. Model

```
Embedding -> 2 x [ multi-head attention + feed-forward ] -> prediction head
```

- **Multi-head attention**: 4 heads run side by side, each free to learn a different kind of question.
- **Feed-forward**: works on each token on its own. It widens 32 → 128, cuts the negatives, then narrows back to 32.
- **Residual + layer norm**: each part adds its result back onto its input, and layer norm keeps the numbers in a sensible range.
- **Prediction head**: turns the last vector into one score for each of the 556 tokens.

`uv run model.py` only builds the model and prints its size. It does not train it.

### 4. Training

```bash
uv run train.py          # 3000 steps
uv run train.py 10000    # longer, better
```

Each step:
1. picks 32 random 64-token windows from the stories,
2. asks the model to predict the next token at every position,
3. measures how wrong it was (the **loss**),
4. works out how each weight contributed (backward pass),
5. nudges every weight a little.

Every 200 steps it prints a **train loss** and a **val loss**. The val loss is measured on the last 10% of the stories, which the model never trains on. A model that guesses at random scores ln(556) ≈ **6.32**, so watch the loss fall below that. If the train loss keeps falling while the val loss stops falling, the model is memorising rather than learning.

The first run encodes the stories once and saves them as `data.pt`. Later runs reuse that file. On a laptop CPU, expect training to take a few minutes.

### 5. Generation

```bash
uv run generate.py "Once upon a"    # a story opening: one tall bar
uv run generate.py "The quantum"    # nothing like the stories: many small bars
uv run generate.py                  # both
```

For each prompt you see the five most likely next tokens with their probabilities, then 40 more tokens written by the model. A language model never refuses to guess; it always gives a spread of probabilities. **A match looks like one tall bar; no match looks like many small ones.**

## Files

| File | In git? | What it is |
|------|---------|------------|
| `stories.txt` | yes | About 446 KB of Aesop, Peter Rabbit and Grimm from Project Gutenberg |
| `tokenizer.json` | yes | Merges and vocabulary written by step 1 |
| `data.pt` | no | The stories as token IDs, cached by step 4 |
| `model.pt` | no | Trained weights and settings, written by step 4 |
| `pyproject.toml`, `uv.lock` | yes | This folder's own environment (PyTorch) |

## Try this

1. Train for 10,000 steps instead of 3,000. How low does the val loss go, and does the generated text get better?
2. In `train.py`, raise `DIM` to 64 and `N_LAYERS` to 4. How many parameters is that now, and how much slower does training get?
3. Retrain the tokenizer with 1000 merges (`uv run bpe_tokenizer.py stories.txt 1000`), then delete `data.pt` and train again. Fewer, longer tokens: does that help?
4. Run `generate.py` with a sentence from the stories, and then with a sentence from your own work. Compare how confident the model is about each.
