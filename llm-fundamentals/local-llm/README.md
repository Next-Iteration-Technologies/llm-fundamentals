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

| Step | File | What it does | Run it | Wiki |
|------|------|--------------|--------|------|
| 1 | `bpe_tokenizer.py` | Learns byte-level BPE merges from `stories.txt` and writes `tokenizer.json` | `uv run bpe_tokenizer.py stories.txt` | [bpe_tokenizer_wiki.md](bpe_tokenizer_wiki.md) |
| 2 | `attention.py` | Embeddings and **one** attention head, every line spelled out | `uv run attention.py` | [attention_wiki.md](attention_wiki.md) |
| 3 | `model.py` | The full transformer: multi-head attention, feed-forward, layers | `uv run model.py` | [model_wiki.md](model_wiki.md) |
| 4 | `train.py` | Trains the model on the stories and writes `model.pt` | `uv run train.py` | [train_wiki.md](train_wiki.md) |
| 5 | `generate.py` | Shows next-token probabilities and lets the model keep writing | `uv run generate.py "in the mor"` | [generate_wiki.md](generate_wiki.md) |
| + | `explore.py` | Untrained vs trained: predictions, temperature, embeddings, attention | `uv run explore.py compare "in the mor"` | [explore_wiki.md](explore_wiki.md) |

The wikis quote real output from our runs. Every log is saved in [`runs/`](runs/), numbered in pipeline order.

Run them in order the first time. Every output is already in the repo (`tokenizer.json`, `data.pt`, the trained models), so you can start at any step. On a slow laptop, skip training and go straight to step 5.

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

The first run encodes the stories once and saves them as `data.pt`. Later runs reuse that file.

Our results on an Intel MacBook Pro (CPU only), from [runs/](runs/):

| Steps | Time | Final train loss | Final val loss |
|---|---|---|---|
| 3,000 | 79 s (+ about 17 s of one-time encoding) | 3.44 | 3.62 |
| 10,000 | 287 s | 3.06 | 3.42 |

The val loss stops improving at about 7,000 steps: that is as far as this small model goes on this much text. Details in [train_wiki.md](train_wiki.md).

### 5. Generation

```bash
uv run generate.py "in the mor"     # 84% sure the next token is 'n' (morning): one tall bar
uv run generate.py "The quantum "   # about 5% at best: many small bars
uv run generate.py                  # the two built-in prompts
```

For each prompt you see the five most likely next tokens with their probabilities, then 40 more tokens written by the model. The built-in prompt `"Once upon a"` gives a surprise; [generate_wiki.md](generate_wiki.md#the-built-in-prompts-a-surprise) explains why. A language model never refuses to guess; it always gives a spread of probabilities. **A match looks like one tall bar; no match looks like many small ones.**

## Files

Everything the pipeline makes is committed, so you can inspect any stage without rerunning it.

| File | Made by | What it is |
|------|---------|------------|
| `stories.txt` | `build_stories.py` | About 446 KB of Aesop, Peter Rabbit and Grimm from Project Gutenberg, licence text removed |
| `tokenizer.json` | step 1 | Merges and vocabulary (556 tokens) |
| `data.pt` | step 4 | The whole of `stories.txt` as 200,797 token IDs. **Not** the embedding table |
| `model_untrained.pt` | `explore.py untrained` | Random starting weights, the same ones training starts from |
| `model_3k.pt` | step 4, 3,000 steps | Trained weights, including the embedding tables |
| `model.pt` | step 4, 10,000 steps | Trained weights, including the embedding tables. Used by `generate.py` |
| `runs/*.txt` | every step | The output of each run, quoted in the wikis |
| `pyproject.toml`, `uv.lock` | | This folder's own environment (PyTorch) |

The embedding tables live inside the model files: `torch.load("model.pt")["state"]["embed.token_table.weight"]`. See [explore_wiki.md](explore_wiki.md#where-the-embedding-table-is-stored).

Running `train.py` overwrites `model.pt`. `git restore model.pt` brings back the committed one.

## Try this

1. Run `uv run explore.py compare "Peter Rab"`. How much surer is the 10,000-step model than the 3,000-step one?
2. In `train.py`, raise `DIM` to 64 and `N_LAYERS` to 4. How many parameters is that now, and how much slower does training get?
3. Retrain the tokenizer with 1000 merges (`uv run bpe_tokenizer.py stories.txt 1000`), then delete `data.pt` and train again. Fewer, longer tokens: does that help? (`git restore tokenizer.json data.pt model.pt` puts everything back.)
4. Run `generate.py` with a sentence from the stories, and then with a sentence from your own work. Compare how confident the model is about each.
