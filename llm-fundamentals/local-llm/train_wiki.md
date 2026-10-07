# Local LLM, step 4: training

We are building a toy language model from scratch, one piece at a time:

tokenizer → embeddings → attention → layers → **training** → generation

Until now every weight in the model was random ([model_wiki.md](model_wiki.md)). `train.py` shows the model thousands of short pieces of the stories. Each time, it nudges all 63,468 weights so the model predicts the next token a little better.

## Run it

```bash
cd llm-fundamentals/local-llm
uv run train.py           # 3,000 steps
uv run train.py 10000     # 10,000 steps
```

We ran both on an Intel MacBook Pro (i9, CPU only). The full output is in the logs:

| Run | Log | Time |
|---|---|---|
| 3,000 steps (first run, includes encoding) | [runs/05_train_3k.txt](runs/05_train_3k.txt) | 71 s of training, plus a few seconds of encoding |
| 10,000 steps | [runs/07_train_10k.txt](runs/07_train_10k.txt) | 267 s of training |

Each run **overwrites `model.pt`**. We copied the 3,000-step model to `model_3k.pt` before running 10,000 steps, so both are kept.

## What it prints, line by line

```
Encoding stories.txt (one-time, may take a minute)...
195,262 tokens, vocabulary 556
model: 63,468 parameters
random-guess loss would be ln(556) = 6.32
```

- **Encoding…**: shown only on the first run. The tokenizer turns all of `stories.txt` into token IDs and saves them as `data.pt`. The tokenizer is pure Python, so later runs load `data.pt` instead.

  `train.py` never checks whether `data.pt` still matches `tokenizer.json`. **If you retrain the tokenizer, delete `data.pt`**, or the model trains on token IDs that mean something else now.
- **195,262 tokens**: the 445,847 bytes of stories became 195,262 tokens (about 2.3 bytes per token, as in [bpe_tokenizer_wiki.md](bpe_tokenizer_wiki.md)).
- **63,468 parameters**: the size of the model ([model_wiki.md](model_wiki.md#where-the-63468-parameters-live)).
- **6.32**: the loss you'd get by giving every one of the 556 tokens the same probability. This is the "knows nothing" line to beat.

### `data.pt` is not the embedding table

These two are easy to mix up:

| | `data.pt` | Embedding table (inside `model.pt`) |
|---|---|---|
| Holds | the whole of `stories.txt` as 195,262 token IDs | one row of 32 numbers per token ID (556 × 32) |
| Looks like | `[79, 110, 345, 491, 258, …]` | row 258 = `[0.63, 0.18, 0.38, …]` |
| Made by | the tokenizer | training (it starts random) |
| Changes during training? | no; it's the text | yes; it's learned |

`data.pt` is what training **reads**. The embedding table is part of what training **learns**.

## The loop

Every step does the same five things (see the docstring at the top of `train.py`):

1. Pick **32 random windows** of 65 tokens from the training text.
2. **Input** = tokens 0–63, **target** = tokens 1–64. So at every one of the 64 positions, the right answer is "the token that actually came next". One window gives 64 practice questions; a step gives 32 × 64 = 2,048.
3. **Forward pass**: the model predicts, and the loss measures how wrong it was ([model_wiki.md](model_wiki.md#the-loss-in-one-line)).
4. **Backward pass** (`loss.backward()`): for every weight, PyTorch works out which direction would lower the loss. These directions are the *gradients*.
5. **Optimizer step** (`AdamW`, learning rate 0.003): move every weight a small step in that direction.

**Train vs val:** the last 10% of the tokens (19,527) are held back as **validation** data. The model never trains on them. The train loss shows how well it fits text it has seen; the val loss shows how well it handles text it hasn't.

## Reading the loss table

From the 3,000-step run ([runs/05_train_3k.txt](runs/05_train_3k.txt)):

```
step     0  train loss 6.490  val loss 6.492  (0s)
step   200  train loss 4.339  val loss 4.412  (5s)
step   400  train loss 3.981  val loss 4.051  (10s)
step   600  train loss 3.832  val loss 3.952  (14s)
step   800  train loss 3.754  val loss 3.907  (19s)
step  1000  train loss 3.697  val loss 3.810  (24s)
...
step  2000  train loss 3.477  val loss 3.720  (46s)
...
step  3000  train loss 3.353  val loss 3.593  (71s)
```

- **Step 0 is 6.49, slightly *worse* than 6.32.** The random model isn't just ignorant; it's confidently wrong about a few tokens. That's normal for random weights.
- **Steps 0 → 400: the big drop (6.5 → 4.1).** The model learns the cheap wins first: which tokens are common overall, that ` the` and ` and` are everywhere, which tokens start a word.
- **Steps 400 → 3,000: slow and steady (4.1 → 3.6).** Now it's learning which token follows which: spelling inside words, short phrases.
- **Every line is an average over 20 random batches**, so small wobbles up and down are noise, not a real change.

What does 3.59 mean? e^−3.59 ≈ 0.028. On average, the model gives the correct next token about a **2.8%** chance. A random guess gives it 0.18%, so that's about 15× better than random. That sounds low, but many next tokens are genuinely unpredictable (which word comes next in a story?). The confident cases, like finishing a word, are where the gains are.

## 3,000 vs 10,000 steps

Both runs start from the same seed (`torch.manual_seed(0)`), so **the first 3,000 steps of the 10,000-step run match the 3,000-step run line for line**. Rerun it and you get the same numbers.

From [runs/07_train_10k.txt](runs/07_train_10k.txt):

| Step | Train loss | Val loss | Gap (val − train) |
|---|---|---|---|
| 0 | 6.490 | 6.492 | 0.00 |
| 1,000 | 3.697 | 3.810 | 0.11 |
| 3,000 | 3.353 | 3.593 | 0.24 |
| 5,000 | 3.136 | 3.484 | 0.35 |
| 7,000 | 3.028 | 3.432 | 0.40 |
| 9,800 | 2.988 | **3.392** (lowest) | 0.40 |
| 10,000 | **2.936** | 3.419 | 0.48 |

- **The train loss keeps falling** all the way to step 10,000.
- **The val loss almost stops improving after about step 7,000.** It creeps from 3.43 to 3.39 over the last 3,000 steps, bouncing up and down by 0.03 along the way.
- **The gap widens** from 0.11 to 0.48. The model is starting to learn things that are true only of the training text (particular stories and names) rather than of English in general. This is the start of **overfitting**.

With 63,468 parameters and 195,262 tokens, this model is close to its limit by about 7,000 steps. More steps won't help much. A bigger model (`DIM`, `N_LAYERS`) or more text would.

| Model | Final val loss | Probability given to the right token | Training time |
|---|---|---|---|
| untrained | 6.49 | 0.15% | none |
| 3,000 steps (`model_3k.pt`) | 3.59 | 2.8% | 71 s |
| 10,000 steps (`model.pt`) | 3.42 | 3.3% | 267 s |

These losses can't be compared one-to-one with a model trained on a different tokenizer. Loss is measured per token, and a different tokenizer cuts the same text into different tokens.

What these improvements look like in generated text is in [generate_wiki.md](generate_wiki.md).

## What's saved

`model.pt` is a dictionary with two keys:

```python
ckpt = torch.load("model.pt")
ckpt["config"]   # {'vocab_size': 556, 'dim': 32, 'n_heads': 4, 'n_layers': 2, 'max_positions': 64}
ckpt["state"]    # every weight, by name: 'embed.token_table.weight', 'layers.0.attn.heads.0.Wq.weight', ...
```

`config` says how to build the model again. `state` holds the learned numbers, embedding tables included. [explore_wiki.md](explore_wiki.md#where-the-embedding-table-is-stored) shows how to look inside.

## Try this

Every training run overwrites `model.pt`. To get the 10,000-step model back afterwards, run `git restore model.pt`.

1. Run `uv run train.py 1000`. Compare its last line with the step-1000 line in the logs above. (It should match, because of the seed.)
2. Change `LR = 3e-3` to `3e-2`, then to `3e-4`. Which learns faster, and which goes wrong?
3. Set `DIM = 64` and `N_LAYERS = 4` and train 3,000 steps. Does val loss beat 3.59? How much slower is it?
4. Change `torch.manual_seed(0)` to `torch.manual_seed(1)`. How different is the final loss?
