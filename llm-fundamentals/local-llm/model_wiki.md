# Local LLM, step 3: the full model

We are building a toy language model from scratch, one piece at a time:

tokenizer → embeddings → attention → **layers** → training → generation

`model.py` stacks the pieces from [attention_wiki.md](attention_wiki.md) into a complete, GPT-shaped model called `ToyGPT`. It only *builds* the model; nothing is trained here.

## Run it

```bash
cd llm-fundamentals/local-llm
uv run model.py
```

The full output is saved in [runs/03_model.txt](runs/03_model.txt):

```
ToyGPT built: 63,468 parameters
logits shape: (1, 10, 556) (batch, tokens, vocab)
```

- **63,468 parameters**: the model has 63,468 numbers that training will adjust. GPT-2 small has 124 million; large modern models have hundreds of billions. The design is the same.
- **logits shape `(1, 10, 556)`**: the demo feeds in 10 random token IDs. For **every** one of those 10 positions, the model gives a score to each of the 556 tokens in the vocabulary: "how likely is this token to come next?"

## The shape of the model

```
token IDs (batch, T)
   │
   ▼
Embedding: token row + position row          (batch, T, 32)
   │
   ▼
Layer 0 ─┬─ LayerNorm → 4-head attention ──(+)   gather from earlier tokens
         └─ LayerNorm → feed-forward ──────(+)   think about it, token by token
   │
   ▼
Layer 1   (same again, its own weights)
   │
   ▼
LayerNorm → prediction head                  (batch, T, 556)  = logits
```

| Piece | What it does |
|---|---|
| **Multi-head attention** | 4 heads of size 8 run side by side, each with its own `Wq`, `Wk`, `Wv`. Each head can learn to look for something different. Their outputs (4 × 8 = 32) are glued together and mixed by `proj`. |
| **Feed-forward** | Works on each token **alone**, with no looking at neighbours. Widens 32 → 128, sets the negatives to 0 (ReLU), then narrows 128 → 32. This is where most of the per-token "thinking" happens. |
| **Residual `x = x + …`** | Each part *adds* its result to what was already there instead of replacing it. Information from the embedding can flow straight through to the end, which makes training much easier. |
| **LayerNorm** | Rescales each token's 32 numbers to a steady range before each part, so values don't blow up or shrink away as they pass through layers. |
| **Prediction head** | A final 32 → 556 step: one score (a *logit*) per vocabulary token. |

## Where the 63,468 parameters live

Calculated from `model.named_parameters()`:

| Part | Shape | Parameters |
|---|---|---|
| Token embedding table | 556 × 32 | 17,792 |
| Position embedding table | 64 × 32 | 2,048 |
| **Each layer** (× 2): | | **12,608** |
|   LayerNorm 1 + 2 | 2 × (32 + 32) | 128 |
|   4 heads × `Wq`, `Wk`, `Wv` | 12 × (8 × 32) | 3,072 |
|   `proj` (glues heads together) | 32 × 32 + 32 | 1,056 |
|   Feed-forward | 32×128 + 128 + 128×32 + 32 | 8,352 |
| Final LayerNorm | 32 + 32 | 64 |
| Prediction head | 556 × 32 + 556 | 18,348 |
| **Total** | | **63,468** |

So 19,840 + 2 × 12,608 + 64 + 18,348 = 63,468.

The two biggest parts are the token embedding table at the start and the prediction head at the end, because both have one row per vocabulary token. In this toy model they make up more than half of all the parameters. In large models the layers dominate instead.

The attention **mask** (a 64 × 64 lower triangle of ones) is also saved in the model file, but it's a fixed *buffer*, not a parameter: training never changes it.

## From logits to a prediction

The logits are raw scores, anywhere from minus to plus infinity. Two methods turn them into something useful:

- **`next_token_probs(ids)`**: takes the logits at the **last** position only and runs them through softmax, giving 556 probabilities that add up to 1. It also cuts the input to the last 64 tokens, because the position table has only 64 rows.
- **`generate(ids, max_new_tokens, temperature)`**: repeats the following: get the probabilities, *sample* one token from them, append it, go again. **Temperature** reshapes the probabilities first. Below 1 sharpens them (safer, more repetitive). Above 1 flattens them (more surprising, more mistakes). [generate_wiki.md](generate_wiki.md) shows this in action.

## The loss, in one line

When `forward` gets `targets` (the true next tokens), it also returns a **loss**: cross-entropy, which is `-log(probability the model gave to the correct next token)`, averaged over every position.

| The model gave the right token… | loss |
|---|---|
| probability 1.0 (certain and right) | 0 |
| probability 0.5 | 0.69 |
| probability 1/556 (a pure guess) | ln(556) = **6.32** |
| probability 0.001 | 6.91 |

Training ([train_wiki.md](train_wiki.md)) is about pushing that number down.

## Try this

1. Build `ToyGPT(556, dim=64, n_layers=4)` in a `uv run python` session and count its parameters. Which parts grew most?
2. Set `n_heads=3`. What goes wrong? (Hint: `dim // n_heads`.)
3. Feed in 100 random IDs instead of 10 and call `model(ids)`. Why does it fail, while `next_token_probs` on the same IDs works?
