# Local LLM, step 2: embeddings and one attention head

We are building a toy language model from scratch, one piece at a time:

tokenizer → **embeddings → attention** → layers → training → generation

The tokenizer ([bpe_tokenizer_wiki.md](bpe_tokenizer_wiki.md)) turns text into token IDs. An ID is only a name tag: ID 412 is not "bigger" than ID 300 in any useful sense. `attention.py` does two things with those IDs:

1. **Embedding**: turns each ID into a list of numbers (a *vector*) that the model can do maths with.
2. **Attention**: lets each token look back at the tokens before it and pull in what it needs.

Every line of the attention maths is written out in the file, so you can match code to idea.

## Run it

```bash
cd llm-fundamentals/local-llm
uv sync                  # once: installs PyTorch into this folder's .venv
uv run attention.py
```

The full output is saved in [runs/02_attention.txt](runs/02_attention.txt).

```
after embedding: (1, 6, 32)   (batch, tokens, dim)
after attention: (1, 6, 16)   (batch, tokens, head_dim)

attention weights (row = which token is asking, col = who it listens to):
tensor([[1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.61, 0.39, 0.00, 0.00, 0.00, 0.00],
        [0.56, 0.20, 0.23, 0.00, 0.00, 0.00],
        [0.24, 0.26, 0.11, 0.39, 0.00, 0.00],
        [0.13, 0.15, 0.21, 0.15, 0.36, 0.00],
        [0.26, 0.10, 0.25, 0.09, 0.17, 0.13]], grad_fn=<SelectBackward0>)

each row sums to: tensor([1.00, 1.00, 1.00, 1.00, 1.00, 1.00], grad_fn=<SumBackward1>)

Note the zeros above the diagonal: no token can see the future.
The numbers are meaningless until training; only the shapes matter now.
```

The demo feeds in six made-up token IDs, `[300, 412, 275, 301, 389, 412]`. With our tokenizer these are `'it'`, `' wa'`, `'on'`, `'om'`, `' c'`, `' wa'`. They're not a real sentence; the demo only cares about the shapes.

## Line 1: after embedding `(1, 6, 32)`

Read the shape as **(batch, tokens, dim)**:

| Number | Meaning |
|---|---|
| 1 | one sentence at a time (a *batch* of 1) |
| 6 | six tokens |
| 32 | each token is now 32 numbers |

The `Embedding` class holds two tables:

| Table | Shape | One row per | Row means |
|---|---|---|---|
| `token_table` | 556 × 32 | token ID | "what this token is" |
| `position_table` | 64 × 32 | position 0–63 | "where in the sentence it is" |

For each token it looks up its row in each table and **adds the two rows together**. Token `' wa'` appears twice (positions 1 and 5). Both copies get the same token row but different position rows, so the model can tell them apart.

Both tables start as random numbers. Training changes them until tokens that behave alike end up with similar rows. Training saves the learned tables inside `model.pt`. [explore_wiki.md](explore_wiki.md) shows how to open them.

## Line 2: after attention `(1, 6, 16)`

Still one sentence and still six tokens, but each token is now 16 numbers: the head's output size (`HEAD_DIM = 16` in the demo). In the full model ([model_wiki.md](model_wiki.md)), four heads of size 8 run side by side and their outputs are glued back together to 32.

What happened in between is `AttentionHead.forward`, in five steps:

| Step | Code | Party analogy |
|---|---|---|
| 1 | `q = Wq(x)`, `k = Wk(x)`, `v = Wv(x)` | Each guest writes a **question** (query), wears a **badge** (key), and prepares **what they'll say** (value) |
| 2 | `scores = q @ k.T / sqrt(head_dim)` | Every question is compared with every badge. A high score means a good match |
| 3 | `masked_fill(mask == 0, -inf)` | You can't listen to guests who haven't arrived yet |
| 4 | `softmax(scores)` | Turn scores into shares of your attention that add up to 1 |
| 5 | `out = weights @ v` | What you take home is a blend of what everyone said, weighted by those shares |

The `/ sqrt(head_dim)` keeps the scores from getting huge. Without it, softmax would give almost all the attention to a single token.

## The table of weights

Row *i* is token *i* asking. Column *j* is token *j* being listened to.

```
             it    ' wa'   on     om    ' c'   ' wa'
it         1.00   0.00   0.00   0.00   0.00   0.00
' wa'      0.61   0.39   0.00   0.00   0.00   0.00
on         0.56   0.20   0.23   0.00   0.00   0.00
om         0.24   0.26   0.11   0.39   0.00   0.00
' c'       0.13   0.15   0.21   0.15   0.36   0.00
' wa'      0.26   0.10   0.25   0.09   0.17   0.13
```

Three things to notice:

1. **Everything above the diagonal is 0.** That's the mask (step 3). Token 2 (`on`) can listen to tokens 0, 1 and 2, but not to 3, 4 or 5. A language model predicts the *next* token, so letting it peek at the next token would be cheating.
2. **The first row is `1.00` then zeros.** The first token has nobody before it, so it gives all its attention to itself.
3. **Every row adds up to 1** (the `each row sums to` line). Softmax shares out one unit of attention. Row 4 (`' c'`) spreads it fairly evenly: 0.13, 0.15, 0.21, 0.15, 0.36.

The *values* are random, because the tables and `Wq`, `Wk`, `Wv` are random. After training, rows become lopsided: a token puts most of its attention on the one or two earlier tokens that help predict what comes next. [explore_wiki.md](explore_wiki.md#attention-in-the-trained-model) shows this on a real sentence.

`grad_fn=<…>` is PyTorch noting that these numbers came from a calculation it can differentiate. Training uses that record to work out how to change each weight. You can ignore it here.

## What's learned and what isn't

| Learned in training | Fixed |
|---|---|
| `token_table` (556 × 32) | the mask (lower-triangle of ones) |
| `position_table` (64 × 32) | the `sqrt(head_dim)` scaling |
| `Wq`, `Wk`, `Wv` (each `dim × head_dim`) | softmax |

## Try this

1. Change the demo IDs to `[79, 110, 401, 334, 275, 316]`, which is "Once upon a" from the tokenizer wiki. Same shapes, different random numbers.
2. Set `MAX_POS = 4` and run it. What error do you get, and why? (Hint: the position table has only 4 rows.)
3. Remove the `masked_fill` line and run again. What happens to the upper triangle?
4. Change `HEAD_DIM` to 4 and to 64. Which shape changes, and which doesn't?
