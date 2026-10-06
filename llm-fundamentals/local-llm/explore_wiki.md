# Local LLM, extras: looking inside the trained model

`explore.py` answers one question: **what did training actually change?** It compares a model with random weights against the trained ones, without changing any of them.

```bash
cd llm-fundamentals/local-llm
uv run explore.py untrained ["prompt"]     # random weights; also saves model_untrained.pt
uv run explore.py compare ["prompt"]       # untrained vs model_3k.pt vs model.pt
uv run explore.py temperature ["prompt"]   # one prompt at temperature 0.5, 0.8, 1.2
uv run explore.py embeddings               # nearest tokens in the embedding table
uv run explore.py attention ["sentence"]   # who listens to whom, in the trained model
```

Add `--model PATH` to use a different model file, for example `--model model_3k.pt`.

Unlike `generate.py`, `explore.py` fixes the random seed before sampling, so every run prints the same continuation. That makes the models easy to compare.

| Subcommand | Log | Explained in |
|---|---|---|
| `untrained` | [runs/04_untrained.txt](runs/04_untrained.txt) | below |
| `compare` | [runs/09_compare.txt](runs/09_compare.txt) | [generate_wiki.md](generate_wiki.md#a-fair-test-match-vs-no-match) |
| `temperature` | [runs/10_temperature.txt](runs/10_temperature.txt) | [generate_wiki.md](generate_wiki.md#temperature) |
| `embeddings` | [runs/11_embeddings.txt](runs/11_embeddings.txt) | below |
| `attention` | [runs/12_attention_trained.txt](runs/12_attention_trained.txt) | below |

## The untrained baseline

`untrained` builds the model exactly as `train.py` does before step 0: same settings, same seed. So `model_untrained.pt` is the true starting point of both training runs.

```
Prompt: "Once upon a"
Top-5 next tokens:
   0.008  'up'
   0.006  ' wi'
   0.006  'le '
   0.006  'to '
   0.006  'est'
   (confidence of best guess: 1%)
Continuation: 'Once upon a c�it ke fromse�rea�d\n\x06ough� pst � they �the�Hould ��oun�The hiAsp: ‘�. The � mfbe de5�'
```

Every probability is close to 1/556 = 0.0018. The model has no preferences, and the text is random tokens. Everything the trained model does better than this was learned from the stories.

## Where the embedding table is stored

There is no separate embedding file. The tables are saved inside each model file, together with every other weight:

```
$ uv run explore.py embeddings
Stored in model.pt -> ['state']['embed.token_table.weight']     shape (556, 32)
Stored in model.pt -> ['state']['embed.position_table.weight']  shape (64, 32)
```

To open them yourself:

```python
import torch
state = torch.load("model.pt")["state"]
table = state["embed.token_table.weight"]     # 556 rows (one per token ID) × 32 numbers
table[397]                                     # the 32 numbers for token 397, ' him'
```

| File | Embedding table |
|---|---|
| `model_untrained.pt` | random, as at the start of training |
| `model_3k.pt` | after 3,000 steps |
| `model.pt` | after 10,000 steps |

## What the embedding table learned

Two tokens with similar rows are tokens the model treats as similar. `embeddings` measures similarity with **cosine similarity**: 1.0 means the rows point the same way, 0 means unrelated. For five tokens, it lists the 5 closest other tokens, before and after training:

```
'she '  (id 394)  first 4 numbers: [-0.27, -1.03, 0.49, 0.47]
   untrained: 'it' 0.56, 'j' 0.45, 't\n' 0.43, 'le ' 0.42, ' th' 0.42
   trained  : ' he ' 0.78, ' she ' 0.77, 'he ' 0.68, ', and ' 0.60, 'it ' 0.58

' him'  (id 397)  first 4 numbers: [0.16, -0.98, 0.18, 1.04]
   untrained: '�' 0.46, '�' 0.44, 'Z' 0.42, 'or' 0.41, ' b' 0.40
   trained  : 'ed him' 0.57, '�' 0.51, 'him' 0.51, 'long' 0.48, 'ound ' 0.42

'when'  (id 482)  first 4 numbers: [0.85, -0.68, -0.47, -0.77]
   untrained: ' an' 0.48, 'tr' 0.47, 'ess' 0.46, 'i' 0.46, 'down' 0.45
   trained  : 'hen' 0.60, 'ow' 0.53, 'if' 0.51, 'king' 0.49, ' for' 0.45
```

(Full output, including `king` and `down`, in [runs/11_embeddings.txt](runs/11_embeddings.txt).)

- **`'she '` is the clearest result.** Untrained, its neighbours are random. Trained, the closest are `' he '`, `' she '` and `'he '`: the other pronouns that start a sentence's action ("she said", "he went"). Nobody told the model these are pronouns. It noticed that they appear in the same places.
- **`' him'`** ends up next to `'ed him'` and `'him'`: the same word in different token shapes.
- **`'when'`** ends up near `'if'`, another word that starts a clause.
- **`'king'` and `'down'`** don't show clear neighbours (see the log). With only 32 numbers per token and a few minutes of training, only the most frequent patterns get learned.

Real models learn embeddings with thousands of numbers per token over billions of words, so their neighbours are much cleaner. This is the same idea, at the start.

## Attention in the trained model

[attention_wiki.md](attention_wiki.md) showed attention weights from random weights: evenly spread, meaningless. `attention` shows the same table from the **trained** model, for layer 0, head 0, on a real sentence. Rows ask, columns are listened to (cut down; full table in [runs/12_attention_trained.txt](runs/12_attention_trained.txt)):

```
             'up'   'on'  ' a '   'ti'  'me '  'ther'  'e '   'wa'   's'
 ' a '       0.01   0.05   0.79   0.00   0.00   0.00   0.00   0.00   0.00
 'ti'        0.00   0.02   0.39   0.56   0.00   0.00   0.00   0.00   0.00
 'me '       0.00   0.01   0.20   0.23   0.54   0.00   0.00   0.00   0.00
 'ther'      0.00   0.00   0.15   0.28   0.49   0.07   0.00   0.00   0.00
 'e '        0.00   0.01   0.17   0.09   0.25   0.06   0.36   0.00   0.00
 'wa'        0.00   0.00   0.02   0.04   0.05   0.01   0.34   0.53   0.00
 's'         0.00   0.00   0.01   0.02   0.03   0.00   0.25   0.39   0.31
```

Compared with the random table:

- **Attention is local.** Almost all the weight sits on the token itself and the one or two just before it. Columns far to the left are near 0. In a tiny model, most of what predicts the next token is the last few tokens, and head 0 has learned exactly that.
- **Tokens look back at the start of their word.** `'ther'` puts 0.49 on `'me '`, the end of the previous word, and 0.28 on `'ti'`. `'s'` (of *was*) puts 0.39 on `'wa'`, the start of the same word. To guess what comes after `s`, it helps to know you're inside *was*.
- **Rows are lopsided, not even.** Random rows spread attention evenly (0.13, 0.15, 0.21, …). Trained rows commit: `' a '` puts 0.79 on itself.

The tool also prints, for **each of the 4 heads**, which token each token listens to most. The heads differ. For example, at the second `' a '`:

```
head 0:  ' a '->' a '         (itself)
head 3:  ' a '->'s'           (the end of "was", just before)
```

and at `'.'`, head 0 looks at itself while head 1 looks back to `'ir'` (inside *girl*). That's why the model has several heads: each can learn to look for something different.

## Try this

1. `uv run explore.py attention "The Fox and the Grapes"`. Which head links `'ox'` back to `'F'`?
2. `uv run explore.py embeddings --model model_3k.pt`. Is `'she '` already next to `' he '` after 3,000 steps?
3. `uv run explore.py compare "Peter Rab"`. How sure is each model about `'b'`/`'le '`?
4. In a `uv run python` session, load `model.pt` and `model_untrained.pt`, and print how far each embedding row moved: `(trained - untrained).norm(dim=1)`. Which tokens moved most, and which least? Bytes that never appear in the stories, like `\x06`, still moved a little (about 1.1). Why, if they never get a gradient? (Hint: the *W* in AdamW is weight decay.)
