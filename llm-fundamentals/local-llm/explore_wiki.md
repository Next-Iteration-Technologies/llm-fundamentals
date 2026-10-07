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
   0.010  '^'
   0.008  ' no'
   0.007  '�'
   0.006  ' as'
   0.006  ' B'
   (confidence of best guess: 1%)
Continuation: 'Once upon aryidce willeredThe�b�\x03ail\u202e my����astK� so�it\x1f st his ofAare��ant�kfself de abess'
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
table[325]                                     # the 32 numbers for token 325, ' him'
```

| File | Embedding table |
|---|---|
| `model_untrained.pt` | random, as at the start of training |
| `model_3k.pt` | after 3,000 steps |
| `model.pt` | after 10,000 steps |

## What the embedding table learned

Two tokens with similar rows are tokens the model treats as similar. `embeddings` measures similarity with **cosine similarity**: 1.0 means the rows point the same way, 0 means unrelated. For five tokens, it lists the 5 closest other tokens, before and after training:

```
' she'  (id 344)  first 4 numbers: [0.57, -0.83, 0.24, 0.12]
   untrained: ' whe' 0.52, '�' 0.49, '7' 0.48, '�' 0.43, ' ne' 0.43
   trained  : ' they' 0.64, ' He' 0.61, 'l' 0.54, ' he' 0.49, 'he' 0.44

' went'  (id 469)  first 4 numbers: [0.11, 0.8, -0.14, -1.37]
   untrained: ' qu' 0.64, '�' 0.48, ' r' 0.46, '�' 0.43, 'X' 0.42
   trained  : ' was' 0.52, '�' 0.49, ' came' 0.49, 'ought' 0.42, 'ried' 0.41

' when'  (id 430)  first 4 numbers: [2.01, -0.76, -0.39, 0.22]
   untrained: ' what' 0.59, "'" 0.49, 'ep' 0.47, ' ‘' 0.44, ' m' 0.39
   trained  : ' which' 0.63, ' if' 0.62, ' what' 0.60, '�' 0.60, '~' 0.51
```

(Full output, including ` him` and ` down`, in [runs/11_embeddings.txt](runs/11_embeddings.txt).)

- **` she` is the clearest result.** Untrained, its neighbours are random. Trained, the closest are ` they`, ` He` and ` he`: the other pronouns that start a sentence's action ("she said", "they went"). Nobody told the model these are pronouns. It noticed that they appear in the same places.
- **` went`** ends up next to ` was` and ` came`: verbs that follow *he*, *she* or *they*.
- **` when`** ends up near ` which`, ` if` and ` what`: words that start a clause. (` what` was already close by chance; training kept it there.)
- **` him` and ` down`** show weaker neighbours (see the log). ` down` lands near ` out` and ` again`, which fits ("sat down", "went out"); ` him` stays muddled. With only 32 numbers per token and a few minutes of training, only the most frequent patterns get learned.

Each of these words is one token, with its leading space, because the tokenizer splits on words first ([bpe_tokenizer_wiki.md](bpe_tokenizer_wiki.md#why-we-split-into-words-first)). With the first version of the tokenizer, *she* was spread over tokens like `'she '` and `' she '`, and the neighbours were much messier.

Real models learn embeddings with thousands of numbers per token over billions of words, so their neighbours are much cleaner. This is the same idea, at the start.

## Attention in the trained model

[attention_wiki.md](attention_wiki.md) showed attention weights from random weights: evenly spread, meaningless. `attention` shows the same table from the **trained** model, for layer 0, head 0, on a real sentence. Rows ask, columns are listened to (cut down; full table in [runs/12_attention_trained.txt](runs/12_attention_trained.txt)):

```
             ' upon'  ' a'   ' t'  'ime' ' there' ' was'  ' a' ' little' ' g'   'ir'
 ' a'         0.16   0.71   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
 ' t'         0.09   0.49   0.27   0.00   0.00   0.00   0.00   0.00   0.00   0.00
 'ime'        0.24   0.13   0.32   0.10   0.00   0.00   0.00   0.00   0.00   0.00
 ' there'     0.05   0.10   0.06   0.48   0.17   0.00   0.00   0.00   0.00   0.00
 ' was'       0.08   0.01   0.06   0.25   0.01   0.46   0.00   0.00   0.00   0.00
 ' a'         0.01   0.03   0.02   0.03   0.03   0.24   0.64   0.00   0.00   0.00
 ' little'    0.00   0.00   0.00   0.01   0.01   0.40   0.14   0.44   0.00   0.00
 ' g'         0.02   0.04   0.07   0.28   0.05   0.24   0.12   0.08   0.05   0.00
 'ir'         0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.92   0.08
 'l'          0.00   0.00   0.00   0.00   0.00   0.00   0.02   0.01   0.20   0.72
```

Compared with the random table:

- **Attention is mostly local.** Most of the weight sits on the token itself and the one or two just before it. In a tiny model, most of what predicts the next token is the last few tokens, and head 0 has learned that.
- **Tokens inside a word look back at the start of the word.** `'ir'` puts 0.92 on `' g'`, and `'l'` puts 0.72 on `'ir'`. To guess what comes after `ir`, it helps to know the word started with *g*. Because every word now starts with a space token, the start of a word is easy to spot.
- **Rows are lopsided, not even.** Random rows spread attention evenly (0.13, 0.15, 0.21, …). Trained rows commit: `'ir'` puts 0.92 on a single token.

The tool also prints, for **each of the 4 heads**, which token each token listens to most. The heads differ. For example, at the second `' a'`:

```
head 0:  ' a'->' a'          (itself)
head 1:  ' a'->' was'        (the word just before)
head 2:  ' a'->' t'          (inside "time", further back)
```

and at `'.'`, head 0 looks at itself while head 1 looks back to `' little'`. That's why the model has several heads: each can learn to look for something different.

## Try this

1. `uv run explore.py attention "The Fox and the Grapes"`. Which head links `'ox'` back to `' F'`?
2. `uv run explore.py embeddings --model model_3k.pt`. Is ` she` already next to ` they` and ` he` after 3,000 steps?
3. `uv run explore.py compare "Little Red"`. How much surer of `'-'` (Little Red-Cap) is the 10,000-step model than the 3,000-step one?
4. In a `uv run python` session, load `model.pt` and `model_untrained.pt`, and print how far each embedding row moved: `(trained - untrained).norm(dim=1)`. Which tokens moved most, and which least? Bytes that never appear in the stories, like `\x06`, still moved a little (about 1.1). Why, if they never get a gradient? (Hint: the *W* in AdamW is weight decay.)
