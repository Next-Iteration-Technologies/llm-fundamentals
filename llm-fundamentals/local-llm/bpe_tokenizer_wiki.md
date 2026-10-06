# Local LLM, step 1: the tokenizer

We are building a toy language model from scratch, one piece at a time:

**tokenizer** → embeddings → attention → layers

A model can't read text. It reads a list of numbers. The tokenizer decides which numbers. `bpe_tokenizer.py` is a byte-level BPE (byte pair encoding) tokenizer in about 140 lines of plain Python, with no packages. It works the same way as the tokenizers behind GPT and Claude, only smaller and slower.

Why not one number per word? There are too many words, and a word the model has never seen would have no number at all. Why not one number per character? The lists get very long, and the model has to learn spelling before it can learn anything else. BPE sits in between. Common pieces like ` the ` or `little ` get their own number. Rare words are built from smaller pieces, and in the worst case from single bytes, so every text can be encoded.

## Run it

```bash
cd llm-fundamentals/local-llm
uv run bpe_tokenizer.py stories.txt          # 300 merges, the default
uv run bpe_tokenizer.py stories.txt 1000     # more merges
```

`python bpe_tokenizer.py stories.txt` works too. The script needs no packages.

Training on `stories.txt` takes about 40 seconds with 300 merges. When it's done, the script writes `tokenizer.json` to the current folder and prints three things:
1. the first 20 merges
2. a summary line
3. a sample sentence, encoded and decoded again

The sections below explain each part.

## The training text: `stories.txt`

The training text is about 446 KB of public-domain stories from [Project Gutenberg](https://www.gutenberg.org):
- *Aesop's Fables* (Townsend translation)
- *The Tale of Peter Rabbit*
- the first part of *Grimm's Fairy Tales*, up to "Sweetheart Roland"

The Gutenberg licence header and footer are removed. Otherwise the tokenizer would spend merges on words like "Gutenberg" and "copyright".

You can train on any UTF-8 text file. The tokenizer learns whatever is common in *that* text.

## How training works

Look at `train()` in `bpe_tokenizer.py`:

1. Turn the text into UTF-8 bytes. Every byte is a number from 0 to 255, so the starting vocabulary is those 256 values.
2. Count every pair of neighbouring tokens (`get_pair_counts`).
3. Take the most frequent pair and give it a new ID: 256 for the first merge, 257 for the second, and so on.
4. Replace every occurrence of that pair with the new ID (`apply_merge`). The text gets shorter.
5. Repeat until you've done `num_merges` merges, or until no pair appears twice.

The result is two things:
- **`merges`**: the ordered list of rules, "pair (a, b) becomes new_id".
- **`vocab`**: for every ID, the bytes it stands for.

### A tiny example

Train on `"the cat sat on the mat"` with 4 merges:

```
merge   1: b'a' + b't' -> b'at'     (id 256, seen 3 times)    cat, sat, mat
merge   2: b't' + b'h' -> b'th'     (id 257, seen 2 times)
merge   3: b'th' + b'e' -> b'the'   (id 258, seen 2 times)    built on merge 2
merge   4: b'the' + b' ' -> b'the ' (id 259, seen 2 times)    built on merge 3

Text went from 22 bytes to 13 tokens.
```

Try it yourself:

```bash
uv run python -c "from bpe_tokenizer import train; train('the cat sat on the mat', 4)"
```

## Reading the merge lines

The real run on `stories.txt` starts like this:

```
merge   1: b'e' + b' ' -> b'e '   (id 256, seen 14434 times)
merge   2: b't' + b'h' -> b'th'   (id 257, seen 10785 times)
merge   3: b'd' + b' ' -> b'd '   (id 258, seen 9665 times)
merge   4: b' ' + b'a' -> b' a'   (id 259, seen 7179 times)
...
merge   8: b' ' + b'th' -> b' th'   (id 263, seen 5181 times)
...
merge  14: b'\xe2' + b'\x80' -> b'\xe2\x80'   (id 269, seen 3431 times)
...
merge  17: b' th' + b'e ' -> b' the '   (id 272, seen 2995 times)
merge  18: b' a' + b'nd ' -> b' and '   (id 273, seen 2913 times)
```

Each part of a line means:
- **`b'e'`**: Python's notation for *bytes*, not a string. The tokenizer works on raw bytes.
- **`b'e' + b' '` → `b'e '`**: the most frequent pair was the letter `e` followed by a space. The two become one token.
- **`id 256`**: IDs 0–255 are the single bytes. Each merge gets the next free ID.
- **`seen 14434 times`**: how often that pair appeared **at that moment**, in the text as it looked after all the earlier merges.

### What the merges show

**English letter statistics come out first.** `e ` is first because so many words end in *e*: *the, he, she, came*. Next come `th`, `d ` (*and, said, had*), ` a`, `t `, `in` and `er`. Nobody told the tokenizer this; it found it by counting.

**Spaces stick to words.** Tokens like `e `, ` a` and `, ` contain a space. This script doesn't split the text into words before counting, so a merge can cross a word boundary. `d ` means "d at the end of a word".

**Later merges build on earlier ones:**

| Merge | Built from | Result |
|---|---|---|
| 2 | `t` + `h` | `th` |
| 8 | ` ` + `th` (merge 2) | ` th` |
| 1 | `e` + ` ` | `e ` |
| 17 | ` th` (merge 8) + `e ` (merge 1) | **` the `** |
| 3 | `d` + ` ` | `d ` |
| 10 | `n` + `d ` (merge 3) | `nd ` |
| 4 | ` ` + `a` | ` a` |
| 18 | ` a` (merge 4) + `nd ` (merge 10) | **` and `** |

By merge 18, the two most common English words are one token each.

**The counts keep falling.** `th` was seen 10,785 times, but ` th` only 5,181 times: many `th`s sit inside words (*other, with, mother*). ` the ` is lower again (2,995). It misses *The* with a capital, *the,* before a comma, and *the* at the end of a line.

**Merge 14 is half a character.** `\xe2\x80` is not a letter. The stories use curly quotes, and in UTF-8 each of those is three bytes that start the same way:

| Character | UTF-8 bytes |
|---|---|
| ’ | `e2 80 99` |
| “ | `e2 80 9c` |
| ” | `e2 80 9d` |

The tokenizer only sees bytes. It noticed that `e2 80` is common and merged it. A later merge (ID 292) adds `99`, so `’` becomes one token. This is why the tokenizer is called *byte-level*: tokens don't have to be whole characters.

### The summary line

```
Training done. Vocabulary size: 556 tokens. Text went from 445847 bytes to 200797 tokens.
```

That's 256 bytes plus 300 merges, which is 556 tokens. Each token now covers about 2.2 bytes on average. More merges give fewer, longer tokens, but also a bigger vocabulary for the model to learn.

## Encoding a sentence

```
Sample:    Once upon a time there was a little girl.
Token IDs: [79, 110, 401, 334, 275, 316, 338, 344, 346, 256, 294, 115, 316, 519, 103, 322, 108, 46]
Tokens:    ['O', 'n', 'ce ', 'up', 'on', ' a ', 'ti', 'me ', 'ther', 'e ', 'wa', 's', ' a ', 'little ', 'g', 'ir', 'l', '.']
Round trip OK: True
```

The two lists line up position by position: ID `79` is `'O'`, ID `110` is `'n'`, and so on. **The list of IDs is all the model will ever see.** The token list is only there so humans can read it.

### How to read an ID

- **0–255**: a single byte. For plain English letters this is the ASCII code: `79` = `O`, `115` = `s`, `46` = `.`.
- **256 and up**: a merged token. **Merge number = ID − 255.** A lower ID means the token was learned earlier, which means its pair was more common.

An ID is only a label. `316` isn't bigger or closer to anything than `275`. Meaning comes later, in step 2, when the model learns an embedding vector for each ID.

### Every token, traced back to bytes

| Token | ID | Merge # | Built from |
|---|---|---|---|
| `'O'` | 79 | – | single byte |
| `'n'` | 110 | – | single byte |
| `'ce '` | 401 | 146 | `c` + `e ` (256) |
| `'up'` | 334 | 79 | `u` + `p` |
| `'on'` | 275 | 20 | `o` + `n` |
| `' a '` | 316 | 61 | ` a` (259) + ` ` |
| `'ti'` | 338 | 83 | `t` + `i` |
| `'me '` | 344 | 89 | `m` + `e ` (256) |
| `'ther'` | 346 | 91 | `th` (257) + `er` (262) |
| `'e '` | 256 | 1 | `e` + ` ` |
| `'wa'` | 294 | 39 | `w` + `a` |
| `'s'` | 115 | – | single byte |
| `' a '` | 316 | 61 | same token as above, so the same ID |
| `'little '` | 519 | 264 | `litt` + `le `, four merges deep |
| `'g'` | 103 | – | single byte |
| `'ir'` | 322 | 67 | `i` + `r` |
| `'l'` | 108 | – | single byte |
| `'.'` | 46 | – | single byte |

### Why the words split where they do

`encode()` replays the merges **in the order they were learned**, not "longest piece first". The pair learned earliest claims the bytes, and later merges can't take them back.

- **`O` `n` `ce `**: lowercase *once* is common, but a capital `O` followed by `n` never got merged. Capitalised words usually cost more tokens.
- **`up` `on`**: there is no `upon` token. `on` can't take the following space either, because merge 4 (` a`) already gave that space to the next word.
- **`' a '`**: the spaces on both sides of *a* end up in one token. That's why *upon* and *was* lose their trailing space.
- **`ther` `e `**: merge 1 (`e `) runs first, so the last `e` of *there* is already taken.
- **`little `**: common in children's stories, so it's one token, trailing space included.
- **`g` `ir` `l`**: *girl* is rare in these stories, so it falls back to near single bytes. Rare words cost more tokens.

41 bytes became 18 tokens, about 2.3 bytes per token. This is also why a model trained mostly on English text uses more tokens, and costs more, for German or for code.

### Decoding

`decode()` looks up each ID's bytes in `vocab`, joins them and turns the bytes back into text. `Round trip OK: True` means decoding the 18 IDs gave exactly the original sentence.

A token can be half a character, like `\xe2\x80` above. If you decode such a token on its own, the bytes aren't valid UTF-8, and `errors="replace"` shows them as `�` instead of crashing.

## `tokenizer.json`

`save()` writes the two things training produced:

```json
{
  "merges": [[[101, 32], 256], [[116, 104], 257], ...],
  "vocab":  {"0": [0], ..., "256": [101, 32], ...}
}
```

- **`merges`**: in learning order, `[[a, b], new_id]`. The order matters, because `encode()` replays the merges in this order.
- **`vocab`**: each ID mapped to its bytes, written as a list of numbers because JSON has no bytes type.

The next step loads it like this:

```python
from bpe_tokenizer import load, encode, decode
merges, vocab = load("tokenizer.json")
ids = encode("The fox and the grapes.", merges)
```

`tokenizer.json` is rebuilt every time you train, so you can delete it at any time.

## What real tokenizers do differently

| This toy | Real tokenizers (GPT-2, tiktoken and others) |
|---|---|
| Counts pairs across the whole text, so merges cross word boundaries (`e `, ` a `) | First split the text into words and punctuation with a regex, then merge only inside each piece. A space can only start a token (` the`). |
| Training recounts every pair after every merge, so 300 merges take about 40 s | Update only the counts that changed. They train on gigabytes with 50,000–200,000 merges. |
| `encode()` replays all 300 merges over the whole text | Look up the best merge in each word directly, and cache common words |
| No special tokens | Reserve IDs for markers like "end of text" or "start of message" |

The idea is the same; the real ones are only faster and larger.

## Try this

1. **Change the number of merges.** Train with 50, 1000 and 3000 merges. Compare the summary lines: how do vocabulary size and bytes per token change? At which merge do whole words appear?
2. **Encode other text.** Load the tokenizer and encode a German sentence, a line of Python and an emoji:
   ```python
   from bpe_tokenizer import load, encode, decode
   merges, vocab = load("tokenizer.json")
   for s in ["Es war einmal ein kleines Mädchen.", "for i in range(10):", "🐇"]:
       ids = encode(s, merges)
       print(len(s.encode()), "bytes ->", len(ids), "tokens:", [decode([i], vocab) for i in ids])
   ```
   Which one costs the most tokens per byte, and why?
3. **Change the training text.** Train only on Aesop, or only on Grimm. Do the first 20 merges change? Does *girl* still split into 4 pieces?
4. **Straighten the quotes.** Replace `’ “ ”` with `' "` in `stories.txt` and train again. What takes the place of merge 14?
