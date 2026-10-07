# Local LLM, step 1: the tokenizer

We are building a toy language model from scratch, one piece at a time:

**tokenizer** → embeddings → attention → layers

A model can't read text. It reads a list of numbers. The tokenizer decides which numbers. `bpe_tokenizer.py` is a byte-level BPE (byte pair encoding) tokenizer in about 160 lines of plain Python, with no packages. It works the same way as the tokenizers behind GPT and Claude, only smaller and slower.

Why not one number per word? There are too many words, and a word the model has never seen would have no number at all. Why not one number per character? The lists get very long, and the model has to learn spelling before it can learn anything else. BPE sits in between. Common words like ` the` or ` little` get their own number. Rare words are built from smaller pieces, and in the worst case from single bytes, so every text can be encoded.

## Run it

```bash
cd llm-fundamentals/local-llm
uv run bpe_tokenizer.py stories.txt          # 300 merges, the default
uv run bpe_tokenizer.py stories.txt 1000     # more merges
```

`python bpe_tokenizer.py stories.txt` works too. The script needs no packages.

Training on `stories.txt` takes about 13 seconds with 300 merges. When it's done, the script writes `tokenizer.json` to the current folder and prints three things:
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

## Why we split into words first

The first version of this script counted pairs across the whole text. Merges then crossed word boundaries, and 104 of the 300 tokens ended up with a space at the end or in the middle: `e `, `d `, ` a `, `to the `, `, and `. The sample sentence came out like this:

```
first version:  ['O', 'n', 'ce ', 'up', 'on', ' a ', 'ti', 'me ', 'ther', 'e ', 'wa', 's', ' a ', 'little ', 'g', 'ir', 'l', '.']
now:            ['O', 'n', 'ce', ' upon', ' a', ' t', 'ime', ' there', ' was', ' a', ' little', ' g', 'ir', 'l', '.']
```

That broke prediction. In the training text, the word *a* almost always came with the space after it glued on: `' a '`. So `' a'` with no space after it only ever appeared at the start of a longer word: *about*, *all*, *after*. When you typed `"Once upon a"`, the prompt ended in `' a'`, and the trained model correctly guessed how that token usually goes on: `'b'`, `'l'`, `'f'`. It never had a chance to say ` time`.

The fix is what GPT-2 and later tokenizers do: cut the text into pieces first, and only merge inside a piece.

```python
SPLIT = re.compile(r"""[’'](?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""")
```

Read it left to right, one alternative at a time:

| Part | Matches | Example |
|---|---|---|
| `[’'](?:[sdmt]\|ll\|ve\|re)` | the end of a contraction | `’ll`, `'s` |
| ` ?[^\W\d_]+` | a word (any letters, so `ä` too), with at most one space in front | ` upon` |
| ` ?\d+` | a number, with at most one space in front | ` 42` |
| ` ?(?:[^\s\w]\|_)+` | punctuation, with at most one space in front | `.”` |
| `\s+(?!\S)` | spaces, except the last one before a word | the extra space in `"a  b"` |
| `\s+` | any spaces that are left, like line breaks | `\n` |

```
'Once upon a time, she said: “I’ll go.”'
-> ['Once', ' upon', ' a', ' time', ',', ' she', ' said', ':', ' “', 'I', '’ll', ' go', '.”']
```

**A space can only start a token, never end one.** Every word now looks the same wherever it stands: ` a` is always the whole word *a*, and the next token always starts a new word.

You can check that no token breaks the rule:

```python
from bpe_tokenizer import load
merges, vocab = load("tokenizer.json")
print([vocab[i] for i in range(256, 556) if b" " in vocab[i][1:]])    # []
```

### What the split does *not* fix: ALL-CAPS

`"ONCE UPON A"` still encodes as `['O', 'N', 'C', 'E', ' ', 'U', 'P', 'O', 'N', ' A']`, ten tokens that are mostly single letters. Only 617 of the 84,000 words in `stories.txt` are all capitals, so no capital-letter pairs got merged, and the model has hardly ever seen capitals follow each other. It will guess more capitals. The tokenizer can only make good tokens for text that looks like its training text.

## How training works

Look at `train()` in `bpe_tokenizer.py`:

1. Cut the text into pieces with the `SPLIT` regex: words, numbers, punctuation and runs of spaces. See [Why we split into words first](#why-we-split-into-words-first).
2. Turn each piece into UTF-8 bytes. Every byte is a number from 0 to 255, so the starting vocabulary is those 256 values.
3. Count every pair of neighbouring tokens **inside each piece** (`get_pair_counts`). A piece that occurs 9,000 times is stored once and its pairs count 9,000 times.
4. Take the most frequent pair and give it a new ID: 256 for the first merge, 257 for the second, and so on.
5. Replace every occurrence of that pair with the new ID (`apply_merge`). The pieces get shorter.
6. Repeat until you've done `num_merges` merges, or until no pair appears twice.

The result is two things:
- **`merges`**: the ordered list of rules, "pair (a, b) becomes new_id".
- **`vocab`**: for every ID, the bytes it stands for.

### A tiny example

Train on `"the cat sat on the mat"` with 4 merges:

```
merge   1: b'a' + b't' -> b'at'     (id 256, seen 3 times)    cat, sat, mat
merge   2: b't' + b'h' -> b'th'     (id 257, seen 2 times)
merge   3: b'th' + b'e' -> b'the'   (id 258, seen 2 times)    built on merge 2

Text went from 22 bytes to 15 tokens.
```

It asked for 4 merges but stopped after 3. The pieces are `the`, ` cat`, ` sat`, ` on`, ` the`, ` mat`, and after merge 3 no pair appears twice any more. `the` and ` the` are different pieces, so `the` + ` ` can't happen: the space belongs to the *next* word.

Try it yourself:

```bash
uv run python -c "from bpe_tokenizer import train; train('the cat sat on the mat', 4)"
```

## Reading the merge lines

The real run on `stories.txt` starts like this:

```
merge   1: b'h' + b'e' -> b'he'   (id 256, seen 13375 times)
merge   2: b' ' + b't' -> b' t'   (id 257, seen 12105 times)
merge   3: b' ' + b'a' -> b' a'   (id 258, seen 9110 times)
merge   4: b' t' + b'he' -> b' the'   (id 259, seen 6550 times)
...
merge   8: b'n' + b'd' -> b'nd'   (id 263, seen 4992 times)
...
merge  12: b' a' + b'nd' -> b' and'   (id 267, seen 3603 times)
merge  13: b'\xe2' + b'\x80' -> b'\xe2\x80'   (id 268, seen 3431 times)
```

Each part of a line means:
- **`b'h'`**: Python's notation for *bytes*, not a string. The tokenizer works on raw bytes.
- **`b'h' + b'e'` → `b'he'`**: the most frequent pair was the letter `h` followed by `e`. The two become one token.
- **`id 256`**: IDs 0–255 are the single bytes. Each merge gets the next free ID.
- **`seen 13375 times`**: how often that pair appeared **at that moment**, after all the earlier merges.

### What the merges show

**English letter statistics come out first.** `he` is first: *the, he, she, her, then*. Next come ` t`, ` a`, ` s`, `in`, ` w`. Nobody told the tokenizer this; it found it by counting.

**Spaces only ever sit at the front.** Many early merges are a space plus a first letter: ` t`, ` a`, ` s`, ` w`, ` h`. They mean "a word starting with this letter". Because of the split, there is no `e ` or `d ` token any more.

**Later merges build on earlier ones:**

| Merge | Built from | Result |
|---|---|---|
| 1 | `h` + `e` | `he` |
| 2 | ` ` + `t` | ` t` |
| 4 | ` t` (merge 2) + `he` (merge 1) | **` the`** |
| 3 | ` ` + `a` | ` a` |
| 8 | `n` + `d` | `nd` |
| 12 | ` a` (merge 3) + `nd` (merge 8) | **` and`** |

By merge 12, the two most common English words are one token each.

**The counts keep falling.** `he` was seen 13,375 times, but ` the` only 6,550 times: many `he`s sit inside other words (*she, her, then*), and *The* with a capital is a different token (` The`, merge 83).

**Merge 13 is half a character.** `\xe2\x80` is not a letter. The stories use curly quotes, and in UTF-8 each of those is three bytes that start the same way:

| Character | UTF-8 bytes |
|---|---|
| ’ | `e2 80 99` |
| “ | `e2 80 9c` |
| ” | `e2 80 9d` |

The tokenizer only sees bytes. It noticed that `e2 80` is common and merged it. Later merges add `99` (ID 298) and `9d` (ID 370), so `’` and `”` become one token each. This is why the tokenizer is called *byte-level*: tokens don't have to be whole characters.

**The longest tokens are whole common words**, all with their leading space: ` little`, ` should`, ` himself`, ` great`, ` again`, ` would`.

### The summary line

```
Training done. Vocabulary size: 556 tokens. Text went from 445847 bytes to 195262 tokens.
```

That's 256 bytes plus 300 merges, which is 556 tokens. Each token now covers about 2.3 bytes on average. More merges give fewer, longer tokens, but also a bigger vocabulary for the model to learn.

## Encoding a sentence

```
Sample:    Once upon a time there was a little girl.
Token IDs: [79, 110, 345, 491, 258, 257, 503, 477, 321, 258, 486, 295, 327, 108, 46]
Tokens:    ['O', 'n', 'ce', ' upon', ' a', ' t', 'ime', ' there', ' was', ' a', ' little', ' g', 'ir', 'l', '.']
Round trip OK: True
```

The two lists line up position by position: ID `79` is `'O'`, ID `110` is `'n'`, and so on. **The list of IDs is all the model will ever see.** The token list is only there so humans can read it.

`encode()` does what training did: split into pieces, then replay the merges inside each piece. It remembers each piece it has already encoded, so a long text with many repeated words is quick.

### How to read an ID

- **0–255**: a single byte. For plain English letters this is the ASCII code: `79` = `O`, `108` = `l`, `46` = `.`.
- **256 and up**: a merged token. **Merge number = ID − 255.** A lower ID means the token was learned earlier, which means its pair was more common.

An ID is only a label. `258` isn't bigger or closer to anything than `257`. Meaning comes later, in step 2, when the model learns an embedding vector for each ID.

### Every token, traced back to bytes

| Token | ID | Merge # | Built from |
|---|---|---|---|
| `'O'` | 79 | – | single byte |
| `'n'` | 110 | – | single byte |
| `'ce'` | 345 | 90 | `c` + `e` |
| `' upon'` | 491 | 236 | ` up` + `on` |
| `' a'` | 258 | 3 | ` ` + `a` |
| `' t'` | 257 | 2 | ` ` + `t` |
| `'ime'` | 503 | 248 | `im` + `e` |
| `' there'` | 477 | 222 | ` the` (259) + `re` (264) |
| `' was'` | 321 | 66 | ` w` (262) + `as` |
| `' a'` | 258 | 3 | same token as above, so the same ID |
| `' little'` | 486 | 231 | ` l` + `ittle`, several merges deep |
| `' g'` | 295 | 40 | ` ` + `g` |
| `'ir'` | 327 | 72 | `i` + `r` |
| `'l'` | 108 | – | single byte |
| `'.'` | 46 | – | single byte |

### Why the words split where they do

`encode()` replays the merges **in the order they were learned**, not "longest piece first". The pair learned earliest claims the bytes, and later merges can't take them back.

- **`O` `n` `ce`**: *Once* starts the sentence, so its piece has no space in front. ` once` in the middle of a sentence is common, but `Once` without a space is rare, and the capital `O` + `n` never got merged. Capitalised words usually cost more tokens.
- **` upon`, ` there`, ` was`, ` little`**: common words are one token each, with their leading space.
- **` t` `ime`**: there is no ` time` token. *time* comes up often, but not often enough to win one of the 300 merges. More merges would give it one.
- **` g` `ir` `l`**: *girl* is rare in these stories, so it falls back to small pieces. Rare words cost more tokens.

41 bytes became 15 tokens, about 2.7 bytes per token. This is also why a model trained mostly on English text uses more tokens, and costs more, for German or for code.

### Decoding

`decode()` looks up each ID's bytes in `vocab`, joins them and turns the bytes back into text. `Round trip OK: True` means decoding the 15 IDs gave exactly the original sentence.

A token can be half a character, like `\xe2\x80` above. If you decode such a token on its own, the bytes aren't valid UTF-8, and `errors="replace"` shows them as `�` instead of crashing.

## `tokenizer.json`

`save()` writes the two things training produced:

```json
{
  "merges": [[[104, 101], 256], [[32, 116], 257], ...],
  "vocab":  {"0": [0], ..., "256": [104, 101], ...}
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
| Splits into words with a regex, like GPT-2 (this one we copied) | The same idea, with a more careful regex (GPT-4's also caps numbers at 3 digits) |
| Training recounts every pair in every distinct word after every merge, about 13 s for 300 merges | Update only the counts that changed. They train on gigabytes with 50,000–200,000 merges. |
| `encode()` replays all 300 merges over each new word | Look up the best merge in each word directly |
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
5. **Turn the split off.** Replace `SPLIT.findall(text)` with `[text]` in `train()` and `encode()`, train again, and look for tokens like `e ` and ` a `. Then retrain the model on them (see [train_wiki.md](train_wiki.md)) and ask it to continue `"Once upon a"`.
3. **Change the training text.** Train only on Aesop, or only on Grimm. Do the first 20 merges change? Does *girl* still split into 3 pieces?
4. **Straighten the quotes.** Replace `’ “ ”` with `' "` in `stories.txt` and train again. What takes the place of merge 13?
