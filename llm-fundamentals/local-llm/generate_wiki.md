# Local LLM, step 5: generation

We are building a toy language model from scratch, one piece at a time:

tokenizer → embeddings → attention → layers → training → **generation**

`generate.py` loads the trained model and asks it two things about a prompt:

1. **What comes next?** It prints the 5 most likely next tokens with their probabilities.
2. **Keep going.** It lets the model write 40 more tokens, one at a time.

## Run it

```bash
cd llm-fundamentals/local-llm
uv run generate.py "in the mor"
uv run generate.py                  # the two built-in prompts
```

`generate.py` always uses `model.pt`. In this repo that's the 10,000-step model.

## How to read the output

```
Prompt: "in the mor"
Top-5 next tokens:
   0.843  'n'
   0.038  'e '
   0.019  'ne'
   0.013  'row'
   0.011  'k'
   (confidence of best guess: 84%)
Continuation: 'in the morning, saw alther old, and the thief. He fearly feaule shibbling up tooken cont'
```

- **Top-5**: the model gives a probability to **every one of the 556 tokens**. Here are the 5 highest. They add up to 92%. The other 551 tokens share the remaining 8%.
- **`'n'` at 0.843**: "in the mor" is almost always followed by "ning" in the stories, so the model is 84% sure. (`'n'` and not `'ning'`, because `ning` isn't a token in our 556-token vocabulary.)
- **Continuation**: made by *sampling*: roll a weighted die over the 556 probabilities, append the token, repeat. A token with 84% gets picked 84% of the time, not every time.

> A language model never "fails to predict". It always produces a spread of probabilities.
> **A match looks like one tall bar. No match looks like many small ones.**

## The built-in prompts: a surprise

The file's docstring expects `"Once upon a"` to spike on `" time"`, and `"The quantum"` to give a flat spread. Here's what really happened ([runs/08_generate_10k.txt](runs/08_generate_10k.txt)):

```
Prompt: "Once upon a"                    Prompt: "The quantum"
   0.118  '\n'                              0.204  'm'
   0.117  'b'                               0.072  'ent'
   0.084  ' s'                              0.067  'p'
   0.068  'l'                               0.065  ' '
   0.063  'f'                               0.055  'en'
   (confidence: 12%)                        (confidence: 20%)
```

The "nonsense" prompt looks *more* confident than the story opening. The tokenizer explains why:

| Prompt | Tokens | The last token is… |
|---|---|---|
| `"Once upon a"` | `'O' 'n' 'ce ' 'up' 'on' ' a'` | `' a'`, **without** a trailing space |
| `"Once upon a "` | `… 'on' ' a '` | `' a '`, a different token (316 instead of 259) |
| `"The quantum"` | `'The ' 'qu' 'an' 't' 'u' 'm'` | `'m'`, inside a word |

1. **`' a'` is not the word "a".** In the stories, `' a'` followed by more letters is the start of *about*, *away*, *and*, *after*… So the model guesses letters: `b`, `l`, `f`. The word "a" on its own is the token `' a '`, with the space.
2. **`' time'` isn't a token at all.** The vocabulary only has `'ti'` and `'me '`. No single token could have spiked.
3. **"upon a time" appears only 8 times** in 446 KB of stories. Grimm prefers "There was once…".
4. **`"The quantum"` ends inside a word** (`'u' 'm'`). The model doesn't know "quantum". It knows that after `m` you often get `m` or `p` (*summer*, *jump*), so it's guessing spelling, not meaning.

The lesson: **the model only ever sees tokens.** The same idea in slightly different tokens can get very different predictions.

## A fair test: match vs no match

Prompts that stop partway through something the model *does* know show the tall bar clearly. Output of `uv run explore.py compare "<prompt>"` for three prompts ([runs/09_compare.txt](runs/09_compare.txt)):

| Prompt | Untrained | 3,000 steps | 10,000 steps |
|---|---|---|---|
| `"Once upon a ti"` | 1% `'ook'` | **65%** `'me '` | **65%** `'me '` |
| `"in the mor"` | 1% `'om'` | **71%** `'n'` | **84%** `'n'` |
| `"The quantum "` | 1% `'lea'` | 6% `'ti'` | 5% `'li'` |

- **Untrained**: every prompt gets about 1% on its best guess. That's the flat, know-nothing spread.
- **"in the mor"** is the clearest match. "in the morning" appears 18 times in the stories. After 3,000 steps the model is 71% sure; 7,000 more steps push it to 84%.
- **"Once upon a ti"** is already at 65% after 3,000 steps and doesn't improve. The phrase is rare, but `ti` → `me ` is common everywhere (*time*, *sometimes*).
- **"The quantum "** stays flat at about 5%. After a space, the model knows a word is starting but has no idea which. That's what "no match" looks like.

## What the text looks like

The same prompt and the same dice (`explore.py` fixes the random seed), from three models:

```
untrained   'in the morge omit ke fromse�ion9 and the Pough� p��y �the��ould �eaoun�The hiAsp: ‘�. The J\x08fbe deesswas '
3,000       'in the morning, saw althe\nold toneie. Now the fox behead him and hile he meattting them conqu'
10,000      'in the morning, saw alther old, and the thief. He fearly feaule shibbling up tooken cont'
```

- **Untrained**: random tokens. The `�` marks are tokens that are only half of a UTF-8 character ([bpe_tokenizer_wiki.md](bpe_tokenizer_wiki.md#decoding)).
- **3,000 steps**: real words (*morning*, *saw*, *old*, *Now the fox*, *him and*), sensible punctuation, made-up words in between.
- **10,000 steps**: slightly more real words and a cleaner sentence shape (*and the thief. He …*). Still not English, but it clearly sounds like these stories.

Don't expect sense from 63,468 parameters. The model has learned spelling, common short words and punctuation habits. It hasn't learned grammar or meaning, which needs much bigger models and much more text.

## Temperature

`generate()` takes a **temperature** that reshapes the probabilities before the die is rolled. `generate.py` uses 0.8. Output of `uv run explore.py temperature "in the mor"` ([runs/10_temperature.txt](runs/10_temperature.txt)):

```
temperature 0.5: 'in the morning, he would not by the stone evening that they are a lives, by mead. If know h'
temperature 0.8: 'in the morning, saw alther old, and the thief. He fearly feaule shibbling up tooken cont'
temperature 1.2: 'in the mor of\nit had herselve, and said, “He! Alada? If live\nGretem–8 mus deess. A'
```

| Temperature | What it does | What you see above |
|---|---|---|
| **0.5** | sharpens: likely tokens get even likelier | the most real words (*he would not by the stone evening that they are*) |
| **0.8** | slightly sharper than the model's own spread | in between |
| **1.2** | flattens: unlikely tokens get more chances | it even skips the 84% `'n'`, writing *mor of*; more made-up words and odd symbols (`–8`) |

Low temperature is safer but gets repetitive. High temperature is more surprising and makes more mistakes. Large chatbots use the same dial.

## Try this

1. Try prompts that stop mid-word: `"The Fox and the Gra"`, `"Peter Rab"`, `"Hansel and Gre"`. Which give a tall bar?
2. Try a modern sentence: `"My laptop compiles "`. What does the model fall back on?
3. Run `uv run explore.py temperature "Once upon a "` and compare 0.5 with 1.2.
4. Run `uv run generate.py "in the mor"` three times. Why is the continuation different each time, while the top-5 numbers stay the same?
