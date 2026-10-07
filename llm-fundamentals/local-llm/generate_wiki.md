# Local LLM, step 5: generation

We are building a toy language model from scratch, one piece at a time:

tokenizer → embeddings → attention → layers → training → **generation**

`generate.py` loads the trained model and asks it two things about a prompt:

1. **What comes next?** It prints the 5 most likely next tokens with their probabilities.
2. **Keep going.** It lets the model write 40 more tokens, one at a time.

## Run it

```bash
cd llm-fundamentals/local-llm
uv run generate.py "Little Red"
uv run generate.py                  # the two built-in prompts
```

`generate.py` always uses `model.pt`. In this repo that's the 10,000-step model.

## How to read the output

```
Prompt: "Little Red"
Top-5 next tokens:
   0.688  '-'
   0.122  ','
   0.019  'i'
   0.018  '.'
   0.015  ' to'
   (confidence of best guess: 69%)
Continuation: 'Little Red-Cap, at the Shepherd blindeather-take him so that\ninquied.\n\nThe Woof and the Tra'
```

(This is the `model.pt` block of [runs/09_compare.txt](runs/09_compare.txt). `explore.py` fixes the dice, so its continuation is repeatable; `generate.py` rolls new dice every time.)

- **Top-5**: the model gives a probability to **every one of the 556 tokens**. Here are the 5 highest. They add up to 86%. The other 551 tokens share the remaining 14%.
- **`'-'` at 0.688**: "Little Red" is followed by "-Cap" 14 times in the stories (*Little Red-Cap*, Grimm's name for Little Red Riding Hood), so the model is 69% sure. The `,` at 12% is the model hedging: maybe the name is already complete.
- **Continuation**: made by *sampling*: roll a weighted die over the 556 probabilities, append the token, repeat. A token with 69% gets picked 69% of the time, not every time.

> A language model never "fails to predict". It always produces a spread of probabilities.
> **A match looks like one tall bar. No match looks like many small ones.**

## The built-in prompts

The file's docstring expects `"Once upon a"` to spike on `" time"`, and `"The quantum"` to give a flat spread. Here's what really happened ([runs/08_generate_10k.txt](runs/08_generate_10k.txt)):

```
Prompt: "Once upon a"                    Prompt: "The quantum"
   0.092  '\n'                              0.542  'p'
   0.062  ' f'                              0.171  'ent'
   0.058  ' c'                              0.087  'b'
   0.049  ' p'                              0.041  'm'
   0.044  ' l'                              0.017  'er'
   (confidence: 9%)                         (confidence: 54%)
```

**"Once upon a"** gives a spread of word starts: a line break, then ` f`, ` c`, ` p`, ` l` (a *forest*, a *cat*, a *poor*, a *long* …). Every one of the top 5 starts a new word. That's right: after the word *a*, a new word comes.

It was not always so. With the first version of the tokenizer, the top guesses were `'b'`, `'l'`, `'f'`: letters that continue a word. That tokenizer glued spaces onto the *end* of tokens, so the word "a" was the token `' a '`, and `' a'` without the trailing space only ever started longer words (*about*, *all*, *after*). The model correctly learned what follows `' a'`, but that wasn't what the prompt meant. [bpe_tokenizer_wiki.md](bpe_tokenizer_wiki.md#why-we-split-into-words-first) has the full story and the fix.

So why not ` time` at the top?

1. **` time` isn't a single token.** It's ` t` + `ime`. ` t` starts many words that have no token of their own (*take*, *two*, *till*, *tree* …), so it can't get a tall bar.
2. **"upon a time" appears only 8 times** in 446 KB of stories. Grimm prefers "There was once…". After "a", *little* (60 times) and *great* (55) are much more common.

**"The quantum"** looks *more* confident than the story opening, at 54% on `'p'`. The model doesn't know "quantum": it's split into ` qu` `ant` `u` `m`, and the prompt ends inside that word. It knows which letters tend to follow `m` inside a word, so it's guessing spelling, not meaning.

The lesson: **the model only ever sees tokens.** The same idea in slightly different tokens can get very different predictions.

## A fair test: match vs no match

Output of `uv run explore.py compare "<prompt>"` for five prompts ([runs/09_compare.txt](runs/09_compare.txt)):

| Prompt | Untrained | 3,000 steps | 10,000 steps |
|---|---|---|---|
| `"Little Red"` | 1% `'\x17'` | 26% `'-'` | **69%** `'-'` |
| `"Once upon a"` | 1% `'^'` | 7% `'fter'` | 9% `'\n'` |
| `"in the mor"` | 1% `'7'` | 26% `'row'` | 35% `'t'` |
| `"Once upon a ti"` | 1% `'@'` | 14% `'es'` | 38% `'red'` |
| `"The quantum "` | 1% `' as'` | 18% `'P'` | 17% `'en'` |

- **Untrained**: every prompt gets about 1% on its best guess. That's the flat, know-nothing spread.
- **"Little Red"** is the clearest match. The model goes from 26% to 69% on `'-'` with 7,000 more steps. That's what a tall bar looks like.
- **"Once upon a"** is many small bars: after *a*, almost any word can come.
- **"The quantum "** stays flat. After a space, the model knows *something* starts, but the trailing space on its own is a rare token (most spaces are glued to the next word), so it guesses capitals and odd pieces.

### The trap: prompts that stop mid-word

`"in the mor"` and `"Once upon a ti"` look like they should be easy. The answers are obviously *morning* and *time*. Instead the model says `'t'` and `'red'` (*mortal*, *tired*). The reason is again the tokenizer:

| Text | Tokens |
|---|---|
| `"in the morning"` (in the stories) | `' m' 'orn' 'ing'` |
| `"in the mor"` (the prompt) | `' m' 'or'` |
| `"Once upon a time"` (in the stories) | `' t' 'ime'` |
| `"Once upon a ti"` (the prompt) | `' t' 'i'` |

In the stories, *morning* never contains the token `'or'`. Merge `orn` already took those letters. So `'or'` after `' m'` is only ever seen in words like *mortal*, *morrow* or *morsel*, and the model guesses accordingly. A prompt that ends inside a token shows the model something it never saw in training. The cut has to fall where the tokenizer would cut.

(Real chat models have the same problem. That's why their prompts end on a whole word or a newline, and why some tools "heal" the last token by backing it up and letting the model redo it.)

## What the text looks like

The same prompt and the same dice (`explore.py` fixes the random seed), from three models:

```
untrained   'Little Redself�ce willeredThe�b�ck\x06 W� my����astK� so�it\x1f st his ofAare��ant�kfself de abess'
3,000       'Little Rediourneent at the dog,\nshe found father, and said, ‘I will kings of a\nmiter, the endfful'
10,000      'Little Red-Cap, at the Shepherd blindeather-take him so that\ninquied.\n\nThe Woof and the Tra'
```

- **Untrained**: random tokens. The `�` marks are tokens that are only half of a UTF-8 character ([bpe_tokenizer_wiki.md](bpe_tokenizer_wiki.md#decoding)).
- **3,000 steps**: real words and phrases (*at the dog*, *she found father, and said, ‘I will*), sensible punctuation, made-up words in between. The 26% `'-'` lost the dice roll, so it wrote *Rediourneent*.
- **10,000 steps**: *Little Red-Cap,* then a story-shaped line and even a fable title (*The Woof and the Tra…*, almost *The Wolf and the …*). Still not English, but it clearly sounds like these stories.

Don't expect sense from 63,468 parameters. The model has learned spelling, common short words and punctuation habits. It hasn't learned grammar or meaning, which needs much bigger models and much more text.

## Temperature

`generate()` takes a **temperature** that reshapes the probabilities before the die is rolled. `generate.py` uses 0.8. Output of `uv run explore.py temperature "Little Red"` ([runs/10_temperature.txt](runs/10_temperature.txt)):

```
temperature 0.5: 'Little Red-Cap, “Theen had you\nsared to find the tree, and soon being stole.\n\nThe Wolf and the Sheep'
temperature 0.8: 'Little Red-Cap, at the Shepherd blindeather-take him so that\ninquied.\n\nThe Woof and the Tra'
temperature 1.2: 'Little Red-Cap, at the Serpose blindeat cellast, and sofiting stied.\n\nThe Weof and and Tra'
```

| Temperature | What it does | What you see above |
|---|---|---|
| **0.5** | sharpens: likely tokens get even likelier | the most real words (*to find the tree, and soon*), and a real fable title: *The Wolf and the Sheep* |
| **0.8** | slightly sharper than the model's own spread | in between |
| **1.2** | flattens: unlikely tokens get more chances | more made-up words (*Serpose*, *sofiting*) and a doubled *and and* |

All three start with *-Cap,*: at 69%, `'-'` is likely enough to win at any of these temperatures. The dice are the same in all three runs, so the texts stay similar where the model is sure and drift apart where it isn't.

Low temperature is safer but gets repetitive. High temperature is more surprising and makes more mistakes. Large chatbots use the same dial.

## Try this

1. Try prompts that end on a whole word: `"Hansel and"`, `"The Wolf and the"`, `"she said to"`. Which give a tall bar?
2. Now cut them mid-word, `"Hansel and Gre"`, `"The Wolf and the L"`, and look at the tokens (see [bpe_tokenizer_wiki.md](bpe_tokenizer_wiki.md#encoding-a-sentence)). When does the cut still work?
3. Run `uv run explore.py temperature "Once upon a"` and compare 0.5 with 1.2.
4. Run `uv run generate.py "Little Red"` three times. Why is the continuation different each time, while the top-5 numbers stay the same?
