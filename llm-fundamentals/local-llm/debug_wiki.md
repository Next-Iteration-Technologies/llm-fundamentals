# Local LLM, the debugger: every number, in words

`debug.py` follows one prompt through the trained model, stage by stage. It prints every number the model computes, and next to each one says what it means in tokens and words. It changes nothing: `model.pt` is only read.

```bash
cd llm-fundamentals/local-llm
uv run debug.py overview "Little Red"          # one line per stage
uv run debug.py trace "Little Red"             # every number, for the token making the guess
uv run debug.py full "Little Red"              # every number, for every token, plus the attention tables
uv run debug.py compare "Little Red"           # untrained vs trained, stage by stage
uv run debug.py off "Little Red"               # switch off each head and neuron: which ones matter?
uv run debug.py off "Little Red" --off 1.ff108 # switch off one part (1.2 = layer 1 head 2)
uv run debug.py train-step "Little Red"        # one training step, in slow motion
```

Without a prompt it uses `"Once upon a time there was a little girl. One day Little Red"`. That is 23 tokens, so attention has plenty to look at. It ends on a whole word on purpose: a prompt that stops mid-word ends on a token the model rarely saw there ([generate_wiki.md](generate_wiki.md#the-trap-prompts-that-stop-mid-word)).

| Option | What it does |
|---|---|
| `--model PATH` | inspect another model, e.g. `model_3k.pt` or `model_untrained.pt` |
| `--token N` | `trace`: follow token N instead of the last one |
| `--layer L`, `--head H` | `trace`, `full`: print only that layer or head |
| `--html PATH` | also write the interactive page (below) |

Layers, heads, neurons and tokens are counted from 0, as in `explore.py`.

| Command | Log |
|---|---|
| `overview` | [runs/13_debug_overview.txt](runs/13_debug_overview.txt) |
| `trace "Little Red"` | [runs/14_debug_trace.txt](runs/14_debug_trace.txt) |
| `full "Little Red"` | [runs/15_debug_full.txt](runs/15_debug_full.txt) |
| `compare` | [runs/16_debug_compare.txt](runs/16_debug_compare.txt) |
| `off` | [runs/17_debug_off.txt](runs/17_debug_off.txt) |
| `train-step` | [runs/18_debug_train_step.txt](runs/18_debug_train_step.txt) |
| `overview --html runs/19_debug_page.html` | [runs/19_debug_page.html](runs/19_debug_page.html) |

## Can you trust the printout?

`debug.py` runs the model's own parts one at a time, in the same order as `ToyGPT.forward` in `model.py`, and keeps every value along the way. Then it checks itself against the real model:

```
Check: this trace and model() agree to within 9.5e-06
```

So the numbers you read are the numbers the model actually computed. For `"Little Red"` it ends on `'-'` at 68.8%, the same as `explore.py compare` in [runs/09_compare.txt](runs/09_compare.txt).

## How a vector becomes words

The model's vectors have no labels. Number 7 of 32 doesn't mean "noun" or "past tense". So each vector is read through **what it matches** or **what it pushes the prediction towards**:

| Thing | Size | Read as |
|---|---|---|
| token card + seat card | 32 + 32 | **nearest cards**: the tokens whose cards point the same way (cosine, 1 = same direction) |
| query | 8 per head | **looks for**: the tokens whose key it matches best |
| key | 8 per head | **answers**: the tokens whose query matches it best |
| value | 8 per head | **passes on**: the next tokens it pushes up when someone listens to it |
| any change added to the vector | 32 | **pushes**: the next tokens it pushes up and down |
| a neuron | 1 of 128 | **fires hardest at**: the places in the stories where it fires most; **pushes**: what its output pushes up |
| any part | | **for the winner**: how much it raises the raw score of the token that ends up predicted |
| the vector so far | 32 | **if it stopped here**: the final steadying and prediction head applied early (known as the "logit lens") |

Three honest notes:

1. **"Pushes" and "for the winner" are straight-line readings.** They pass the change through the final steadying's scale and the prediction head, but ignore the steadying's overall resizing. They tell you the direction (helps `'-'`, hurts `'i'`), not exact percentages. "If it stopped here" is exact.
2. **"Looks for" and "answers" are the roughest readings.** They take each of the 556 tokens on its own, at this position, and measure how well it would match. In layer 0 that is the real situation. In layer 1, a real token's vector has already mixed in its neighbours, which a lone token can't do, so the printout marks it `(approx)`. The **share of attention** lines below them are exact: those are the real tokens in your sentence.
3. **Readings only list tokens seen at least 20 times in the stories** (364 of 556). The others, mostly lone bytes of rare characters, were hardly trained, so their numbers are leftovers from the random start and would crowd out the rest.

How tokens are written: `·` is a space, `⏎` a new line and `⟨e2⟩` a lone byte. So `'·upon'` is the single token " upon".

## Overview: the guess forming, layer by layer

[runs/13_debug_overview.txt](runs/13_debug_overview.txt), last token `'ed'` (of *Red*):

```
TOKENS        'O' 'n' 'ce' '·upon' '·a' '·t' 'ime' '·there' '·was' '·a' '·little' '·g' 'ir' 'l' '.' '·O' 'ne' '·day' '·L' 'ittle' '·' 'R' 'ed'
EMBEDDING     'ed' = card #271 + seat 22.  Nearest cards: 'ack' +0.60  'ght' +0.53  'ing' +0.52  'ved' +0.52  'hed' +0.50
                 if it stopped here: ':' 23%  '·upon' 17%  '·into' 12%  '·and' 7%  '·away' 5%
L0 ATTENTION  'ed' listens: h0 → 'R' 0.79 · h1 → '·' 0.44 · h2 → '·' 0.39 · h3 → 'R' 0.40
                 if it stopped here: 'i' 14%  ':' 12%  'al' 12%  '·and' 11%  'ed' 8%
L0 FEED-FWD   48 of 128 neurons fire; key neuron #67 (+1.3 for '-', fires for … he was [ve])
                 if it stopped here: 'e' 52%  'i' 20%  ',' 6%  '·to' 4%  '·the' 4%
L1 ATTENTION  'ed' listens: h0 → 'R' 0.67 · h1 → 'ittle' 0.54 · h2 → '·' 0.44 · h3 → 'R' 0.48
                 if it stopped here: 'i' 68%  'it' 4%  'le' 3%  '·to' 2%  ',' 2%
L1 FEED-FWD   38 of 128 neurons fire; key neuron #108 (+4.0 for '-', fires for … OLD[ M])
                 if it stopped here: '-' 81%  ',' 8%  'ge' 1%  'i' 1%  '.' 1%
PREDICTION    '-' 81%  ',' 8%  'ge' 1%  'i' 1%  '.' 1%   (other 551 tokens share 8%)
```

*Red* is not one token. The tokenizer never merged ` R` (capital R after a space is rare), so it's `'·'` `'R'` `'ed'`. The model has to work out from `'R'` + `'ed'` that this is the word *Red*.

Read the "if it stopped here" lines from top to bottom:

- **The card alone** only knows `'ed'` is a word ending (its nearest cards are `'ing'`, `'ved'`, `'hed'`), so it guesses what follows a finished verb: `:`, ` upon`, ` into`.
- **After layer 0's attention**, heads 0 and 3 look back at `'R'`. Now it's no longer a verb ending, but the guess is still mostly spelling.
- **After layer 1's attention**, head 1 looks back at `'ittle'`, the *Little* in front. The guess becomes `'i'` 68%: it's spelling a word that starts with *Red*, such as *Redi…*.
- **After layer 1's feed-forward**, `'-'` jumps from almost nothing to 81%: "Red-Cap". **The answer appears in the very last step**, the second thinking room.

With the short prompt `"Little Red"` the same happens: `'i'` 44% after layer 1's attention, then `'-'` 69% after its feed-forward ([runs/14_debug_trace.txt](runs/14_debug_trace.txt)). The long prompt is surer (81%) because the story before it gives more context.

## Trace: every number for one token

[runs/14_debug_trace.txt](runs/14_debug_trace.txt) follows `'ed'` in `"Little Red"`. Each stage prints its numbers, then the reading.

### Embedding

```
token card  +0.64 +0.93 +0.71 -0.06 -0.98 +0.38 -0.55 +0.89 ...
seat 4      -0.38 +0.50 -0.16 -0.54 -0.51 -0.23 +1.43 -0.11 ...
sum         +0.26 +1.43 +0.55 -0.60 -1.49 +0.15 +0.88 +0.78 ...
meaning: nearest cards to 'ed' (cosine, 1 = same direction): 'ack' +0.60  'ght' +0.53  'ing' +0.52  'ved' +0.52  'hed' +0.50
```

Row 271 of the token table, plus row 4 of the seat table (because `'ed'` sits in seat 4), number by number. The sum is what goes upstairs.

### Steadying, then one head

```
--- LAYER 0 · STEADYING (ln1): mean +0.11, spread 1.12  ->  mean -0.13, spread 0.71 ---

--- LAYER 0 · HEAD 0 ---
query       -1.21 +0.41 +1.57 +2.63 -1.36 -0.04 -0.31 +6.36   looks for: ']' -10.70  '·H' -11.95  '·T' -12.05  ...
key         +2.68 -0.96 -2.55 -1.90 +1.34 -1.47 -1.74 -4.43   answers:   '·my' +3.26  '·was' +2.05  ...
value       -0.71 -0.90 -1.15 +0.00 +0.01 +0.75 +0.37 -0.21   passes on: 'ch' +4.13  'ut' +3.54  'v' +3.27  'et' +2.98  'ved' +2.68
Q·K score (÷√8) and share of attention for each token so far:
            'L'  score -19.70  ->  0.002
        'ittle'  score -17.27  ->  0.019 █
            '·'  score -16.26  ->  0.051 ██
            'R'  score -13.61  ->  0.723 ██████████████████████
           'ed'  score -14.87  ->  0.205 ██████
blend       -0.21 +0.18 -0.36 -0.44 +1.03 -0.32 +0.43 +0.19   (the values, mixed by those shares)
this head pushes: up 'on' +2.71  'ch' +2.71  'i' +2.57  '·H' +2.42  'le' +2.34   down 'ard' -4.61  'ght' -4.21  ⟨98⟩ -4.15
for the winner '-': +1.04
```

- **Steadying** rescales the 32 numbers to a calm range (mean near 0, spread near 1) before the heads see them.
- **Query, key and value** are 8 numbers each, made from the steadied vector by the head's three matrices `Wq`, `Wk`, `Wv`.
- **The score** for each earlier token is its key dotted with this query, divided by √8. The raw scores are negative here, but only the **differences** matter. `'R'` at -13.61 is 1.26 higher than `'ed'` at -14.87, and softmax turns that gap into 0.723 vs 0.205 (e^1.26 ≈ 3.5 times as much).
- **The blend** is the values of all five tokens mixed by those shares, mostly `'R'`. That is what this head hands upstairs: "the letter before me was R".
- **For the winner** says this head raises `'-'` by +1.04. Head 1 of layer 0 lowers it a little (-0.10). Heads can work against the final answer.

Future tokens are shown as `— future, hidden by the mask` when you trace a token in the middle (`--token`).

### Stitch and add

```
--- LAYER 1 · STITCH 4 heads x 8 -> 32 (proj), then ADD to the input ---
pushes: up '[' +22.58  ']' +19.58  '(' +14.82  'M' +13.96  'nt' +13.80   down 'ess' -11.40  'ell' -11.17  'ered' -11.08
for the winner '-': +2.20
if it stopped here: 'i' 44%  '·to' 9%  'it' 5%  ',' 4%  'le' 4%
```

The four 8-number blends are joined into 32 and mixed by `proj`, then **added** to what the token already had (the residual). Layer 1's attention, taken together, pushes `'-'` up by 2.20, but `'i'` still leads.

### Feed-forward: widen, cut, shrink

```
--- LAYER 1 · FEED-FORWARD: steady (ln2), widen 32 -> 128, cut negatives, shrink 128 -> 32 ---
widened (128 neurons, after cutting negatives): 36 fire
   | ▃  ▄ ▄▂  ▄ ▂         ▆    ▅            ▆  ▄      ▁   ▄   ▂ ▆▁         ▂  ▅▁  ▃▁ ▂   ▃   ▇ ▁ ...|
the neurons that move the winner '-' most (firing x output):
   #108 fires 2.65  -> +3.76 for '-'   fires hardest at: … OLD[ M]  … RO[L]  …LTS[K]
   #89  fires 2.39  -> -2.63 for '-'   fires hardest at: …’s no p[la]  …irst p[la]  …ne of the[ w]
   #27  fires 1.82  -> -1.98 for '-'   fires hardest at: … Ne[ver]  …⏎Gret[el]  … went the whe[el]
   #6   fires 1.47  -> +1.98 for '-'   fires hardest at: …BOD[,]  …MIN[,]  …THY[,]
   ...
pushes: up 'ore' +13.92  'd' +12.84  'self' +12.39  'st' +11.57  'T' +11.02   down 'ut' -14.33  'hen' -13.62  'ter' -13.30
for the winner '-': +6.99
if it stopped here: '-' 69%  ',' 12%  'i' 2%  '.' 2%  '·to' 1%
```

- **Widen**: 32 numbers become 128 neurons. **Cut**: negatives become 0 (ReLU), so 92 of the 128 stay silent. The block line shows all 128: blank = silent, █ = the strongest.
- **The neuron table** ranks neurons by what they add to `'-'`: how hard each fires, times its output column. "Fires hardest at" comes from one pass over the held-out stories. Neurons #108 and #6 fire hardest inside words written in capitals, like the openings of Aesop's fables ("AN OLD MAN …"). They seem to react to capital letters in the middle of a word or name, and *Red*, built from `'R'` + `'ed'`, sets them off. Neurons #89 and #27 work against `'-'`.
- **Shrink**: 128 back to 32, then added on. This one step adds **+6.99** to `'-'`, more than everything before it.

### Prediction

```
highest raw scores: '-' +10.62  ',' +8.89  'i' +7.02  '.' +6.99  '·to' +6.79
            '-'   68.8% ████████████████████████████
            ','   12.2% █████
      other 546    8.2%
```

A final steadying, then the prediction head gives 556 raw scores. Softmax turns them into chances. `'-'` leads `','` by 1.73 raw points, which softmax turns into e^1.73 ≈ 5.6 times the chance.

## Full: every token

`full` prints the same trace for every token, then the whole attention table for each head ([runs/15_debug_full.txt](runs/15_debug_full.txt)). Run it on the long sentence with `--layer 0 --head 0` and the first 15 rows are the table from [runs/12_attention_trained.txt](runs/12_attention_trained.txt) again, with the first `'·a'` giving 0.71 to itself.

## Compare: before and after training

[runs/16_debug_compare.txt](runs/16_debug_compare.txt) prints each stage twice:

```
L1 FEED-FWD
   untrained  62 of 128 neurons fire; key neuron #9 (+0.1 for '⟨17⟩', fires for …s soon[ as])
   trained    38 of 128 neurons fire; key neuron #108 (+4.0 for '-', fires for … OLD[ M])
PREDICTION
   untrained  '⟨17⟩' 1%  '·whe' 1%  'Z' 1%  ⟨ea⟩ 1%  '␍' 1%   (other 551 tokens share 97%)
   trained    '-' 81%  ',' 8%  'ge' 1%  'i' 1%  '.' 1%   (other 551 tokens share 8%)
```

The untrained model has the same machinery, but nothing moves the guess: its best neuron adds +0.1. About half of its neurons fire (62 to 63 of 128). After training, fewer fire (38 to 48), and some of those really matter.

## Off: which parts matter?

[runs/17_debug_off.txt](runs/17_debug_off.txt) silences one head, or one neuron, and reruns:

```
Normal best guess: '-' 81.2%
   layer 0 head 0     81.2% -> 25.6%  █████████████████████████████████   now: '-' 26%  'i' 23%  'ge' 7%
   layer 1 head 0     81.2% -> 85.0%     now: '-' 85%  ',' 5%  'i' 1%
   layer 1 head 3     81.2% -> 12.3%  █████████████████████████████████████████   now: 'i' 50%  '-' 12%  ',' 7%
   ...
The 8 neurons (of 256) whose silence hurts '-' most:
   layer 1 neuron #60     81.2% -> 24.2%   now: 'ge' 61%  '-' 24%  ',' 5%
   layer 1 neuron #108    81.2% -> 24.7%   now: 'i' 32%  '-' 25%  'ge' 13%
```

- **Two heads carry the answer.** Layer 0 head 0 (the one that looks at `'R'`) and layer 1 head 3. Without either, `'-'` falls from 81% to 26% or 12%, and `'i'` takes over.
- **One neuron out of 256** (#60 in layer 1) takes `'-'` from 81% to 24%. Then `'ge'` wins: "Little Red**ge**…", a spelling guess.
- **Some heads work against the answer**: without layer 1 head 0, `'-'` goes *up* to 85%.
- "For the winner" in the trace and "switched off" here can rank parts differently. The first measures only a part's own push. The second also catches everything downstream that depended on it. Neuron #60 isn't in the trace's top list at all, but silencing it changes what the later parts see.

## Train-step: one lesson in slow motion

[runs/18_debug_train_step.txt](runs/18_debug_train_step.txt) does one turn of `train.py`'s loop on a copy of the model:

```
1. FORWARD. Loss 3.005: on average the right next token got e^-3.005 = 5.0% of the chance.
           'ak' -> 'en'          90.5%   surprise  0.10
          '·to' -> '·in'          0.2%   surprise  6.03
2. BACKWARD.
   layer 0 feed-forward shrink        gradient  0.3532   moved  0.1920
   Token cards with a gradient: 328 of 556 (only tokens that appear in the batch).
3. STEP. Loss on the same batch: 3.005 -> 2.789
   Prompt "...One day Little Red":  '-' 81.2% -> 79.4%
```

- **Surprise** is −ln(chance). "tak" → `'en'` was easy (0.10): *taken*. After "to", ` in` was a surprise (6.03). The loss is the average surprise.
- **Only cards of tokens in the batch get a gradient**: a lesson can't teach a card it never shows.
- **One step** lowers the loss on that batch from 3.005 to 2.789. That is more than a real late-training step would, because `train.py` didn't save its optimizer's memory: this step starts AdamW fresh, and the loss is measured on the batch the step was made for.
- **Our prompt got slightly worse** (81.2% → 79.4%). The batch had nothing about Little Red-Cap in it, and every weight moved a little for other texts. Training is a tug of war between all the examples.

## The page

`--html PATH` writes one self-contained page: open it in any browser, phone included. Whatever command made it, the page holds everything for that prompt: the trained and untrained traces, the switch-off ranking and the training step. [runs/19_debug_page.html](runs/19_debug_page.html) is the default prompt.

- **Tokens**: tap one to follow it.
- **Overview**: one card per stage, each with its "if it stopped here" bars. Tap a card for every number of that stage, or tick "show all stages in full".
- **Numbers** are coloured cells: orange positive, blue negative, stronger colour for bigger. Hover or tap one, and the bar at the bottom says what it is (`query #3 = +6.35`).
- **Attention**: each head's bars, plus a fold-out table of the whole sentence.
- **Model tabs** switch between `model.pt` and `model_untrained.pt`.

The page template is `debug_page.html`; `debug.py` fills in the data.

## Try this

1. `uv run debug.py overview "The quantum "`: at which stage does the model give up, and what does "many small bars" look like inside?
2. `uv run debug.py trace "Little Red" --model model_3k.pt`: is neuron #108 already there after 3,000 steps?
3. `uv run debug.py off "Little Red" --off 1.ff108`: neuron #108 likes capitals. What wins when it's silent?
4. `uv run debug.py overview "in the mor"`, then `"in the m"`: why does the shorter prompt get closer to *morning*? (Hint: [generate_wiki.md](generate_wiki.md#the-trap-prompts-that-stop-mid-word).)
5. `uv run debug.py trace "Once upon a time there was a little girl." --token 10 --layer 0`: follow `'·little'`. Which heads look back at `'·was'`?
