# Trainer notes for LAB.md

*Not for students. Expected results and talking points per chapter, from the build runs. Your numbers may differ a little
with another embedding or chat model; the shape should hold.*

Every command runs in `rag-exercises/python`. The figures below say which embedding model and which chat model they
came from. "hash" is the offline stand-in (`--embed hash`): no model, word matching only.

**Before the session:** run each chapter once on the trainer machine. Embeddings, reranker scores and model replies are
saved (`python/.store/lab-cache.sqlite`), so the run in the room is fast, and the same command gives the same output.

---

## Chapter 1. Ask it something

**Goal:** name each step of a RAG pipeline and say what goes in and out of it.

**What they see** (`walkthrough --question 1`):

- **D&D, question 1** (easy-0): "Compare the restoration abilities of the Pit Fiend, Planetar, Rakshasa, and Solar…".
  This is a four-part question: four gold passages in one file, each a short "Restoration" paragraph from a different
  monster's entry. Fixed-size chunks of 200 tokens are big enough to hold each one. Whether all four are found depends on the
  embedding model. Expect step 8 to show some **found (rank N)** and some **not found**, a natural lead into Chapter 2
  (one question proves nothing) and Chapter 8 (half found).
- **Contoso, question 1** (g01, "Why did we drop vendor A?"): the minutes of 2026-10-01 §Decision come near the top. Step 5
  shows the reranker would put §Decision first (score ≈ 0.98), a good teaser for Chapter 4.
- Step 6 shows the full prompt, with its size in tokens. Point out that this is *all* the model sees: if a fact isn't in
  these chunks, a grounded model can't use it.
- A typed question skips step 8: there is no answer key for it.

**Talking points**
- What goes in and out of each step: files → text; text → chunks (with positions); chunks → vectors; question → ranked
  chunks; ranked chunks → reordered chunks; chunks + question → prompt; prompt → answer; answer key → found / not found.
- "Which step goes wrong most often?" The rest of the lab answers this: cutting (Ch. 3), searching (Ch. 4–5), and the
  answer itself (Ch. 7).

**A good notebook line:** the eight steps with an in/out for each, e.g. "Chunk: documents in, small pieces with their
place in the file out."

---

## Chapter 2. Was that luck?

**Goal:** explain why one good answer proves nothing, and read Recall@3 and MRR.

**What they see** (`eval`, the starting setup: the dataset's own chunking, meaning search, no rerank, 5 chunks):

| | Questions scored | Recall@3 | Recall@5 | MRR | Embedding |
|---|---|---|---|---|---|
| D&D | 55 (easy 25, medium 30) | 0.44 | 0.45 | 0.58 | bge-small |
| D&D easy / medium | | 0.59 / 0.31 | 0.63 / 0.30 | 0.76 / 0.43 | bge-small |
| Contoso | 17 | 0.58 | 0.61 | 0.69 | nomic-embed-text |

- Perfect student: 1.00 on every score, both datasets. Random guesser: D&D 0.00 / 0.00 / 0.01, Contoso 0.02 / 0.03 / 0.04
  (Recall@3 / Recall@5 / MRR).
- D&D scores 55 questions, not 56: question medium-4 ("How can a player recover spell slots except using Long Rest?") has
  no gold passages in the source dataset, so it can't be scored. The 4 unanswerable questions are for Chapter 7.
- With the hash stand-in (`--embed hash`) the numbers are lower. That's fine: the point is the method.

**Talking points**
- One question is 2% on D&D, 6% on Contoso. A "better" setup that wins by one question on Contoso is noise.
- Easy vs medium: medium questions need several passages (up to 24), so 5 chunks can't hold them all. Recall is
  counted out of at most 5 passages (out of 3 for Recall@3), so a perfect top 5 still scores 1.
- Why test the test: if the perfect student scored 0.8, the answer key or the matching would be broken, and every
  number after it would be meaningless.
- How a match is decided: a chunk counts for a right passage when they share at least half of the smaller of the two.
  One rule for every score.

**A good notebook line:** "Starting point on the rulebook: Recall@3 0.44, MRR 0.58. Perfect = 1, random ≈ 0, so the
test works. 55 questions: a 2% change is one question."

---

## Chapter 3. The answer was cut in half

**Goal:** pick a chunking for a source and explain why, from what the chunks actually look like.

**What they see**

`show 3`: on D&D it's easy-2, the seven layers of a Prismatic Wall. The layers sit in a table, and fixed-size chunks
cut through it. Some layers come at rank 19, and one chunk holds only 34% of the Green layer ("the cut went through it").
On Contoso it's g03 ("What happened with 7f3a-91c2?").

`chunks` (D&D):

| Chunking | Chunks | Median tokens | Gold passages cut in two |
|---|---|---|---|
| fixed-200 | 1,999 | 200 | 144 of 264 (55%) |
| recursive | 3,151 | 98 | 108 (41%) |
| sentence | 6,197 | 59 | 191 (72%) |
| semantic | 6,286 (bge-small) | 39 | 157 (59%) |
| hierarchical | 2,806 sections, searched as 13,228 pieces | 93 | 100 (38%) |

`compare --phase chunking` (semantic search, no rerank, 5 chunks):

| D&D (bge-small) | R@3 | R@5 | MRR | R@~2k tok |
|---|---|---|---|---|
| fixed-200 | 0.44 | 0.45 | 0.58 | 0.51 |
| **recursive** | **0.50** | **0.52** | **0.65** | **0.58** |
| sentence | 0.37 | 0.43 | 0.55 | 0.53 |
| semantic | 0.45 | 0.45 | 0.57 | 0.50 |
| hierarchical | 0.38 | 0.41 | 0.52 | 0.48 |

| Contoso (nomic) | R@3 | R@5 | MRR | R@~2k tok |
|---|---|---|---|---|
| structure | 0.58 | 0.61 | 0.69 | 0.89 |
| naive-500 | 0.45 | 0.47 | 0.49 | 0.73 |
| fixed-200 | 0.36 | 0.50 | 0.46 | 0.75 |
| recursive | 0.52 | 0.60 | 0.61 | 0.74 |
| sentence | 0.40 | 0.42 | 0.54 | 0.86 |
| semantic | 0.59 | 0.64 | 0.67 | 0.87 |
| **hierarchical** | **0.66** | **0.80** | **0.73** | 0.77 |

**Talking points**
- D&D: recursive wins. It cuts at headings and paragraphs, so a spell or monster entry stays together. Sentence chunks
  are too small: a rule needs several sentences, and 72% of the gold passages get cut in two.
- **Hierarchical doesn't win on D&D, and that's worth discussing.** The rulebook's headings include "Page 12" leftovers
  from the PDF, so its "sections" are often a page, not a topic. Handing over a whole page brings noise and pushes the
  right section down. On Contoso, where sections are real (SOP §4, minutes §Decision), hierarchical is the clear winner.
  Same method, different documents, different result: that's the chapter's lesson.
- Contoso: structure-aware beats naive-500 (0.61 vs 0.47, the E1 lesson), and it's best on the same-size view (R@~2k
  0.89). Its chunks are small (median 26 tokens), so five of them carry little text.
- With 17 questions, Contoso differences under ~6% are one question. Read the won/lost lists, not just the averages.
- Contextual chunking needs a model call per chunk, so it's on request: `compare --phase chunking --rows contextual`.
  To show what it writes, cheaply: `chunks --dataset contoso --chunking contextual --limit 10 --show 5`.

**A good notebook line:** "Recursive, because it cuts the rulebook at headings and paragraphs, so a whole rule stays
in one chunk; sentences cut 72% of the right passages in two."
