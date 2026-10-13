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
| semantic | 6,286 | 39 | 157 (59%) |
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
| semantic | 0.37 | 0.54 | 0.54 | 0.84 |
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
- Semantic chunking sees topic changes with bge-small, whatever model searches, so its chunks stay the same in Chapter 5.
- Contextual chunking needs a model call per chunk, so it's on request: `compare --phase chunking --rows contextual`.
  To show what it writes, cheaply: `chunks --dataset contoso --chunking contextual --limit 10 --show 5`.

**A good notebook line:** "Recursive, because it cuts the rulebook at headings and paragraphs, so a whole rule stays
in one chunk; sentences cut 72% of the right passages in two."

---

## Chapter 4. It was there, just too far down

**Goal:** say when keyword search beats semantic search, and what rerank can and can't fix.

**What they see** (`compare --phase retrieval`; rows grouped by chunking: the two best from Chapter 3, plus fixed-200)

| D&D (bge-small) | R@3 | R@5 | MRR | R@50 |
|---|---|---|---|---|
| recursive, keyword | 0.35 | 0.38 | 0.51 | 0.54 |
| recursive, **semantic** | **0.50** | **0.52** | **0.65** | 0.65 |
| recursive, hybrid | 0.44 | 0.44 | 0.64 | 0.68 |
| recursive, hybrid + rerank | 0.47 | 0.48 | 0.63 | 0.68 |
| fixed-200, semantic | 0.44 | 0.45 | 0.58 | 0.63 |
| fixed-200, hybrid | 0.44 | 0.43 | 0.62 | 0.65 |
| fixed-200, hybrid + rerank | 0.45 | 0.48 | 0.64 | 0.65 |

| Contoso (nomic) | R@3 | R@5 | MRR | R@50 |
|---|---|---|---|---|
| hierarchical, keyword | 0.41 | 0.60 | 0.56 | 0.94 |
| hierarchical, **semantic** | **0.66** | **0.80** | **0.73** | 1.00 |
| hierarchical, hybrid | 0.58 | 0.76 | 0.70 | 1.00 |
| hierarchical, hybrid + rerank | 0.54 | 0.71 | 0.68 | 1.00 |
| hierarchical, hybrid + rerank + rewrite (llama3.2:3b) | 0.54 | 0.73 | 0.68 | 1.00 |
| structure, semantic | 0.58 | 0.61 | 0.69 | 0.89 |
| structure, hybrid | 0.55 | 0.54 | 0.65 | 0.90 |
| structure, hybrid + rerank | 0.60 | 0.60 | **0.84** | 0.90 |
| fixed-200, hybrid + rerank | 0.47 | 0.61 | 0.64 | 1.00 |

**Talking points**
- **R@50 is the same with and without rerank, in every group.** Rerank only reorders the 50 it is given; it can't find a
  chunk the first search missed. That's the answer to the chapter's question.
- **Hybrid did not beat semantic on either dataset here.** The questions are written in plain words, and a good
  embedding model already handles them; BM25 adds noise that equal-weight RRF lets in. Hybrid still has a higher R@50
  on D&D (0.68 vs 0.65): it finds more, but ranks it worse. Keyword search alone lost on average in every group here.
  Where it helps is exact strings: use E2's `ask "transfer TIMEOUT 7f3a-91c2" --keyword-only` to show it, and ask the
  room which questions in their own golden set are like that. (Checked: on g08 "KI-047" keyword and semantic both find
  the row; keyword ranks it lower.)
- **Rerank helps small chunks and hurts long ones.** On Contoso's structure-aware chunks, rerank lifts MRR from 0.65 to
  0.84 (E3: "Why did we drop vendor A?" goes to rank 1). On hierarchical sections (up to 800 tokens) it lowers the
  scores: the reranker reads only the first ~512 tokens of each, so a section whose answer sits at the end looks
  irrelevant. "What rerank can't fix" includes this.
- Rewriting the question with a 3B local model barely moved anything (one question). With Sonnet it's worth trying.
- Recall@3 can be higher than Recall@5 for a row: recall counts out of at most 3 (or 5) passages, so a question with 4
  gold passages and 2 in the top 3 scores 0.67 on R@3 but 0.50 on R@5.

**A good notebook line:** "Semantic on recursive chunks. Rerank didn't change R@50 (it only reorders) and helped only
where chunks were small. Keyword search is for exact strings like IDs and error codes."

---

## Chapter 5. Same meaning, different words

**Goal:** explain what an embedding model changes, and when a better one is worth it.

**What they see** (`compare --phase embedding`, keeping the best setup from Chapter 4)

| | hash | nomic-embed-text (Ollama) | bge-small |
|---|---|---|---|
| D&D (recursive, semantic): R@5 / MRR | 0.27 / 0.34 | **0.53 / 0.66** | 0.52 / 0.65 |
| Contoso (hierarchical, semantic): R@5 / MRR | 0.59 / 0.62 | **0.80 / 0.73** | 0.73 / 0.77 |

Time on the build machine (Intel Mac, CPU only) to embed the D&D recursive chunks (3,151) once: nomic about 33 min,
bge-small about 4 min, hash a few seconds. After that, both are cached and a rerun takes seconds.

**Talking points**
- A real model doubles D&D's scores over the word-matching stand-in. That's the size of "meaning, not words".
- nomic and bge-small are a tie on D&D (one question apart), and nomic leads on Contoso by about one question. bge-small
  is about 8 times faster on this laptop. "When is a better one worth it?" The numbers here say: going from no model to
  any model is worth it; between these two, speed decides.
- The spec's follow-up: re-run Chapter 3 with the chosen model to check the winner still wins. On D&D the chunking
  comparison was already run with bge-small.

**A good notebook line:** "bge-small: as good as nomic on the rulebook, much faster here. hash is half as good, so a
real model matters; which real model matters less."

---

## Chapter 6. Two documents disagree (Contoso)

**Goal:** explain why a RAG answer can be correct and still out of date.

**What they see** (`compare --phase filters --dataset contoso`, on the best setup so far: hierarchical, semantic, nomic)

| Row | R@3 | R@5 | MRR | R@50 | Log question (g10): which comes first? |
|---|---|---|---|---|---|
| no-filters | 0.66 | 0.80 | 0.73 | 1.00 | the 2025 SOP (14 days) rank 1, the 2026 config (30 days) rank 4 |
| asker | 0.69 | 0.77 | 0.67 | 0.94 | SOP first |
| platform | 0.66 | 0.80 | 0.74 | 1.00 | SOP first |
| after-2026 | 0.66 | 0.77 | 0.70 | 0.97 | config only: the SOP is filtered out |
| prefer-newest | 0.34 | 0.80 | 0.44 | 1.00 | **config rank 2, SOP rank 5** |
| asker + prefer-newest | 0.40 | 0.77 | 0.39 | 0.94 | config first |

**Talking points**
- Without help, the out-of-date SOP comes first. An answer from it ("14 days") is grounded, cited, and wrong today.
- "prefer-newest" sends the same five chunks, newest first. Recall@5 doesn't move, but MRR drops (0.73 → 0.44): for
  most questions newest-first is *not* most-relevant-first. Freshness is a trade-off, not a free fix. That's why rag.py
  also says it in the prompt ("If passages disagree, prefer the newest") and leaves the call to the model.
- "after-2026" removes the SOP entirely. But the golden set's must-say for g10 is "14 days in SOP-OPS-02, 30 days in the
  config, **the two disagree**". Filtering the old one out hides the disagreement, and the best answer names it.
- "asker" lowers R@50 (0.94): Sam, Priya and the vendor see different things (E6). The score drops because the vendor
  can't see some passages; that's correct behaviour, not a worse search.
- Where in their own documents could this happen: SOPs vs config, old wiki pages, superseded policies.

**A good notebook line:** "The SOP says 14 days and is found first, but the 2026 config says 30. The answer is grounded
and still out of date. Newest-first fixes this question but costs ranking everywhere else."
