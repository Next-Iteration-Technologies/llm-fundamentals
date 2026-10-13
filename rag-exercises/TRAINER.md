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
