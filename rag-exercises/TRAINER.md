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
