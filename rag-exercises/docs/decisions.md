# Decisions

One line per choice, and why. The spec's own decisions are in `.plan/SPEC.md` ("Decisions taken"); these are the ones
made while building.

## Data

- **Gold is always file + start + end.** D&D gives positions; Contoso's quotes (`lab_match`) are found in the file, whitespace ignored. Why: one scoring rule for every dataset (SPEC).
- **Contoso quotes that name a record are widened to the record**: a known-issues row to its CSV line, a ticket or Jira quote ("TCK-4471") to its JSON object. Why: the quote is only the record's id; the passage is the record.
- **Known-issues quotes are written as rag.py rewrites the row ("Known issue KI-014.")**, so they're found by the row's id. Why: `corpus-index.json` was written against the chunk text, not the file.
- **D&D question medium-4 has no gold passages in the source data: answerable, but not scored.** D&D retrieval scores are over 55 questions. Why: no answer key, no score; dropping it silently would hide it.
- **Four unanswerable D&D questions added** (`qa/unanswerable.json`: Artificer, Warforged, the game's history, book prices), checked absent with grep. Why: Chapter 7 needs refusal questions on both datasets.
- **Contoso g14 is a refusal question that also has background gold (c05).** It's scored for retrieval and checked for refusal. Why: that's what the golden set says; the old eval also scores 17 questions.

## Scores

- **A chunk matches a gold passage when they share at least half of the smaller of the two.** Why: SPEC, one rule.
- **Recall@k counts out of min(k, number of gold passages), capped at 1.** Why: a perfect top k must score 1.00 even for questions with 24 gold passages (SPEC check: perfect = 1.00 everywhere). Side effect: R@3 can be higher than R@5 for one question.
- **"All found" = the top 5 hold min(5, n) gold passages.** Same reason.
- **nDCG@5 counts each gold passage once, at its first rank, with its grade (D&D: all 1), capped at 1.** Why: one chunk holding two passages would otherwise score above 1.
- **The fair-size view (R@~2k) takes chunks in rank order up to about 2,000 tokens.** The perfect retriever scores 0.96 on it for D&D, because some questions' gold adds up to more than 2,000 tokens. Why: SPEC asks for it to compare chunk sizes fairly.
- **Tokens are counted as words plus punctuation marks** (`\w+|[^\w\s]`). Why: close enough to compare sizes, and no tokenizer download. It undercounts Claude's tokens by about 1.6× (the D&D full-context prompt: ~398k by this count, 627,656 by Claude's), so the `final` preview also shows characters / 2.4, which came within 0.4%.
- **"Best" means highest Recall@5, then MRR, then nDCG@5**, among real setups (not perfect or random). Why: Recall@5 is what's sent to the model.
- **won / lost compares Recall@5 per question with the row above, within the same chunking.** Why: SPEC "which questions changed".

## Setups

- **The starting setup: the dataset's chunking (Contoso structure, D&D fixed-200), semantic search, no rerank, 5 chunks.** Why: SPEC phase 2 keeps "retrieval (semantic)"; each later chapter turns one thing on.
- **The walkthrough always shows step 5 (rerank), and says the prompt keeps the search order when rerank is off.** Why: all eight steps visible in Chapter 1, without changing the setup.
- **200-token fixed size** (SPEC), **recursive up to 300 tokens, sentence = 4 sentences, semantic cuts at the lowest quarter of neighbour similarities (max 300 tokens), hierarchical sections up to 800 tokens searched as 2-sentence children.**
- **Semantic chunking finds topic changes with one fixed model: bge-small (hash if it's missing).** Why: with the run's own model, Chapter 5 would change the chunks and the search model at once, and nomic would need hours for the ~17,000 D&D sentences.
- **Search keeps the top 50; rerank reorders those 50; hybrid fuses the top 100 of each list with RRF (k = 60).** Why: SPEC "rerank the top 50"; R@50 is then the same with and without rerank.
- **No access filter in Chapters 2–5; the asker filter is a phase 5 option.** Why: generic chunkings don't know section-level permissions; comparing chunkings with different visibility wouldn't be fair.
- **Chunkings that don't know a file's structure take its most restrictive data class** (and the narrowest access). Why: never send a model more than it may read.
- **Rows that need a model call per chunk or per question (contextual, query rewrites) only run when named with `--rows`.** Why: a student's `compare` must not make 3,000 model calls by surprise.
- **"Prefer the newest" keeps the same top-k chunks and sorts them newest first.** Why: a date bonus on every candidate (tried first) pulled recent but unrelated mail and log lines into the top 5 (Contoso R@3 0.66 → 0.26). Sorting only what's sent keeps Recall@5, puts the 2026 config above the 2025 SOP, and makes the cost visible: MRR drops.
- **D&D's recorded results use bge-small; Contoso's use nomic-embed-text.** Why: nomic in Ollama embeds ~2 chunks/s on the Intel Mac (16 min for the 1,999 fixed-200 chunks); bge-small does 7–30/s. This is the spec's own "bge-small for D&D".

## Models

- **Model choice: `--chat`, else `LLM_CHAT`, else `chat` in providers.toml. Judge: `--judge`, else `LLM_JUDGE`, else `judge` in providers.toml, else the chat model.** Why: always a setting (SPEC); `judge` is a new top-level key.
- **`LAB_NO_API=1` blocks any `anthropic` or `openai` block in `models.py`, before a call.** Why: the overnight build must not spend money, even by mistake.
- **Ollama gets `num_ctx` (block key, default 8192) and `truncate: false`.** Verified on Ollama 0.32.4: without it, a 1,229-token prompt was silently cut to 258 tokens; with it, Ollama returns 400 with the real counts. A prompt over twice the window (a whole corpus) is stopped before sending.
- **Every model reply, embedding and rerank score is cached in `python/.store/lab-cache.sqlite`.** Why: same input, same output, and reruns are free.

## Setup

- **`setup.py` asks two things: the chat provider (which also sets the judge), then the embedding.** nexus → nexus answers and judges; anthropic → sonnet answers, opus judges; ollama → ollama for both; offline → `--dry`. Why: the trainer runs Claude + Ollama, students run Nexus with whatever embedding their laptop allows; one script serves both.
- **Only embeddings this laptop can use are offered**, with the reason for the others: `ollama-embed` if Ollama answers (pull offered), `nexus-embed` with the nexus provider and a model name, `bge-small` and `hash` always. The first usable one is the default. Why: a choice that can't work shouldn't be offered.
- **Students pick the Nexus chat and embedding models from the gateway's list** (`GET /models`, with the key asked first; names with "embed" go in the embedding list, plus "0. none"). A gateway that doesn't list them gets typed names. With a Nexus embedding model, `nexus-embed` is the default embedding. Why: students don't know the gateway's model names, and its own embedding model needs no extra download.
- **Ollama is optional.** `setup.py` never installs it; nothing in the lab needs it. Why: it is blocked on some company laptops.
- **`compare` leaves out a row whose embed block doesn't answer, and says why; named in `--rows`, it runs and fails loudly.** Chapter 5's rows: hash, bge-small, ollama-embed, bge-base, nexus-embed. Why: every laptop gets a chapter 5 with at least one "better model" (`bge-base` needs only Hugging Face).
- **An older `providers.toml` gets the example's missing blocks and the `judge` line appended; nothing is rewritten.** Why: `setup.py` used to copy the example only when the file was missing, so old files lacked `bge-small`, `sonnet`, `opus` and dnd-srd silently fell back to hash.
- **Hugging Face is the one download every setup needs** (bge-small; bge-base and the reranker later); `setup.py` fetches bge-small up front and prints the `HF_ENDPOINT` / cache-copy hint if it fails.
- **`rag.py check` also checks the judge block** when it differs from the chat block. Why: chapters 7 and 9 need it, and a missing key there shows up late.

## Answers and the final comparison

- **Must-say counts a phrase as said when 70% of its content words are in the answer.** Why: the golden set's phrases are descriptions ("no on-premise parser before next year"), not exact wording.
- **Refusal = the answer matches "don't find / not in the passages / cannot answer / no information …".**
- **In `final`, faithfulness is judged against the gold passages for both rows.** Why: the judge can't reread the whole corpus for every claim; one yardstick keeps the two rows comparable.
- **The full-context corpus for Contoso is rag.py's structure-aware text; for D&D it's the files as they are.** Why: Contoso's data classes live in its loaders, and E9's rule must hold in both rows.
- **`--limit N` takes N questions spread evenly over the golden set**, first and last included. Why: a small sample still has easy, medium and unanswerable questions.
- **`final` asks before sending (y/N), also when the answer comes from a pipe; `--yes` skips it.** Why: the token preview is the point; a script must say yes on purpose.
- **Chunkings ranked best-1/best-2 for the retrieval phase come from the saved chunking runs, by Recall@5 then MRR.** Why: "the best two chunkings from phase 2" (SPEC) without typing names.
