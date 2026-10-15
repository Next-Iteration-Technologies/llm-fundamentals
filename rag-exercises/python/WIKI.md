# rag.py: how it works

A code wiki for `rag.py`, the Python rung of the RAG lab. The lab `README.md` (one folder up) says what to run and what to expect in each exercise; this page says what the code does when you run it. Flow numbers refer to `../../content-src/rag-day.md`.

One file, ~600 lines, five commands, no framework. Every stage of the RAG loop is a plain function you can read top to bottom, print from, and break on purpose.

## Contents

1. [The five commands](#1-the-five-commands)
2. [Configuration](#2-configuration)
3. [The chunk record](#3-the-chunk-record)
4. [Loading and chunking](#4-loading-and-chunking)
5. [Models: embeddings and chat](#5-models-embeddings-and-chat)
6. [Ingest](#6-ingest)
7. [Retrieve](#7-retrieve)
8. [Ask](#8-ask)
9. [Agent](#9-agent)
10. [Eval](#10-eval)
11. [Files on disk](#11-files-on-disk)
12. [Flags and what they switch](#12-flags-and-what-they-switch)
13. [Things that bite](#13-things-that-bite)
14. [Extending it](#14-extending-it)

## 1. The five commands

```
python rag.py check    # can this laptop reach the chosen chat and embed blocks?
python rag.py ingest   # pack -> chunks -> vectors in Qdrant, plus a JSON sidecar
python rag.py ask      # question -> retrieve -> prompt -> answer, one shot
python rag.py agent    # question -> the model picks tools (search_docs, find_tickets, jira_search) -> answer
python rag.py eval     # golden set -> Recall@5, MRR, nDCG@5, faithfulness
```

The pipeline, with the function that owns each step:

```
                 load_pack / naive_chunks        embed("doc")          QdrantClient.upsert
  contoso-pack ───────────────────────► chunks ─────────────► vectors ─────────────────► .store/qdrant
                 split_children (--children)                                             .store/<collection>.json
                 stable_ids, tag_corpus_ids

  question ──► transform ──► BM25 + Qdrant (access-filtered) ──► RRF ──► parent swap ──► rerank ──► top k
                                                 retrieve()
            ──► live_jira (--live-jira) ──► residency ──► build_prompt ──► chat ──► answer + CITED ──► trace.jsonl
                                                 ask()
```

`ingest` writes; `ask`, `agent` and `eval` only read the store, and `ask` and `agent` append to the trace. `check` touches nothing: it probes both blocks with `models.reachable`, confirms the pack exists, and prints which classes the chat model may read.

At the bottom of the file, `__main__` builds one subparser per command, sets `CHAT` and `EMBED` from `--chat` / `--embed`, sets `WHERE` (`local` under `--dry`, otherwise `models.where(CHAT)`) and `MAY_READ`, then dispatches to `check`, `ingest`, `ask`, `agent` or `evaluate`.

## 2. Configuration

Models come from `providers.toml` (fallback `providers.example.toml`), keys from `.env` or the environment; both are read by `models.py`. One block per model, two defaults at the top:

| Key | Where | Meaning |
|---|---|---|
| `chat = "nexus"` | top of the file | the chat block when no `--chat` / `LLM_CHAT` is given: answers, contextual prefixes, query rewrites, the judge, the agent |
| `embed = "nexus-embed"` | top of the file | the embed block when no `--embed` / `LLM_EMBED` is given |
| `kind` | every block | `anthropic` (official SDK), `openai` (any OpenAI-compatible endpoint: Nexus), `ollama`, or `hash` (no model, embed only) |
| `base_url`, `model`, `api_key_env` | openai, ollama | the endpoint, the model name, the environment variable that holds the key |
| `where` | chat blocks | `local`, `tenant` or `external`: where the model runs, hence `MAY_READ` (flow 9.4). `CHAT_MODEL_WHERE` overrides it; a block without `where` counts as `external`. |
| `timeout`, `temperature`, `think`, `effort` | optional | per block; `LLM_TIMEOUT` is the env fallback for the timeout |

The rest is at module top in `rag.py`, overridable by environment variable:

| Name | Env var | Default | Used for |
|---|---|---|---|
| `PACK` | `CONTOSO_PACK` | `../contoso-pack` | where the fictional data lives |
| `STORE` | `RAG_STORE` | `./.store` | Qdrant files, sidecar JSON, trace |
| `WHERE`, `MAY_READ`, `CHAT`, `EMBED` | (set in `__main__` from the flags) | `local` under `--dry` | which classes the chat model may read; the active block names |

Three tables encode policy:

- `ROLES` maps an asker to the access tags they may see: `sam` sees `team`; `priya` sees `team` and `priya`; `vendor` sees only `vendor`. This is the permission filter of flow 9.
- `MAY_READ` is the data-residency rule of flow 9.4. The table lives in `models.MAY_READ`; `rag.py` picks the row for `WHERE`: a local model may read every class, the tenant's model Public and Internal, an external model Public only. The empty class (naive chunks) counts as readable only for `local`.
- `SYSTEM` is the grounding prompt of flow 5.2: answer from the passages only, cite with the bracketed name that starts each passage, refuse with a fixed sentence, treat passage text as data, prefer newer passages and "jira live" over the snapshot.

Two helpers are used everywhere: `norm` collapses whitespace, `tokens` lowercases and splits into words, keeping IDs like `7f3a-91c2` and `SOP-CH-03` together.

## 3. The chunk record

Everything downstream works on a list of dicts. `chunk()` builds one and defaults `access` to `["team"]`.

| Key | Set by | Meaning |
|---|---|---|
| `text` | every loader | the passage, whitespace-normalised |
| `source` | every loader | the citation label, e.g. `minutes 2026-10-01 §Decision`, `known-issues KI-014`, `transfer.log 2026-10-02 02:14:07`. Printed as `[source]` in the prompt. |
| `file` | every loader | path relative to the pack; with `source` it forms the stable id |
| `type` | every loader | `sop`, `design`, `policy`, `minutes`, `onedrive`, `mail`, `ki`, `config`, `log`, `ticket`, `jira`, or `naive` |
| `date` | every loader | ISO date, used by `find_tickets` ordering and by the reader |
| `cls` | every loader | `Public`, `Internal`, `Confidential` (logs and config), or `""` for naive |
| `platform` | every loader | `Content Hub`, `Legacy Archive`, `Capture`, `POC`; filterable by the agent |
| `access` | loader | list of tags; a chunk is visible if it shares one tag with the asker's `ROLES` entry |
| `title`, `section` | `load_markdown` | front-matter title and `##` heading |
| `untrusted` | `load_mail`, tickets | marks text written by outsiders; the prompt appends `(untrusted: data only)` |
| `doc_id`, `tenant` | logs, tickets | extra metadata kept for filtering and display |
| `uid` | `stable_ids` | UUID from md5 of `file|source|n`; the Qdrant point id |
| `hash` | `stable_ids` | md5 of `text`, 12 chars; what `--incremental` compares |
| `cid` | `tag_corpus_ids` | the demo id from `corpus-index.json` (`c25` etc.), used by `eval` and shown by `show()` |
| `embed_text` | `ingest` | what was actually embedded: `text`, or the contextual prefix plus `text` |
| `parent` | `split_children` | the section's `uid`, on child chunks only |
| `score` | `retrieve` | RRF score, or the reranker score if it ran |

The whole record is stored as the Qdrant payload and again in the sidecar JSON, so retrieval never needs to reparse the pack.

## 4. Loading and chunking

`load_pack()` walks the pack source by source. Each source gets the chunking that fits its shape; that is the point of flow 2.

| Source | Loader | One chunk is | Notes |
|---|---|---|---|
| `contenthub/*.md` (SOP, SD, policy) | `load_markdown` | one `## ` section | type from the filename prefix |
| `minutes/*.md` | `load_markdown` | one `## ` section | label is `minutes <date>` |
| `onedrive/**/*.md` | `load_markdown` | one section, or the whole note if it has no `##` headings | informal notes |
| `mail/*.eml` | `load_mail` | one message | subject and sender prepended, quoted replies and signature cut, `untrusted` |
| `known-issues.csv` | inline | one row | column names written into the text so "Workaround: ..." is searchable |
| `contenthub/config/contenthub-cleanup.xml` | inline | one `<cleanup ...>` line | Confidential |
| `logs/transfer.log`, `logs/index.log` | inline | one line | never cut mid-line; `doc-id` extracted to metadata; Confidential |
| `tickets/tickets.json` | inline | one ticket (`as_text`) | `tenant` kept, `untrusted` |
| `jira/snapshot-2026-10-06.json` | inline | one issue | the deliberately stale nightly snapshot of flow 6.9 |

`load_markdown` details worth knowing:

- `front_matter` reads the `---` block as `key: value` lines. `access` may be a comma list (`team, vendor`). `class`, `platform`, `title`, `id`, `date`/`revised`/`modified` are picked up.
- A section whose heading contains "internal note" is forced to `access: ["team"]` whatever the file says. That is the section-level permission of E6: the vendor sees the 2026-09-24 minutes but not that section.
- HTML comments inside a section are stripped before chunking.
- Empty sections are dropped.

`strip_mail` keeps lines until the first signature marker (`--`, `Best,`, `Regards,`...) and drops every line starting with `>`.

Four transformations run after loading:

- `naive_chunks` (with `--naive`) ignores all of the above: every text file in the pack is flattened and cut every 500 characters, with `access` open to everyone and no class or date. It is the flow 6 baseline the structured loader is measured against.
- `split_children` (with `--children`) cuts markdown-derived sections into roughly 12-word pieces at sentence boundaries, each carrying `parent` = the section's `uid`. Short sections and non-markdown types pass through unchanged. Flow 2.5.
- `stable_ids` gives every chunk a `uid` that is the same on every run (so Qdrant upserts replace rather than duplicate) and a `hash` of its text. A counter disambiguates two chunks with the same `file|source`.
- `tag_corpus_ids` looks each chunk up in `corpus-index.json` by file and a substring match, and stores the demo id as `cid`. Only chunks with a `cid` count in `eval`.

## 5. Models: embeddings and chat

Everything goes through `models.py`, which knows three APIs and translates one neutral message format for each:

```
{"role": "user", "content": "..."}
{"role": "assistant", "content": "...", "tool_calls": [{"id", "name", "arguments"}]}
{"role": "tool", "tool_call_id": "...", "name": "...", "content": "the result"}
```

A tool is `{"name", "description", "parameters": a JSON schema}`. `anthropic_messages`, `openai_messages` and `ollama_messages` turn that into tool_use blocks, OpenAI `tool_calls` or Ollama's shape; `openai_tools` wraps the tool list for the two that take OpenAI's format. It is the same module shape as `../ai-application-architecture/llm_client.py` in the earlier track, without demo and replay.

**Embeddings.** `models.embed(texts, kind, block)` sends batches of 32 to the block's model: `POST /embeddings` for `openai`, `POST /api/embed` for `ollama`, `hash_embed` for `hash`. `nomic-embed-text` wants a task prefix, so when the model name starts with `nomic`, queries get `search_query: ` and documents `search_document: `; `kind` is `"query"` or `"doc"`. HyDE passes `"doc"` for its drafted answer on purpose, since it is searching document-to-document. Anthropic has no embeddings endpoint, so there is no anthropic embed kind.

`pick_embedder(dry)` in `rag.py` runs once per ingest. It probes the embed block (`models.reachable`) and returns its id, `<kind>:<model>`, for the sidecar. If the block is unreachable it exits with the advice, unless `--dry`, where it returns `"hash:hash"`.

`hash_embed` is that fallback: feature hashing of words and character trigrams into 768 dimensions, L2-normalised. It is lexical, not semantic, so under `--dry` semantic search behaves like a fuzzy keyword search. The embedder id is stored in the sidecar; `retrieve` refuses to run if the current `--embed` block would produce a different id, so a store is always queried with the model that built it.

**Chat.** `models.chat(messages, system, tools, block)` is one request; `models.text(reply)` strips `<think>...</think>` blocks so thinking models do not leak reasoning into answers or citations. For `anthropic` it is `client.beta.messages.create` with `output_config.effort` (default `low`), the server-side fallback for refusals, and `system` as the top-level field; for `openai` and `ollama` the system prompt becomes the first message. `rag.py`'s `chat(prompt, system)` wraps a single user turn. Five callers use it with their own system prompt: the answer, contextual prefixes, query rewrites, the faithfulness judge, and the agent's fallback when it runs out of steps.

`models.reachable(block, role)` backs `python rag.py check`: a model lookup for Anthropic, one tiny real call for an OpenAI-compatible gateway (a gateway may not serve `/models`), the tag list for Ollama.

## 6. Ingest

`ingest(a)` in order:

1. **Load**: `naive_chunks` or `load_pack`, then `tag_corpus_ids`, then `stable_ids`.
2. **Children**: with `--children`, keep the sections as `parents` (a dict by `uid`) and replace `chunks` with the children, re-running `stable_ids` so children get their own ids.
3. **Pick the embedder** and the collection name: `contoso`, `contoso_naive` or `contoso_children`. The three live side by side in the same Qdrant folder.
4. **Diff** (with `--incremental`): read the previous sidecar. If it was built with a different embedder, fall back to a full rebuild. Otherwise `todo` = chunks whose `uid` is new or whose `hash` changed; `gone` = ids in the old sidecar that no longer exist. Unchanged chunks keep their old `embed_text`, so a contextual prefix is not regenerated.
5. **Contextual prefix** (with `--contextual`, not `--dry`): for each `todo` chunk except logs and naive slices, ask the chat model for one sentence saying where the chunk sits and what it is about, and prepend it to `embed_text`. Flow 2.6. The chunk is only sent if its `cls` is in `MAY_READ`: the residency rule applies at ingest too.
6. **Write to Qdrant** (local path mode, no server): on a full rebuild drop the collection; create it with cosine distance and the vector width the embedder returned; upsert `todo` with the full chunk as payload; delete `gone`.
7. **Write the sidecar** `.store/<collection>.json` with `embedder`, every chunk (including `embed_text`), and `parents`.
8. **Print** a per-type count, how many chunks carry a demo id, and for incremental runs the added/changed/removed/unchanged counts.

The sidecar is not a cache of Qdrant; it is the primary store for BM25, `find_tickets`, parent lookup and `eval`. Qdrant only serves the vector side.

## 7. Retrieve

`retrieve(question, a, asker)` is the heart of flows 4 and 9. In order:

1. **Load the sidecar** for the collection the flags name. `pos` maps `uid` to list index.
2. **Permission filter before ranking.** `ok` is the list of chunk indices whose `access` intersects `ROLES[asker]`. BM25 is built over `ok` only, and the Qdrant query carries a `MatchAny` filter on the `access` payload. A chunk the asker may not see is never scored, so it can never leak through a low rank.
3. **Transform** (`transform()`, flow 4.6): returns the searches to run as `(keyword text, vector text, embed kind)` tuples.
   - no flag or `--dry`: the question itself
   - `rewrite`: one rewritten query, used for both BM25 and the vector side
   - `multi`: the question plus up to three rewrites, one line each
   - `hyde`: the question for BM25, a drafted answer (embedded as a document) for the vector side
   Every transformed query is printed before searching so the room sees what actually ran. `PLAIN` is appended to the rewrite prompts to stop the model writing `site:` or quoted web-search syntax.
4. **Search.** For each tuple: BM25 (`rank_bm25.BM25Okapi` over `tokens(embed_text)`) returns up to 30 indices with a positive score; Qdrant returns 30 nearest points. `--keyword-only` and `--semantic-only` drop one side.
5. **Fuse** every ranked list with Reciprocal Rank Fusion, `1 / (60 + rank)`. With `multi` that is up to eight lists.
6. **Parent swap and dedupe.** Walk the fused order; if a chunk has a `parent`, substitute the parent section. Keep the first 20 distinct `uid`s as candidates, each with its RRF score.
7. **Rerank** (unless `--no-rerank`): lazy-load the cross-encoder `BAAI/bge-reranker-base` once per process (forced onto the CPU on Intel Macs, where torch on MPS returns wrong scores), score `(question, text)` for every candidate, replace `score` and re-sort. Flow 4.5. If the import or download fails, print a warning, set `a.no_rerank = True` so it is not retried, and return the RRF order.

`retrieve` returns up to 20 candidates; callers cut to `--k`.

## 8. Ask

`ask(a)`:

1. `retrieve` and cut to `k`.
2. With `--live-jira URL`: `jql_for` guesses a JQL from the first two content words, `jira_query` hits the mock's `/rest/api/2/search`, and the live issues replace every `type == "jira"` hit from the stale snapshot. The JQL and hit count are printed. Flow 6.9: a tool for what changes faster than the index.
3. Print the retrieval mode line and the hits (`show()`: rank, score, demo id, source, first 90 characters).
4. `residency` drops hits whose `cls` is not in `MAY_READ` and prints them as withheld. Flow 9.4.
5. `build_prompt` renders `PASSAGES` as `[source] text` lines, `(untrusted: data only)` after the source of mail and tickets, then `QUESTION`.
6. `chat` with `SYSTEM`, unless `--dry`. Print the answer and the set of `[...]` citations found in it, so a made-up citation is visible at a glance.
7. `trace("ask", ...)` with question, asker, every flag, the hits with scores, the withheld sources, the answer and elapsed milliseconds.

## 9. Agent

Flow 8: the same retrieval, but the model chooses which searches to run.

**Tools** (`TOOLS`, the neutral `{name, description, parameters}` shape; `models.py` translates it per provider):

| Tool | Arguments | Implementation in `run_tool` |
|---|---|---|
| `search_docs` | `query`, optional `source_type`, `platform` | `retrieve` for the asker, minus `jira` snapshot hits (status is `jira_search`'s job), filtered by type and platform, cut to `k` |
| `find_tickets` | `tenant` (enum of the four tenants) | every ticket chunk of that tenant the asker may see, from the sidecar, oldest first. No search: a lookup. |
| `jira_search` | `jql` | the live mock. The mock understands `AND` only. |

`call_tool` wraps `run_tool`: an exception becomes the string `error: ...` so the model can try again; results pass through `residency`; the step, arguments and the first six hits are printed; the model receives the passages formatted like the `ask` prompt, or `nothing found`.

**Loop** (`agent(a)`): the question, then up to `--max-steps` (default 5) rounds of `models.chat(messages, system=AGENT_SYSTEM, tools=TOOLS)`. The reply is appended as an assistant message with its `tool_calls`; every call is executed and appended as a `tool` message with the call's id (Anthropic needs the id to match, OpenAI too; Ollama ignores it). A reply with no tool calls is the answer. If the steps run out, `chat` is called once with everything found so far (`used`) and the plain `SYSTEM` prompt.

**`--dry`** has no model to choose, so it runs the fixed workflow instead: `search_docs` with the question, `find_tickets` for every tenant named in the question, `jira_search` with `jql_for`. Comparing its calls with the model's is exercise E7.

The trace row for `agent` records every call with its arguments, every source used, and the answer.

## 10. Eval

`evaluate(a)` reads `golden-set.json` and, per question:

- retrieves as `sam`, or as the question's own `asker` with `--as-in-set`
- `cids` = the demo ids of the hits in rank order, deduplicated
- `expected_ids` is a dict `cid -> grade` (3 = the answer, 2 = needed, 1 = background)
- the three retrieval metrics are computed only when `expected_ids` is non-empty (refusal questions have none)
- **R@5** = relevant ids in the top 5 / relevant ids
- **MRR** = 1 / rank of the first relevant id, 0 if none
- **nDCG@5** = DCG over the top 5 with the grades as gain, divided by the ideal DCG
- without `--dry`: `residency`, `chat` the answer, then `judge_faithfulness` asks the chat model to split the answer into claims and mark each SUPPORTED or UNSUPPORTED; the score is the supported share. Questions with `should_refuse` also get `refused`, a regex over the answer for the refusal phrases.
- with `--export runs.jsonl`: one line per question with `question`, `answer`, `contexts`, `ground_truth` (the `must_say` list joined), the shape RAGAS expects.

One line per question is printed, then the mean of each metric over the questions that have it. `ms` is retrieval only under `--dry`, retrieval plus answer plus judge otherwise.

The judge uses the same `--chat` block as the answer. For the room that is fine; for a real eval the judge should be a stronger model than the one being judged.

## 11. Files on disk

All under `STORE` (`python/.store/` by default), created on first use:

| Path | Written by | Read by |
|---|---|---|
| `qdrant/` | `ingest` | `retrieve` (vector side). Qdrant local mode: one process at a time, so do not run two commands concurrently. |
| `contoso.json`, `contoso_naive.json`, `contoso_children.json` | `ingest` | `retrieve`, `find_tickets`, incremental diff. Holds `embedder`, `chunks`, `parents`. |
| `trace.jsonl` | `ask`, `agent` | you, and a future eval. One JSON object per line: `ts`, `kind`, `chat` (the block name), `where`, question, asker, flags, hits, withheld, answer, `ms`. The backbone's log box from flow 1.8. |

Deleting `.store/` is a clean reset. The reranker model is cached by Hugging Face under `~/.cache/huggingface`, not here.

## 12. Flags and what they switch

| Flag | Commands | Effect |
|---|---|---|
| `--chat NAME` / `--embed NAME` | all | the blocks from `providers.toml` (default: its `chat` / `embed`, or `LLM_CHAT` / `LLM_EMBED`) |
| `--dry` | all but check | no chat calls; hashed embedding if the embed block is unreachable; nothing withheld; `agent` runs the fixed workflow; `transform` is skipped |
| `--naive` | all but check | collection `contoso_naive`, 500-character slices (ingest) or query it (others) |
| `--children` | all but check | collection `contoso_children`, sentence-sized children with parent swap on retrieval |
| `--contextual` | ingest | one-sentence LLM prefix per chunk in `embed_text` |
| `--incremental` | ingest | diff against the previous sidecar by `uid` and `hash` |
| `--no-rerank` | ask, agent, eval | keep the RRF order, skip the cross-encoder |
| `--semantic-only` / `--keyword-only` | ask, agent, eval | drop BM25 or drop Qdrant |
| `--k N` | ask, agent, eval | passages handed to the model (default 5) |
| `--transform rewrite\|multi\|hyde` | ask, eval | query transformation before search |
| `--as sam\|priya\|vendor` | ask, agent | the asker, hence the permission filter |
| `--live-jira URL` | ask | replace snapshot hits with the live mock's issues |
| `--jira URL` | agent | the live Jira behind `jira_search` (default `http://localhost:8080`) |
| `--max-steps N` | agent | tool-calling rounds before answering from what was found (default 5) |
| `--as-in-set` | eval | ask each golden question as its own asker |
| `--export FILE` | eval | RAGAS-shaped JSONL |

`--naive` and `--children` select the collection, so the flag you ingested with is the flag you must ask with.

## 13. Things that bite

- **Collection mismatch.** `ask --children` on a store ingested without `--children` fails with a missing `contoso_children.json`. Ingest once per variant you want to query.
- **Embedder mismatch.** The sidecar records which embed block built the index. `retrieve` exits if the current `--embed` block differs (query and document vectors would not match); pass the matching block or re-ingest. A store built with a hashed embedding is queried with `hash` automatically, so `--dry` indexes work offline.
- **BM25 is rebuilt on every retrieve** from the sidecar. Fine for a few hundred chunks; the first thing to cache if the corpus grows.
- **The reranker download** (~1.1 GB, pulls in torch) happens on the first `ask` without `--no-rerank`. If it fails, retrieval still returns the RRF order and says so.
- **Naive chunks have no class.** `MAY_READ` includes `""` only for `local`, so with a `tenant` or `external` chat block (the default `nexus` block is `tenant`) every naive chunk is withheld. That is correct by the policy (unknown class is not Public) and surprising in the room. The same applies to logs and config (Confidential) in E2 and E9.
- **`jql_for` is deliberately dumb.** Two content words, no understanding. It exists so the fixed workflow has something to compare against the agent's own JQL.
- **The Jira mock ignores `OR`.** A model that writes `OR` gets every issue back; this shows up in E7.
- **`tag_corpus_ids` is a substring match** on `lab_match` within the chunk text. A chunk that is split differently (naive, children) may lose its `cid` and therefore disappear from the eval metrics. Children inherit their parent's `cid` through `dict(c, ...)`, and the parent swap returns the section, so `--children` keeps eval working.
- **`a.no_rerank` is mutated** on rerank failure, so a single `eval` run does not retry the download seventeen times.
- **`chat` has no retry.** Set `temperature` in a block to pin it; otherwise the provider's default applies and answers vary between runs. A gateway that does not expose tool calling makes `agent` fail on the first round; `ask` and `eval` still work.

## 14. Extending it

- **A new source.** Add a loop in `load_pack` that produces `chunk(text, source=..., file=..., type=..., date=..., cls=..., platform=...)`. Choose the chunk boundary that matches the source's shape. Add the new `type` to the `source_type` enum in `TOOLS` if the agent should filter on it. Add it to `split_children`'s type list only if it is prose.
- **A new role.** Add an entry to `ROLES` and put the tag in the pack's `access:` front matter. Nothing else changes: the filter and the Qdrant query read `ROLES`.
- **A new tool.** Add a schema to `TOOLS` and a branch in `run_tool` returning a list of chunk-like dicts with at least `source`, `text`, `cls`. `call_tool` handles errors, residency and formatting.
- **A different policy.** Edit `MAY_READ` in `models.py`. `rag.py` consults it in two places: contextual prefixes at ingest, and `residency` before every prompt (`check` only prints it).
- **A different embedding model.** Add or edit an embed block in `providers.toml`. The `search_query:` / `search_document:` prefixes are applied only to models whose name starts with `nomic`. Re-ingest: the vector width is read from the first batch.
- **A different provider.** Add a `kind` branch in `models.chat` and, if it embeds, in `models.embed`, plus a message translator if its wire format is new. `rag.py` does not change.
- **A different store.** Everything Qdrant-specific is in `ingest` (create, upsert, delete) and one `query_points` call in `retrieve`. The payload filter on `access` must be reproduced in whatever replaces it, or the permission filter degrades to post-filtering.
- **Metrics.** `evaluate` builds a `row` dict per question; add a key and it appears in the per-question line. Add the key to the final loop for a mean.
