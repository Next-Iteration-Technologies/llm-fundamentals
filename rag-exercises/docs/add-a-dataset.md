# Add a dataset

The lab works on any set of documents that has questions with known answers. One small settings file plugs it in.

## 1. Put the files somewhere

```
datasets/my-docs/docs/        the documents (markdown, text, csv, eml, json, …: anything readable as text)
datasets/my-docs/qa/          the questions
```

## 2. Write the questions

Pick the simplest format that holds what you know. All three end the same way: every right passage becomes a file
and a start and end position in it, and every score uses those positions.

**quotes** (the easiest to write by hand): quote the right passage, the lab finds it in the file.

```json
[
  {"id": "q1", "question": "How long do we keep backups?",
   "passages": [{"file": "policies/backup.md", "quote": "Backups are kept for 35 days"}],
   "must_say": ["35 days"]},
  {"id": "q2", "question": "Who approves a restore?",
   "passages": [{"file": "sops/restore.md"}]},
  {"id": "q3", "question": "What is our office's wifi password?", "refuse": true}
]
```

- `quote`: the text of the right passage, as in the file. Line breaks and extra spaces don't matter.
- No `quote`, only `file`: the whole file is the right passage (scores get coarse).
- `"refuse": true` and no passages: the answer isn't in the documents; Chapter 7 checks it says so.
- Optional: `level` (e.g. easy, medium: reported apart), `must_say` (facts the answer must contain), `answer`.

**positions** (the D&D format, from the datapizza dataset builder): `start_char` / `end_char` in the file's text.

```json
[{"id": 0, "question": "…", "answer": "…",
  "passages": [{"document_path": "chapter1.md", "start_char": 8877, "end_char": 9087, "content": "…"}]}]
```

The file name (`easy.json`, `medium.json`) becomes the level. `content` is checked against the file.

**contoso**: Contoso's own `golden-set.json` (graded chunk ids, resolved through `corpus-index.json`). Use it only if
your data already looks like that.

## 3. Write the settings file

`datasets/my-docs.toml`:

```toml
title = "My documents"
docs = "my-docs/docs"                  # relative to this file
include = ["**/*.md", "**/*.txt"]      # which files are documents
questions = ["my-docs/qa/questions.json"]
format = "quotes"                      # quotes | positions | contoso
chunking = "recursive"                 # the starting point (Chapter 2): fixed-200, recursive, sentence, …
embed = "bge-small"                    # optional: used when neither --embed nor LLM_EMBED is set
refusal = "I don't find this in our documents"   # what the model should say when the answer isn't there
class = "Internal"                     # optional: data class of every document (E9); default Public
```

## 4. Check it

```
python rag.py check-data --dataset my-docs
```

Every gold passage must be "found at its position". A quote that can't be found is listed with its question id:
fix the quote (copy it from the file), then run it again.

Then the whole lab works with `export RAG_DATASET=my-docs`. (Settings files live in `datasets/`; `RAG_DATASETS=/other/folder`
points the lab at another folder of them.) Chapter 6 (dates and platforms) needs that metadata and
uses Contoso.

## What counts as a match

A chunk counts as the right one for a gold passage when they share at least half of the smaller of the two. So keep
gold passages to what really answers the question: a whole chapter as the "passage" makes every chunk in it a match.
