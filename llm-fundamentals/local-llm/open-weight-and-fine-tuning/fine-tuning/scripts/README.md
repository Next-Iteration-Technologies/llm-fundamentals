# Dataset scripts

Four helpers you can run on your own documents after you have tried the
[Qwen demo](../qwen-finetune-demo/README.md) on the bundled policy set.

## Prerequisites

1. Install [uv](https://docs.astral.sh/uv/) and Python 3.11+.
2. From the `open-weight-and-fine-tuning/` folder, run `uv sync`.
3. `01_prepare_enterprise_docs.py` and `02_prepare_code_repo.py` do not need
   Ollama. They only read files and write JSONL.
4. `04_evaluate_model.py` does need a running Ollama server and a pulled
   model. Check with `curl http://localhost:11434/api/version`, then
   `ollama pull qwen3.5:0.8b` (or the tuned tag you created).
5. Optional: `export OLLAMA_HOST=http://HOST:11434` for a remote server.

Every script accepts `--dry-run`.

From the `open-weight-and-fine-tuning/` folder:

```bash
uv run python fine-tuning/scripts/01_prepare_enterprise_docs.py --dry-run
uv run python fine-tuning/scripts/02_prepare_code_repo.py --dry-run
uv run python fine-tuning/scripts/03_validate_dataset.py --dry-run
uv run python fine-tuning/scripts/04_evaluate_model.py --dry-run
```

## 01_prepare_enterprise_docs.py

Reads Markdown or text files from `fine-tuning/sample_raw_docs/` (or a
folder you pass) and writes a draft SFT JSONL. The script builds one
"what does this policy say?" pair per document, plus a refusal pair.
You will still want to write more natural questions by hand, the way the
demo `data/train.jsonl` does, before you train.

```bash
uv run python fine-tuning/scripts/01_prepare_enterprise_docs.py \
  --input fine-tuning/sample_raw_docs \
  --output /tmp/enterprise.jsonl
```

## 02_prepare_code_repo.py

Walks a source tree, picks Python (and a few other) files, and writes
chat rows whose answer is a short description plus a path citation.
This is a starting point for a style or navigation dataset. It is not
a substitute for RAG over the live repo. See
[RAG versus fine-tuning](../rag-vs-fine-tuning.md).

```bash
uv run python fine-tuning/scripts/02_prepare_code_repo.py \
  --input fine-tuning/sample_raw_docs/sample_repo \
  --output /tmp/repo.jsonl
```

## 03_validate_dataset.py

Checks that every line is JSON, that `messages` has a user and an
assistant, that the assistant reply ends with the required JSON footer,
and that `policy_id` / `escalation_needed` types are right.

```bash
uv run python fine-tuning/scripts/03_validate_dataset.py fine-tuning/qwen-finetune-demo/data/train.jsonl
uv run python fine-tuning/scripts/03_validate_dataset.py fine-tuning/qwen-finetune-demo/data/*.jsonl
```

Expected output: a pass line with the row count, or a list of errors
and a non-zero exit.

## 04_evaluate_model.py

Runs each user prompt in a JSONL file against an Ollama model and
checks whether the reply contains the expected `policy_id` and a
parseable footer. Use this on the base model, then again on the tuned
tag.

```bash
uv run python fine-tuning/scripts/04_evaluate_model.py \
  --data fine-tuning/qwen-finetune-demo/data/test.jsonl \
  --model qwen3.5:0.8b
```
