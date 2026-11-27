# Runnable examples

Three small scripts that talk to a local Ollama server. They share the same prerequisites and the same failure behaviour: if the server is down or the model is missing, the script exits with a short message rather than a stack trace.

Run them from the `open-weight-and-fine-tuning/` folder, which holds this workshop's `pyproject.toml`.

## Prerequisites

These apply to every script in this folder.

1. Install [uv](https://docs.astral.sh/uv/) and Python 3.11+.
2. Install [Ollama](https://ollama.com) and start the server (the desktop app, or `ollama serve`).
3. Check the server:

   ```bash
   curl http://localhost:11434/api/version
   ```

   You should see a JSON body with a `version` field. If this fails, the server is not running.
4. From the `open-weight-and-fine-tuning/` folder:

   ```bash
   uv sync
   ```
5. Pull the models listed in each script below.
6. Optional: if Ollama is on another machine, `export OLLAMA_HOST=http://HOST:11434` and pull the model on that machine.

`--dry-run` prints the planned calls and exits 0 without talking to the server. Use it to confirm the scripts parse on a machine that has no Ollama.

```bash
uv run python using-local-models/examples/01_chat_api.py --dry-run
uv run python using-local-models/examples/02_tool_calling_agent.py --dry-run
uv run python using-local-models/examples/03_local_pdf_rag.py --dry-run
```

## 01_chat_api.py

A single chat turn, then a structured JSON reply.

```bash
ollama pull qwen3.5:0.8b
uv run python using-local-models/examples/01_chat_api.py
```

Expected output: a one-sentence greeting, then a JSON object with `city` and `status` keys. To try the 4B model, pass `--model qwen3.5:4b` after `ollama pull qwen3.5:4b`.

## 02_tool_calling_agent.py

A short loop that offers a `read_file` tool, limited to this examples folder, and asks the model to summarise `README.md`.

```bash
ollama pull qwen3.5:0.8b
uv run python using-local-models/examples/02_tool_calling_agent.py
```

Expected output: a log line showing the tool call, then a short summary of this README. If the 0.8B model skips the tool, rerun with `--model qwen3.5:4b` (and a 16K context, which the script already sets). Check `ollama show qwen3.5:0.8b` for a `tools` capability.

## 03_local_pdf_rag.py

Embeds `sample_policy.pdf` with `embeddinggemma-2:270m`, retrieves the top chunks for a question, and answers with `qwen3.5:0.8b`. This is the retrieval half of the [RAG versus fine-tuning](../../fine-tuning/rag-vs-fine-tuning.md) comparison.

```bash
ollama pull qwen3.5:0.8b
ollama pull embeddinggemma-2:270m
uv run python using-local-models/examples/03_local_pdf_rag.py
uv run python using-local-models/examples/03_local_pdf_rag.py --question "How many days of paternity leave do employees get?"
```

Expected output: the top chunk scores, then an answer that refers to the Harbor Labs paternity-leave rule (10 working days). If the answer is missing, the model is instructed to say so.

When you are done, free the memory:

```bash
ollama stop qwen3.5:0.8b
ollama stop embeddinggemma-2:270m
ollama ps
```

See [Loading and unloading models](../ollama-guide.md#loading-and-unloading-models-from-ram).
