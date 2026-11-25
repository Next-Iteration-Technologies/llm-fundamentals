# Open-weight models and fine-tuning

Workshop material for running open-weight models on your own machine, then
fine-tuning a small Qwen 3.5 model when that is the right tool.

Based on November 2026 internet data.

## Open the docs as a site

From the `open-weight-and-fine-tuning/` folder:

```bash
uv sync --group docs
NO_MKDOCS_2_WARNING=1 uv run mkdocs serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The Markdown files in this
folder are the source, so they also read cleanly on GitHub. The generated
`site/` folder is gitignored. `docs/` holds relative symlinks so MkDocs can
use a child docs directory without copying those files; `ensure_docs_links.py`
recreates them if a copy tool drops the links.

To build once, without serving:

```bash
NO_MKDOCS_2_WARNING=1 uv run mkdocs build --strict
```

## Layout

| Path | What it is |
| --- | --- |
| [using-local-models/ollama-guide.md](using-local-models/ollama-guide.md) | Install, pull, chat, APIs, RAM load and unload, networking |
| [using-local-models/chat-vs-agent-harnesses.md](using-local-models/chat-vs-agent-harnesses.md) | Ollama as a model runner, plus official `ollama launch` targets |
| [using-local-models/lm-studio-guide.md](using-local-models/lm-studio-guide.md) | GUI alternative, MCP, Projects, `lms` CLI |
| [using-local-models/hardware-and-vram-guide.md](using-local-models/hardware-and-vram-guide.md) | APXML VRAM calculator, field by field, with screenshots |
| [using-local-models/examples/](using-local-models/examples/README.md) | Chat API, tool-calling agent, local PDF RAG |
| [fine-tuning/README.md](fine-tuning/README.md) | LoRA, QLoRA, SFT, preference training, export to Ollama |
| [fine-tuning/rag-vs-fine-tuning.md](fine-tuning/rag-vs-fine-tuning.md) | When to retrieve, when to train, when to combine |
| [fine-tuning/scripts/](fine-tuning/scripts/README.md) | Dataset prep, validation, and evaluation helpers |
| [fine-tuning/qwen-finetune-demo/](fine-tuning/qwen-finetune-demo/README.md) | Hands-on Qwen 3.5 0.8B policy Q&A demo |

This folder has its own `pyproject.toml`. Use this environment for the
guides and scripts here. Do not mix it with another project's virtualenv.

## Quick start

1. Install [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com).
2. Confirm the server is up: `curl http://localhost:11434/api/version`.
3. Pull a small model: `ollama pull qwen3.5:0.8b`.
4. From the `open-weight-and-fine-tuning/` folder, `uv sync`, then run an example:

```bash
uv run python using-local-models/examples/01_chat_api.py --dry-run
uv run python using-local-models/examples/01_chat_api.py
```

The fine-tune demo has a short [how to read this folder](fine-tuning/qwen-finetune-demo/HOW_TO_READ.md)
page, then a run guide for Mac (MLX) and NVIDIA (Unsloth).
