# Open-weight models and fine-tuning

A workshop for senior engineers who already ship software, and who now want to
run open-weight models on their own machines and, when it is worth it, fine-tune
them.

The Markdown in this folder is the source. You can read it on GitHub, or open
it as a site:

From the `open-weight-and-fine-tuning/` folder:

```bash
uv sync --group docs
NO_MKDOCS_2_WARNING=1 uv run mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

Based on November 2026 internet data.

## What you will be able to do

- Install Ollama, pull a model, and talk to it from the terminal and from a
  short Python script.
- Load a model into RAM and unload it again, so a 16 GB laptop stays usable.
- Point Claude Code, Codex CLI, OpenCode, Hermes Agent, and the other official
  `ollama launch` integrations at a local model.
- Use LM Studio when you want a GUI, MCP tools, or a one-click local server.
- Size a model to your machine with the APXML VRAM calculator, field by field.
- Decide between RAG and fine-tuning with official guidance and what
  practitioners actually report.
- Prepare a small policy dataset, fine-tune Qwen 3.5 on a Mac (MLX) or an
  NVIDIA GPU (Unsloth), and compare the result in Ollama.

## Map

### Using models locally

1. [Ollama guide](using-local-models/ollama-guide.md). Install, pull, chat,
   APIs, load and unload from RAM, serve on a network, and shut the server
   down on each OS.
2. [Chat vs agent harnesses](using-local-models/chat-vs-agent-harnesses.md).
   What Ollama is, and how Claude Code, Codex, OpenCode, Hermes Agent, and the
   other official launch targets use it.
3. [LM Studio](using-local-models/lm-studio-guide.md). The GUI alternative,
   including MCP, Projects, and the `lms` CLI.
4. [Hardware and VRAM](using-local-models/hardware-and-vram-guide.md). A walk
   through every field in the APXML VRAM calculator, with annotated
   screenshots.
5. [Runnable examples](using-local-models/examples/README.md). Chat API, a
   small tool-calling agent, and local PDF Q&A.

### Fine-tuning

1. [How fine-tuning works](fine-tuning/README.md). LoRA, QLoRA, SFT, preference
   training, tool choices, and the export path into Ollama.
2. [RAG vs fine-tuning](fine-tuning/rag-vs-fine-tuning.md). When to retrieve,
   when to train, and when to combine them.
3. [Dataset scripts](fine-tuning/scripts/README.md). Turn internal docs or a
   code repo into JSONL, validate it, and evaluate a model.
4. [Qwen 3.5 demo](fine-tuning/qwen-finetune-demo/README.md). A short policy
   Q&A fine-tune, with a beginner [walk through the folder](fine-tuning/qwen-finetune-demo/HOW_TO_READ.md).

## Two hardware tracks

The examples are written so you can follow one track without ignoring the
other. Tabbed install and training pages cover both.

=== "Apple Silicon (Mac)"

    Use Ollama or LM Studio to run GGUF or MLX models. Fine-tune with
    `mlx-lm`. 16 GB unified memory is enough for the 0.8B demo and for
    inference of mid-size Qwen 3 models at 4-bit.

=== "NVIDIA GPU"

    Use Ollama or LM Studio to run GGUF models. Fine-tune with Unsloth.
    The 0.8B demo fits in about 3 GB of VRAM. A 4B run needs about 10 GB.

## Prerequisites that apply everywhere

- Python 3.11 or newer, and [uv](https://docs.astral.sh/uv/).
- [Ollama](https://ollama.com) installed, with the server running. Check
  with `curl http://localhost:11434/api/version`.
- For the fine-tune demo, either a Mac with Apple Silicon or a machine with
  an NVIDIA GPU. The Linux workspace used to build these docs cannot run
  Metal or CUDA training. Use `--dry-run` there to confirm the scripts parse.

If a script cannot reach Ollama, or the model has not been pulled, it exits
with a short message that tells you what to start or pull. Set
`OLLAMA_HOST` if the server is not on localhost.
