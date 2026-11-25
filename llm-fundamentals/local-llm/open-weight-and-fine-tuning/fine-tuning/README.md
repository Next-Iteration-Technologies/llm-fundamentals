# How fine-tuning works

Fine-tuning continues training on a prepared dataset so the model picks up a habit: a format, a tone, a tool pattern, or a tighter way of doing one task. It is a poor default for "make the model remember this handbook." For that comparison, read [RAG versus fine-tuning](rag-vs-fine-tuning.md) first.

Based on November 2026 internet data.

This page covers the methods, the memory-saving adapters (LoRA and QLoRA), the local tools you can actually run, and the export path into Ollama. The hands-on work is in the [dataset scripts](scripts/README.md) and the [Qwen 3.5 demo](qwen-finetune-demo/README.md). If this folder is new to you, start with [How to read the demo folder](qwen-finetune-demo/HOW_TO_READ.md).

## What you are changing

A base model is a large pile of weights. Full fine-tuning updates all of them. That is accurate and extremely memory-hungry: optimiser states and gradients sit next to the weights. The VRAM calculator's Fine-tuning tab, with Training Mode set to Full, is the way to see that number before you try. A 32B full run on an M4 is hundreds of gigabytes. See the annotated screenshot in the [VRAM guide](../using-local-models/hardware-and-vram-guide.md#the-fine-tuning-tab).

**LoRA** (Low-Rank Adaptation) freezes the base weights and trains a small pair of matrices next to selected layers. The adapter is a few tens of megabytes. At inference you either load it alongside the base model, or merge it into a new set of weights.

**QLoRA** is LoRA plus a 4-bit frozen base, so a 7B or 8B model can be tuned on a single 16 to 24 GB GPU. This is the usual NVIDIA path. It is not the path Unsloth currently recommends for Qwen 3.5: they ask for **bf16 LoRA**, with 4-bit QLoRA turned off, because of how that family is quantised. The demo follows that official rule. The 0.8B model needs about 3 GB of VRAM in that setup; the 4B needs about 10 GB.

On Apple Silicon, **mlx-lm** is the usual LoRA trainer. Unsloth's PEFT-style default learning rate (often `1e-4`) is too high for mlx-lm; official mlx-lm examples use a lower rate, around `1e-5`, with LoRA scale 20. The demo config follows that.

## Methods, in the order you will meet them

- **SFT (supervised fine-tuning).** The workhorse. Each row is an input and the target reply. The demo is SFT: policy questions with a short answer and a strict JSON footer.
- **CPT (continued pretraining).** Next-token training on raw domain text, without question/answer structure. Sometimes used to expose a model to a specialised vocabulary. It is easy to overdo, and it does not replace RAG for facts that must stay current.
- **Preference training (DPO, ORPO).** Each row is a prompt plus a chosen reply and a rejected reply. Use this after SFT when you can label "better / worse" pairs, for tone or safety.
- **GRPO and similar verifiable-reward methods.** The model generates several replies, a scorer checks a rule (did the JSON parse? did the tests pass?), and the model is updated from that signal. Useful when you can check the output automatically. Not required for the demo.

Start with SFT. Add preference training only when a held-out eval shows a specific failure SFT did not fix.

## What a training file looks like

The demo uses a chat JSONL row, one JSON object per line:

```json
{"messages": [
  {"role": "system", "content": "You are Harbor Labs' internal policy assistant. Answer only from Harbor Labs policy. End every reply with a JSON object on its own line: {\"policy_id\": \"...\" or null, \"escalation_needed\": true or false}."},
  {"role": "user", "content": "How many days of paternity leave do employees get?"},
  {"role": "assistant", "content": "Eligible employees receive 10 working days of paternity leave, to be taken within 60 days of the child's birth.\n\n{\"policy_id\": \"HR-14\", \"escalation_needed\": false}"}
]}
```

Alpaca-style `{ "instruction", "input", "output" }` files also work with most trainers. Keep one schema and stick to it. Validate every row before you train: see `scripts/03_validate_dataset.py`.

## Tools you can run locally (November 2026)

| Tool | Platform | Notes |
| --- | --- | --- |
| **mlx-lm** | Apple Silicon | The Mac path in the demo. CLI: `mlx_lm.lora`, `mlx_lm.fuse`. |
| **Unsloth** | NVIDIA (and Colab) | The NVIDIA path in the demo. Fast LoRA, official Qwen 3.5 guide uses bf16, not 4-bit QLoRA. |
| **Unsloth Studio** | Browser UI on top of Unsloth | Useful if you want a form instead of a script. |
| **LLaMA-Factory** | NVIDIA, some Apple support | Broad trainer with a lot of model cards and a web UI. |
| **Axolotl** | NVIDIA | YAML-driven, common in serious open-weight training setups. |
| **Transformer Lab** | Desktop app | A GUI around local training and inference. |
| **Silicon Studio** | Mac GUI | A Mac-oriented training UI if you do not want the mlx-lm CLI. |

This workshop uses mlx-lm and Unsloth because they are documented for Qwen 3.5, they export in a way you can get to GGUF, and they fit the two hardware tracks.

## Getting a fine-tune into Ollama

Ollama v0.34.1 removed the `ADAPTER` keyword from the Modelfile. You cannot point Ollama at a loose LoRA file any more. Merge the adapter into the base weights, export or convert to GGUF (usually `Q4_K_M` for a laptop), then:

```dockerfile
FROM ./harbor-policy-qwen35-0.8b-q4_k_m.gguf
PARAMETER temperature 0.1
PARAMETER num_ctx 8192
SYSTEM """You are Harbor Labs' internal policy assistant. ..."""
```

```bash
ollama create harbor-policy -f Modelfile
ollama run harbor-policy "How many days of paternity leave do employees get?"
```

**Apple / mlx-lm path.** `mlx_lm.fuse --export-gguf` does not currently support Qwen 3.5. Fuse and dequantize to Hugging Face weights (the demo script does this), then convert with llama.cpp's `convert_hf_to_gguf.py` and quantize with `llama-quantize Q4_K_M`.

**NVIDIA / Unsloth path.** `save_pretrained_gguf(..., quantization_method="q4_k_m")` writes a GGUF you can import. Qwen 3.5 does not need a separate `mmproj` file for Ollama in the current Unsloth guide.

After `ollama create`, compare the base model and the tuned model with `qwen-finetune-demo/compare_in_ollama.py`.

## A realistic training loop

1. Write an eval set first (the demo already has `data/test.jsonl`).
2. Start from a prompt-only baseline on that set.
3. Prepare and validate the JSONL (`scripts/`).
4. Train LoRA on 0.8B. If the eval is clearly short of the bar, swap the IDs to 4B (one-line change in the demo).
5. Look at the failures. If they are facts that were never in the documents, add RAG rather than more epochs.
6. Merge, convert to GGUF, import into Ollama, and compare again.

Recommended first learning rates:

=== "Apple Silicon (mlx-lm)"

    LoRA rank 16, scale 20, learning rate `1e-5`, a few hundred iters on the small demo set. The values are in `qwen-finetune-demo/mac_mlx/lora_config.yaml`.

=== "NVIDIA (Unsloth)"

    bf16 LoRA, `load_in_4bit=False`, learning rate `2e-4` is a common Unsloth starting point for LoRA (their docs, not mlx-lm's). The 0.8B script in `nvidia_unsloth/train_and_export.py` uses a short epoch count so a first run finishes in minutes.

## Sources

- [Unsloth: Fine-tune Qwen 3.5](https://docs.unsloth.ai/models/qwen-3.5)
- [Unsloth FAQ](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/faq)
- [mlx-lm](https://github.com/ml-explore/mlx-lm)
- [Ollama Modelfile](https://docs.ollama.com/modelfile) and [import](https://docs.ollama.com/import)
- [APXML VRAM calculator](https://apxml.com/tools/vram-calculator)
