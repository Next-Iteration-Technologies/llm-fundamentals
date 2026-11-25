# Qwen 3.5 fine-tune demo

Fine-tune Qwen 3.5 so it answers Harbor Labs policy questions and ends every reply with a JSON footer:

```json
{"policy_id": "HR-14", "escalation_needed": false}
```

When the policy set has no answer, the model should say so and set `"policy_id": null, "escalation_needed": true`.

Default model: **Qwen 3.5 0.8B**. To use 4B, change the model id in the config or script as noted below, and pull or download the 4B weights instead. If this folder is new to you, read [HOW_TO_READ.md](HOW_TO_READ.md) first.

Based on November 2026 internet data.

## Prerequisites

These apply to both hardware tracks.

1. Install [uv](https://docs.astral.sh/uv/) and Python 3.11+.
2. From the `open-weight-and-fine-tuning/` folder, run `uv sync` (for the comparison script and the dataset validator).
3. Work through [How fine-tuning works](../README.md) and [RAG versus fine-tuning](../rag-vs-fine-tuning.md) so the goal is clear.
4. Validate the bundled data:

   From the `open-weight-and-fine-tuning/` folder:

   ```bash
   uv run python fine-tuning/scripts/03_validate_dataset.py \
     fine-tuning/qwen-finetune-demo/data/train.jsonl \
     fine-tuning/qwen-finetune-demo/data/valid.jsonl \
     fine-tuning/qwen-finetune-demo/data/test.jsonl
   ```

   Expected output: three `ok:` lines and exit 0.
5. After export, you need [Ollama](https://ollama.com) running to compare models:
   `curl http://localhost:11434/api/version` should return a JSON `version`.
6. Optional: `export OLLAMA_HOST=http://HOST:11434`.

Live training needs Apple Silicon (mlx-lm) or an NVIDIA GPU (Unsloth). The Linux environment used to build these docs can only run `--dry-run` and syntax checks.

## The dataset

36 training rows, 6 validation rows, 8 held-out test rows, all in `data/`. Policies covered: HR-14 parental leave, HR-07 expenses, HR-03 PTO, IT-11 laptops, SEC-02 data handling, plus refusals for questions the handbook does not answer.

The raw Markdown lives in `../sample_raw_docs/` if you want to extend the set with [the prep scripts](../scripts/README.md).

## Track A: Apple Silicon (mlx-lm)

About 16 GB unified memory is enough for the 0.8B demo. The 4B model is tighter; close other apps and check the [VRAM guide](../../using-local-models/hardware-and-vram-guide.md) first.

From `fine-tuning/qwen-finetune-demo/mac_mlx/` (inside this folder):

```bash
uv sync
# optional dry run of the shell wrapper (prints the commands):
./run_mac_mlx.sh --dry-run
./run_mac_mlx.sh
```

The script will:

1. Train LoRA with `lora_config.yaml` (rank 16, scale 20, learning rate `1e-5`).
2. Fuse the adapter back into weights, and dequantize them, because `mlx_lm.fuse --export-gguf` does not currently support Qwen 3.5.
3. Print the llama.cpp commands to run next: `convert_hf_to_gguf.py` and `llama-quantize Q4_K_M`.

To swap to 4B, edit `lora_config.yaml` and set:

```yaml
model: mlx-community/Qwen3.5-4B-4bit
```

Use the matching 4B id if the community name has moved; check [mlx-community on Hugging Face](https://huggingface.co/mlx-community) before you start.

You need a checkout of [llama.cpp](https://github.com/ggml-org/llama.cpp) for the convert and quantize steps. The script prints exact commands with the output directory it just wrote.

## Track B: NVIDIA GPU or Colab (Unsloth)

The 0.8B bf16 LoRA run wants about 3 GB of VRAM. The 4B run wants about 10 GB. Unsloth's official Qwen 3.5 guide says to use **bf16 LoRA**, with 4-bit QLoRA off (`load_in_4bit=False`).

From `fine-tuning/qwen-finetune-demo/nvidia_unsloth/` (inside this folder):

```bash
uv sync
uv run train_and_export.py --dry-run
uv run train_and_export.py
```

To swap to 4B:

```bash
uv run train_and_export.py --model Qwen/Qwen3.5-4B
```

The script writes a GGUF with `save_pretrained_gguf(..., quantization_method="q4_k_m")`. Qwen 3.5 does not need a separate `mmproj` file for Ollama. Point `Modelfile.template` at that GGUF.

If `uv sync` cannot resolve Unsloth on your platform, follow the current install block on [the Unsloth Qwen 3.5 page](https://docs.unsloth.ai/models/qwen-3.5) and run the same script inside that environment.

## Import into Ollama and compare

1. Copy `Modelfile.template` to `Modelfile` in the directory that contains the GGUF.
2. Set the `FROM` path to your `.gguf` file. Leave `ADAPTER` out.
3. Create and test:

   ```bash
   ollama create harbor-policy -f Modelfile
   ollama run harbor-policy "How many days of paternity leave do employees get?"
   ```

   You want a short factual answer and a footer with `"policy_id": "HR-14"`.
4. Compare against the base model (pull it first):

   From the `open-weight-and-fine-tuning/` folder:

   ```bash
   ollama pull qwen3.5:0.8b
   uv run python fine-tuning/qwen-finetune-demo/compare_in_ollama.py \
     --base qwen3.5:0.8b --tuned harbor-policy
   ```

   `--dry-run` is supported. Expected output: for each test question, both replies, then a short score summary.

5. You can also score a single model with [04_evaluate_model.py](../scripts/README.md).
6. Unload when you are finished: `ollama stop harbor-policy` and `ollama stop qwen3.5:0.8b`. See [Loading and unloading models](../../using-local-models/ollama-guide.md#loading-and-unloading-models-from-ram).

## Sources

- [Unsloth: Fine-tune Qwen 3.5](https://docs.unsloth.ai/models/qwen-3.5)
- [mlx-lm](https://github.com/ml-explore/mlx-lm)
- [Ollama import](https://docs.ollama.com/import) and [Modelfile](https://docs.ollama.com/modelfile)
