# How to read this folder

This is a beginner map of the Qwen 3.5 fine-tune demo, for engineers who ship software and have not trained a model before.

## What you will do

You will teach a small open-weight model one habit: answer Harbor Labs policy questions, and end every reply with a strict JSON footer. You will not teach it "all of Harbor Labs." The policies are a stand-in for an internal handbook. The [RAG versus fine-tuning](../rag-vs-fine-tuning.md) page explains why those are different jobs.

The default model is **Qwen 3.5 0.8B**, chosen because it is fast enough for a live demo. A one-line swap, documented in the [run guide](README.md), moves you to the 4B model if the 0.8B habit is too weak.

## The pieces

| Path | Role |
| --- | --- |
| `data/train.jsonl` | Examples the trainer learns from (36 rows). |
| `data/valid.jsonl` | A small check set used during training (6 rows). |
| `data/test.jsonl` | Held-out questions you score after training (8 rows). Do not train on these. |
| `mac_mlx/` | Apple Silicon path: `lora_config.yaml` and `run_mac_mlx.sh`. |
| `nvidia_unsloth/` | NVIDIA or Colab path: `train_and_export.py`. |
| `Modelfile.template` | How the merged GGUF becomes an Ollama model. There is no `ADAPTER` line. |
| `compare_in_ollama.py` | Asks the base model and the tuned model the same test questions. |

A JSONL file is just one JSON object per line. Each object has a `messages` list: system, user, assistant. The assistant's last line is always a JSON footer with `policy_id` and `escalation_needed`.

## The flow, in prose

1. Read two or three rows of `data/train.jsonl` so the footer is concrete.
2. From the `open-weight-and-fine-tuning/` folder, run the validator: `uv run python fine-tuning/scripts/03_validate_dataset.py fine-tuning/qwen-finetune-demo/data/*.jsonl`.
3. Pick your hardware track in the [run guide](README.md) and run the trainer. The first run should be short.
4. Merge the LoRA adapter into a full set of weights, convert to GGUF, and `ollama create` from the Modelfile.
5. Compare base vs tuned with `compare_in_ollama.py`. You are looking for a stable footer and the right `policy_id`, including refusals.

If training is not possible on the machine in front of you (no Metal, no CUDA), `--dry-run` still checks that the scripts parse. Do the live run on a Mac or an NVIDIA box.

## Words that show up in the configs

- **Base model.** The starting weights, before your examples.
- **LoRA adapter.** A small file of extra weights. This is what training actually writes.
- **Merge / fuse.** Copy the adapter into the base weights so you have one model file.
- **GGUF.** The file format Ollama imports. `Q4_K_M` is a common 4-bit laptop quantisation.
- **Eval.** A fixed list of questions with answers you already accept. Always score the same list before and after training.
