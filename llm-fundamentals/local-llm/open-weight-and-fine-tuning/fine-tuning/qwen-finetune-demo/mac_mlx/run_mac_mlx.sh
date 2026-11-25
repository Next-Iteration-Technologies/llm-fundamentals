#!/usr/bin/env bash
# Train a LoRA adapter with mlx-lm, fuse it, and print the GGUF conversion steps.
#
# Prerequisites:
#   1. An Apple Silicon Mac with enough unified memory (16 GB is enough for 0.8B).
#   2. uv and Python 3.11+.
#   3. From this folder: uv sync
#   4. A llama.cpp checkout later, for convert_hf_to_gguf.py and llama-quantize.
#   5. Ollama is only needed after the GGUF exists.
#
# Usage:
#   ./run_mac_mlx.sh
#   ./run_mac_mlx.sh --dry-run
#   LLAMA_CPP=/path/to/llama.cpp ./run_mac_mlx.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

echo "config: $ROOT/lora_config.yaml"
echo "data:   $ROOT/../data"
echo "adapters will be written to $ROOT/adapters"
echo "fused weights will be written to $ROOT/fused_model"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "dry-run: would run: uv run mlx_lm.lora --config lora_config.yaml"
  echo "dry-run: would run: uv run mlx_lm.fuse --model <config model> --adapter-path ./adapters --save-path ./fused_model --de-quantize"
  echo "dry-run: would then print llama.cpp convert + llama-quantize Q4_K_M commands"
  exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not on PATH. Install it from https://docs.astral.sh/uv/" >&2
  exit 2
fi

uv sync
uv run mlx_lm.lora --config lora_config.yaml

# Qwen 3.5 is not supported by mlx_lm.fuse --export-gguf (as of Nov 2026).
# Fuse and dequantize to Hugging Face weights, then convert with llama.cpp.
uv run mlx_lm.fuse \
  --model mlx-community/Qwen3.5-0.8B-4bit \
  --adapter-path ./adapters \
  --save-path ./fused_model \
  --de-quantize

echo
echo "Fused dequantized weights are in $ROOT/fused_model"
echo "Convert and quantize with llama.cpp next. From your llama.cpp checkout:"
echo
LLAMA_CPP="${LLAMA_CPP:-/path/to/llama.cpp}"
echo "  python ${LLAMA_CPP}/convert_hf_to_gguf.py $ROOT/fused_model --outfile $ROOT/harbor-policy-qwen35-0.8b-f16.gguf"
echo "  ${LLAMA_CPP}/llama-quantize $ROOT/harbor-policy-qwen35-0.8b-f16.gguf $ROOT/harbor-policy-qwen35-0.8b-q4_k_m.gguf Q4_K_M"
echo
echo "Then copy ../Modelfile.template to $ROOT/Modelfile, set FROM to the Q4_K_M file,"
echo "and run:  ollama create harbor-policy -f $ROOT/Modelfile"
