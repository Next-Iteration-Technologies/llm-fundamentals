#!/usr/bin/env python3
"""Fine-tune Qwen 3.5 with Unsloth (bf16 LoRA) and export a Q4_K_M GGUF.

Prerequisites
-------------
1. A machine with an NVIDIA GPU (about 3 GB VRAM for 0.8B, about 10 GB for 4B)
   or a Colab GPU runtime.
2. uv and Python 3.11+, or the environment from Unsloth's current install docs:
   https://docs.unsloth.ai/models/qwen-3.5
3. From this folder: `uv sync` (if resolution fails, use Unsloth's install block).
4. Official Qwen 3.5 rule: bf16 LoRA, load_in_4bit=False. Do not switch on
   4-bit QLoRA for this family unless Unsloth's guide has changed.
5. Ollama is only needed after the GGUF exists.
6. Expected output: adapters plus a .gguf file under ./export_gguf, and printed
   instructions for `ollama create`.

Usage
-----
    uv run train_and_export.py --dry-run
    uv run train_and_export.py
    uv run train_and_export.py --model Qwen/Qwen3.5-4B
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = Path(__file__).resolve().parent / "export_gguf"
DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face id. Swap to Qwen/Qwen3.5-4B for the larger demo.")
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> int:
    args = parse_args()
    train_path = args.data / "train.jsonl"
    valid_path = args.data / "valid.jsonl"
    train_n = sum(1 for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()) if train_path.is_file() else 0
    valid_n = sum(1 for line in valid_path.read_text(encoding="utf-8").splitlines() if line.strip()) if valid_path.is_file() else 0

    if args.dry_run:
        print(f"dry-run: would load {args.model}")
        print("dry-run: FastModel.from_pretrained(..., load_in_4bit=False, dtype=bfloat16)")
        print(f"dry-run: would train LoRA on {train_n} rows, validate on {valid_n} rows")
        print(f"dry-run: would write GGUF to {args.output} with quantization_method='q4_k_m'")
        return 0

    if train_n == 0:
        print(f"no training rows in {train_path}", file=sys.stderr)
        return 1

    try:
        import torch
        from datasets import Dataset
        from unsloth import FastModel
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        print(
            "Could not import Unsloth, datasets, or trl.\n"
            f"Underlying error: {exc}\n"
            "Install from this folder with `uv sync`, or follow "
            "https://docs.unsloth.ai/models/qwen-3.5",
            file=sys.stderr,
        )
        return 2

    if not torch.cuda.is_available():
        print(
            "No CUDA GPU is visible. Run this script on an NVIDIA machine or Colab, "
            "or pass --dry-run to check the arguments.",
            file=sys.stderr,
        )
        return 2

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        dtype=torch.bfloat16,
    )
    model = FastModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        use_gradient_checkpointing="unsloth",
    )

    def to_text(row: dict) -> dict:
        text = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_ds = Dataset.from_list(load_rows(train_path)).map(to_text)
    valid_ds = Dataset.from_list(load_rows(valid_path)).map(to_text) if valid_n else None

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        args=SFTConfig(
            output_dir=str(Path(__file__).resolve().parent / "outputs"),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=args.epochs,
            learning_rate=2e-4,
            logging_steps=5,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            max_seq_length=args.max_seq_length,
            dataset_text_field="text",
        ),
    )
    trainer.train()

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_gguf(
        str(args.output),
        tokenizer,
        quantization_method="q4_k_m",
    )
    print(f"Wrote GGUF under {args.output}")
    print("Point Modelfile.template FROM at the .gguf file in that folder,")
    print("then run:  ollama create harbor-policy -f Modelfile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
