"""Emit the self-contained T4 training notebook (Phase 3, step 3).

The notebook is plain JSON (nbformat 4.5) and loads ``qlora.json`` and the
chat-formatted splits, so it stays the only place the manual GPU run is
described in executable form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .qlora import QLoRAConfig

RUNBOOK_PATH = "docs/qlora-runbook.md"
GGUF_RUNBOOK_PATH = "docs/gguf-runbook.md"

__all__ = ["GGUF_RUNBOOK_PATH", "RUNBOOK_PATH", "emit_notebook"]


def _code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


def _markdown(source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }


def emit_notebook(
    config: QLoRAConfig,
    *,
    book_id: str,
    runbook_path: str = RUNBOOK_PATH,
) -> dict[str, Any]:
    """Build the T4 notebook payload.

    The cells install the training stack, load ``qlora.json`` and the splits,
    train with Unsloth + TRL's ``SFTTrainer``, save the LoRA adapter and write
    ``adapter/run.json`` for ``distiller train --register``.

    Args:
        config: Pinned QLoRA configuration the notebook refers to.
        book_id: Book the adapter is trained for.
        runbook_path: Repository path of the manual runbook linked from the
            notebook.

    Returns:
        Notebook payload ready for ``json.dumps``.
    """
    cells = [
        _markdown(
            f"""# QLoRA training — {book_id}

Self-contained T4 runbook: installs the stack, loads `qlora.json` and the
chat-formatted splits, trains with Unsloth + TRL `SFTTrainer`, saves the LoRA
adapter and writes `adapter/run.json` for registration.

Base model `{config.base_model}`, LoRA rank {config.lora_rank}, {config.epochs}
epochs. Every hyperparameter is read from `qlora.json`.

Manual steps (upload, train, download, register, compare):
`{runbook_path}`.

The last cell also merges the adapter and exports a Q4_K_M GGUF; follow
`{GGUF_RUNBOOK_PATH}` to download and serve it locally."""
        ),
        _code(
            """# 1. Install the training stack. Unsloth pins its own torch/CUDA build.
!pip install -q unsloth trl peft transformers bitsandbytes datasets
"""
        ),
        _code(
            """# 2. Load the pinned configuration and the chat-formatted splits.
import json
import pathlib

from datasets import load_dataset

config = json.loads(pathlib.Path("qlora.json").read_text(encoding="utf-8"))
manifest = json.loads(pathlib.Path("manifest.json").read_text(encoding="utf-8"))
train_dataset = load_dataset("json", data_files="train.jsonl", split="train")
validation_dataset = load_dataset("json", data_files="validation.jsonl", split="train")
print(f"{len(train_dataset)} train / {len(validation_dataset)} validation examples")
"""
        ),
        _code(
            """# 3. Load the 4-bit base model with Unsloth.
import torch
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config["base_model"],
    max_seq_length=config["max_seq_length"],
    load_in_4bit=config["load_in_4bit"],
    dtype=getattr(torch, config["compute_dtype"]),
)
"""
        ),
        _code(
            """# 4. Attach the LoRA adapter with the pinned hyperparameters.
model = FastLanguageModel.get_peft_model(
    model,
    r=config["lora_rank"],
    lora_alpha=config["lora_alpha"],
    lora_dropout=config["lora_dropout"],
    target_modules=config["target_modules"],
    use_gradient_checkpointing="unsloth",
)
"""
        ),
        _code(
            """# 5. Train with TRL's SFTTrainer on the chat messages.
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    args=TrainingArguments(
        output_dir="checkpoints",
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["epochs"],
        fp16=config["compute_dtype"] == "float16",
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        seed=config["seed"],
    ),
)
trainer.train()
"""
        ),
        _code(
            """# 6. Save the LoRA adapter.
model.save_pretrained("adapter")
tokenizer.save_pretrained("adapter")
print("Adapter saved to adapter/")
"""
        ),
        _code(
            """# 7. Write adapter/run.json for `distiller train <book> --register`.
import datetime
import hashlib

config_hash = hashlib.sha256(pathlib.Path("qlora.json").read_bytes()).hexdigest()[:40]
loss_history = [
    float(entry["loss"]) for entry in trainer.state.log_history if "loss" in entry
]
report = {
    "book_id": manifest["book_id"],
    "base_model": config["base_model"],
    "dataset_hash": manifest["dataset_hash"],
    "config_hash": config_hash,
    "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
    "train_count": manifest["train_count"],
    "validation_count": manifest["validation_count"],
    "epochs": config["epochs"],
    "learning_rate": config["learning_rate"],
    "seed": config["seed"],
    "loss_history": loss_history,
    "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
adapter_dir = pathlib.Path("adapter")
adapter_dir.mkdir(parents=True, exist_ok=True)
(adapter_dir / "run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"Wrote {adapter_dir / 'run.json'}")
"""
        ),
        _code(
            f"""# 8. Merge the adapter and export a Q4_K_M GGUF for local serving.
# Download the gguf/ folder afterwards and follow `{GGUF_RUNBOOK_PATH}`.
model.save_pretrained_gguf("gguf", tokenizer, quantization_method="q4_k_m")
print("GGUF written to gguf/ (see docs/gguf-runbook.md)")
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
