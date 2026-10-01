"""Pinned QLoRA configuration and the heavy-stack gate (Phase 3, step 2).

The training stack (torch, unsloth, trl, peft, transformers, bitsandbytes)
lives in the optional ``training`` extra. Nothing here imports it at module
level: :func:`require_training_stack` loads it through ``optional_deps.require``
so a plain install gets one consistent, actionable error.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ..models import DomainModel
from ..optional_deps import require

TRAINING_MODULES = (
    "torch",
    "unsloth",
    "trl",
    "transformers",
    "peft",
    "bitsandbytes",
)

__all__ = [
    "TRAINING_MODULES",
    "QLoRAConfig",
    "require_training_stack",
    "training_stack_versions",
]


class QLoRAConfig(DomainModel):
    """Pinned QLoRA hyperparameters (the single source of truth for training).

    Written to ``training/qlora.json``; the emitted T4 notebook loads it instead
    of hardcoding values, so changing a hyperparameter is editing one document.

    Attributes:
        base_model: Hugging Face checkpoint to fine-tune (Apache 2.0).
        load_in_4bit: Load the base model in 4-bit quantization.
        quant_type: 4-bit quantization scheme (NF4 for QLoRA).
        compute_dtype: Compute dtype; fp16 because the T4 has no bf16.
        lora_rank: LoRA rank (r).
        lora_alpha: LoRA scaling alpha.
        lora_dropout: LoRA dropout probability.
        target_modules: Attention and MLP projections that receive adapters.
        learning_rate: Optimizer learning rate.
        epochs: Number of training epochs.
        max_seq_length: Maximum sequence length in tokens.
        per_device_train_batch_size: Batch size per device.
        gradient_accumulation_steps: Micro-batches per optimizer step.
        seed: Random seed for training.
    """

    base_model: str = "Qwen/Qwen3-4B"
    load_in_4bit: bool = True
    quant_type: Literal["nf4", "fp4"] = "nf4"
    compute_dtype: Literal["float16", "bfloat16"] = "float16"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    learning_rate: float = 2e-4
    epochs: int = 3
    max_seq_length: int = 4096
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    seed: int = 13


def require_training_stack() -> None:
    """Load every heavy training module, one actionable error when missing.

    Raises:
        MissingDependencyError: When the ``training`` extra is not installed.
    """
    for module_name in TRAINING_MODULES:
        require(module_name, extra="training", purpose="QLoRA training")


def training_stack_versions() -> dict[str, str]:
    """Return the installed version of every training-stack module.

    Returns:
        Mapping module name -> version string.

    Raises:
        MissingDependencyError: When the ``training`` extra is not installed.
    """
    require_training_stack()
    versions: dict[str, str] = {}
    for module_name in TRAINING_MODULES:  # pragma: no cover - needs the GPU stack
        module = require(module_name, extra="training", purpose="QLoRA training")
        versions[module_name] = str(getattr(module, "__version__", "unknown"))
    return versions
