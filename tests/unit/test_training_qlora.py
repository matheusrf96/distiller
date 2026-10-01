"""Unit tests for the pinned QLoRA configuration and the stack gate."""

from __future__ import annotations

import sys

import pytest

from distiller.exceptions import MissingDependencyError
from distiller.optional_deps import is_available
from distiller.training.qlora import (
    TRAINING_MODULES,
    QLoRAConfig,
    require_training_stack,
    training_stack_versions,
)


def test_qlora_config_round_trips_with_pinned_defaults() -> None:
    """Every approved hyperparameter is pinned and serializes (REQ-TR-006)."""
    config = QLoRAConfig()

    assert QLoRAConfig.model_validate_json(config.model_dump_json()) == config
    assert config.base_model == "Qwen/Qwen3-4B"
    assert config.load_in_4bit is True
    assert config.quant_type == "nf4"
    assert config.compute_dtype == "float16"
    assert config.lora_rank == 16
    assert config.lora_alpha == 32
    assert config.lora_dropout == 0.0
    assert set(config.target_modules) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    assert config.learning_rate == 2e-4
    assert config.epochs == 3
    assert config.max_seq_length == 4096
    assert config.seed == 13


def test_require_training_stack_names_the_training_extra() -> None:
    """Missing heavy modules raise the standard missing-extra error (REQ-TR-013)."""
    if all(is_available(module) for module in TRAINING_MODULES):
        pytest.skip("training stack is installed in this environment")

    with pytest.raises(MissingDependencyError, match="--extra training"):
        require_training_stack()
    with pytest.raises(MissingDependencyError, match="--extra training"):
        training_stack_versions()


def test_harness_imports_without_the_training_stack() -> None:
    """Importing the harness never loads the heavy stack (REQ-TR-013)."""
    import distiller.training  # noqa: F401 - the import is the behaviour under test

    heavy = {"torch", "unsloth", "trl", "peft", "transformers"} & set(sys.modules)
    assert not heavy, f"the training harness imported the heavy stack: {sorted(heavy)}"


def test_training_stack_versions_with_stubbed_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Versions are collected when every module loads."""

    class StubModule:
        __version__ = "1.2.3"

    monkeypatch.setattr(
        "distiller.training.qlora.require", lambda name, **kwargs: StubModule
    )

    versions = training_stack_versions()

    assert set(versions) == set(TRAINING_MODULES)
    assert versions[TRAINING_MODULES[0]] == "1.2.3"
