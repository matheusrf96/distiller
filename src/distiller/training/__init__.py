"""QLoRA training harness (Phase 3): chat formatting, dataset prep, registry.

The harness runs fully offline: preparing the dataset, registering an adapter
and comparing base vs adapter never import torch, unsloth, trl, peft or
transformers. Training itself is the manual T4 step documented in
``docs/qlora-runbook.md``.
"""

from .chat import (
    ChatMessage,
    TrainingExample,
    citation_indices,
    format_example,
    format_examples,
    prompt_contract_hash,
)
from .compare import (
    COMPARISON_METRICS,
    GeneratorResult,
    GeneratorVariant,
    TrainingComparison,
    TrainingRun,
    run_comparison,
)
from .dataset import (
    TrainingBundle,
    TrainingDatasetManifest,
    ValidationReport,
    load_raft_examples,
    prepare_dataset,
    split_examples,
    validate_examples,
)
from .notebook import RUNBOOK_PATH, emit_notebook
from .qlora import (
    TRAINING_MODULES,
    QLoRAConfig,
    require_training_stack,
    training_stack_versions,
)
from .registry import (
    ADAPTER_FILES,
    TrainingReport,
    load_adapter_report,
    register_adapter,
)

__all__ = [
    "ADAPTER_FILES",
    "COMPARISON_METRICS",
    "RUNBOOK_PATH",
    "TRAINING_MODULES",
    "ChatMessage",
    "GeneratorResult",
    "GeneratorVariant",
    "QLoRAConfig",
    "TrainingBundle",
    "TrainingComparison",
    "TrainingDatasetManifest",
    "TrainingExample",
    "TrainingReport",
    "TrainingRun",
    "ValidationReport",
    "citation_indices",
    "emit_notebook",
    "format_example",
    "format_examples",
    "load_adapter_report",
    "load_raft_examples",
    "prepare_dataset",
    "prompt_contract_hash",
    "register_adapter",
    "require_training_stack",
    "run_comparison",
    "split_examples",
    "training_stack_versions",
    "validate_examples",
]
