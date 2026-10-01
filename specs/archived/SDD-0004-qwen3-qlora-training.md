---
id: SDD-0004
status: done
supersedes:
owner: matheus
created: 2026-09-30
archived: 2026-10-01
---

# Qwen3-4B QLoRA Training (Phase 3)

## Problem

Phase 2 produces a RAFT dataset for a book, but nothing turns it into a model —
and nothing proves the fine-tune helps. The payoff of the "distiller" name is a
4B student that answers from this book better than the base model. This spec
ships the training **harness**: chat formatting of the RAFT examples, a pinned
QLoRA configuration, a ready-to-run notebook for a free T4, adapter
registration, and a base-vs-adapter evaluation over the same index and golden
set. Training itself runs manually on the free GPU; the repository, its tests
and CI stay GPU-free.

## Context

- **Phase 2 dataset**: `src/distiller/synthesis/` writes
  `dataset/{raft,qa,rejected}.jsonl` + `dataset/manifest.json` (SDD-0003).
  `RaftExample` carries `question`, ordered `contexts` (each with `chunk_id`,
  `text`, `is_golden`), a target `answer` (quoted evidence + `[n]` citation, or
  the refusal) and `answerable`.
- **Prompt contract**: `src/distiller/rag/prompts.py`
  (`build_system_prompt`, `build_user_prompt`, `render_documents`) and the
  shared refusal in `src/distiller/models.py::refusal_text` (moved there by
  SDD-0003). Training data must match inference behaviour, so the formatter
  imports these instead of duplicating them.
- **Evaluation**: `src/distiller/evaluation/metrics.py` (`evaluate_item`,
  `summarize`) and `ablation.py` (`metric_deltas`); the `index` identity block
  in `eval/report.json` (SDD-0001, SDD-0002) is the pattern the `generator`
  identity copies.
- **LLM abstraction**: `src/distiller/llm/` — any OpenAI-compatible endpoint,
  plus the deterministic `FakeLLM`. A served LoRA adapter is just another
  endpoint, so the harness needs no local GPU.
- **Optional dependencies**: `src/distiller/optional_deps.require()` gates
  heavy extras; `pyproject.toml` carries `pdf-ai`, `embed`, `index`, `eval`.
- **Related specs**: builds on SDD-0003 (the dataset); copies the report
  identity pattern from SDD-0001/0002; Phase 4 (GGUF serving) and Phase 5
  (thematic layer) consume the adapter this phase produces.

### Data model deltas

| Model / field | Kind | Notes |
|---|---|---|
| `QLoRAConfig` | pydantic (`DomainModel`) | Base model, quantization, LoRA and trainer hyperparameters; emitted as `training/qlora.json` |
| `ChatMessage` / `TrainingExample` | pydantic | Chat roles and the serialized training rows (`id`, `question`, `messages`, `answerable`) |
| `TrainingDatasetManifest` | pydantic | Source hash, prompt hash, split configuration, counts, dataset hash |
| `ValidationReport` | pydantic | Errors, warnings and counts from the pre-training checks |
| `TrainingReport` | pydantic | `run.json`: base model, dataset/config hashes, counts, epochs, loss history, hardware |
| `GeneratorVariant` / `GeneratorResult` / `TrainingComparison` | pydantic | Base-vs-adapter comparison report |
| `TrainingRun` | dataclass (runtime only) | Variant + answer callable + skip reason (mirrors `AblationRun`) |
| `TrainingBundle` | dataclass (runtime only) | Manifest + examples + splits + config (mirrors `SynthesisRun`) |
| `TrainingSettings` / `AdapterSettings` | pydantic settings | Root `Settings.training` / `Settings.adapter` |

### Module deltas

```
src/distiller/training/
├── __init__.py     # exports
├── chat.py         # RAFT -> chat examples (RAG prompt contract reuse)
├── dataset.py      # split, validation, manifest, prepare orchestration
├── qlora.py        # QLoRAConfig + training-stack gate
├── notebook.py     # self-contained T4 notebook emission
├── registry.py     # TrainingReport, adapter register/load
└── compare.py      # base vs adapter golden-set comparison
```

### Configuration

`TrainingSettings(seed=13, val_ratio=0.1)` under `Settings.training`, with
validated bounds. `AdapterSettings(base_url=None, api_key=None, model=None,
timeout=120.0)` under `Settings.adapter`; `model=None` means "no adapter
configured" and the sentinel `"fake"` selects the offline client.

Environment: `DISTILLER_TRAINING__SEED`, `DISTILLER_TRAINING__VAL_RATIO`,
`DISTILLER_ADAPTER__MODEL`, `DISTILLER_ADAPTER__BASE_URL`.

`pyproject.toml` gains `training = ["torch>=2.2", "unsloth>=2024.12",
"trl>=0.11", "peft>=0.12", "transformers>=4.44", "bitsandbytes>=0.43"]`
(minimum bounds, matching the other extras). It is deliberately **not** part of
`all`: Unsloth pins its own torch/CUDA stack, so `make install-all` stays
portable.

### CLI

- `distiller train <book> [--seed] [--val-ratio] [--register DIR]
  [--check-runtime] [--json]` — prepares the dataset and emits the notebook;
  `--register` validates and installs a T4 adapter instead; `--check-runtime`
  reports whether the heavy stack is installed.
- `distiller train-eval <book> [--golden] [--limit] [--json]` — evaluates base
  and adapter over the same index and golden set; writes `eval/training.json`.
- `distiller ask ... --adapter` / `distiller eval ... --adapter` — use the
  configured adapter endpoint instead of the base generator.

### Artifact layout

```
artifacts/<book>/
├── dataset/raft.jsonl            # SDD-0003 output (input to this phase)
├── training/
│   ├── train.jsonl               # chat-formatted training split
│   ├── validation.jsonl          # held-out split (same format)
│   ├── manifest.json             # TrainingDatasetManifest
│   ├── qlora.json                # QLoRAConfig (single source of truth)
│   ├── train_t4.ipynb            # emitted T4 notebook
│   └── adapter/                  # registered T4 output
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── run.json              # TrainingReport (identity + loss history)
└── eval/
    ├── report.json               # + generator block
    └── training.json             # base vs adapter comparison
```

### Failure modes

| Failure | Behaviour |
|---|---|
| `dataset/raft.jsonl` missing | Friendly error: run `distiller synth` first |
| Context chunk id absent from `chunks.jsonl` | Actionable error: re-run `distiller index` and `distiller synth` |
| Dataset validation errors | Nothing is written; errors are listed in the CLI message |
| No unanswerable examples | Validation error naming the missing refusal negatives |
| Adapter directory invalid (no/invalid `run.json`, wrong book or base model) | Register fails with a friendly message; nothing is copied |
| Adapter endpoint unconfigured | `ask`/`eval`/`train-eval` fail with the config hint |
| No registered adapter | Friendly message pointing at `distiller train <book> --register <dir>` |
| One comparison variant raises mid-run | Recorded as that variant's skip reason; the other still reports |
| Heavy stack absent | `MissingDependencyError` naming the `training` extra (`uv sync --extra training`) |

### Design decisions

- **Unsloth + TRL `SFTTrainer`, pinned.** Unsloth's Qwen3 kernels and 4-bit
  loading are what make a 4B QLoRA fit a free T4 (16 GB) with usable speed;
  TRL's `SFTTrainer` is the standard supervised loop and supports chat-template
  datasets natively. Rejected: vanilla `peft` + `transformers` (slower, more
  VRAM) and axolotl (extra config surface, no notebook-first workflow).
- **QLoRA, not full fine-tuning.** A 4-bit NF4 base plus LoRA adapters trains
  within 16 GB (and the maintainer's 4 GB GTX 1650 stays out of the loop);
  full fine-tuning a 4B model needs tens of GB.
- **Qwen3-4B (Apache 2.0).** Best fine-tunability at this size per the project
  thesis, a permissive license, and a strong multilingual base. The exact
  checkpoint id is pinned in `QLoRAConfig`.
- **The repo ships the harness; the T4 run is manual.** CI and the offline
  suite never train. The GPU deliverable is a LoRA adapter directory plus
  `run.json`; the runbook documents the manual steps.
- **Training data reuses the RAG prompt contract.** The formatter calls
  `build_system_prompt` / `build_user_prompt`, so training and inference cannot
  drift; the manifest records a prompt hash to make drift visible.
- **`qlora.json` is the single source of truth.** The notebook loads it instead
  of hardcoding values, so changing a hyperparameter is editing one document.
- **Filesystem registry over config.** `training/adapter/run.json` mirrors
  `index/metadata.json`; `Settings.adapter` only carries the endpoint, so
  re-training a book needs no configuration churn. Registry ownership is split
  from Phase 4 on purpose: this phase registers adapters for *evaluation*;
  Phase 4's serving registry (merged weights, GGUF, modelfiles) builds on the
  same `run.json` identity — the README roadmap will note the split.
- **No automatic winner; deltas shared with ablation.** The comparison reports
  metrics and deltas and leaves judgement to the maintainer (SDD-0002
  precedent). The delta computation is extracted from `evaluation/ablation.py`
  into a shared helper so both reports use identical semantics.
- **Only the generator changes.** The comparison report records one shared
  `index` + `retrieval` identity and one `generator` identity per variant, so a
  reader can verify that fine-tuned-student + RAG and base + RAG differ in
  exactly one variable.

## Requirements

1. **REQ-TR-001** — WHEN `distiller train <book>` runs for a book with a
   synthesized dataset, THEN the system SHALL format every RAFT example into a
   chat example whose system prompt and document rendering come from the shared
   RAG prompt contract, preserving context order and `[n]` citation indices.
2. **REQ-TR-002** — WHEN formatting a RAFT context, THEN the system SHALL
   resolve its chunk id against `chunks.jsonl` so `<doc>` blocks carry the
   chapter/section/page provenance the generator sees at inference time, and
   SHALL fail with an actionable error when a context chunk id is unknown.
3. **REQ-TR-003** — WHEN preparing the dataset, THEN the system SHALL split
   examples deterministically from the configured seed and validation ratio
   such that no question appears in both splits.
4. **REQ-TR-004** — BEFORE writing training artifacts, THEN the system SHALL
   validate every example and SHALL abort without writing when an example is
   malformed (empty message, no contexts, an answerable answer without a valid
   `[n]`, a negative answer differing from the shared refusal sentence) or when
   the dataset contains no unanswerable examples.
5. **REQ-TR-005** — WHEN preparation completes, THEN the system SHALL write
   `training/train.jsonl`, `training/validation.jsonl` and
   `training/manifest.json` recording counts, split configuration, source hash,
   dataset hash and the prompt-contract hash (so drift between the training
   prompt and the inference prompt is visible).
6. **REQ-TR-006** — WHEN emitting training configuration, THEN the system SHALL
   write `training/qlora.json` from a single `QLoRAConfig` model pinning the
   base model, 4-bit NF4 quantization, LoRA rank/alpha/dropout/target modules,
   learning rate, epochs, sequence length and seed.
7. **REQ-TR-007** — WHEN `distiller train` completes, THEN the system SHALL
   emit a self-contained T4 notebook that installs its stack, loads the dataset
   and `qlora.json`, trains with Unsloth + TRL `SFTTrainer`, saves the LoRA
   adapter, writes `run.json` with the loss history and metadata, and links to
   the runbook.
8. **REQ-TR-008** — WHEN `distiller train <book> --register <dir>` runs, THEN
   the system SHALL validate the adapter directory (`run.json` present and
   valid, book id and base model matching this book's training configuration)
   and copy it to `training/adapter/`, recording base model, dataset hash,
   config hash and training metadata; otherwise it SHALL fail with a friendly
   error without copying.
9. **REQ-TR-009** — WHEN `distiller ask` or `distiller eval` is given
   `--adapter`, THEN the system SHALL answer with the configured adapter
   endpoint and SHALL fail with an actionable message when no adapter is
   registered or the endpoint is unconfigured.
10. **REQ-TR-010** — WHEN `distiller eval` runs, THEN the system SHALL record a
    `generator` block in `eval/report.json` identifying the generator (kind
    `base` or `adapter`, model, adapter provenance) next to the existing
    `index` block; the legacy top-level `model` key SHALL be kept for backward
    compatibility, with `generator.model` as the authoritative field.
11. **REQ-TR-011** — WHEN `distiller train-eval <book>` runs, THEN the system
    SHALL evaluate the base model and the registered adapter over the same
    loaded index, retrieval settings and golden items, record a variant that
    fails as skipped without aborting the other, and write `eval/training.json`
    with shared index/retrieval identity, per-variant generator identity,
    per-variant metrics and deltas versus the base variant.
12. **REQ-TR-012** — WHEN preparation or a comparison finishes, THEN the system
    SHALL print a summary table (counts and artifact paths, or metrics and
    deltas) and `--json` SHALL print the report instead.
13. **REQ-TR-013** — The heavy training stack (torch, unsloth, peft, trl,
    transformers) SHALL live in an optional `training` extra and SHALL be
    loaded through `optional_deps.require(..., extra="training", ...)` when
    repo-side code needs it; `distiller train --check-runtime` SHALL report the
    stack or raise the standard missing-extra error, and the core harness and
    offline suite SHALL run with none of it installed.
14. **REQ-TR-014** — WHEN the offline fake LLM is configured, THEN the system
    SHALL complete prepare, register and comparison end to end without a GPU,
    network or heavy dependencies.
15. **REQ-TR-015** — WHEN a prerequisite artifact is missing, THEN the system
    SHALL fail with an actionable message naming the next command
    (`distiller synth`, `distiller train`, or `--register`).
16. **REQ-TR-016** — The repository SHALL ship `docs/qlora-runbook.md`
    documenting the manual T4 steps (upload, train, download, register,
    compare), linked from the docs index and referenced by the emitted
    notebook.

## Acceptance Criteria

- **AC1** (Req 1) A formatted positive example has a system message equal to
  `build_system_prompt(book.title)`, `<doc id="n" ...>` blocks in RAFT context
  order, and the RAFT target as the assistant message.
- **AC2** (Req 2) `<doc>` blocks carry chapter/section/page provenance from
  `chunks.jsonl`; an unknown context chunk id aborts with the re-index/re-synth
  message.
- **AC3** (Req 3) The same seed and ratio produce byte-identical splits; no
  question appears in both; the validation size is
  `max(1, floor(n * val_ratio))` (and the train split gets the rest).
- **AC4** (Req 4) Malformed examples and a dataset without unanswerable
  examples produce validation errors, and prepare writes no files.
- **AC5** (Req 5) Prepare writes both splits and a manifest whose counts, seed,
  ratio, source hash, dataset hash and prompt hash match the written files.
- **AC6** (Req 6) `qlora.json` round-trips through `QLoRAConfig` and pins
  Qwen3-4B, NF4 4-bit, rank/alpha/dropout, target modules, learning rate,
  epochs, sequence length and seed.
- **AC7** (Req 7) The emitted `.ipynb` parses as JSON and contains install,
  model-loading, training, adapter-saving and report-writing cells that
  reference `qlora.json`, `SFTTrainer`, `save_pretrained`, `run.json` and the
  runbook path.
- **AC8** (Req 8) Registering a valid adapter copies it into `training/adapter/`
  and records its identity; a wrong book id or base model is rejected with a
  friendly error and nothing is copied.
- **AC9** (Req 9) `ask --adapter` / `eval --adapter` use the adapter endpoint;
  without a registration or endpoint config they exit non-zero with an
  actionable message and no traceback.
- **AC10** (Req 10) An adapter eval writes `generator` with `kind: adapter` and
  adapter provenance; a plain eval writes `kind: base`.
- **AC11** (Req 11) `train-eval` writes `eval/training.json` with `base` and
  `adapter` variants over the same item count, one shared `index`/`retrieval`
  identity, per-variant generator identity, and deltas; differing answers yield
  non-zero deltas, and a failing variant is recorded as skipped.
- **AC12** (Req 12) The prepare summary lists counts and artifact paths, the
  comparison table renders both variants with a Δ column, and `--json` output
  equals the written report.
- **AC13** (Req 13) Importing the harness loads none of torch/unsloth/trl/peft/
  transformers; `require_training_stack()` and `train --check-runtime` raise
  the missing-extra error naming the `training` extra.
- **AC14** (Req 14) The offline CLI chain ingest → index → synth → train →
  register → train-eval completes with the fake LLM and hash embedder and
  writes a well-formed comparison report.
- **AC15** (Req 15) A missing RAFT dataset names `distiller synth`; a missing
  adapter registry names `--register`; neither prints a traceback.
- **AC16** (Req 16) `docs/qlora-runbook.md` exists, names the register and
  compare commands, and is linked from `docs/README.md` and referenced by the
  emitted notebook.

## Non-Goals

- Serving, adapter merging and GGUF export (Phase 4).
- Thematic / GraphRAG retrieval (Phase 5).
- DPO/RLHF or any preference optimisation.
- Multi-book training or training several adapters in one run.
- Hyperparameter search or sweeps.
- Executing the T4 job in CI or from the test suite (manual runbook step).
- Full fine-tuning (QLoRA only) and dataset mixing with general instruction
  data.

## File-change Plan

| Action | Path | Purpose |
|--------|------|---------|
| create | `src/distiller/training/__init__.py` | Package exports |
| create | `src/distiller/training/chat.py` | RAFT → chat examples; RAG prompt contract reuse |
| create | `src/distiller/training/dataset.py` | Split, validation, manifest, prepare orchestration |
| create | `src/distiller/training/qlora.py` | `QLoRAConfig` and the training-stack gate |
| create | `src/distiller/training/notebook.py` | Self-contained T4 notebook emission |
| create | `src/distiller/training/registry.py` | `TrainingReport`, adapter register/load |
| create | `src/distiller/training/compare.py` | Base-vs-adapter comparison and deltas |
| modify | `src/distiller/config.py` | `TrainingSettings`, `AdapterSettings`, `Settings` fields |
| modify | `src/distiller/paths.py` | Training artifact paths |
| modify | `src/distiller/llm/__init__.py` | `get_adapter_llm` factory |
| modify | `src/distiller/evaluation/metrics.py` | Extract the shared delta helper |
| modify | `src/distiller/evaluation/ablation.py` | Delegate `metric_deltas` to the shared helper |
| modify | `src/distiller/evaluation/__init__.py` | Export the shared delta helper |
| modify | `src/distiller/cli/context.py` | `build_training_runs`, adapter-aware pipeline builder |
| modify | `src/distiller/cli/main.py` | `train` / `train-eval` commands; `--adapter`; `generator` block |
| modify | `src/distiller/cli/render.py` | `render_training`, `render_comparison` |
| modify | `pyproject.toml` | The `training` optional extra |
| create | `docs/qlora-runbook.md` | Manual T4 steps (upload, train, download, register, compare) |
| modify | `docs/README.md` | Link the runbook |
| create | `tests/unit/test_training_chat.py` | Formatting and provenance tests |
| create | `tests/unit/test_training_dataset.py` | Split, validation, manifest, prepare tests |
| create | `tests/unit/test_training_qlora.py` | Config round-trip and heavy-dependency gate tests |
| create | `tests/unit/test_training_notebook.py` | Notebook and runbook emission tests |
| create | `tests/unit/test_training_registry.py` | Register/load validation tests |
| create | `tests/unit/test_training_compare.py` | Comparison, deltas and skip tests |
| modify | `tests/conftest.py` | Shared RAFT example factory fixture |
| modify | `tests/integration/test_cli_end_to_end.py` | CLI training coverage |
| modify | `README.md`, `AGENTS.md` | Quickstart, roadmap and status updates |

## Test Plan

| AC | Test file | Test name |
|----|-----------|-----------|
| AC1 | `tests/unit/test_training_chat.py` | `test_format_example_matches_the_rag_prompt_contract` |
| AC1 | `tests/unit/test_training_chat.py` | `test_format_negative_targets_the_refusal_sentence` |
| AC2 | `tests/unit/test_training_chat.py` | `test_format_example_renders_context_provenance` |
| AC2 | `tests/unit/test_training_chat.py` | `test_format_example_requires_known_chunks` |
| AC3 | `tests/unit/test_training_dataset.py` | `test_split_is_deterministic_and_leak_free` |
| AC3 | `tests/unit/test_training_dataset.py` | `test_split_honours_the_validation_ratio` |
| AC4 | `tests/unit/test_training_dataset.py` | `test_validation_flags_malformed_examples` |
| AC4 | `tests/unit/test_training_dataset.py` | `test_validation_requires_refusal_negatives` |
| AC4 | `tests/unit/test_training_dataset.py` | `test_prepare_writes_nothing_when_validation_fails` |
| AC5 | `tests/unit/test_training_dataset.py` | `test_prepare_writes_splits_and_manifest` |
| AC5 | `tests/integration/test_cli_end_to_end.py` | `test_train_command_writes_training_artifacts` |
| AC6 | `tests/unit/test_training_qlora.py` | `test_qlora_config_round_trips_with_pinned_defaults` |
| AC6 | `tests/unit/test_training_dataset.py` | `test_prepare_writes_the_qlora_config` |
| AC7 | `tests/unit/test_training_notebook.py` | `test_emit_notebook_is_valid_json_with_expected_cells` |
| AC8 | `tests/unit/test_training_registry.py` | `test_register_adapter_copies_and_validates` |
| AC8 | `tests/unit/test_training_registry.py` | `test_register_rejects_wrong_book_or_base_model` |
| AC9 | `tests/integration/test_cli_end_to_end.py` | `test_ask_and_eval_require_a_registered_adapter` |
| AC9 | `tests/integration/test_cli_end_to_end.py` | `test_ask_and_eval_select_a_registered_adapter` |
| AC10 | `tests/integration/test_cli_end_to_end.py` | `test_eval_records_generator_identity` |
| AC11 | `tests/unit/test_training_compare.py` | `test_run_comparison_computes_deltas_against_base` |
| AC11 | `tests/unit/test_training_compare.py` | `test_run_comparison_records_a_failing_variant_as_skipped` |
| AC11 | `tests/integration/test_cli_end_to_end.py` | `test_train_eval_command_end_to_end` |
| AC12 | `tests/unit/test_training_compare.py` | `test_comparison_round_trips_through_json` |
| AC12 | `tests/integration/test_cli_end_to_end.py` | `test_train_eval_command_end_to_end` |
| AC13 | `tests/unit/test_training_qlora.py` | `test_require_training_stack_names_the_training_extra` |
| AC13 | `tests/unit/test_training_qlora.py` | `test_harness_imports_without_the_training_stack` |
| AC13 | `tests/integration/test_cli_end_to_end.py` | `test_train_check_runtime_requires_the_extra` |
| AC14 | `tests/integration/test_cli_end_to_end.py` | `test_training_pipeline_end_to_end` |
| AC15 | `tests/integration/test_cli_end_to_end.py` | `test_train_requires_dataset` |
| AC15 | `tests/integration/test_cli_end_to_end.py` | `test_ask_and_eval_require_a_registered_adapter` |
| AC16 | `tests/unit/test_training_notebook.py` | `test_qlora_runbook_documents_the_manual_steps` |

## Open Questions

Decided at approval (2026-09-30):

- **Hyperparameters**: rank 16, alpha 32, dropout 0.0, all seven attention/MLP
  projections, lr 2e-4, 3 epochs, `max_seq_length` 4096, fp16 compute (Turing T4
  has no bf16). Checkpoint: `Qwen/Qwen3-4B` (fall back to Unsloth's mirror only
  if the emitted notebook's loader requires it).
- **Emitted artifact**: a notebook (Colab/Kaggle-friendly). A plain script stays
  a future option behind `--emit`.
- **Which metric decides "better"**: `contains_rate` primary with
  `refusal_accuracy` and `citation_coverage` as guards; no automatic winner.
- **Adapter registry**: filesystem-driven. A dataset-hash mismatch on register
  is recorded in the report, not fatal.
- **Positives-only datasets**: hard validation error.
- **`--check-runtime`**: stays on `train`.

## Outcomes

- **Implemented in**: `de8122e` (training extra), `ab4105f` (shared delta helper), `26cb2a4` (training harness), `504befa` (CLI), `a943daf` (runbook + docs)
- **Spec archived**: 2026-10-01
- **Post-mortem notes**: three deviations surfaced during implementation and were
  accepted: a new `TrainingError(DistillerError)` for the failure domain,
  `pipeline_model_name` moved from `cli/main.py` to `cli/context.py` to break a
  circular import, and `render_comparison` split into metrics + deltas tables for
  80-column readability. The register→compare chain was validated without a GPU
  by registering a hand-built fake adapter directory (real `run.json` schema), so
  the only unverified step is the T4 training run itself — covered by the
  runbook, not by CI.
