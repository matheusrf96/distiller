---
id: SDD-0005
status: approved
supersedes:
owner: matheus
created: 2026-10-01
---

# GGUF Export + Local Serving (Phase 4)

## Problem

Phase 3 produces a LoRA adapter, but nothing runs on the maintainer's machine:
the 4 GB GTX 1650 cannot train, and it cannot hold a full-precision 4B model. The
distilled model becomes usable only as a quantized GGUF served with partial GPU
offload — and nothing in the repository validates the downloaded GGUF, configures
llama.cpp or Ollama for that machine, or records which file produced a set of
evaluation numbers. This spec closes the loop: register the Q4_K_M export, emit
the exact serving commands, and evaluate the served model against the base and
adapter baselines over the same golden set.

## Context

- **Phase 3 output**: `training/adapter/` (LoRA weights plus `run.json`,
  `TrainingReport`) and the base-vs-adapter comparison in `eval/training.json`
  (SDD-0004). The T4 notebook `training/train_t4.ipynb` is the manual GPU step.
- **Merging and quantization happen on the training machine.** Unsloth's
  `save_pretrained_gguf` merges the adapter into the base model, converts and
  quantizes to Q4_K_M in one call. The repository never merges or quantizes: its
  job starts at the downloaded `.gguf`.
- **Serving plugs into the existing LLM abstraction**: `src/distiller/llm/`
  speaks OpenAI-compatible HTTP, and both llama-server
  (`http://localhost:8080/v1`) and Ollama (`http://localhost:11434/v1`) expose
  that API. A served GGUF is therefore another client factory, exactly like
  `get_adapter_llm` (SDD-0004).
- **Prompt contract**: `src/distiller/rag/prompts.py::build_system_prompt`. The
  Modelfile's SYSTEM prompt must be the same contract the pipeline sends, so a
  served model is prompted identically to the evaluated one.
- **Report identity**: `eval/report.json` already carries an `index` block
  (SDD-0001) and a `generator` block with kind `base`/`adapter` (SDD-0004); this
  phase adds kind `gguf` and the registered file identity.
- **Optional dependencies**: `optional_deps.require` gates heavy extras
  (SDD-0004). Phase 4 adds **no new dependency**: the GGUF reader is plain
  Python, and llama.cpp/Ollama are external binaries the repo never bundles or
  launches.
- **Related specs**: builds on SDD-0004 (adapter registry, comparison); Phase 5
  (thematic layer) consumes the served model.

### Data model deltas

| Model / field | Kind | Notes |
|---|---|---|
| `GgufMetadata` | pydantic (`DomainModel`) | Parsed header: version, architecture, model name, file type/quantization, tensor count, parameter count |
| `GgufReport` | pydantic (`DomainModel`) | `gguf.json`: metadata, file hash, size, source name, served model name, registration timestamp |
| `ServingProfile` | pydantic (`DomainModel`) | Pinned Ollama/llama.cpp parameters for 4 GB partial offload (temperature, context, GPU layers, slots, host, port, stop tokens) |
| `GgufSettings` | pydantic settings | Root `Settings.gguf`: endpoint base URL, key, model, timeout |
| `GeneratorVariant` | pydantic (extend) | `kind` gains `gguf`; new `gguf` provenance block |

### Module deltas

```
src/distiller/gguf/
├── __init__.py     # exports
├── reader.py       # dependency-free GGUF header/metadata parser
├── registry.py     # GgufReport, register_gguf, load_gguf_report
└── serving.py      # ServingProfile, Modelfile and serve.sh emission
```

### Configuration

`GgufSettings(base_url=None, api_key=None, model=None, timeout=120.0)` under
`Settings.gguf`. `base_url=None` means "no endpoint configured" and produces the
actionable error naming `DISTILLER_GGUF__BASE_URL` (the two intended values are
Ollama's `http://localhost:11434/v1` and llama-server's
`http://localhost:8080/v1`). `model=None` means "use the registered local name"
(`distiller-<book-id>`, the same name the emitted commands use); the sentinel
`"fake"` selects the offline client, like `LLMSettings` and `AdapterSettings`.
Environment: `DISTILLER_GGUF__BASE_URL`, `DISTILLER_GGUF__API_KEY`,
`DISTILLER_GGUF__MODEL`, `DISTILLER_GGUF__TIMEOUT`.

The client factory takes the resolved name explicitly:
`get_gguf_llm(settings.gguf, model_name)` where the CLI resolves the registered
`distiller-<book-id>` (or the `DISTILLER_GGUF__MODEL` override) through the
registry loader before building the pipeline.

Connection failures are converted once, in the shared client:
`openai_compat.py` wraps `openai.APIConnectionError` in `ConfigurationError`
naming the endpoint and the local-server hint (`ollama serve`, or the emitted
`serve.sh`), so every caller gets a friendly message instead of a traceback.

`ServingProfile` carries the pinned 4 GB defaults (temperature 0.1, `num_ctx`
4096, `num_gpu` 20 of Qwen3-4B's 36 layers, one parallel slot,
`127.0.0.1:8080`, stop tokens `<|im_end|>` and `<|endoftext|>`). It is not
env-configurable in this phase; changing it is editing one model.

`pyproject.toml` is untouched: no new dependency, no new extra.

### CLI

- `distiller gguf register <book> <file> [--json]` — validate, copy to
  `training/gguf/model.gguf`, write `gguf.json`, emit `Modelfile` + `serve.sh`,
  print the summary.
- `distiller gguf serve <book>` — print the registered model path, the Modelfile
  and the exact Ollama/llama.cpp commands; launches nothing.
- `distiller ask ... --gguf` / `distiller eval ... --gguf` — use the configured
  GGUF endpoint with the registered identity.
- `distiller train-eval <book> --gguf` — add the GGUF variant to the existing
  base-vs-adapter comparison.

### Artifact layout

```
artifacts/<book>/
├── training/
│   ├── adapter/               # Phase 3 (unchanged)
│   └── gguf/
│       ├── model.gguf         # registered copy of the Q4_K_M export
│       ├── gguf.json          # GgufReport: metadata + hash + size + identity
│       ├── Modelfile          # Ollama: SYSTEM = build_system_prompt(book title)
│       └── serve.sh           # llama-server invocation + Ollama commands
└── eval/
    ├── report.json            # generator.kind == "gguf" + GGUF identity
    └── training.json          # base + adapter + gguf variants
```

### Failure modes

| Failure | Behaviour |
|---|---|
| File is not GGUF (bad magic) | `GGUFError` → friendly CLI error naming the file |
| Truncated header or metadata | `GGUFError` naming the file and the truncated section |
| Unsupported GGUF version | `GGUFError` naming the version |
| Book not ingested | Friendly error: run `distiller ingest` first |
| No registered GGUF | Friendly error: run `distiller gguf register <book> <file>` |
| GGUF endpoint unconfigured (`DISTILLER_GGUF__BASE_URL` unset) | Configuration hint naming `DISTILLER_GGUF__BASE_URL` with the Ollama/llama-server example values |
| Endpoint unreachable | `ConfigurationError` from the shared client naming the endpoint and the local-server hint (`ollama serve` / the emitted `serve.sh`) → friendly CLI error |
| `--adapter` and `--gguf` together | Rejected: choose one generator |
| One comparison variant raises | Recorded as skipped; the other variants still report (SDD-0004 behaviour) |
| Re-registration | Replaces the previous GGUF and report |

### Design decisions

- **Merging and quantizing stay on the training machine.** Unsloth's
  `save_pretrained_gguf` merges the adapter, converts and quantizes in one step
  on the free T4; doing it in-repo would add torch plus a llama.cpp build, none
  of it testable offline. The repo's job is everything after the download:
  validate, register, configure, evaluate.
- **Dependency-free GGUF reader.** GGUF is a small, documented binary container;
  parsing its header ourselves keeps the core install light (no `gguf` package,
  no `llama-cpp-python`) and lets tests build tiny synthetic files in
  `tmp_path`. Rejected: the `gguf` PyPI package (a new runtime dependency for a
  header read) and shelling out to llama.cpp tools (binaries absent in CI).
- **No process supervision.** The repo never launches llama-server or Ollama: it
  emits the exact commands and a Modelfile, and the runbook documents them. This
  keeps the feature fully offline-testable and avoids platform-specific process
  management. Rejected: subprocess management, Docker Compose, systemd units.
- **Serving is just another OpenAI-compatible endpoint.** `DISTILLER_GGUF__*`
  mirrors `DISTILLER_ADAPTER__*`; llama-server and Ollama already speak the
  protocol `LLMClient` expects, so the phase adds a factory and a report kind,
  not a client. The same served model name (`distiller-<book-id>`) is the Ollama
  model name and the llama-server `--alias`, so one setting points at either
  backend.
- **The GGUF registry builds on the adapter identity.** `gguf.json` records the
  file hash, quantization and the registered adapter's base model/dataset hash
  when available, so a served file is traceable to the training run that
  produced it (SDD-0004's `run.json`).
- **One variable per comparison, three ways.** `train-eval --gguf` shares the
  index, retrieval settings and golden items and adds a `gguf` variant with
  deltas versus base, so base vs adapter vs GGUF-Q4 line up in one table
  (SDD-0004 precedent).
- **Modelfile SYSTEM = the RAG prompt contract.** The Modelfile embeds
  `build_system_prompt(book.title)`; serving the model with any other system
  prompt would silently change behaviour between evaluation and use.
- **Q4_K_M only.** One pinned quantization keeps the export, the emitted
  commands and the runbook single-path; Q5/Q8 are a future knob.
- **A pinned 4 GB profile, not auto-detection.** `num_gpu 20` / `-ngl 20`,
  `num_ctx 4096`, one slot; the runbook explains how to raise or lower them.
  Auto-detection would need the binaries running, which the offline suite cannot
  exercise.
- **Registration copies the file.** `training/gguf/model.gguf` makes the
  artifacts directory self-contained and the report's hash is computed over the
  registered copy; the original file name is recorded in `gguf.json`.

## Requirements

1. **REQ-GG-001** — WHEN the system reads a GGUF file, THEN it SHALL parse the
   binary header, key-value metadata and tensor-info table with no third-party
   dependency, extracting at least the GGUF version, architecture, model name,
   file type (quantization), tensor count and parameter count (the sum of the
   tensor dimensions).
2. **REQ-GG-002** — IF a file is not a GGUF file, uses an unsupported version,
   or is truncated inside the header or metadata, THEN the system SHALL raise a
   domain error naming the file and the specific problem.
3. **REQ-GG-003** — WHEN `distiller gguf register <book> <file>` runs for an
   ingested book, THEN the system SHALL validate the file, copy it into
   `artifacts/<book>/training/gguf/`, write `gguf.json` with the parsed
   metadata, file hash, size, source name and registration timestamp, and print
   a summary table (`--json` SHALL print the report instead); re-registering
   SHALL replace the previous copy and report.
4. **REQ-GG-004** — IF the GGUF file is invalid or the book has no ingested
   artifacts, THEN the system SHALL reject the registration with an actionable
   message and SHALL NOT copy anything.
5. **REQ-GG-005** — WHEN `distiller ask` or `distiller eval` is given `--gguf`,
   THEN the system SHALL answer or evaluate through the configured GGUF endpoint
   using the registered local model name, SHALL fail with an actionable message
   when no GGUF is registered or `DISTILLER_GGUF__BASE_URL` is unset, and SHALL
   reject `--gguf` combined with `--adapter` with a friendly error.
6. **REQ-GG-006** — WHEN `distiller eval` writes `eval/report.json`, THEN the
   `generator` block SHALL support kind `gguf` carrying the registered GGUF
   identity (file hash, quantization, architecture, model name) next to the
   existing `base` and `adapter` kinds.
7. **REQ-GG-007** — WHEN registration completes, THEN the system SHALL emit
   `training/gguf/Modelfile` whose SYSTEM prompt is
   `build_system_prompt(book.title)` and whose parameters set temperature,
   context length, GPU offload and Qwen3 stop tokens for 4 GB VRAM, together
   with the `ollama create` / `ollama run` commands.
8. **REQ-GG-008** — WHEN registration completes, THEN the system SHALL emit
   `training/gguf/serve.sh` containing the exact `llama-server` invocation
   (model, alias, `-ngl`, `-c`, `-np`, `--host`, `--port`) for partial GPU
   offload on a 4 GB card and SHALL state the OpenAI-compatible endpoint it
   serves.
9. **REQ-GG-009** — WHEN `distiller gguf serve <book>` runs, THEN the system
   SHALL print the registered model path, the Modelfile, the Ollama commands and
   the llama-server command without launching any process.
10. **REQ-GG-010** — WHEN `distiller train-eval <book> --gguf` runs, THEN the
    comparison SHALL include a `gguf` variant alongside base and adapter, over
    the same index, retrieval settings and golden items, with deltas versus the
    base variant in `eval/training.json`.
11. **REQ-GG-011** — WHEN `distiller train` emits the T4 notebook, THEN the
    notebook SHALL include a GGUF export cell that merges the adapter and
    quantizes to Q4_K_M (`save_pretrained_gguf`) and SHALL reference
    `docs/gguf-runbook.md` for the download and registration steps.
12. **REQ-GG-012** — The repository SHALL ship `docs/gguf-runbook.md`
    documenting the Colab export, download, `distiller gguf register`, serving
    with Ollama and llama.cpp, and the evaluation against the base and adapter
    baselines; the runbook SHALL be linked from the docs index.
13. **REQ-GG-013** — The GGUF feature SHALL require no new runtime dependency
    and no heavy extra, and WHEN the offline fake LLM is configured, THEN
    register → emit → eval SHALL complete without a GPU, network, llama.cpp
    binary or running server.
14. **REQ-GG-014** — WHEN a prerequisite is missing or the configured endpoint
    is unreachable, THEN the system SHALL fail with a friendly CLI error naming
    the next command (`distiller ingest`, `distiller gguf register`, `ollama
    serve` or the emitted `serve.sh`) and SHALL NOT print a traceback.

## Acceptance Criteria

- **AC1** (Req 1) A synthetic GGUF (header, metadata keys and tensor descriptors)
  built in `tmp_path` parses to the expected version, architecture, model name,
  quantization label (Q4_K_M from file type 15), tensor count and parameter
  count.
- **AC2** (Req 2) Non-GGUF bytes, an unsupported version and a truncated header
  each raise a `GGUFError` whose message names the file and the problem.
- **AC3** (Req 3) Registering a valid GGUF copies it to `training/gguf/model.gguf`,
  writes `gguf.json` whose file hash, size and metadata match the source, prints
  the summary, and `--json` prints the same report; re-registering replaces the
  previous file and report.
- **AC4** (Req 4) An invalid GGUF and an unknown book id are rejected with an
  actionable message and nothing is copied.
- **AC5** (Req 5) `ask --gguf` and `eval --gguf` answer through the configured
  GGUF endpoint and local model name; without a registration or with
  `DISTILLER_GGUF__BASE_URL` unset they exit non-zero with an actionable message
  and no traceback; `--adapter` together with `--gguf` is rejected with a
  friendly error.
- **AC6** (Req 6) A `--gguf` eval writes `generator.kind == "gguf"` with the
  registered file hash and quantization, while a plain eval keeps
  `kind: "base"`.
- **AC7** (Req 7) The emitted Modelfile's SYSTEM block equals
  `build_system_prompt(book.title)`, carries temperature/`num_ctx`/`num_gpu`/stop
  parameters, and the printed commands include `ollama create` and `ollama run`.
- **AC8** (Req 8) `serve.sh` contains a `llama-server` command with `-ngl`, `-c`,
  `-np`, `--host` and `--port`, and names the `http://localhost:8080/v1`
  endpoint.
- **AC9** (Req 9) `gguf serve <book>` prints the model path, the Modelfile path,
  the Ollama commands and the llama-server command without launching a process.
- **AC10** (Req 10) `train-eval --gguf` writes `eval/training.json` with `base`,
  `adapter` and `gguf` variants over the same item count and one shared
  `index`/`retrieval` identity, with GGUF deltas versus base.
- **AC11** (Req 11) The emitted notebook parses as JSON and contains a
  `save_pretrained_gguf` cell with `q4_k_m` and the `docs/gguf-runbook.md`
  reference.
- **AC12** (Req 12) `docs/gguf-runbook.md` exists, names the register, serve and
  evaluate commands, and is linked from `docs/README.md`.
- **AC13** (Req 13) The offline chain ingest → index → synth → train → register
  GGUF → eval `--gguf` completes with the fake LLM and no GPU, network, binaries
  or new dependency; importing `distiller.gguf` loads no third-party GGUF
  library.
- **AC14** (Req 14) A missing registration, a missing book and an unreachable
  endpoint each produce an actionable message naming the next command
  (`distiller gguf register`, `distiller ingest`, `ollama serve`) with no
  traceback.

## Non-Goals

- Local model merging or quantization (training-machine step, documented in the
  runbook).
- Process supervision (starting/stopping llama-server or Ollama), Docker, or
  systemd units.
- Quantizations other than Q4_K_M (Q5_K_M/Q8_0 are future).
- Vision/multimodal models, mobile/edge deployment, and throughput benchmarking
  beyond the existing golden-set eval.
- Fine-tuning again (Phase 3) or the thematic layer (Phase 5).
- Downloading models or integrating with the Hugging Face Hub (the user
  downloads the GGUF manually).
- Comparing several quantizations or registering more than one GGUF per book.
- Serving the raw adapter (Phase 3 already covers adapter endpoints via
  `DISTILLER_ADAPTER__*`).

## File-change Plan

| Action | Path | Purpose |
|--------|------|---------|
| create | `src/distiller/gguf/__init__.py` | Package exports |
| create | `src/distiller/gguf/reader.py` | Dependency-free GGUF header/metadata parser |
| create | `src/distiller/gguf/registry.py` | `GgufReport`, `register_gguf`, `load_gguf_report` |
| create | `src/distiller/gguf/serving.py` | `ServingProfile`, Modelfile and serve.sh emission |
| modify | `src/distiller/exceptions.py` | `GGUFError(DistillerError)` failure domain |
| modify | `src/distiller/config.py` | `GgufSettings` + `Settings.gguf` |
| modify | `src/distiller/paths.py` | `gguf_dir`, `gguf_file`, `gguf_report_json`, `gguf_modelfile`, `gguf_serve_script` |
| modify | `src/distiller/llm/__init__.py` | `get_gguf_llm` factory |
| modify | `src/distiller/llm/openai_compat.py` | Wrap `APIConnectionError` in `ConfigurationError` with the local-server hint |
| modify | `src/distiller/training/compare.py` | `gguf` generator kind + provenance field |
| modify | `src/distiller/training/notebook.py` | Q4_K_M export cell + GGUF runbook link |
| modify | `src/distiller/cli/context.py` | GGUF register/serve/load helpers; GGUF-aware runs |
| modify | `src/distiller/cli/main.py` | `gguf` sub-app; `--gguf` on `ask`/`eval`/`train-eval`; generator block |
| modify | `src/distiller/cli/render.py` | `render_gguf_registration`, `render_gguf_serving` |
| create | `docs/gguf-runbook.md` | Manual export, download, register, serve, evaluate steps |
| modify | `docs/README.md` | Link the GGUF runbook |
| modify | `docs/qlora-runbook.md` | Cross-link the GGUF runbook |
| modify | `README.md`, `AGENTS.md` | Quickstart, artifact layout, roadmap, package map |
| create | `tests/unit/test_gguf_reader.py` | Parser tests |
| create | `tests/unit/test_gguf_registry.py` | Registration tests |
| create | `tests/unit/test_gguf_serving.py` | Modelfile, serve.sh and runbook tests |
| create | `tests/unit/test_gguf_compare.py` | GGUF generator variant and deltas tests |
| modify | `tests/unit/test_training_notebook.py` | GGUF export cell test |
| modify | `tests/unit/test_ingest_and_llm_errors.py` | Connection-error → `ConfigurationError` test |
| modify | `tests/conftest.py` | `gguf_factory` fixture |
| modify | `tests/integration/test_cli_end_to_end.py` | CLI GGUF coverage |

## Test Plan

| AC | Test file | Test name |
|----|-----------|-----------|
| AC1 | `tests/unit/test_gguf_reader.py` | `test_read_metadata_extracts_identity_and_counts` |
| AC2 | `tests/unit/test_gguf_reader.py` | `test_rejects_non_gguf_bad_version_and_truncated_files` |
| AC3 | `tests/unit/test_gguf_registry.py` | `test_register_copies_and_writes_the_report` |
| AC3 | `tests/integration/test_cli_end_to_end.py` | `test_gguf_register_command_end_to_end` |
| AC4 | `tests/unit/test_gguf_registry.py` | `test_register_rejects_invalid_file_without_copying` |
| AC4 | `tests/integration/test_cli_end_to_end.py` | `test_gguf_register_requires_an_ingested_book` |
| AC5 | `tests/integration/test_cli_end_to_end.py` | `test_ask_and_eval_select_a_registered_gguf` |
| AC5 | `tests/integration/test_cli_end_to_end.py` | `test_ask_and_eval_require_a_registered_gguf` |
| AC6 | `tests/integration/test_cli_end_to_end.py` | `test_eval_records_the_gguf_generator_identity` |
| AC7 | `tests/unit/test_gguf_serving.py` | `test_modelfile_uses_the_rag_prompt_and_serving_parameters` |
| AC8 | `tests/unit/test_gguf_serving.py` | `test_serve_script_uses_partial_offload_for_four_gb` |
| AC9 | `tests/integration/test_cli_end_to_end.py` | `test_gguf_serve_prints_commands_without_launching` |
| AC10 | `tests/unit/test_gguf_compare.py` | `test_comparison_records_a_gguf_variant_with_deltas` |
| AC10 | `tests/integration/test_cli_end_to_end.py` | `test_train_eval_with_gguf_command_end_to_end` |
| AC11 | `tests/unit/test_training_notebook.py` | `test_emit_notebook_exports_a_q4_k_m_gguf` |
| AC12 | `tests/unit/test_gguf_serving.py` | `test_gguf_runbook_documents_the_serving_steps` |
| AC13 | `tests/integration/test_cli_end_to_end.py` | `test_gguf_serving_pipeline_end_to_end` |
| AC14 | `tests/integration/test_cli_end_to_end.py` | `test_gguf_commands_report_errors_without_tracebacks` |
| AC14 | `tests/integration/test_cli_end_to_end.py` | `test_ask_and_eval_require_a_registered_gguf` |
| AC14 | `tests/unit/test_ingest_and_llm_errors.py` | `test_connection_errors_become_configuration_errors` |

## Open Questions

Decided at approval (2026-10-01):

- **CLI shape**: a Typer sub-app — `distiller gguf register` / `distiller gguf
  serve` — leaving room for future `verify`/`list` verbs.
- **`num_gpu` default for 4 GB**: pinned `num_gpu 20` of Qwen3-4B's 36 layers
  with `num_ctx 4096` and one parallel slot; tunable in a later phase.
- **`--gguf` endpoint source**: a dedicated `DISTILLER_GGUF__BASE_URL` with no
  default (set it to Ollama's `http://localhost:11434/v1` or llama-server's
  `http://localhost:8080/v1`), leaving `DISTILLER_LLM__BASE_URL` alone.
- **Emission location**: `training/gguf/`, next to the model and `gguf.json`.
- **Comparison continuity**: both `eval --gguf` and the three-way
  `train-eval --gguf`.
- **Served model name**: deterministic `distiller-<book-id>`, overridable via
  `DISTILLER_GGUF__MODEL`.

## Outcomes

<!-- Filled on archival. -->

- **Implemented in**: <commit SHAs>
- **Spec archived**: <date>
- **Post-mortem notes**: <optional>
