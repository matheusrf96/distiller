# QLoRA runbook — manual T4 training

Phase 3 ships the **harness**; the GPU run is manual. CI, the test suite and
`make check` never train a model, so this page is the only place the T4 steps
live. The repository prepares the data and the configuration, then validates
the adapter that comes back.

```
distiller train <book>              # prepare train.jsonl + validation.jsonl + qlora.json + notebook
       │
       ▼  upload training/ to Colab/Kaggle (T4)
   train_t4.ipynb                   # install → train → save adapter + run.json
       │
       ▼  download adapter/
distiller train <book> --register <dir>   # validate + install into training/adapter/
       │
       ▼
distiller train-eval <book>         # base vs adapter on the same golden set
```

## 0. Prerequisites

```bash
uv run distiller ingest book.epub
uv run distiller index <book>
uv run distiller synth <book> --max-chunks 100     # needs an LLM (DISTILLER_LLM__*)
uv run distiller train <book>                      # writes artifacts/<book>/training/
```

`training/` now contains:

| File | Purpose |
|------|---------|
| `train.jsonl` | chat-formatted training split (system + user + assistant) |
| `validation.jsonl` | held-out split, same format |
| `manifest.json` | counts, split seed/ratio, source/dataset/prompt hashes |
| `qlora.json` | pinned QLoRA hyperparameters (the single source of truth) |
| `train_t4.ipynb` | the notebook you run on the GPU |

The training stack is an optional extra and is deliberately **not** part of
`make install-all` (Unsloth pins its own torch/CUDA build):

```bash
uv sync --extra training
uv run distiller train <book> --check-runtime   # prints versions, or the missing-extra error
```

## 1. Upload

Upload the whole `training/` folder to Google Colab or Kaggle (both offer a
free Tesla T4, 16 GB). Keep the files flat: the notebook reads `qlora.json`,
`manifest.json`, `train.jsonl` and `validation.jsonl` from its working
directory.

## 2. Install

Open `train_t4.ipynb` and run the first cell. It installs the stack with
`pip install unsloth trl peft transformers bitsandbytes datasets`. Restart the
runtime if the environment asks for it.

## 3. Train

Run the remaining cells in order. The notebook:

1. loads `qlora.json` and the two splits;
2. loads `Qwen/Qwen3-4B` in 4-bit NF4 with Unsloth;
3. attaches the LoRA adapter (rank 16, alpha 32, all attention/MLP projections);
4. trains with TRL's `SFTTrainer` (fp16, 3 epochs, `max_seq_length` 4096);
5. saves the adapter to `adapter/`;
6. writes `adapter/run.json` with the identity hashes, loss history and GPU.

Change any hyperparameter by editing `qlora.json` before re-running — the
notebook never hardcodes values.

## 4. Download

Download the `adapter/` folder from the notebook's file browser. It must
contain:

- `adapter_config.json`
- `adapter_model.safetensors`
- `run.json`

## 5. Register

Back on the workstation:

```bash
uv run distiller train <book> --register ./adapter
```

Registration validates that `run.json` is present and valid, that the adapter
belongs to this book and that its base model matches this book's
`training/qlora.json`, then copies the folder to `training/adapter/`. A
dataset-hash mismatch (the adapter was trained on an older split) is recorded
in the registered `run.json` as `"dataset_hash_matches": false` and warned
about, but does not block registration. A wrong book or base model fails
without copying anything.

## 6. Compare

Serve the adapter with any OpenAI-compatible server (llama.cpp, vLLM, Ollama)
and point distiller at it:

```bash
export DISTILLER_ADAPTER__MODEL=<served-model-name>
export DISTILLER_ADAPTER__BASE_URL=http://localhost:8000/v1
# offline smoke test instead of a real server:
# export DISTILLER_ADAPTER__MODEL=fake

uv run distiller train-eval <book>          # base vs adapter, writes eval/training.json
uv run distiller eval <book> --adapter      # full eval report with the adapter generator
uv run distiller ask <book> "..." --adapter
```

`train-eval` evaluates both generators over the **same** index, retrieval
settings and golden items, so the only variable is the generator. The report
records one shared `index`/`retrieval` identity, one `generator` identity per
variant and metric deltas versus the base. A variant that raises mid-run is
recorded as skipped and the other still reports. `contains_rate` is the primary
signal; `refusal_accuracy` and `citation_coverage` are the guards. No automatic
winner is declared.

Once the adapter is registered, the T4 notebook can also merge it and export a
Q4_K_M GGUF for local serving. See [gguf-runbook.md](gguf-runbook.md) for the
download, `distiller gguf register`, the Ollama/llama.cpp commands and the
`eval --gguf` / `train-eval --gguf` comparison.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `QLoRA training requires the optional dependency 'torch'` | `uv sync --extra training` |
| `No RAFT dataset ... Run distiller synth` | run `distiller synth <book>` |
| `Training context '...' is not in chunks.jsonl` | re-run `distiller index <book>` and `distiller synth <book>` |
| `the dataset has no unanswerable examples` | raise `--negative-ratio` and re-run `distiller synth <book> --regenerate` |
| `No registered adapter ... --register` | download `adapter/` from the T4 and run `distiller train <book> --register <dir>` |
| `No adapter endpoint configured` | set `DISTILLER_ADAPTER__MODEL` / `DISTILLER_ADAPTER__BASE_URL` |

Related: [README](../README.md) (pipeline overview),
[gguf-runbook.md](gguf-runbook.md) (Phase 4: export, serve, evaluate),
[CONTRIBUTING](CONTRIBUTING.md) (conventions),
[specs/SDD-0004](../specs/active/SDD-0004-qwen3-qlora-training.md)
(the approved spec).
