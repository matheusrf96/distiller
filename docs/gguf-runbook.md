# GGUF runbook — export, serve and evaluate (Phase 4)

Phase 3 produces a LoRA adapter; Phase 4 makes it runnable on a 4 GB machine.
Merging and quantization happen on the training machine (the free T4): the
repository never builds llama.cpp, loads torch or launches a server. Its job
starts at the downloaded `.gguf`: validate it, register it, emit the exact
serving commands and evaluate the served model.

```
distiller train <book>                      # Phase 3: splits + qlora.json + notebook
       │
       ▼  run train_t4.ipynb on a T4        # trains the adapter and exports a Q4_K_M GGUF
   gguf/  (download from the notebook)
       │
       ▼  download the .gguf file
distiller gguf register <book> <file.gguf>  # validate + copy + gguf.json + Modelfile + serve.sh
       │
       ▼
serve.sh (llama.cpp) or ollama create -f Modelfile
       │
       ▼
DISTILLER_GGUF__BASE_URL=... distiller eval <book> --gguf
```

## 0. Prerequisites

```bash
uv run distiller ingest book.epub
uv run distiller index <book>
uv run distiller synth <book> --max-chunks 100     # needs an LLM (DISTILLER_LLM__*)
uv run distiller train <book>                      # Phase 3 artifacts + notebook
```

The Phase 3 steps (upload, train, register the adapter) are documented in
[qlora-runbook.md](qlora-runbook.md); this page continues from the exported
GGUF file.

## 1. Export on the T4 (Colab/Kaggle)

Open `training/train_t4.ipynb` and run every cell. The last cell merges the
LoRA adapter into the base model and quantizes it in one call:

```python
model.save_pretrained_gguf("gguf", tokenizer, quantization_method="q4_k_m")
```

Q4_K_M is the only pinned quantization: it fits a 4 GB card with partial
offload and keeps the commands single-path.

## 2. Download

Download the exported file from the notebook's file browser. Unsloth writes it
under the output directory chosen in the cell (for example
`gguf/model-Q4_K_M.gguf`). Copy it to the workstation, e.g.
`~/Downloads/model-Q4_K_M.gguf`.

## 3. Register

```bash
uv run distiller gguf register <book> ~/Downloads/model-Q4_K_M.gguf
```

Registration is validate-then-copy:

1. parses the GGUF header, metadata and tensor table (no third-party library);
2. rejects a bad magic, an unsupported version or a truncated file, naming the
   problem — nothing is copied on failure;
3. copies the file to `artifacts/<book>/training/gguf/model.gguf`;
4. writes `gguf.json` with the file hash (computed over the registered copy),
   size, parsed metadata, the source file name, the registration timestamp and
   the registered adapter identity when `training/adapter/run.json` exists;
5. emits `Modelfile` and `serve.sh`.

`--json` prints the report instead of the summary table. Re-running the
command replaces the previous model, report, Modelfile and script. Use
`DISTILLER_GGUF__MODEL` to override the served name (default
`distiller-<book-id>`).

## 4. Serve

Both backends expose the OpenAI-compatible API `distiller` already speaks, and
both use the same served model name. Pick one.

### Ollama

```bash
cd artifacts/<book>/training/gguf
ollama create distiller-<book-id> -f Modelfile
ollama run distiller-<book-id>
```

The Modelfile's SYSTEM block is exactly
`build_system_prompt(book.title)` — the same contract the pipeline sends — and
pins temperature 0.1, `num_ctx` 4096, `num_gpu` 20 of Qwen3-4B's 36 layers and
the Qwen3 stop tokens.

### llama.cpp (partial GPU offload)

```bash
uv run distiller gguf serve <book>          # prints the files and commands, launches nothing
./artifacts/<book>/training/gguf/serve.sh   # or run the emitted script
```

`serve.sh` runs the exact command for a 4 GB card:

```bash
llama-server -m "$script_dir/model.gguf" --alias distiller-<book-id> \
    -ngl 20 -c 4096 -np 1 --host 127.0.0.1 --port 8080
```

which serves `http://localhost:8080/v1`. Raise or lower `-ngl`/`num_gpu`
(and `num_ctx`) if the card fits more or less; the runbook pins one profile
instead of auto-detecting hardware.

## 5. Evaluate

Point distiller at the running server and evaluate the served model against
the same golden set used for the base and adapter baselines:

```bash
export DISTILLER_GGUF__BASE_URL=http://localhost:8080/v1   # llama-server
# or: export DISTILLER_GGUF__BASE_URL=http://localhost:11434/v1   # Ollama
# the served name defaults to the registered distiller-<book-id>;
# override with DISTILLER_GGUF__MODEL only if you renamed the model

uv run distiller eval <book> --gguf        # eval/report.json, generator.kind == "gguf"
uv run distiller ask <book> "..." --gguf

# three-way comparison: base vs adapter vs GGUF on one golden set
uv run distiller train-eval <book> --gguf  # eval/training.json
```

`eval --gguf` records the registered file identity (hash, quantization,
architecture, model name) in the `generator` block. `train-eval --gguf` adds
the `gguf` variant next to base and adapter over the same index, retrieval
settings and golden items, with deltas versus base; a variant that raises
mid-run is recorded as skipped and the others still report. `--gguf` and
`--adapter` are mutually exclusive.

Offline smoke test without any server (uses the deterministic fake LLM):

```bash
export DISTILLER_GGUF__MODEL=fake
uv run distiller eval <book> --gguf
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `not a GGUF file (bad magic ...)` | you registered something that is not the exported file; check the download |
| `uses unsupported version ...` | re-export with a current llama.cpp/Unsloth stack |
| `is truncated ...` | the download is incomplete; copy the file again |
| `No ingested book ... distiller ingest` | run `distiller ingest book.epub` first |
| `No registered GGUF ... distiller gguf register` | register the downloaded `.gguf` (step 3) |
| `No GGUF endpoint configured ... DISTILLER_GGUF__BASE_URL` | start a server and export the base URL (step 5) |
| `Cannot reach the LLM endpoint ... ollama serve` | start Ollama (`ollama serve`) or the emitted `serve.sh`, then retry |
| `Choose one generator: --adapter or --gguf` | pass exactly one of the two flags |

Related: [README](../README.md) (pipeline overview),
[qlora-runbook.md](qlora-runbook.md) (Phase 3 training),
[CONTRIBUTING.md](CONTRIBUTING.md) (conventions),
[specs/SDD-0005](../specs/active/SDD-0005-gguf-serving.md) (the spec).
