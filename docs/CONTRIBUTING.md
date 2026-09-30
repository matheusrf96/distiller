# Contributing to distiller

**Reference**: mirrors the conventions established in the Devotion project
(`~/dev/py/devotion`), adapted to this codebase.

## Documentation Rules

1. **Repository pattern** - all markdown lives in `docs/`. Only `AGENTS.md` and
   `README.md` may sit in the project root.
2. **Naming** - lowercase, hyphenated file names (e.g. `retrieval-tuning.md`).
3. **No temporary files** - finished investigations become docs; abandoned notes
   get deleted, not left behind.
4. **Update the index** - new docs must be linked from `docs/README.md`.
5. **Cross-reference** - link related docs and source modules.

---

## Code Conventions

### Formatting and linting

- **Line length: 88 characters** (black-compatible).
- **`ruff format`** is the single formatter and **`ruff check`** the single linter
  (it also enforces isort import ordering). There is no separate black/isort/pylint
  step: ruff covers those standards, plus mccabe complexity, type-checking imports,
  logging, pathlib and bandit security rules.
- Import order: standard library -> third-party -> local, blank line between groups.

```bash
make format    # format + typecheck + lint + security
```

### Type safety

- **`mypy --strict`** on `src/` (see `[tool.mypy]` in `pyproject.toml`).
- 100% of functions and methods have parameter and return type hints.
- Modern syntax only: `list[str]`, `X | None`, PEP 695 generics - never `typing.List`.

### Docstrings

Google style, mandatory on public API:

```python
def build_index(book_id: str, settings: Settings) -> dict:
    """Chunk, embed and persist a retrieval index for one ingested book.

    Args:
        book_id: Slug of a previously ingested book.
        settings: Pipeline settings (chunking, embedding, store backend).

    Returns:
        Index metadata written to ``index/metadata.json``.

    Raises:
        BookNotFoundError: If the book was never ingested.
    """
```

One-line docstrings are acceptable for obvious functions.

### Data models: pydantic vs dataclass

- **Pydantic** for boundary data: anything validated, serialized or crossing
  module/CLI/artifact boundaries (`BookDocument`, `Chunk`, `Answer`, `Settings`,
  `ItemResult`).
- **Dataclass** (`frozen=True, slots=True` where possible) for internal hot-path
  values (`_Piece`, `SearchHit`), filesystem helpers (`BookPaths`) and runtime
  containers holding live objects (`IndexBundle`).

### Exceptions

- Domain, I/O and configuration failures raise subclasses of `DistillerError`
  (`IngestError`, `BookNotFoundError`, `IndexNotFoundError`, `IndexBuildError`,
  `ConfigurationError`, `MissingDependencyError`).
- Internal programming invariants keep raising plain `ValueError`/`TypeError`.
- CLI commands catch `DistillerError` and convert it to a friendly
  `typer.BadParameter` message.

### Imports and optional dependencies

- Libraries the core pipeline always needs are imported at module top level.
- Heavy optional libraries (docling, PyMuPDF4LLM, sentence-transformers, qdrant,
  ragas) are loaded dynamically through `distiller.optional_deps.require()`.
  Never write a literal `import` statement inside a function.
- Use `optional_deps.is_available()` to select backends without importing.

### Naming

- No abbreviations: `doc` -> `book`, `meta` -> `metadata`, `n_chunks` ->
  `chunk_count`. Well-established domain terms are fine (`toc`, `href`, `ctx`).
- `snake_case` functions/variables, `PascalCase` classes, `UPPER_CASE` module
  constants, `_leading_underscore` for private helpers.

---

## Testing Standards

- Layout: `tests/unit/` for single-module behavior, `tests/integration/` for
  multi-stage flows (the CLI pipeline test).
- **The entire suite is offline**: generated EPUB/PDF fixtures, the `hash`
  embedder, the `fake` LLM. No network, no model downloads, no binary fixtures.
- Functional style (plain functions), never class-based tests.
- Shared data goes through **factory fixtures** (`epub_factory`, `pdf_factory`),
  not module-level helpers imported between test files.
- Fixtures have return type annotations and docstrings.
- Mark tests needing network/heavy models with `@pytest.mark.integration`.
- Coverage: `make coverage` fails under **80%** (`[tool.coverage.report]`).
  Critical paths (ingest, chunking, retrieval, evaluation) target 90%+.

```bash
make unit-test   # tests/unit, parallel
make test        # everything
make coverage    # report + threshold
```

## Code Review Checklist

- [ ] Formatting and lint clean (`make format`)
- [ ] Strict typing passes (`make typecheck`)
- [ ] Docstrings on new public functions and classes
- [ ] Domain errors use the `DistillerError` hierarchy
- [ ] Pydantic for boundary data, dataclass for internals
- [ ] No literal imports inside functions
- [ ] Tests cover new behavior and the suite stays offline
- [ ] Docs updated (`docs/` + `docs/README.md` link)
- [ ] No hardcoded secrets; configuration via `Settings`
