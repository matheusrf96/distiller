# Specs

Kiro-style spec folders. Every non-trivial feature starts here, before any test
or code (see `AGENTS.md`, RULE #2).

Each folder contains:

| File | Purpose |
|------|---------|
| `requirements.md` | EARS-style functional requirements (`REQ-*`) and constraints (`CON-*`) |
| `design.md` | Data model, module, configuration and CLI deltas |
| `tasks.md` | Checklist driving the TDD implementation and verification |

## Index

| Feature | Status | Summary |
|---------|--------|---------|
| [contextual-retrieval](contextual-retrieval/) | ✅ implemented | LLM-generated situating context per chunk for embedding/BM25 (Phase 1) |
| [reranking-ablations](reranking-ablations/) | ✅ implemented | Compare retrieval configs (rerank on/off, top-k sweep) on one golden set (Phase 1) |
