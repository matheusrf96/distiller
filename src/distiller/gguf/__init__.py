"""GGUF export validation, registration and local serving (Phase 4).

The repository never merges, quantizes or launches anything: it validates the
Q4_K_M file downloaded from the training machine, records its identity and
emits the exact Ollama/llama.cpp commands. The reader is dependency-free.
"""

from .reader import FILE_TYPE_LABELS, GgufMetadata, read_gguf
from .registry import MODEL_FILE_NAME, GgufReport, load_gguf_report, register_gguf
from .serving import ServingProfile, render_modelfile, render_serve_script

__all__ = [
    "FILE_TYPE_LABELS",
    "MODEL_FILE_NAME",
    "GgufMetadata",
    "GgufReport",
    "ServingProfile",
    "load_gguf_report",
    "read_gguf",
    "register_gguf",
    "render_modelfile",
    "render_serve_script",
]
