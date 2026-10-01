"""Unit tests for Modelfile/serve.sh emission and the GGUF runbook."""

from __future__ import annotations

from pathlib import Path

from distiller.gguf import ServingProfile, render_modelfile, render_serve_script
from distiller.rag.prompts import build_system_prompt

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_modelfile_uses_the_rag_prompt_and_serving_parameters() -> None:
    """SYSTEM is the shared RAG prompt; parameters pin 4 GB serving (AC7)."""
    modelfile = render_modelfile("The Lantern Keeper", ServingProfile())

    assert f'SYSTEM """{build_system_prompt("The Lantern Keeper")}"""' in modelfile
    assert "FROM ./model.gguf" in modelfile
    assert "PARAMETER temperature 0.1" in modelfile
    assert "PARAMETER num_ctx 4096" in modelfile
    assert "PARAMETER num_gpu 20" in modelfile
    assert 'PARAMETER stop "<|im_end|>"' in modelfile
    assert 'PARAMETER stop "<|endoftext|>"' in modelfile
    assert "ollama create" in modelfile
    assert "ollama run" in modelfile


def test_serve_script_uses_partial_offload_for_four_gb() -> None:
    """serve.sh carries the exact llama-server invocation and endpoint (AC8)."""
    script = render_serve_script("distiller-the-lantern-keeper", ServingProfile())

    assert "llama-server" in script
    assert "--alias distiller-the-lantern-keeper" in script
    assert "-ngl 20" in script
    assert "-c 4096" in script
    assert "-np 1" in script
    assert "--host 127.0.0.1" in script
    assert "--port 8080" in script
    assert "http://localhost:8080/v1" in script
    assert "ollama create distiller-the-lantern-keeper" in script
    assert "ollama run distiller-the-lantern-keeper" in script


def test_gguf_runbook_documents_the_serving_steps() -> None:
    """The runbook exists, documents register/serve/eval, and is linked (AC12)."""
    runbook = REPO_ROOT / "docs" / "gguf-runbook.md"
    assert runbook.exists()

    text = runbook.read_text(encoding="utf-8")
    assert "distiller gguf register" in text
    assert "distiller gguf serve" in text
    assert "distiller eval" in text
    assert "--gguf" in text
    assert "ollama create" in text
    assert "llama-server" in text

    docs_index = (REPO_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    assert "gguf-runbook.md" in docs_index
