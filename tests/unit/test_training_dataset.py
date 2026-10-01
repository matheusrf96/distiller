"""Unit tests for training dataset preparation (split, validation, writing)."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import pytest

from distiller.config import AdapterSettings, TrainingSettings
from distiller.exceptions import TrainingError
from distiller.models import refusal_text
from distiller.paths import BookPaths
from distiller.training.chat import ChatMessage, TrainingExample, prompt_contract_hash
from distiller.training.dataset import (
    prepare_dataset,
    split_examples,
    validate_examples,
)
from distiller.training.qlora import QLoRAConfig
from distiller.utils import read_jsonl, stable_hash_hex

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from distiller.models import BookDocument, Chunk
    from distiller.synthesis import RaftExample

    CorpusFactory = Callable[..., tuple[BookDocument, list[Chunk]]]
    RaftFactory = Callable[..., RaftExample]

BOOK_TITLE = "The Lantern Keeper"


def make_example(
    index: int,
    *,
    answerable: bool = True,
    question: str | None = None,
    user: str | None = None,
    target: str | None = None,
) -> TrainingExample:
    """Build one chat example with a valid document block by default."""
    if user is None:
        user = (
            '<doc id="1" chapter="Chapter One">\nlantern text\n</doc>\n\n'
            "Question: q\n\nAnswer with citations:"
        )
    if target is None:
        target = (
            'The relevant passage says: "lantern text" [1]'
            if answerable
            else refusal_text(BOOK_TITLE)
        )
    return TrainingExample(
        id=f"example-{index}",
        book_id="book",
        question=question or f"question {index}?",
        messages=[
            ChatMessage(role="system", content="system"),
            ChatMessage(role="user", content=user),
            ChatMessage(role="assistant", content=target),
        ],
        answerable=answerable,
    )


def test_split_is_deterministic_and_leak_free() -> None:
    """The same seed produces identical, disjoint splits (REQ-TR-003)."""
    examples = [make_example(index) for index in range(10)]

    train, validation = split_examples(examples, seed=7, val_ratio=0.2)
    again_train, again_validation = split_examples(examples, seed=7, val_ratio=0.2)

    assert [row.id for row in train] == [row.id for row in again_train]
    assert [row.id for row in validation] == [row.id for row in again_validation]
    assert {row.question for row in train}.isdisjoint(
        {row.question for row in validation}
    )
    assert sorted(row.id for row in [*train, *validation]) == sorted(
        row.id for row in examples
    )


def test_split_honours_the_validation_ratio() -> None:
    """Validation size is ``max(1, floor(n * val_ratio))`` (REQ-TR-003)."""
    examples = [make_example(index) for index in range(10)]
    train, validation = split_examples(examples, seed=1, val_ratio=0.2)
    assert len(validation) == math.floor(10 * 0.2) == 2
    assert len(train) == 8

    train, validation = split_examples([make_example(0)], seed=1, val_ratio=0.1)
    assert len(validation) == 1
    assert train == []

    few = [make_example(index) for index in range(3)]
    _, validation = split_examples(few, seed=1, val_ratio=0.1)
    assert len(validation) == max(1, math.floor(3 * 0.1)) == 1


def test_validation_flags_malformed_examples() -> None:
    """Empty messages, missing contexts, bad citations, wrong refusals (REQ-TR-004)."""
    examples = [
        make_example(0),
        make_example(1, answerable=False),
        TrainingExample(
            id="empty",
            book_id="book",
            question="q?",
            messages=[ChatMessage(role="user", content="")],
            answerable=True,
        ),
        make_example(3, user="Question: q?\n\nAnswer with citations:"),
        make_example(4, target="answer without a citation"),
        make_example(5, answerable=False, target="not the refusal"),
    ]

    report = validate_examples(examples, BOOK_TITLE)

    assert not report.ok
    assert report.example_count == 6
    assert report.answerable_count == 4
    assert report.unanswerable_count == 2
    assert any("empty" in error for error in report.errors)
    assert any("context" in error for error in report.errors)
    assert any("citation" in error for error in report.errors)
    assert any("refusal" in error for error in report.errors)

    valid = validate_examples(
        [make_example(0), make_example(1, answerable=False)], BOOK_TITLE
    )
    assert valid.ok


def test_validation_requires_refusal_negatives() -> None:
    """A positives-only dataset is a hard validation error (REQ-TR-004)."""
    report = validate_examples([make_example(0), make_example(1)], BOOK_TITLE)

    assert not report.ok
    assert any("unanswerable" in error for error in report.errors)
    assert report.unanswerable_count == 0


def test_validation_flags_an_empty_dataset() -> None:
    """An empty RAFT file cannot be prepared (REQ-TR-004)."""
    report = validate_examples([], BOOK_TITLE)

    assert not report.ok
    assert any("no examples" in error for error in report.errors)


def test_training_and_adapter_settings_validate_bounds() -> None:
    """Split and adapter settings default as documented and reject bad values."""
    from pydantic import ValidationError

    assert TrainingSettings().seed == 13
    assert TrainingSettings().val_ratio == 0.1
    assert AdapterSettings().model is None
    assert AdapterSettings().timeout == 120.0

    with pytest.raises(ValidationError):
        TrainingSettings(val_ratio=0.0)
    with pytest.raises(ValidationError):
        TrainingSettings(val_ratio=1.0)
    with pytest.raises(ValidationError):
        TrainingSettings(seed=-1)
    with pytest.raises(ValidationError):
        AdapterSettings(timeout=0)
    with pytest.raises(ValidationError):
        AdapterSettings(model="  ")


def test_prepare_writes_splits_and_manifest(
    corpus_factory: CorpusFactory,
    raft_factory: RaftFactory,
    tmp_path: Path,
) -> None:
    """Prepare writes both splits and a manifest matching them (REQ-TR-005)."""
    book, chunks = corpus_factory()
    examples = [
        raft_factory(chunks, suffix=str(index), question=f"question {index}?")
        for index in range(4)
    ] + [
        raft_factory(
            chunks,
            suffix=f"n{index}",
            question=f"negative {index}?",
            answerable=False,
        )
        for index in range(2)
    ]
    paths = BookPaths.for_book(tmp_path / "artifacts", book.book_id)

    bundle = prepare_dataset(
        book, examples, chunks, TrainingSettings(seed=5, val_ratio=0.34), paths=paths
    )

    manifest = json.loads(paths.training_manifest.read_text(encoding="utf-8"))
    assert manifest["train_count"] == bundle.manifest.train_count == 4
    assert manifest["validation_count"] == bundle.manifest.validation_count == 2
    assert manifest["source_count"] == 6
    assert manifest["seed"] == 5
    assert manifest["val_ratio"] == 0.34
    assert manifest["answerable_count"] == 4
    assert manifest["unanswerable_count"] == 2
    assert manifest["prompt_hash"] == prompt_contract_hash(book.title)

    written_train = [
        TrainingExample.model_validate(row)
        for row in read_jsonl(paths.training_train_jsonl)
    ]
    written_validation = [
        TrainingExample.model_validate(row)
        for row in read_jsonl(paths.training_validation_jsonl)
    ]
    assert [row.id for row in written_train] == [row.id for row in bundle.train]
    assert [row.id for row in written_validation] == [
        row.id for row in bundle.validation
    ]

    # dataset_hash is recomputable from the written files
    payload = "\n".join(
        row.model_dump_json() for row in [*written_train, *written_validation]
    )
    assert manifest["dataset_hash"] == stable_hash_hex(payload, 40)
    assert manifest["dataset_hash"] == bundle.manifest.dataset_hash
    source_payload = "\n".join(example.model_dump_json() for example in examples)
    assert manifest["source_hash"] == stable_hash_hex(source_payload, 40)


def test_prepare_writes_the_qlora_config(
    corpus_factory: CorpusFactory,
    raft_factory: RaftFactory,
    tmp_path: Path,
) -> None:
    """qlora.json round-trips and pins the approved hyperparameters (REQ-TR-006)."""
    book, chunks = corpus_factory()
    examples = [
        raft_factory(chunks, suffix="0", question="q0?"),
        raft_factory(chunks, suffix="1", question="q1?", answerable=False),
    ]
    paths = BookPaths.for_book(tmp_path / "artifacts", book.book_id)

    bundle = prepare_dataset(
        book, examples, chunks, TrainingSettings(seed=9, val_ratio=0.5), paths=paths
    )

    config = QLoRAConfig.model_validate(
        json.loads(paths.training_qlora_json.read_text(encoding="utf-8"))
    )
    assert config == bundle.config
    assert config.base_model == "Qwen/Qwen3-4B"
    assert config.load_in_4bit is True
    assert config.quant_type == "nf4"
    assert config.seed == 9
    assert paths.training_notebook.exists()


def test_prepare_writes_nothing_when_validation_fails(
    corpus_factory: CorpusFactory,
    raft_factory: RaftFactory,
    tmp_path: Path,
) -> None:
    """Validation aborts before any artifact is written (REQ-TR-004)."""
    book, chunks = corpus_factory()
    paths = BookPaths.for_book(tmp_path / "artifacts", book.book_id)

    # a positives-only dataset is rejected
    with pytest.raises(TrainingError, match="validation failed"):
        prepare_dataset(
            book,
            [raft_factory(chunks)],
            chunks,
            TrainingSettings(seed=3),
            paths=paths,
        )
    assert not paths.training_dir.exists()

    # an unknown context chunk aborts before writing too
    broken = raft_factory(chunks)
    broken.contexts[0].chunk_id = "missing:0000:xxxx"
    with pytest.raises(TrainingError, match=r"chunks\.jsonl"):
        prepare_dataset(book, [broken], chunks, TrainingSettings(seed=3), paths=paths)
    assert not paths.training_dir.exists()
