"""Synthetic QA generation and RAFT dataset formatting (Phase 2)."""

from .dataset import (
    DatasetManifest,
    SynthesisRun,
    load_cached_pairs,
    sample_chunks,
    synthesize,
)
from .filtering import FilterOutcome, RejectedPair, filter_pairs
from .qa import QAPair, generate_pairs, parse_pairs
from .raft import RaftContext, RaftExample, build_examples

__all__ = [
    "DatasetManifest",
    "FilterOutcome",
    "QAPair",
    "RaftContext",
    "RaftExample",
    "RejectedPair",
    "SynthesisRun",
    "build_examples",
    "filter_pairs",
    "generate_pairs",
    "load_cached_pairs",
    "parse_pairs",
    "sample_chunks",
    "synthesize",
]
