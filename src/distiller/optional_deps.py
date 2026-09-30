"""Centralized access to optional (extra-gated) dependencies.

Import policy
-------------
* Libraries the core pipeline always needs are imported at module top level.
* Heavy or platform-specific libraries live in extras. They are imported
  dynamically, on first use, through :func:`require`, so importing
  ``distiller`` stays cheap and a missing extra produces one consistent,
  actionable :class:`~distiller.exceptions.MissingDependencyError`.
* :func:`is_available` answers "can this backend run here?" without importing
  anything, which keeps backend selection out of exception-driven control flow.
"""

from __future__ import annotations

import importlib
import importlib.util
from functools import cache
from typing import TYPE_CHECKING

from .exceptions import MissingDependencyError

if TYPE_CHECKING:
    from types import ModuleType

__all__ = ["MissingDependencyError", "is_available", "require"]


def is_available(module_name: str) -> bool:
    """Return True when ``module_name`` can be imported, without importing it.

    Args:
        module_name: Dotted import path to check (e.g. ``"docling"``).

    Returns:
        True when the module is importable in the current environment.
    """
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken parent package
        return False


@cache
def require(module_name: str, *, extra: str, purpose: str) -> ModuleType:
    """Import and return an optional module (cached after the first call).

    Args:
        module_name: Dotted import path to load (e.g. ``"qdrant_client.models"``).
        extra: Project extra that provides the module.
        purpose: Human-readable feature that needs the module.

    Returns:
        The imported module.

    Raises:
        MissingDependencyError: When the module is not installed.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise MissingDependencyError(module_name, extra, purpose) from exc
