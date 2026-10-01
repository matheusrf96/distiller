"""Exception hierarchy for the distiller project.

Follows the Devotion convention: one base exception per project, typed
subclasses for each failure domain, so callers can catch precisely.

Policy: domain, I/O and configuration failures raise subclasses of
:class:`DistillerError`. Internal programming invariants (for example a vector
matrix with the wrong shape) keep raising plain ``ValueError``/``TypeError``.
"""

from __future__ import annotations


class DistillerError(Exception):
    """Base exception for every distiller failure domain."""


class ConfigurationError(DistillerError):
    """Raised when runtime configuration is invalid or inconsistent."""


class IngestError(DistillerError):
    """Raised when a book cannot be parsed into a usable document."""


class BookNotFoundError(DistillerError):
    """Raised when a book id has no ingested artifacts on disk."""


class IndexNotFoundError(DistillerError):
    """Raised when a book has not been indexed yet."""


class IndexBuildError(DistillerError):
    """Raised when an index cannot be built or loaded consistently."""


class TrainingError(DistillerError):
    """Raised when training data, configuration or an adapter is unusable."""


class GGUFError(DistillerError):
    """Raised when a GGUF file or its registration is unusable."""


class MissingDependencyError(DistillerError):
    """Raised when a feature is used without its optional dependency installed."""

    def __init__(self, module: str, extra: str, purpose: str) -> None:
        """Record what is missing and how to install it.

        Args:
            module: Import name of the missing module.
            extra: Project extra that provides the module.
            purpose: Human-readable feature that needs the module.
        """
        super().__init__(
            f"{purpose} requires the optional dependency '{module}'.\n"
            f"Install it with: uv sync --extra {extra}"
        )
        self.module = module
        self.extra = extra
        self.purpose = purpose
