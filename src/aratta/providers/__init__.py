"""Aratta provider adapters — one interface, every LLM backend."""

from .base import (
    AuthenticationError,
    BaseProvider,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
)

__all__ = ["BaseProvider", "ProviderError", "RateLimitError", "AuthenticationError", "ModelNotFoundError"]
