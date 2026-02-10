"""
Base provider — abstract class all AI provider adapters inherit from.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import httpx

from aratta.config import ProviderConfig
from aratta.core.types import (
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelCapabilities,
    Tool,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ProviderError(Exception):
    def __init__(self, message: str, provider: str, status_code: int | None = None):
        self.message = message
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"[{provider}] {message}")


class RateLimitError(ProviderError):
    pass


class AuthenticationError(ProviderError):
    pass


class ModelNotFoundError(ProviderError):
    pass


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseProvider(ABC):
    """
    Abstract base for all AI provider adapters.

    Subclasses implement chat / chat_stream / embed / get_models and the
    two format-conversion helpers.
    """

    name: str = "base"
    display_name: str = "Base Provider"

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=self._get_headers(),
                timeout=httpx.Timeout(self.config.timeout),
            )
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "User-Agent": "Aratta/0.1.0"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    # --- Abstract ---

    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse: ...

    @abstractmethod
    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]: ...

    @abstractmethod
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse: ...

    @abstractmethod
    def get_models(self) -> list[ModelCapabilities]: ...

    @abstractmethod
    def convert_tools(self, tools: list[Tool]) -> Any: ...

    @abstractmethod
    def convert_messages(self, messages: list[Any]) -> Any: ...

    # --- Helpers ---

    async def health_check(self) -> dict[str, Any]:
        try:
            resp = await self.client.get("/models")
            return {
                "status": "healthy" if resp.status_code == 200 else "degraded",
                "provider": self.name,
                "latency_ms": resp.elapsed.total_seconds() * 1000,
            }
        except Exception as e:
            return {"status": "unhealthy", "provider": self.name, "error": str(e)}

    def _handle_error(self, response: httpx.Response) -> None:
        if response.status_code == 401:
            raise AuthenticationError("Authentication failed.", self.name, 401)
        if response.status_code == 429:
            raise RateLimitError("Rate limit exceeded.", self.name, 429)
        if response.status_code == 404:
            raise ModelNotFoundError("Model not found.", self.name, 404)
        if response.status_code >= 400:
            try:
                msg = response.json().get("error", {}).get("message", response.text)
            except Exception:
                msg = response.text
            raise ProviderError(msg, self.name, response.status_code)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name})>"
