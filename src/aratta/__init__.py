"""
Aratta — A sovereignty layer for AI.

Your local models are the foundation. Cloud providers are callable services.
SCRI is the language your system speaks. Aratta handles the rest.

Quick start::

    pip install aratta
    aratta init    # pick providers, set keys
    aratta serve   # starts on :8084

    # Invoke any provider — your system speaks SCRI:
    httpx.post("http://localhost:8084/api/v1/chat", json={
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "local",  # or "sonnet", "gpt", "gemini", "grok"
    })
"""

__version__ = "0.1.0"

from aratta.config import ArattaConfig, ProviderConfig, get_config, load_config
from aratta.core.types import (
    ChatRequest,
    ChatResponse,
    Content,
    ContentType,
    EmbeddingRequest,
    EmbeddingResponse,
    FinishReason,
    Lineage,
    Message,
    ModelCapabilities,
    Role,
    ThinkingBlock,
    Tool,
    ToolCall,
    Usage,
)
from aratta.providers.base import BaseProvider, ProviderError

__all__ = [
    # Core types
    "ChatRequest",
    "ChatResponse",
    "Content",
    "ContentType",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "FinishReason",
    "Lineage",
    "Message",
    "ModelCapabilities",
    "Role",
    "ThinkingBlock",
    "Tool",
    "ToolCall",
    "Usage",
    # Config
    "load_config",
    "get_config",
    "ArattaConfig",
    "ProviderConfig",
    # Providers
    "BaseProvider",
    "ProviderError",
]
