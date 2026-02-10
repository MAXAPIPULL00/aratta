"""
SCRI — the language your system speaks.

These types define SCRI: one set of structures for messages, tool calls,
responses, usage, and streaming — regardless of which AI provider is on the
other end. Your code speaks SCRI. Aratta handles the translation to and from
every provider's native format. Provider-specific structures never leak into
your application logic.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(str, Enum):
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------

@dataclass
class Content:
    """A content block within a message."""
    type: ContentType
    text: str | None = None
    image_url: str | None = None
    image_base64: str | None = None
    tool_use_id: str | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_result: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": self.type.value}
        for key in ("text", "image_url", "image_base64", "tool_use_id", "tool_name", "tool_input", "tool_result"):
            val = getattr(self, key)
            if val is not None:
                result[key] = val
        return result


@dataclass
class ThinkingBlock:
    """Extended thinking / reasoning block."""
    thinking: str
    signature: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": "thinking", "thinking": self.thinking}
        if self.signature:
            d["signature"] = self.signature
        return d


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A message in the conversation."""
    role: Role
    content: str | list[Content]
    name: str | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"role": self.role.value}
        if isinstance(self.content, str):
            result["content"] = self.content
        else:
            result["content"] = [c.to_dict() for c in self.content]
        if self.name:
            result["name"] = self.name
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        role = Role(data["role"])
        content = data["content"]
        if isinstance(content, list):
            content = [
                Content(
                    type=ContentType(c["type"]),
                    text=c.get("text"),
                    image_url=c.get("image_url"),
                    image_base64=c.get("image_base64"),
                    tool_use_id=c.get("tool_use_id"),
                    tool_name=c.get("tool_name"),
                    tool_input=c.get("tool_input"),
                    tool_result=c.get("tool_result"),
                )
                for c in content
            ]
        return cls(role=role, content=content, name=data.get("name"), tool_call_id=data.get("tool_call_id"))


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    """Universal tool definition (JSON Schema parameters)."""
    name: str
    description: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Tool":
        return cls(name=data["name"], description=data["description"], parameters=data["parameters"])


@dataclass
class ToolCall:
    """A tool call made by the model."""
    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


# ---------------------------------------------------------------------------
# Usage / Lineage
# ---------------------------------------------------------------------------

@dataclass
class Usage:
    """Token usage statistics."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens, "total_tokens": self.total_tokens}
        for k in ("cache_read_tokens", "cache_write_tokens", "reasoning_tokens"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        return d


@dataclass
class Lineage:
    """Provenance tracking for a response."""
    provider: str
    model: str
    model_version: str | None = None
    request_id: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    latency_ms: float | None = None
    source_system: str = "aratta"
    source_version: str = "0.1.0"

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ---------------------------------------------------------------------------
# Request / Response
# ---------------------------------------------------------------------------

@dataclass
class ChatRequest:
    """Unified chat request."""
    messages: list[Message]
    model: str = "local"
    provider: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    tools: list[Tool] | None = None
    tool_choice: str | dict[str, Any] | None = None
    stream: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    sovereign: bool = False

    # Extended thinking
    thinking_enabled: bool = False
    thinking_budget: int = 10000

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "messages": [m.to_dict() for m in self.messages],
            "model": self.model,
            "temperature": self.temperature,
            "stream": self.stream,
        }
        if self.provider:
            d["provider"] = self.provider
        if self.max_tokens:
            d["max_tokens"] = self.max_tokens
        if self.tools:
            d["tools"] = [t.to_dict() for t in self.tools]
        if self.thinking_enabled:
            d["thinking"] = {"enabled": True, "budget_tokens": self.thinking_budget}
        return d


@dataclass
class ChatResponse:
    """Unified chat response."""
    id: str
    content: str
    role: str = "assistant"
    tool_calls: list[ToolCall] | None = None
    thinking: list[ThinkingBlock] | None = None
    model: str = ""
    provider: str = ""
    finish_reason: FinishReason = FinishReason.STOP
    usage: Usage | None = None
    lineage: Lineage | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "content": self.content,
            "role": self.role,
            "model": self.model,
            "provider": self.provider,
            "finish_reason": self.finish_reason.value,
            "timestamp": self.timestamp,
        }
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.thinking:
            d["thinking"] = [t.to_dict() for t in self.thinking]
        if self.usage:
            d["usage"] = self.usage.to_dict()
        if self.lineage:
            d["lineage"] = self.lineage.to_dict()
        return d


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingRequest:
    """Unified embedding request."""
    input: str | list[str]
    model: str = "embed"
    provider: str | None = None
    dimensions: int | None = None


@dataclass
class Embedding:
    embedding: list[float]
    index: int


@dataclass
class EmbeddingResponse:
    embeddings: list[Embedding]
    model: str
    provider: str
    usage: Usage
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "embeddings": [{"embedding": e.embedding, "index": e.index} for e in self.embeddings],
            "model": self.model,
            "provider": self.provider,
            "usage": self.usage.to_dict(),
        }


# ---------------------------------------------------------------------------
# Model capabilities
# ---------------------------------------------------------------------------

@dataclass
class ModelCapabilities:
    """Model capability metadata."""
    model_id: str
    provider: str
    display_name: str
    supports_tools: bool = False
    supports_vision: bool = False
    supports_streaming: bool = True
    supports_json_mode: bool = False
    supports_thinking: bool = False
    context_window: int = 4096
    max_output_tokens: int | None = None
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    categories: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
