"""Agent framework types — states, config, messages, results."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class AgentState(Enum):
    IDLE = "idle"
    REASONING = "reasoning"
    EXECUTING = "executing"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class AgentConfig:
    model: str = "local"
    provider: str | None = None
    temperature: float = 0.7
    max_tokens: int = 4096
    max_iterations: int = 10
    timeout_seconds: int = 300
    retry_on_error: bool = True
    max_retries: int = 3
    allowed_tools: list[str] | None = None
    blocked_tools: list[str] | None = None
    require_confirmation: list[str] = field(default_factory=list)
    enable_thinking: bool = False
    thinking_budget: int = 10000
    stream: bool = False


@dataclass
class AgentMessage:
    role: str
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_results: list[dict[str, Any]] | None = None
    thinking: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content, "timestamp": self.timestamp}
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_results:
            d["tool_results"] = self.tool_results
        if self.thinking:
            d["thinking"] = self.thinking
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentMessage":
        return cls(role=data["role"], content=data["content"], tool_calls=data.get("tool_calls"),
                   tool_results=data.get("tool_results"), thinking=data.get("thinking"),
                   timestamp=data.get("timestamp", datetime.now(UTC).isoformat()))


@dataclass
class LoopResult:
    success: bool
    content: str
    iterations: int
    tool_calls: list[dict[str, Any]]
    reasoning: list[str] | None = None
    error: str | None = None
    stopped_reason: str = "completed"
    elapsed_ms: float = 0
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class ToolCallSpec:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    call_id: str
    tool_name: str
    success: bool
    output: Any
    error: str | None = None
    execution_time_ms: float = 0

    def to_dict(self) -> dict[str, Any]:
        return {"call_id": self.call_id, "tool_name": self.tool_name, "success": self.success,
                "output": self.output, "error": self.error, "execution_time_ms": self.execution_time_ms}
