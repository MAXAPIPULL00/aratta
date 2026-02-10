"""Agent — autonomous AI execution unit with ReAct loop."""

from __future__ import annotations

import logging
import uuid
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .types import AgentConfig, AgentMessage, AgentState, LoopResult

if TYPE_CHECKING:
    from .context import AgentContext

logger = logging.getLogger("aratta.agent")
MAX_MESSAGES = 100


class Agent:
    """
    Provider-agnostic autonomous agent.

    Usage:
        agent = Agent(config=AgentConfig(model="local"), context=ctx)
        result = await agent.run("Research quantum computing")
    """

    def __init__(self, agent_id: str = None, config: AgentConfig = None, context: AgentContext = None,
                 tools: list[Any] = None, system_prompt: str = None):
        self.agent_id = agent_id or f"agent-{uuid.uuid4().hex[:8]}"
        self.config = config or AgentConfig()
        self.context = context
        self.tools = tools if tools is not None else (context.get_enabled_tools() if context else [])
        self._apply_tool_filters()
        self.system_prompt = system_prompt or self._default_prompt()
        self.state = AgentState.IDLE
        self.messages: deque = deque(maxlen=MAX_MESSAGES)
        self.iterations = 0
        self.created_at = datetime.now(UTC)
        self.last_activity = self.created_at
        self._on_complete: list[Callable] = []
        self._on_tool_call: list[Callable] = []
        self.metrics = {"tool_calls": 0, "tokens_used": 0, "reasoning_time_ms": 0, "execution_time_ms": 0, "errors": 0}
        if self.context:
            self.context.agent_id = self.agent_id

    def _apply_tool_filters(self):
        if not self.tools:
            return
        if self.config.allowed_tools:
            allowed = set(self.config.allowed_tools)
            self.tools = [t for t in self.tools if t.name in allowed]
        if self.config.blocked_tools:
            blocked = set(self.config.blocked_tools)
            self.tools = [t for t in self.tools if t.name not in blocked]

    def _default_prompt(self) -> str:
        names = ", ".join(t.name for t in self.tools) if self.tools else "none"
        return f"You are an autonomous AI agent.\n\nAvailable tools: {names}\n\nReason through problems, use tools when needed, and provide clear final answers."

    async def run(self, task: str) -> dict[str, Any]:
        from .loop import AgentLoop
        self.add_message(AgentMessage(role="user", content=task))
        result = await AgentLoop(self, self.context).run()
        for cb in self._on_complete:
            try:
                cb(result)
            except Exception:
                pass
        return result.to_dict() if isinstance(result, LoopResult) else result

    def add_message(self, msg: AgentMessage):
        self.messages.append(msg)
        self.last_activity = datetime.now(UTC)

    def set_state(self, state: AgentState):
        self.state = state
        self.last_activity = datetime.now(UTC)

    def on_complete(self, cb: Callable):
        self._on_complete.append(cb)

    def on_tool_call(self, cb: Callable):
        self._on_tool_call.append(cb)

    def get_status(self) -> dict[str, Any]:
        return {"agent_id": self.agent_id, "state": self.state.value, "iterations": self.iterations,
                "messages": len(self.messages), "tools": len(self.tools), "metrics": self.metrics}

    def to_dict(self) -> dict[str, Any]:
        return {"agent_id": self.agent_id, "state": self.state.value, "iterations": self.iterations,
                "messages": [m.to_dict() for m in self.messages], "metrics": self.metrics,
                "config": {"model": self.config.model, "max_iterations": self.config.max_iterations}}

    @classmethod
    def from_dict(cls, data: dict[str, Any], context: AgentContext = None) -> Agent:
        agent = cls(agent_id=data["agent_id"], config=AgentConfig(**data.get("config", {})), context=context)
        agent.state = AgentState(data.get("state", "idle"))
        agent.iterations = data.get("iterations", 0)
        agent.metrics = data.get("metrics", agent.metrics)
        for m in data.get("messages", []):
            agent.messages.append(AgentMessage.from_dict(m))
        return agent
