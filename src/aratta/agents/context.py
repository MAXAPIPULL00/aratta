"""Agent context — scoped service access following least-privilege."""

from dataclasses import dataclass
from typing import Any


@dataclass
class AgentContext:
    """Scoped context providing only the services an agent needs."""

    aratta: Any = None  # AI client
    tool_registry: Any | None = None
    tool_executor: Any | None = None
    agent_id: str | None = None

    @classmethod
    def create(cls, aratta_client: Any) -> "AgentContext":
        from aratta.agents.executor import ToolExecutor
        from aratta.tools.registry import get_registry

        ctx = cls(aratta=aratta_client, tool_registry=get_registry())
        ctx.tool_executor = ToolExecutor(ctx)
        return ctx

    @classmethod
    def minimal(cls, aratta_client: Any) -> "AgentContext":
        from aratta.tools.registry import get_registry
        return cls(aratta=aratta_client, tool_registry=get_registry())

    def get_enabled_tools(self):
        if self.tool_registry:
            return self.tool_registry.list_all()
        return []

    def has_tool(self, name: str) -> bool:
        return self.tool_registry.get(name) is not None if self.tool_registry else False

    async def execute_tool(self, name: str, arguments: dict) -> dict:
        if self.tool_executor:
            result = await self.tool_executor.execute(name, arguments)
            return result.to_dict()
        return {"success": False, "output": None, "error": "No executor available"}
