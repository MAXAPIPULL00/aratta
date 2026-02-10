"""
Tool registry — unified tool management with provider format translation.

Register tools once in a universal format, export to any provider's
expected schema on demand.
"""

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

Provider = Literal["anthropic", "openai", "google", "xai"]


class ToolDef(BaseModel):
    """Universal tool definition."""
    name: str = Field(..., max_length=64)
    description: str = Field(..., max_length=500)
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    strict: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolRegistry:
    """In-memory tool registry with provider translation."""

    def __init__(self):
        self._tools: dict[str, ToolDef] = {}

    def register(self, tool: ToolDef) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def list_all(self) -> list[ToolDef]:
        return list(self._tools.values())

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def count(self) -> int:
        return len(self._tools)

    def export_for_provider(self, provider: Provider) -> list[dict[str, Any]]:
        tools = self.list_all()
        if provider == "anthropic":
            return [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]
        if provider in ("openai", "xai"):
            return [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}} for t in tools]
        if provider == "google":
            return [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in tools]
        return [t.model_dump() for t in tools]

    # --- Import from provider formats ---

    @staticmethod
    def from_anthropic(d: dict) -> ToolDef:
        return ToolDef(name=d["name"], description=d["description"], parameters=d.get("input_schema", {}), metadata={"provider": "anthropic"})

    @staticmethod
    def from_openai(d: dict) -> ToolDef:
        f = d.get("function", d)
        return ToolDef(name=f["name"], description=f["description"], parameters=f.get("parameters", {}), strict=f.get("strict", False), metadata={"provider": "openai"})

    @staticmethod
    def from_google(d: dict) -> ToolDef:
        return ToolDef(name=d["name"], description=d["description"], parameters=d.get("parameters", {}), metadata={"provider": "google"})

    def import_from_provider(self, d: dict, provider: Provider) -> ToolDef:
        if provider == "anthropic":
            tool = self.from_anthropic(d)
        elif provider in ("openai", "xai"):
            tool = self.from_openai(d)
        elif provider == "google":
            tool = self.from_google(d)
        else:
            tool = ToolDef(**d)
        self.register(tool)
        return tool


_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
