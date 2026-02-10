"""
Cloud capability registry — loads provider capabilities from JSON definitions.

Enables intelligent routing, model selection, and feature detection.
"""

import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

Provider = Literal["anthropic", "openai", "google", "xai"]


class ModelSpec(BaseModel):
    name: str
    model_id: str
    context_window: int | None = None
    max_output_tokens: int | None = None
    pricing: dict[str, float] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ServerTool(BaseModel):
    name: str
    type: str = "server_tool"
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class ProviderCapabilities(BaseModel):
    provider: str
    models: list[ModelSpec] = Field(default_factory=list)
    server_tools: list[ServerTool] = Field(default_factory=list)
    capabilities: list[dict[str, Any]] = Field(default_factory=list)


class CapabilityRegistry:
    def __init__(self):
        self._providers: dict[str, ProviderCapabilities] = {}
        self._models_by_id: dict[str, ModelSpec] = {}

    def load_provider(self, provider: str, data: dict[str, Any]) -> None:
        models = [ModelSpec(name=m.get("name", ""), model_id=m.get("model_id", m.get("name", "")),
                            context_window=m.get("context_window"), max_output_tokens=m.get("max_output_tokens"),
                            pricing=m.get("pricing", {}), capabilities=m.get("capabilities", []))
                  for m in data.get("models", [])]
        for m in models:
            self._models_by_id[m.model_id] = m
        tools = [ServerTool(name=t.get("name", ""), description=t.get("description", ""), parameters=t.get("parameters", {}))
                 for t in data.get("server_tools", [])]
        self._providers[provider] = ProviderCapabilities(provider=provider, models=models, server_tools=tools, capabilities=data.get("capabilities", []))

    def get_models(self, provider: str) -> list[ModelSpec]:
        c = self._providers.get(provider)
        return c.models if c else []

    def get_model(self, model_id: str) -> ModelSpec | None:
        return self._models_by_id.get(model_id)

    def get_server_tools(self, provider: str) -> list[ServerTool]:
        c = self._providers.get(provider)
        return c.server_tools if c else []

    def has_capability(self, provider: str, name: str) -> bool:
        return any(t.name == name for t in self.get_server_tools(provider))

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())

    def stats(self) -> dict[str, Any]:
        return {"providers": len(self._providers), "total_models": len(self._models_by_id)}


_cap_registry: CapabilityRegistry | None = None


def get_capability_registry() -> CapabilityRegistry:
    global _cap_registry
    if _cap_registry is None:
        _cap_registry = CapabilityRegistry()
    return _cap_registry


def load_capabilities(directory: Path | None = None) -> CapabilityRegistry:
    registry = get_capability_registry()
    dir_path = directory or (Path(__file__).parent / "builtin")
    if not dir_path.exists():
        return registry
    for provider, filename in {"anthropic": "anthropic_capabilities.json", "openai": "openai_capabilities.json",
                                "google": "google_capabilities.json", "xai": "xai_capabilities.json"}.items():
        fp = dir_path / filename
        if fp.exists():
            try:
                with open(fp, encoding="utf-8") as f:
                    registry.load_provider(provider, json.load(f))
            except Exception as e:
                logger.error(f"Failed to load {filename}: {e}")
    return registry
