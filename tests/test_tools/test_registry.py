"""Tests for the tool registry and cross-provider translation."""

import pytest

from aratta.tools.registry import ToolDef, ToolRegistry


@pytest.fixture
def registry():
    return ToolRegistry()


@pytest.fixture
def sample_tool():
    return ToolDef(
        name="search",
        description="Search the web",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    )


class TestToolRegistry:
    def test_register_and_get(self, registry, sample_tool):
        registry.register(sample_tool)
        assert registry.get("search") is not None
        assert registry.get("search").name == "search"

    def test_unregister(self, registry, sample_tool):
        registry.register(sample_tool)
        assert registry.unregister("search") is True
        assert registry.get("search") is None

    def test_unregister_missing(self, registry):
        assert registry.unregister("nonexistent") is False

    def test_list_all(self, registry, sample_tool):
        registry.register(sample_tool)
        registry.register(ToolDef(name="calc", description="Calculate", parameters={}))
        assert registry.count() == 2
        assert len(registry.list_all()) == 2

    def test_list_names(self, registry, sample_tool):
        registry.register(sample_tool)
        assert "search" in registry.list_names()


class TestProviderExport:
    def test_anthropic_format(self, registry, sample_tool):
        registry.register(sample_tool)
        exported = registry.export_for_provider("anthropic")
        assert len(exported) == 1
        assert "input_schema" in exported[0]
        assert exported[0]["name"] == "search"

    def test_openai_format(self, registry, sample_tool):
        registry.register(sample_tool)
        exported = registry.export_for_provider("openai")
        assert exported[0]["type"] == "function"
        assert exported[0]["function"]["name"] == "search"

    def test_google_format(self, registry, sample_tool):
        registry.register(sample_tool)
        exported = registry.export_for_provider("google")
        assert exported[0]["name"] == "search"
        assert "parameters" in exported[0]

    def test_xai_same_as_openai(self, registry, sample_tool):
        registry.register(sample_tool)
        openai_fmt = registry.export_for_provider("openai")
        xai_fmt = registry.export_for_provider("xai")
        assert openai_fmt == xai_fmt


class TestProviderImport:
    def test_from_anthropic(self):
        tool = ToolRegistry.from_anthropic({
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        })
        assert tool.name == "get_weather"
        assert tool.parameters["type"] == "object"

    def test_from_openai(self):
        tool = ToolRegistry.from_openai({
            "type": "function",
            "function": {"name": "calc", "description": "Calculate", "parameters": {"type": "object"}},
        })
        assert tool.name == "calc"

    def test_from_google(self):
        tool = ToolRegistry.from_google({
            "name": "translate",
            "description": "Translate text",
            "parameters": {"type": "object"},
        })
        assert tool.name == "translate"

    def test_import_registers(self, registry):
        registry.import_from_provider(
            {"name": "test_tool", "description": "A test", "input_schema": {}},
            "anthropic",
        )
        assert registry.get("test_tool") is not None
