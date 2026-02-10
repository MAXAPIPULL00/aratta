"""Tests for the Anthropic provider adapter."""

import pytest

from aratta.config import ProviderConfig, ProviderPriority
from aratta.core.types import Content, ContentType, Message, Role, Tool
from aratta.providers.anthropic import AnthropicProvider


@pytest.fixture
def provider():
    cfg = ProviderConfig(
        name="anthropic", base_url="https://api.anthropic.com",
        api_key_env=None, default_model="claude-sonnet-4-5-20250929",
        priority=ProviderPriority.PRIMARY.value,
    )
    return AnthropicProvider(cfg)


class TestMessageConversion:
    def test_system_extracted(self, provider):
        msgs = [Message(role=Role.SYSTEM, content="Be helpful"), Message(role=Role.USER, content="hi")]
        system, converted = provider.convert_messages(msgs)
        assert system == "Be helpful"
        assert len(converted) == 1
        assert converted[0]["role"] == "user"

    def test_text_message(self, provider):
        msgs = [Message(role=Role.USER, content="hello")]
        _, converted = provider.convert_messages(msgs)
        assert converted[0]["content"] == "hello"

    def test_image_block(self, provider):
        blocks = [Content(type=ContentType.IMAGE, image_base64="abc123")]
        msgs = [Message(role=Role.USER, content=blocks)]
        _, converted = provider.convert_messages(msgs)
        assert converted[0]["content"][0]["type"] == "image"
        assert converted[0]["content"][0]["source"]["type"] == "base64"

    def test_tool_result_block(self, provider):
        blocks = [Content(type=ContentType.TOOL_RESULT, tool_use_id="tu_1", tool_result={"answer": 42})]
        msgs = [Message(role=Role.USER, content=blocks)]
        _, converted = provider.convert_messages(msgs)
        assert converted[0]["content"][0]["type"] == "tool_result"


class TestToolConversion:
    def test_converts_to_anthropic_format(self, provider):
        tools = [Tool(name="search", description="Search", parameters={"type": "object"})]
        converted = provider.convert_tools(tools)
        assert converted[0]["input_schema"] == {"type": "object"}


class TestModels:
    def test_lists_models(self, provider):
        models = provider.get_models()
        assert len(models) >= 3
        names = [m.model_id for m in models]
        assert "claude-sonnet-4-5-20250929" in names

    def test_models_have_capabilities(self, provider):
        for m in provider.get_models():
            assert m.supports_tools is True
            assert m.context_window > 0
