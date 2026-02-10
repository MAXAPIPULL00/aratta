"""Tests for the xAI (Grok) provider adapter."""

import pytest

from aratta.config import ProviderConfig, ProviderPriority
from aratta.core.types import Message, Role, Tool
from aratta.providers.xai import XAIProvider


@pytest.fixture
def provider():
    cfg = ProviderConfig(
        name="xai", base_url="https://api.x.ai/v1",
        api_key_env=None, default_model="grok-4-1-fast",
        priority=ProviderPriority.FALLBACK.value,
    )
    return XAIProvider(cfg)


class TestMessageConversion:
    def test_text_message(self, provider):
        msgs = [Message(role=Role.USER, content="hello")]
        converted = provider.convert_messages(msgs)
        assert converted[0]["content"] == "hello"
        assert converted[0]["role"] == "user"

    def test_system_passes_through(self, provider):
        msgs = [Message(role=Role.SYSTEM, content="Be helpful")]
        converted = provider.convert_messages(msgs)
        assert converted[0]["role"] == "system"


class TestToolConversion:
    def test_openai_compatible_format(self, provider):
        tools = [Tool(name="search", description="Search", parameters={"type": "object"})]
        converted = provider.convert_tools(tools)
        assert converted[0]["type"] == "function"
        assert converted[0]["function"]["name"] == "search"


class TestModels:
    def test_lists_models(self, provider):
        models = provider.get_models()
        assert len(models) >= 2
        ids = [m.model_id for m in models]
        assert "grok-4-1-fast" in ids
