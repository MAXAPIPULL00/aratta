"""Tests for the OpenAI provider adapter."""

import pytest

from aratta.config import ProviderConfig, ProviderPriority
from aratta.core.types import Message, Role, Tool
from aratta.providers.openai import OpenAIProvider


@pytest.fixture
def provider():
    cfg = ProviderConfig(
        name="openai", base_url="https://api.openai.com/v1",
        api_key_env=None, default_model="gpt-4.1",
        priority=ProviderPriority.SECONDARY.value,
    )
    return OpenAIProvider(cfg)


class TestMessageConversion:
    def test_system_as_instructions(self, provider):
        msgs = [Message(role=Role.SYSTEM, content="Be concise"), Message(role=Role.USER, content="hi")]
        instructions, converted = provider.convert_messages(msgs)
        assert instructions == "Be concise"
        assert len(converted) == 1

    def test_text_message(self, provider):
        msgs = [Message(role=Role.USER, content="hello")]
        _, converted = provider.convert_messages(msgs)
        assert converted[0]["content"] == "hello"


class TestToolConversion:
    def test_openai_function_format(self, provider):
        tools = [Tool(name="calc", description="Calculate", parameters={"type": "object"})]
        converted = provider.convert_tools(tools)
        assert converted[0]["type"] == "function"
        assert converted[0]["function"]["name"] == "calc"


class TestModels:
    def test_lists_models(self, provider):
        models = provider.get_models()
        assert len(models) >= 3
        ids = [m.model_id for m in models]
        assert "gpt-4.1" in ids
        assert "o3" in ids
