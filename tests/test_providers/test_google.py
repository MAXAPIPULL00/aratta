"""Tests for the Google (Gemini) provider adapter."""

import pytest

from aratta.config import ProviderConfig, ProviderPriority
from aratta.core.types import Message, Role, Tool
from aratta.providers.google import GoogleProvider


@pytest.fixture
def provider():
    cfg = ProviderConfig(
        name="google", base_url="https://generativelanguage.googleapis.com",
        api_key_env=None, default_model="gemini-3-flash-preview",
        priority=ProviderPriority.TERTIARY.value,
    )
    return GoogleProvider(cfg)


class TestMessageConversion:
    def test_system_instruction(self, provider):
        msgs = [Message(role=Role.SYSTEM, content="Be helpful"), Message(role=Role.USER, content="hi")]
        sys_instr, contents = provider.convert_messages(msgs)
        assert sys_instr == "Be helpful"
        assert len(contents) == 1
        assert contents[0]["role"] == "user"

    def test_gemini_parts_format(self, provider):
        msgs = [Message(role=Role.USER, content="hello")]
        _, contents = provider.convert_messages(msgs)
        assert contents[0]["parts"][0]["text"] == "hello"

    def test_assistant_becomes_model(self, provider):
        msgs = [Message(role=Role.ASSISTANT, content="sure")]
        _, contents = provider.convert_messages(msgs)
        assert contents[0]["role"] == "model"


class TestToolConversion:
    def test_google_function_declarations(self, provider):
        tools = [Tool(name="search", description="Search", parameters={"type": "object"})]
        converted = provider.convert_tools(tools)
        assert "functionDeclarations" in converted[0]
        assert converted[0]["functionDeclarations"][0]["name"] == "search"


class TestModels:
    def test_lists_models(self, provider):
        models = provider.get_models()
        assert len(models) >= 4
        ids = [m.model_id for m in models]
        assert "gemini-3-flash-preview" in ids
