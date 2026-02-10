"""Tests for the local provider adapter (Ollama / vLLM / llama.cpp)."""

import pytest

from aratta.config import ProviderConfig, ProviderPriority
from aratta.core.types import Message, Role, Tool
from aratta.providers.local import LocalProvider


@pytest.fixture
def ollama_provider():
    cfg = ProviderConfig(
        name="ollama", base_url="http://localhost:11434",
        api_key_env=None, default_model="llama3.1:8b",
        priority=ProviderPriority.LOCAL.value,
    )
    return LocalProvider(cfg)


@pytest.fixture
def vllm_provider():
    cfg = ProviderConfig(
        name="vllm", base_url="http://localhost:8000",
        api_key_env=None, default_model="meta-llama/Llama-3.1-8B-Instruct",
        priority=ProviderPriority.LOCAL.value,
    )
    return LocalProvider(cfg)


class TestLocalDetection:
    def test_ollama_detected(self, ollama_provider):
        assert ollama_provider._is_ollama is True

    def test_vllm_not_ollama(self, vllm_provider):
        assert vllm_provider._is_ollama is False

    def test_chat_path(self, ollama_provider):
        assert ollama_provider._chat_path == "/v1/chat/completions"


class TestMessageConversion:
    def test_text_message(self, ollama_provider):
        msgs = [Message(role=Role.USER, content="hello")]
        converted = ollama_provider.convert_messages(msgs)
        assert converted[0]["role"] == "user"
        assert converted[0]["content"] == "hello"

    def test_system_passes_through(self, ollama_provider):
        msgs = [Message(role=Role.SYSTEM, content="Be helpful"), Message(role=Role.USER, content="hi")]
        converted = ollama_provider.convert_messages(msgs)
        assert len(converted) == 2
        assert converted[0]["role"] == "system"


class TestToolConversion:
    def test_openai_compatible_format(self, ollama_provider):
        tools = [Tool(name="search", description="Search", parameters={"type": "object"})]
        converted = ollama_provider.convert_tools(tools)
        assert converted[0]["type"] == "function"
        assert converted[0]["function"]["name"] == "search"


class TestModels:
    def test_returns_configured_default(self, ollama_provider):
        models = ollama_provider.get_models()
        assert len(models) == 1
        assert models[0].model_id == "llama3.1:8b"
        assert "local" in models[0].categories

    def test_vllm_model(self, vllm_provider):
        models = vllm_provider.get_models()
        assert models[0].model_id == "meta-llama/Llama-3.1-8B-Instruct"


class TestNoAuth:
    def test_no_auth_header(self, ollama_provider):
        headers = ollama_provider._get_headers()
        assert "Authorization" not in headers
        assert "x-api-key" not in headers
