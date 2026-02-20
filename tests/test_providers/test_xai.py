"""Tests for the xAI (Grok) provider adapter — SDK-based rewrite."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aratta.config import ProviderConfig, ProviderPriority
from aratta.core.types import (
    ChatRequest,
    Content,
    ContentType,
    FinishReason,
    Message,
    Role,
    Tool,
)


# ---------------------------------------------------------------------------
# Helpers — fake xai_sdk objects
# ---------------------------------------------------------------------------

def _make_sdk_response(
    *,
    content="Hello!",
    finish_reason="stop",
    tool_calls=None,
    thinking=None,
    usage=None,
    response_id="resp-123",
    model="grok-4-1-fast",
):
    """Build a SimpleNamespace that mimics an xai_sdk response."""
    if usage is None:
        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            input_tokens=10,
            output_tokens=20,
            reasoning_tokens=None,
        )
    return SimpleNamespace(
        id=response_id,
        content=content,
        finish_reason=finish_reason,
        tool_calls=tool_calls,
        thinking=thinking,
        encrypted_content=None,
        usage=usage,
        model=model,
    )


# ---------------------------------------------------------------------------
# Fixture — provider with mocked SDK client
# ---------------------------------------------------------------------------

@pytest.fixture
def provider():
    """Create an XAIProvider with a mocked xai_sdk.Client."""
    import aratta.providers.xai.adapter as adapter_mod

    cfg = ProviderConfig(
        name="xai",
        base_url="https://api.x.ai/v1",
        api_key_env=None,
        default_model="grok-4-1-fast",
        priority=ProviderPriority.FALLBACK.value,
    )
    # Inject a mock xai_sdk into the adapter module directly
    mock_sdk = MagicMock()
    original_sdk = getattr(adapter_mod, "xai_sdk", None)
    original_flag = adapter_mod._HAS_XAI_SDK

    adapter_mod.xai_sdk = mock_sdk
    adapter_mod._HAS_XAI_SDK = True
    try:
        p = adapter_mod.XAIProvider(cfg)
        p._sdk_client = mock_sdk.Client.return_value
        yield p
    finally:
        adapter_mod.xai_sdk = original_sdk
        adapter_mod._HAS_XAI_SDK = original_flag


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------

class TestMessageConversion:
    def test_text_message(self, provider):
        msgs = [Message(role=Role.USER, content="hello")]
        converted = provider.convert_messages(msgs)
        assert converted[0]["content"] == "hello"
        assert converted[0]["role"] == "user"

    def test_system_message(self, provider):
        msgs = [Message(role=Role.SYSTEM, content="Be helpful")]
        converted = provider.convert_messages(msgs)
        assert converted[0]["role"] == "system"
        assert converted[0]["content"] == "Be helpful"

    def test_assistant_message(self, provider):
        msgs = [Message(role=Role.ASSISTANT, content="Sure!")]
        converted = provider.convert_messages(msgs)
        assert converted[0]["role"] == "assistant"

    def test_tool_message_with_id(self, provider):
        msgs = [Message(role=Role.TOOL, content="result", tool_call_id="tc-1")]
        converted = provider.convert_messages(msgs)
        assert converted[0]["role"] == "tool"
        assert converted[0]["tool_call_id"] == "tc-1"

    def test_multimodal_content(self, provider):
        msgs = [Message(
            role=Role.USER,
            content=[
                Content(type=ContentType.TEXT, text="Describe this"),
                Content(type=ContentType.IMAGE, image_url="https://example.com/img.png"),
            ],
        )]
        converted = provider.convert_messages(msgs)
        parts = converted[0]["content"]
        assert len(parts) == 2
        assert parts[0]["type"] == "text"
        assert parts[0]["text"] == "Describe this"
        assert parts[1]["type"] == "image_url"
        assert parts[1]["image_url"]["url"] == "https://example.com/img.png"

    def test_empty_content_blocks(self, provider):
        msgs = [Message(role=Role.USER, content=[])]
        converted = provider.convert_messages(msgs)
        assert converted[0]["content"] == ""


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------

class TestToolConversion:
    def test_function_tool_format(self, provider):
        tools = [Tool(name="search", description="Search the web", parameters={"type": "object"})]
        converted = provider.convert_tools(tools)
        assert converted[0]["type"] == "function"
        assert converted[0]["name"] == "search"
        assert converted[0]["description"] == "Search the web"
        assert converted[0]["parameters"] == {"type": "object"}

    def test_multiple_tools(self, provider):
        tools = [
            Tool(name="a", description="A", parameters={}),
            Tool(name="b", description="B", parameters={}),
        ]
        converted = provider.convert_tools(tools)
        assert len(converted) == 2
        assert converted[0]["name"] == "a"
        assert converted[1]["name"] == "b"


# ---------------------------------------------------------------------------
# _build_tools
# ---------------------------------------------------------------------------

class TestBuildTools:
    def test_builtin_tools(self, provider):
        tools = provider._build_tools(None, ["web_search", "x_search"], None)
        assert len(tools) == 2
        assert tools[0] == {"type": "web_search"}
        assert tools[1] == {"type": "x_search"}

    def test_collection_ids(self, provider):
        tools = provider._build_tools(None, None, ["col-1", "col-2"])
        assert len(tools) == 1
        assert tools[0]["type"] == "collections_search"
        assert tools[0]["collection_ids"] == ["col-1", "col-2"]

    def test_user_tools_merged(self, provider):
        user_tools = [Tool(name="calc", description="Calculate", parameters={})]
        tools = provider._build_tools(user_tools, ["web_search"], None)
        assert len(tools) == 2
        assert tools[0]["type"] == "web_search"
        assert tools[1]["name"] == "calc"

    def test_no_tools_returns_none(self, provider):
        assert provider._build_tools(None, None, None) is None

    def test_code_execution_builtin(self, provider):
        tools = provider._build_tools(None, ["code_execution"], None)
        assert tools[0] == {"type": "code_execution"}


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

class TestModels:
    def test_returns_four_models(self, provider):
        models = provider.get_models()
        assert len(models) == 4

    def test_model_ids(self, provider):
        ids = [m.model_id for m in provider.get_models()]
        assert "grok-4" in ids
        assert "grok-4-fast" in ids
        assert "grok-4-1-fast" in ids
        assert "grok-4-1-fast-reasoning" in ids

    def test_grok4_metadata(self, provider):
        m = next(m for m in provider.get_models() if m.model_id == "grok-4")
        assert m.context_window == 256_000
        assert m.supports_thinking is True
        assert m.input_cost_per_million == 3.0
        assert m.output_cost_per_million == 15.0
        assert "reasoning" in m.categories

    def test_grok4_fast_metadata(self, provider):
        m = next(m for m in provider.get_models() if m.model_id == "grok-4-fast")
        assert m.context_window == 131_072
        assert m.input_cost_per_million == 2.0
        assert m.output_cost_per_million == 10.0
        assert "fast" in m.categories

    def test_grok41_fast_metadata(self, provider):
        m = next(m for m in provider.get_models() if m.model_id == "grok-4-1-fast")
        assert m.context_window == 2_000_000
        assert "agentic" in m.categories
        assert "research" in m.categories

    def test_grok41_fast_reasoning_metadata(self, provider):
        m = next(m for m in provider.get_models() if m.model_id == "grok-4-1-fast-reasoning")
        assert m.supports_thinking is True
        assert "reasoning" in m.categories
        assert m.context_window == 2_000_000

    def test_all_models_have_pricing(self, provider):
        for m in provider.get_models():
            assert m.input_cost_per_million is not None and m.input_cost_per_million > 0
            assert m.output_cost_per_million is not None and m.output_cost_per_million > 0

    def test_all_models_provider_is_xai(self, provider):
        for m in provider.get_models():
            assert m.provider == "xai"


# ---------------------------------------------------------------------------
# Response normalization
# ---------------------------------------------------------------------------

class TestNormalizeResponse:
    def test_basic_response(self, provider):
        sdk_resp = _make_sdk_response(content="Hello!")
        result = provider._normalize_response(sdk_resp, "grok-4-1-fast", 42.0)
        assert result.id == "resp-123"
        assert result.content == "Hello!"
        assert result.provider == "xai"
        assert result.model == "grok-4-1-fast"
        assert result.finish_reason == FinishReason.STOP
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 20
        assert result.lineage.provider == "xai"
        assert result.lineage.latency_ms == 42.0

    def test_tool_calls_response(self, provider):
        tc = SimpleNamespace(
            id="tc-1",
            function=SimpleNamespace(name="search", arguments='{"q": "test"}'),
        )
        sdk_resp = _make_sdk_response(tool_calls=[tc], finish_reason="tool_calls")
        result = provider._normalize_response(sdk_resp, "grok-4", 10.0)
        assert result.finish_reason == FinishReason.TOOL_CALLS
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments == {"q": "test"}

    def test_thinking_blocks(self, provider):
        thinking = [SimpleNamespace(thinking="Let me think...", signature="sig-abc")]
        sdk_resp = _make_sdk_response(thinking=thinking)
        result = provider._normalize_response(sdk_resp, "grok-4", 5.0)
        assert result.thinking is not None
        assert len(result.thinking) == 1
        assert result.thinking[0].thinking == "Let me think..."
        assert result.thinking[0].signature == "sig-abc"

    def test_content_as_list_of_blocks(self, provider):
        sdk_resp = _make_sdk_response(
            content=[{"type": "text", "text": "Part 1"}, {"type": "text", "text": "Part 2"}],
        )
        result = provider._normalize_response(sdk_resp, "grok-4", 1.0)
        assert result.content == "Part 1Part 2"

    def test_none_finish_reason_defaults_to_stop(self, provider):
        sdk_resp = _make_sdk_response(finish_reason=None)
        result = provider._normalize_response(sdk_resp, "grok-4", 1.0)
        assert result.finish_reason == FinishReason.STOP

    def test_no_usage_defaults_to_zero(self, provider):
        sdk_resp = _make_sdk_response(usage=None)
        # Remove usage attr to simulate missing
        sdk_resp.usage = None
        result = provider._normalize_response(sdk_resp, "grok-4", 1.0)
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0


# ---------------------------------------------------------------------------
# Chat (integration with mocked SDK)
# ---------------------------------------------------------------------------

class TestChat:
    @pytest.mark.asyncio
    async def test_chat_calls_sdk(self, provider):
        sdk_resp = _make_sdk_response()
        mock_conversation = MagicMock()
        mock_conversation.sample = AsyncMock(return_value=sdk_resp)
        provider._sdk_client.chat.create.return_value = mock_conversation

        request = ChatRequest(
            messages=[Message(role=Role.USER, content="Hi")],
            model="grok-4-1-fast",
        )
        result = await provider.chat(request)

        assert result.content == "Hello!"
        assert result.provider == "xai"
        provider._sdk_client.chat.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_passes_previous_response_id(self, provider):
        sdk_resp = _make_sdk_response()
        mock_conversation = MagicMock()
        mock_conversation.sample = AsyncMock(return_value=sdk_resp)
        provider._sdk_client.chat.create.return_value = mock_conversation

        request = ChatRequest(
            messages=[Message(role=Role.USER, content="Continue")],
            model="grok-4-1-fast",
            metadata={"previous_response_id": "prev-456"},
        )
        await provider.chat(request)

        call_kwargs = provider._sdk_client.chat.create.call_args
        assert call_kwargs.kwargs.get("previous_response_id") == "prev-456"

    @pytest.mark.asyncio
    async def test_chat_passes_store_messages(self, provider):
        sdk_resp = _make_sdk_response()
        mock_conversation = MagicMock()
        mock_conversation.sample = AsyncMock(return_value=sdk_resp)
        provider._sdk_client.chat.create.return_value = mock_conversation

        request = ChatRequest(
            messages=[Message(role=Role.USER, content="Save this")],
            model="grok-4-1-fast",
            metadata={"store_messages": True},
        )
        await provider.chat(request)

        call_kwargs = provider._sdk_client.chat.create.call_args
        assert call_kwargs.kwargs.get("store") is True

    @pytest.mark.asyncio
    async def test_chat_with_thinking_enabled(self, provider):
        thinking = [SimpleNamespace(thinking="Reasoning...", signature="sig")]
        sdk_resp = _make_sdk_response(thinking=thinking)
        mock_conversation = MagicMock()
        mock_conversation.sample = AsyncMock(return_value=sdk_resp)
        provider._sdk_client.chat.create.return_value = mock_conversation

        request = ChatRequest(
            messages=[Message(role=Role.USER, content="Think hard")],
            model="grok-4",
            thinking_enabled=True,
        )
        result = await provider.chat(request)

        call_kwargs = provider._sdk_client.chat.create.call_args
        assert call_kwargs.kwargs.get("use_encrypted_content") is True
        assert result.thinking is not None

    @pytest.mark.asyncio
    async def test_chat_with_builtin_tools(self, provider):
        sdk_resp = _make_sdk_response()
        mock_conversation = MagicMock()
        mock_conversation.sample = AsyncMock(return_value=sdk_resp)
        provider._sdk_client.chat.create.return_value = mock_conversation

        request = ChatRequest(
            messages=[Message(role=Role.USER, content="Search for X")],
            model="grok-4-1-fast",
        )
        await provider.chat(request, builtin_tools=["web_search", "x_search"])

        call_kwargs = provider._sdk_client.chat.create.call_args
        tools = call_kwargs.kwargs.get("tools")
        assert any(t["type"] == "web_search" for t in tools)
        assert any(t["type"] == "x_search" for t in tools)

    @pytest.mark.asyncio
    async def test_chat_with_collection_ids(self, provider):
        sdk_resp = _make_sdk_response()
        mock_conversation = MagicMock()
        mock_conversation.sample = AsyncMock(return_value=sdk_resp)
        provider._sdk_client.chat.create.return_value = mock_conversation

        request = ChatRequest(
            messages=[Message(role=Role.USER, content="Search collections")],
            model="grok-4-1-fast",
        )
        await provider.chat(request, collection_ids=["col-abc"])

        call_kwargs = provider._sdk_client.chat.create.call_args
        tools = call_kwargs.kwargs.get("tools")
        assert any(t["type"] == "collections_search" for t in tools)
