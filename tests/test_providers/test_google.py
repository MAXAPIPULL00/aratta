"""Tests for the Google (Gemini) provider adapter — SDK-based rewrite."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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
# Helpers — fake google-genai SDK response objects
# ---------------------------------------------------------------------------

def _make_sdk_response(
    *,
    text="Hello!",
    finish_reason="STOP",
    function_calls=None,
    usage=None,
    model_version="gemini-3-pro-preview",
):
    """Build a SimpleNamespace that mimics a google-genai GenerateContentResponse."""
    parts = []
    if text:
        parts.append(SimpleNamespace(
            text=text,
            function_call=None,
            executable_code=None,
            code_execution_result=None,
        ))
    for fc in (function_calls or []):
        parts.append(SimpleNamespace(
            text=None,
            function_call=SimpleNamespace(name=fc["name"], args=fc.get("args", {})),
            executable_code=None,
            code_execution_result=None,
        ))

    if usage is None:
        usage = SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=20,
            total_token_count=30,
            cached_content_token_count=None,
        )

    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=parts),
        finish_reason=finish_reason,
    )
    return SimpleNamespace(
        candidates=[candidate],
        usage_metadata=usage,
        model_version=model_version,
    )


# ---------------------------------------------------------------------------
# Fixture — provider with mocked SDK client
# ---------------------------------------------------------------------------

@pytest.fixture
def provider():
    """Create a GoogleProvider with a mocked google.genai.Client."""
    import aratta.providers.google.adapter as adapter_mod

    # Inject a mock genai module
    mock_genai = MagicMock()
    original_genai = getattr(adapter_mod, "genai", None)
    original_flag = adapter_mod._HAS_GENAI_SDK

    adapter_mod.genai = mock_genai
    adapter_mod._HAS_GENAI_SDK = True

    cfg = ProviderConfig(
        name="google",
        base_url="https://generativelanguage.googleapis.com",
        api_key_env=None,
        default_model="gemini-3-flash-preview",
        priority=ProviderPriority.TERTIARY.value,
    )
    try:
        p = adapter_mod.GoogleProvider(cfg)
        p._sdk_client = mock_genai.Client.return_value
        yield p
    finally:
        adapter_mod.genai = original_genai
        adapter_mod._HAS_GENAI_SDK = original_flag


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

class TestImportGuard:
    def test_raises_when_sdk_missing(self):
        import aratta.providers.google.adapter as adapter_mod

        original_flag = adapter_mod._HAS_GENAI_SDK
        adapter_mod._HAS_GENAI_SDK = False
        try:
            cfg = ProviderConfig(
                name="google",
                base_url="https://generativelanguage.googleapis.com",
                api_key_env=None,
                default_model="gemini-3-flash-preview",
                priority=ProviderPriority.TERTIARY.value,
            )
            with pytest.raises(ImportError, match="google-genai is required"):
                adapter_mod.GoogleProvider(cfg)
        finally:
            adapter_mod._HAS_GENAI_SDK = original_flag


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------

class TestMessageConversion:
    def test_system_instruction(self, provider):
        msgs = [
            Message(role=Role.SYSTEM, content="Be helpful"),
            Message(role=Role.USER, content="hi"),
        ]
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

    def test_tool_role_becomes_user(self, provider):
        msgs = [Message(role=Role.TOOL, content="result")]
        _, contents = provider.convert_messages(msgs)
        assert contents[0]["role"] == "user"

    def test_multimodal_content(self, provider):
        msgs = [Message(
            role=Role.USER,
            content=[
                Content(type=ContentType.TEXT, text="Describe this"),
                Content(type=ContentType.IMAGE, image_base64="base64data"),
            ],
        )]
        _, contents = provider.convert_messages(msgs)
        parts = contents[0]["parts"]
        assert len(parts) == 2
        assert parts[0]["text"] == "Describe this"
        assert parts[1]["inlineData"]["data"] == "base64data"

    def test_tool_result_content(self, provider):
        msgs = [Message(
            role=Role.TOOL,
            content=[
                Content(type=ContentType.TOOL_RESULT, tool_name="search", tool_result={"data": "found"}),
            ],
        )]
        _, contents = provider.convert_messages(msgs)
        fr = contents[0]["parts"][0]["functionResponse"]
        assert fr["name"] == "search"
        assert fr["response"] == {"data": "found"}

    def test_tool_use_content(self, provider):
        msgs = [Message(
            role=Role.ASSISTANT,
            content=[
                Content(type=ContentType.TOOL_USE, tool_name="calc", tool_input={"x": 1}),
            ],
        )]
        _, contents = provider.convert_messages(msgs)
        fc = contents[0]["parts"][0]["functionCall"]
        assert fc["name"] == "calc"
        assert fc["args"] == {"x": 1}

    def test_no_system_returns_none(self, provider):
        msgs = [Message(role=Role.USER, content="hi")]
        sys_instr, _ = provider.convert_messages(msgs)
        assert sys_instr is None


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------

class TestToolConversion:
    def test_google_function_declarations(self, provider):
        tools = [Tool(name="search", description="Search", parameters={"type": "object"})]
        converted = provider.convert_tools(tools)
        assert "functionDeclarations" in converted[0]
        assert converted[0]["functionDeclarations"][0]["name"] == "search"

    def test_multiple_tools(self, provider):
        tools = [
            Tool(name="a", description="A", parameters={}),
            Tool(name="b", description="B", parameters={}),
        ]
        converted = provider.convert_tools(tools)
        decls = converted[0]["functionDeclarations"]
        assert len(decls) == 2
        assert decls[0]["name"] == "a"
        assert decls[1]["name"] == "b"


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

class TestModels:
    def test_returns_eight_models(self, provider):
        models = provider.get_models()
        assert len(models) == 8

    def test_existing_model_ids_preserved(self, provider):
        ids = [m.model_id for m in provider.get_models()]
        assert "gemini-3-pro-preview" in ids
        assert "gemini-3-flash-preview" in ids
        assert "gemini-2.5-pro" in ids
        assert "gemini-2.5-flash" in ids
        assert "gemini-2.5-flash-lite" in ids

    def test_new_model_ids(self, provider):
        ids = [m.model_id for m in provider.get_models()]
        assert "gemini-3.1-pro-preview" in ids
        assert "gemini-3.1-pro-preview-customtools" in ids
        assert "gemini-3-pro-image-preview" in ids

    def test_gemini31_pro_metadata(self, provider):
        m = next(m for m in provider.get_models() if m.model_id == "gemini-3.1-pro-preview")
        assert m.context_window == 1_000_000
        assert m.max_output_tokens == 65_000
        assert m.input_cost_per_million == 2.0
        assert m.output_cost_per_million == 12.0
        assert "reasoning" in m.categories
        assert "agentic" in m.categories

    def test_gemini31_customtools_metadata(self, provider):
        m = next(m for m in provider.get_models() if m.model_id == "gemini-3.1-pro-preview-customtools")
        assert "agentic" in m.categories
        assert "tools" in m.categories

    def test_gemini3_image_metadata(self, provider):
        m = next(m for m in provider.get_models() if m.model_id == "gemini-3-pro-image-preview")
        assert "image_generation" in m.categories

    def test_all_models_provider_is_google(self, provider):
        for m in provider.get_models():
            assert m.provider == "google"

    def test_all_models_have_pricing(self, provider):
        for m in provider.get_models():
            assert m.input_cost_per_million is not None and m.input_cost_per_million > 0
            assert m.output_cost_per_million is not None and m.output_cost_per_million > 0


# ---------------------------------------------------------------------------
# Response normalization
# ---------------------------------------------------------------------------

class TestNormalizeResponse:
    def test_basic_response(self, provider):
        sdk_resp = _make_sdk_response(text="Hello!")
        result = provider._normalize(sdk_resp, "gemini-3-pro-preview", 42.0)
        assert result.content == "Hello!"
        assert result.provider == "google"
        assert result.model == "gemini-3-pro-preview"
        assert result.finish_reason == FinishReason.STOP
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 20
        assert result.lineage.provider == "google"
        assert result.lineage.latency_ms == 42.0

    def test_tool_calls_response(self, provider):
        sdk_resp = _make_sdk_response(
            text=None,
            function_calls=[{"name": "search", "args": {"q": "test"}}],
        )
        result = provider._normalize(sdk_resp, "gemini-3-pro-preview", 10.0)
        assert result.finish_reason == FinishReason.TOOL_CALLS
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments == {"q": "test"}

    def test_no_candidates_raises(self, provider):
        resp = SimpleNamespace(candidates=[], usage_metadata=None, model_version=None)
        with pytest.raises(Exception, match="No candidates"):
            provider._normalize(resp, "gemini-3-pro-preview", 1.0)

    def test_no_usage_defaults_to_zero(self, provider):
        sdk_resp = _make_sdk_response()
        sdk_resp.usage_metadata = None
        result = provider._normalize(sdk_resp, "gemini-3-pro-preview", 1.0)
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0


# ---------------------------------------------------------------------------
# Chat (integration with mocked SDK)
# ---------------------------------------------------------------------------

class TestChat:
    @pytest.mark.asyncio
    async def test_chat_calls_sdk(self, provider):
        sdk_resp = _make_sdk_response()
        provider._sdk_client.aio.models.generate_content = AsyncMock(return_value=sdk_resp)

        request = ChatRequest(
            messages=[Message(role=Role.USER, content="Hi")],
            model="gemini-3-pro-preview",
        )
        result = await provider.chat(request)

        assert result.content == "Hello!"
        assert result.provider == "google"
        provider._sdk_client.aio.models.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_passes_system_instruction(self, provider):
        sdk_resp = _make_sdk_response()
        provider._sdk_client.aio.models.generate_content = AsyncMock(return_value=sdk_resp)

        request = ChatRequest(
            messages=[
                Message(role=Role.SYSTEM, content="Be concise"),
                Message(role=Role.USER, content="Hi"),
            ],
            model="gemini-3-pro-preview",
        )
        await provider.chat(request)

        call_kwargs = provider._sdk_client.aio.models.generate_content.call_args
        config = call_kwargs.kwargs.get("config", {})
        assert config.get("system_instruction") == "Be concise"

    @pytest.mark.asyncio
    async def test_chat_with_tools(self, provider):
        sdk_resp = _make_sdk_response()
        provider._sdk_client.aio.models.generate_content = AsyncMock(return_value=sdk_resp)

        request = ChatRequest(
            messages=[Message(role=Role.USER, content="Search")],
            model="gemini-3-pro-preview",
            tools=[Tool(name="search", description="Search", parameters={"type": "object"})],
        )
        await provider.chat(request)

        call_kwargs = provider._sdk_client.aio.models.generate_content.call_args
        config = call_kwargs.kwargs.get("config", {})
        assert "tools" in config

    @pytest.mark.asyncio
    async def test_chat_sdk_error_raises_provider_error(self, provider):
        from aratta.providers.base import ProviderError

        provider._sdk_client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("API error")
        )

        request = ChatRequest(
            messages=[Message(role=Role.USER, content="Hi")],
            model="gemini-3-pro-preview",
        )
        with pytest.raises(ProviderError, match="API error"):
            await provider.chat(request)
