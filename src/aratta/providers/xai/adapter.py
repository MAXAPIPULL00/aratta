"""
xAI (Grok) provider adapter.

Uses the official xai_sdk with the Responses API for gRPC-based communication.
Supports agentic server-side tools (web_search, x_search, code_execution,
collections_search), encrypted thinking traces, conversation chaining via
previous_response_id, and server-side message persistence via store_messages.
"""

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

try:
    import xai_sdk
    _HAS_XAI_SDK = True
except ImportError:
    xai_sdk = None  # type: ignore[assignment]
    _HAS_XAI_SDK = False

from aratta.core.types import (
    ChatRequest,
    ChatResponse,
    Embedding,
    EmbeddingRequest,
    EmbeddingResponse,
    FinishReason,
    Lineage,
    Message,
    ModelCapabilities,
    Role,
    ThinkingBlock,
    Tool,
    ToolCall,
    Usage,
)
from aratta.providers.base import BaseProvider, ProviderError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Server-side builtin tool types supported by xAI
# ---------------------------------------------------------------------------
_BUILTIN_TOOL_TYPES = frozenset({"web_search", "x_search", "code_execution"})


class XAIProvider(BaseProvider):
    """xAI (Grok) provider using the official xai_sdk."""

    name = "xai"
    display_name = "xAI (Grok)"

    def __init__(self, config):
        if not _HAS_XAI_SDK:
            raise ImportError(
                "xai-sdk is required for the xAI provider. "
                "Install it with: pip install aratta[xai]"
            )
        super().__init__(config)
        self._sdk_client = xai_sdk.Client(api_key=config.api_key)

    # ------------------------------------------------------------------
    # Message conversion  (SCRI → xAI SDK format)
    # ------------------------------------------------------------------

    def convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert SCRI Messages to xAI SDK message dicts."""
        converted: list[dict[str, Any]] = []
        for msg in messages:
            m: dict[str, Any] = {"role": msg.role.value}

            if isinstance(msg.content, str):
                m["content"] = msg.content
            else:
                parts: list[dict[str, Any]] = []
                for block in msg.content:
                    if block.type.value == "text" and block.text:
                        parts.append({"type": "text", "text": block.text})
                    elif block.type.value == "image" and block.image_url:
                        parts.append({
                            "type": "image_url",
                            "image_url": {"url": block.image_url},
                        })
                m["content"] = parts if parts else ""

            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            if msg.name:
                m["name"] = msg.name

            converted.append(m)
        return converted

    # ------------------------------------------------------------------
    # Tool conversion  (SCRI → xAI SDK format)
    # ------------------------------------------------------------------

    def convert_tools(self, tools: list[Tool]) -> list[dict[str, Any]]:
        """Convert SCRI Tools to xAI SDK function tool definitions."""
        return [
            {
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in tools
        ]

    def _build_tools(
        self,
        user_tools: list[Tool] | None,
        builtin_tools: list[str] | None,
        collection_ids: list[str] | None,
    ) -> list[dict[str, Any]] | None:
        """Merge user-defined tools with server-side builtin tools."""
        tools: list[dict[str, Any]] = []

        # Server-side builtin tools
        for bt in builtin_tools or []:
            if bt in _BUILTIN_TOOL_TYPES:
                tools.append({"type": bt})
            else:
                logger.warning("Unknown builtin tool type: %s", bt)

        # Collections search with IDs
        if collection_ids:
            tools.append({
                "type": "collections_search",
                "collection_ids": collection_ids,
            })

        # User-defined function tools
        if user_tools:
            tools.extend(self.convert_tools(user_tools))

        return tools if tools else None


    # ------------------------------------------------------------------
    # Chat  (non-streaming)
    # ------------------------------------------------------------------

    async def chat(
        self,
        request: ChatRequest,
        *,
        builtin_tools: list[str] | None = None,
        collection_ids: list[str] | None = None,
    ) -> ChatResponse:
        start = time.time()
        messages = self.convert_messages(request.messages)
        tools = self._build_tools(request.tools, builtin_tools, collection_ids)

        # Build SDK conversation kwargs
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
        }
        if request.temperature != 0.7:
            kwargs["temperature"] = request.temperature
        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens
        if tools:
            kwargs["tools"] = tools
        if request.tool_choice:
            kwargs["tool_choice"] = request.tool_choice

        # Metadata-driven features
        meta = request.metadata or {}
        if "previous_response_id" in meta:
            kwargs["previous_response_id"] = meta["previous_response_id"]
        if meta.get("store_messages"):
            kwargs["store"] = True

        # Thinking / encrypted content
        if request.thinking_enabled:
            kwargs["use_encrypted_content"] = True

        try:
            conversation = self._sdk_client.chat.create(**kwargs)
            response = await conversation.sample()
        except Exception as exc:
            raise ProviderError(str(exc), self.name) from exc

        latency = (time.time() - start) * 1000
        return self._normalize_response(response, request.model, latency)

    # ------------------------------------------------------------------
    # Chat  (streaming)
    # ------------------------------------------------------------------

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        messages = self.convert_messages(request.messages)
        tools = self._build_tools(request.tools, None, None)

        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": True,
        }
        if request.temperature != 0.7:
            kwargs["temperature"] = request.temperature
        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens
        if tools:
            kwargs["tools"] = tools

        if request.thinking_enabled:
            kwargs["use_encrypted_content"] = True

        try:
            conversation = self._sdk_client.chat.create(**kwargs)
            async for chunk in conversation.stream():
                # Yield SSE-formatted chunks
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            raise ProviderError(str(exc), self.name) from exc


    # ------------------------------------------------------------------
    # Embeddings  (not supported by xAI SDK — stub for interface)
    # ------------------------------------------------------------------

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """xAI does not currently expose an embedding endpoint via the SDK."""
        raise NotImplementedError("xAI does not support embeddings via xai_sdk.")

    # ------------------------------------------------------------------
    # Model registry
    # ------------------------------------------------------------------

    def get_models(self) -> list[ModelCapabilities]:
        return [
            ModelCapabilities(
                model_id="grok-4",
                provider="xai",
                display_name="Grok 4",
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_thinking=True,
                context_window=256_000,
                max_output_tokens=16_384,
                input_cost_per_million=3.0,
                output_cost_per_million=15.0,
                categories=["reasoning"],
            ),
            ModelCapabilities(
                model_id="grok-4-fast",
                provider="xai",
                display_name="Grok 4 Fast",
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_thinking=False,
                context_window=131_072,
                max_output_tokens=16_384,
                input_cost_per_million=2.0,
                output_cost_per_million=10.0,
                categories=["agentic", "fast"],
            ),
            ModelCapabilities(
                model_id="grok-4-1-fast",
                provider="xai",
                display_name="Grok 4.1 Fast",
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_thinking=False,
                context_window=2_000_000,
                max_output_tokens=16_384,
                input_cost_per_million=2.0,
                output_cost_per_million=10.0,
                categories=["agentic", "research"],
            ),
            ModelCapabilities(
                model_id="grok-4-1-fast-reasoning",
                provider="xai",
                display_name="Grok 4.1 Fast Reasoning",
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_thinking=True,
                context_window=2_000_000,
                max_output_tokens=16_384,
                input_cost_per_million=2.0,
                output_cost_per_million=10.0,
                categories=["reasoning", "agentic"],
            ),
        ]

    # ------------------------------------------------------------------
    # Response normalization  (xAI SDK → SCRI ChatResponse)
    # ------------------------------------------------------------------

    def _normalize_response(
        self,
        response: Any,
        model: str,
        latency_ms: float,
    ) -> ChatResponse:
        """Map an xAI SDK response object to a SCRI ChatResponse."""
        # Extract content
        content = getattr(response, "content", "") or ""
        if isinstance(content, list):
            # SDK may return content blocks — join text parts
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)
            content = "".join(text_parts)

        # Extract tool calls
        tool_calls = None
        raw_tool_calls = getattr(response, "tool_calls", None)
        if raw_tool_calls:
            tool_calls = []
            for tc in raw_tool_calls:
                if hasattr(tc, "function"):
                    args = tc.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            args = {"raw": args}
                    tool_calls.append(ToolCall(
                        id=getattr(tc, "id", ""),
                        name=tc.function.name,
                        arguments=args,
                    ))
            tool_calls = tool_calls or None

        # Extract thinking blocks (encrypted content)
        thinking = None
        raw_thinking = getattr(response, "thinking", None) or getattr(response, "encrypted_content", None)
        if raw_thinking:
            if isinstance(raw_thinking, list):
                thinking = [
                    ThinkingBlock(
                        thinking=getattr(t, "thinking", "") or getattr(t, "content", "") or str(t),
                        signature=getattr(t, "signature", None),
                    )
                    for t in raw_thinking
                ]
            elif isinstance(raw_thinking, str):
                thinking = [ThinkingBlock(thinking=raw_thinking)]

        # Finish reason mapping
        raw_finish = getattr(response, "finish_reason", "stop")
        if raw_finish is None:
            raw_finish = "stop"
        fr_map = {
            "stop": FinishReason.STOP,
            "tool_calls": FinishReason.TOOL_CALLS,
            "length": FinishReason.LENGTH,
            "content_filter": FinishReason.CONTENT_FILTER,
        }
        finish_reason = fr_map.get(str(raw_finish), FinishReason.STOP)

        # Usage
        raw_usage = getattr(response, "usage", None)
        if raw_usage:
            usage = Usage(
                input_tokens=getattr(raw_usage, "prompt_tokens", 0) or getattr(raw_usage, "input_tokens", 0),
                output_tokens=getattr(raw_usage, "completion_tokens", 0) or getattr(raw_usage, "output_tokens", 0),
                total_tokens=getattr(raw_usage, "total_tokens", 0),
                reasoning_tokens=getattr(raw_usage, "reasoning_tokens", None),
            )
        else:
            usage = Usage(input_tokens=0, output_tokens=0, total_tokens=0)

        response_id = getattr(response, "id", "") or ""
        response_model = getattr(response, "model", model) or model

        return ChatResponse(
            id=response_id,
            content=content,
            tool_calls=tool_calls,
            thinking=thinking,
            model=response_model,
            provider="xai",
            finish_reason=finish_reason,
            usage=usage,
            lineage=Lineage(
                provider="xai",
                model=response_model,
                request_id=response_id,
                latency_ms=latency_ms,
            ),
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self):
        """Clean up both the SDK client and inherited httpx client."""
        if hasattr(self._sdk_client, "close"):
            try:
                await self._sdk_client.close()
            except Exception:
                pass  # Best-effort cleanup
        await super().close()
