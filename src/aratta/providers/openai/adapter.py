"""
OpenAI provider adapter.

Uses the official openai Python SDK for communication with the OpenAI API.
Supports Responses API (primary) and Chat Completions API (legacy).
GPT-5.2, GPT-4.1, O-series reasoning models, built-in tools.
"""

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

try:
    import openai
    _HAS_OPENAI_SDK = True
except ImportError:
    openai = None  # type: ignore[assignment]
    _HAS_OPENAI_SDK = False

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


class OpenAIProvider(BaseProvider):
    """OpenAI provider using the official openai Python SDK."""

    name = "openai"
    display_name = "OpenAI"

    def __init__(self, config, *, use_responses_api: bool = True):
        if not _HAS_OPENAI_SDK:
            raise ImportError(
                "openai is required for the OpenAI provider. "
                "Install it with: pip install aratta[openai-sdk]"
            )
        super().__init__(config)
        self.use_responses_api = use_responses_api
        self._sdk_client = openai.AsyncOpenAI(api_key=config.api_key)

    # ------------------------------------------------------------------
    # Message conversion  (SCRI → OpenAI format)
    # ------------------------------------------------------------------

    def convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict]]:
        """Convert SCRI Messages to OpenAI message dicts.

        Returns (instructions, converted_msgs) where SYSTEM messages are
        extracted into the instructions string (for Responses API) and all
        other messages become standard OpenAI message dicts.
        """
        instructions = None
        converted: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                instructions = (
                    msg.content
                    if isinstance(msg.content, str)
                    else " ".join(c.text for c in msg.content if c.text)
                )
                continue

            m: dict[str, Any] = {"role": msg.role.value}

            if isinstance(msg.content, str):
                m["content"] = msg.content
            else:
                parts: list[dict[str, Any]] = []
                for b in msg.content:
                    if b.type.value == "text" and b.text:
                        parts.append({"type": "text", "text": b.text})
                    elif b.type.value == "image" and b.image_url:
                        parts.append({
                            "type": "image_url",
                            "image_url": {"url": b.image_url},
                        })
                m["content"] = parts or msg.content

            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id

            converted.append(m)

        return instructions, converted

    # ------------------------------------------------------------------
    # Tool conversion  (SCRI → OpenAI format)
    # ------------------------------------------------------------------

    def convert_tools(self, tools: list[Tool]) -> list[dict]:
        """Convert SCRI Tools to OpenAI function tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    # ------------------------------------------------------------------
    # Chat  (dispatch)
    # ------------------------------------------------------------------

    async def chat(
        self,
        request: ChatRequest,
        *,
        reasoning_effort: str | None = None,
    ) -> ChatResponse:
        start = time.time()
        instructions, msgs = self.convert_messages(request.messages)

        if self.use_responses_api:
            return await self._responses_chat(
                request, instructions, msgs, reasoning_effort, start,
            )
        return await self._completions_chat(
            request, instructions, msgs, reasoning_effort, start,
        )

    # ------------------------------------------------------------------
    # Responses API  (primary path)
    # ------------------------------------------------------------------

    async def _responses_chat(
        self,
        request: ChatRequest,
        instructions: str | None,
        msgs: list[dict],
        reasoning_effort: str | None,
        start: float,
    ) -> ChatResponse:
        kwargs: dict[str, Any] = {"model": request.model}

        # Input: single string for simple user messages, list otherwise
        if (
            len(msgs) == 1
            and msgs[0]["role"] == "user"
            and isinstance(msgs[0]["content"], str)
        ):
            kwargs["input"] = msgs[0]["content"]
        else:
            kwargs["input"] = msgs

        if instructions:
            kwargs["instructions"] = instructions
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}
        if request.max_tokens:
            kwargs["max_output_tokens"] = request.max_tokens
        if request.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                    "strict": True,
                }
                for t in request.tools
            ]
        if request.tool_choice:
            kwargs["tool_choice"] = request.tool_choice

        try:
            response = await self._sdk_client.responses.create(**kwargs)
        except Exception as exc:
            raise ProviderError(str(exc), self.name) from exc

        latency = (time.time() - start) * 1000

        # Extract content
        content = getattr(response, "output_text", "") or ""

        # Extract tool calls and thinking blocks from output items
        tool_calls: list[ToolCall] = []
        thinking: list[ThinkingBlock] = []

        for item in getattr(response, "output", []) or []:
            item_type = getattr(item, "type", None)
            if item_type == "function_call":
                raw_args = getattr(item, "arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": raw_args}
                tool_calls.append(ToolCall(
                    id=getattr(item, "call_id", getattr(item, "id", "")),
                    name=getattr(item, "name", ""),
                    arguments=args,
                ))
            elif item_type == "reasoning":
                thinking.append(ThinkingBlock(
                    thinking=getattr(item, "content", ""),
                    signature=getattr(item, "encrypted_content", None),
                ))

        # Finish reason
        finish = FinishReason.STOP
        status = getattr(response, "status", None)
        if status == "incomplete":
            details = getattr(response, "incomplete_details", None)
            reason = getattr(details, "reason", None) if details else None
            if reason == "max_output_tokens":
                finish = FinishReason.LENGTH
            elif reason == "content_filter":
                finish = FinishReason.CONTENT_FILTER
        elif tool_calls:
            finish = FinishReason.TOOL_CALLS

        # Usage
        u = getattr(response, "usage", None)
        usage = Usage(
            input_tokens=getattr(u, "input_tokens", 0),
            output_tokens=getattr(u, "output_tokens", 0),
            total_tokens=getattr(u, "total_tokens", 0),
            reasoning_tokens=getattr(
                getattr(u, "output_tokens_details", None),
                "reasoning_tokens", None,
            ),
            cache_read_tokens=getattr(
                getattr(u, "input_tokens_details", None),
                "cached_tokens", None,
            ),
        ) if u else Usage(input_tokens=0, output_tokens=0, total_tokens=0)

        resp_id = getattr(response, "id", "") or ""
        resp_model = getattr(response, "model", request.model) or request.model

        return ChatResponse(
            id=resp_id,
            content=content,
            tool_calls=tool_calls or None,
            thinking=thinking or None,
            model=resp_model,
            provider="openai",
            finish_reason=finish,
            usage=usage,
            lineage=Lineage(
                provider="openai",
                model=resp_model,
                request_id=resp_id,
                latency_ms=latency,
            ),
        )

    # ------------------------------------------------------------------
    # Chat Completions API  (legacy path)
    # ------------------------------------------------------------------

    async def _completions_chat(
        self,
        request: ChatRequest,
        instructions: str | None,
        msgs: list[dict],
        reasoning_effort: str | None,
        start: float,
    ) -> ChatResponse:
        # Inject system message for completions API
        if instructions:
            msgs.insert(0, {"role": "system", "content": instructions})

        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": msgs,
        }

        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        elif request.temperature != 0.7:
            kwargs["temperature"] = request.temperature

        if request.max_tokens:
            kwargs["max_completion_tokens"] = request.max_tokens
        if request.tools:
            kwargs["tools"] = self.convert_tools(request.tools)
        if request.tool_choice:
            kwargs["tool_choice"] = request.tool_choice

        try:
            response = await self._sdk_client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise ProviderError(str(exc), self.name) from exc

        latency = (time.time() - start) * 1000

        choice = response.choices[0]
        msg = choice.message

        # Tool calls
        tool_calls = None
        if msg.tool_calls:
            tool_calls = []
            for tc in msg.tool_calls:
                raw_args = tc.function.arguments
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": raw_args}
                tool_calls.append(ToolCall(tc.id, tc.function.name, args))

        # Finish reason
        fr_map = {
            "stop": FinishReason.STOP,
            "tool_calls": FinishReason.TOOL_CALLS,
            "length": FinishReason.LENGTH,
            "content_filter": FinishReason.CONTENT_FILTER,
        }
        finish = fr_map.get(choice.finish_reason or "stop", FinishReason.STOP)

        # Usage
        u = response.usage
        usage = Usage(
            input_tokens=u.prompt_tokens if u else 0,
            output_tokens=u.completion_tokens if u else 0,
            total_tokens=u.total_tokens if u else 0,
        ) if u else Usage(input_tokens=0, output_tokens=0, total_tokens=0)

        resp_id = response.id or ""
        resp_model = response.model or request.model

        return ChatResponse(
            id=resp_id,
            content=msg.content or "",
            tool_calls=tool_calls,
            model=resp_model,
            provider="openai",
            finish_reason=finish,
            usage=usage,
            lineage=Lineage(
                provider="openai",
                model=resp_model,
                request_id=resp_id,
                latency_ms=latency,
            ),
        )

    # ------------------------------------------------------------------
    # Chat  (streaming)
    # ------------------------------------------------------------------

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        _, msgs = self.convert_messages(request.messages)

        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": msgs,
            "stream": True,
        }
        if request.tools:
            kwargs["tools"] = self.convert_tools(request.tools)

        try:
            stream = await self._sdk_client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield f"data: {json.dumps({'content': delta.content})}\n\n"
        except Exception as exc:
            raise ProviderError(str(exc), self.name) from exc

        yield "data: [DONE]\n\n"

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "input": request.input,
        }
        if request.dimensions:
            kwargs["dimensions"] = request.dimensions

        try:
            response = await self._sdk_client.embeddings.create(**kwargs)
        except Exception as exc:
            raise ProviderError(str(exc), self.name) from exc

        embeddings = [
            Embedding(item.embedding, item.index) for item in response.data
        ]

        u = getattr(response, "usage", None)
        usage = Usage(
            input_tokens=getattr(u, "prompt_tokens", 0),
            output_tokens=0,
            total_tokens=getattr(u, "total_tokens", 0),
        ) if u else Usage(input_tokens=0, output_tokens=0, total_tokens=0)

        return EmbeddingResponse(
            embeddings, response.model, "openai", usage,
        )

    # ------------------------------------------------------------------
    # Model registry
    # ------------------------------------------------------------------

    def get_models(self) -> list[ModelCapabilities]:
        return [
            # --- Existing models (preserved) ---
            ModelCapabilities(
                model_id="gpt-4.1",
                provider="openai",
                display_name="GPT-4.1",
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_thinking=False,
                context_window=1_000_000,
                max_output_tokens=32768,
                input_cost_per_million=2.0,
                output_cost_per_million=8.0,
                categories=["chat", "code"],
            ),
            ModelCapabilities(
                model_id="gpt-4.1-mini",
                provider="openai",
                display_name="GPT-4.1 Mini",
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_thinking=False,
                context_window=1_000_000,
                max_output_tokens=32768,
                input_cost_per_million=0.4,
                output_cost_per_million=1.6,
                categories=["chat", "fast"],
            ),
            ModelCapabilities(
                model_id="gpt-4.1-nano",
                provider="openai",
                display_name="GPT-4.1 Nano",
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_thinking=False,
                context_window=1_000_000,
                max_output_tokens=32768,
                input_cost_per_million=0.1,
                output_cost_per_million=0.4,
                categories=["fast"],
            ),
            ModelCapabilities(
                model_id="o3",
                provider="openai",
                display_name="O3",
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True,
                supports_json_mode=False,
                supports_thinking=True,
                context_window=200_000,
                max_output_tokens=100_000,
                input_cost_per_million=2.0,
                output_cost_per_million=8.0,
                categories=["reasoning"],
            ),
            # --- New GPT-5.2 models ---
            ModelCapabilities(
                model_id="gpt-5.2",
                provider="openai",
                display_name="GPT-5.2",
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_thinking=False,
                context_window=1_000_000,
                max_output_tokens=32768,
                input_cost_per_million=1.75,
                output_cost_per_million=14.0,
                categories=["chat", "reasoning"],
            ),
            ModelCapabilities(
                model_id="gpt-5.2-pro",
                provider="openai",
                display_name="GPT-5.2 Pro",
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_thinking=True,
                context_window=1_000_000,
                max_output_tokens=32768,
                input_cost_per_million=3.0,
                output_cost_per_million=18.0,
                categories=["reasoning", "pro"],
            ),
            ModelCapabilities(
                model_id="gpt-5.2-codex",
                provider="openai",
                display_name="GPT-5.2 Codex",
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_thinking=False,
                context_window=1_000_000,
                max_output_tokens=32768,
                input_cost_per_million=1.75,
                output_cost_per_million=14.0,
                categories=["code"],
            ),
        ]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self):
        """Clean up both the SDK client and inherited httpx client."""
        if hasattr(self, "_sdk_client") and hasattr(self._sdk_client, "close"):
            try:
                await self._sdk_client.close()
            except Exception:
                pass  # Best-effort cleanup
        await super().close()
