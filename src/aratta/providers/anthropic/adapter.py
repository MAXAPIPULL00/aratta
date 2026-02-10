"""
Anthropic (Claude) provider adapter.

Supports Claude 4.5 series with extended thinking, effort control,
prompt caching, tool calling, vision, and streaming.
"""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from aratta.core.types import (
    ChatRequest,
    ChatResponse,
    ContentType,
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

FINISH_MAP = {
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "tool_use": FinishReason.TOOL_CALLS,
    "max_tokens": FinishReason.LENGTH,
}


class AnthropicProvider(BaseProvider):
    name = "anthropic"
    display_name = "Anthropic (Claude)"
    API_VERSION = "2023-06-01"

    def _get_headers(self, request: ChatRequest | None = None) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key or "",
            "anthropic-version": self.API_VERSION,
            "User-Agent": "Aratta/0.1.0",
        }
        betas = []
        if request and request.thinking_enabled:
            betas.append("extended-thinking-2025-01-24")
        if betas:
            headers["anthropic-beta"] = ",".join(sorted(set(betas)))
        return headers

    def get_models(self) -> list[ModelCapabilities]:
        return [
            ModelCapabilities("claude-opus-4-5-20251101", "anthropic", "Claude Opus 4.5",
                              True, True, True, True, True, 200000, 64000, 5.0, 25.0, ["chat", "reasoning", "code"]),
            ModelCapabilities("claude-sonnet-4-5-20250929", "anthropic", "Claude Sonnet 4.5",
                              True, True, True, True, True, 200000, 64000, 3.0, 15.0, ["chat", "code"]),
            ModelCapabilities("claude-haiku-4-5-20251001", "anthropic", "Claude Haiku 4.5",
                              True, True, True, True, True, 200000, 64000, 1.0, 5.0, ["chat", "fast"]),
        ]

    def convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict]]:
        system_msg = None
        converted = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_msg = msg.content if isinstance(msg.content, str) else "\n".join(
                    c.text for c in msg.content if c.type == ContentType.TEXT and c.text)
                continue
            m: dict[str, Any] = {"role": msg.role.value}
            if isinstance(msg.content, str):
                m["content"] = msg.content
            else:
                blocks = []
                for b in msg.content:
                    if b.type == ContentType.TEXT:
                        blocks.append({"type": "text", "text": b.text})
                    elif b.type == ContentType.IMAGE and b.image_base64:
                        blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b.image_base64}})
                    elif b.type == ContentType.IMAGE and b.image_url:
                        blocks.append({"type": "image", "source": {"type": "url", "url": b.image_url}})
                    elif b.type == ContentType.TOOL_RESULT:
                        blocks.append({"type": "tool_result", "tool_use_id": b.tool_use_id,
                                       "content": json.dumps(b.tool_result) if isinstance(b.tool_result, dict) else str(b.tool_result)})
                m["content"] = blocks or msg.content
            converted.append(m)
        return system_msg, converted

    def convert_tools(self, tools: list[Tool]) -> list[dict]:
        return [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        start = time.time()
        system_msg, messages = self.convert_messages(request.messages)
        body: dict[str, Any] = {"model": request.model, "messages": messages, "max_tokens": request.max_tokens or 4096}
        if system_msg:
            body["system"] = system_msg
        if not request.thinking_enabled and request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.stop:
            body["stop_sequences"] = request.stop
        if request.tools:
            body["tools"] = self.convert_tools(request.tools)
            if request.tool_choice:
                body["tool_choice"] = {"type": "auto"} if request.tool_choice == "auto" else (
                    {"type": "any"} if request.tool_choice == "required" else request.tool_choice)
        if request.thinking_enabled:
            body["thinking"] = {"type": "enabled", "budget_tokens": max(1024, request.thinking_budget)}

        headers = self._get_headers(request)
        resp = await self.client.post("/v1/messages", json=body, headers=headers)
        if resp.status_code != 200:
            self._handle_error(resp)
        return self._normalize(resp.json(), request.model, (time.time() - start) * 1000)

    def _normalize(self, data: dict, model: str, latency: float) -> ChatResponse:
        text = ""
        tool_calls: list[ToolCall] = []
        thinking: list[ThinkingBlock] = []
        for block in data.get("content", []):
            bt = block.get("type")
            if bt == "text":
                text += block.get("text", "")
            elif bt == "thinking":
                thinking.append(ThinkingBlock(block.get("thinking", ""), block.get("signature")))
            elif bt == "tool_use":
                tool_calls.append(ToolCall(block.get("id", f"tool_{uuid.uuid4().hex[:8]}"), block.get("name", ""), block.get("input", {})))
        u = data.get("usage", {})
        return ChatResponse(
            id=data.get("id", f"msg_{uuid.uuid4().hex[:12]}"), content=text,
            tool_calls=tool_calls or None, thinking=thinking or None,
            model=data.get("model", model), provider="anthropic",
            finish_reason=FINISH_MAP.get(data.get("stop_reason", "end_turn"), FinishReason.STOP),
            usage=Usage(u.get("input_tokens", 0), u.get("output_tokens", 0),
                        u.get("input_tokens", 0) + u.get("output_tokens", 0),
                        u.get("cache_read_input_tokens"), u.get("cache_creation_input_tokens")),
            lineage=Lineage("anthropic", data.get("model", model), request_id=data.get("id"), latency_ms=latency),
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        system_msg, messages = self.convert_messages(request.messages)
        body: dict[str, Any] = {"model": request.model, "messages": messages, "max_tokens": request.max_tokens or 4096, "stream": True}
        if system_msg:
            body["system"] = system_msg
        if not request.thinking_enabled and request.temperature is not None:
            body["temperature"] = request.temperature
        if request.tools:
            body["tools"] = self.convert_tools(request.tools)
        if request.thinking_enabled:
            body["thinking"] = {"type": "enabled", "budget_tokens": max(1024, request.thinking_budget)}
        headers = self._get_headers(request)
        async with self.client.stream("POST", "/v1/messages", json=body, headers=headers) as resp:
            if resp.status_code != 200:
                self._handle_error(resp)
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    raw = line[6:]
                    if raw == "[DONE]":
                        break
                    try:
                        evt = json.loads(raw)
                        chunk = self._stream_chunk(evt)
                        if chunk:
                            yield f"data: {json.dumps(chunk)}\n\n"
                    except json.JSONDecodeError:
                        continue
        yield "data: [DONE]\n\n"

    def _stream_chunk(self, evt: dict) -> dict | None:
        t = evt.get("type")
        if t == "content_block_delta":
            d = evt.get("delta", {})
            dt = d.get("type")
            if dt == "text_delta":
                return {"type": "content", "content": d.get("text", "")}
            if dt == "thinking_delta":
                return {"type": "thinking", "thinking": d.get("thinking", "")}
        elif t == "message_start":
            return {"type": "start", "id": evt.get("message", {}).get("id"), "model": evt.get("message", {}).get("model")}
        elif t == "message_stop":
            return {"type": "stop", "finish_reason": "stop"}
        return None

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        raise ProviderError("Anthropic does not support embeddings. Use OpenAI.", self.name)

    async def health_check(self) -> dict[str, Any]:
        try:
            resp = await self.client.post("/v1/messages", json={
                "model": "claude-haiku-4-5-20251001", "max_tokens": 1,
                "messages": [{"role": "user", "content": "hi"}],
            }, headers=self._get_headers())
            return {"status": "healthy" if resp.status_code == 200 else "degraded",
                    "provider": self.name, "latency_ms": resp.elapsed.total_seconds() * 1000}
        except Exception as e:
            return {"status": "unhealthy", "provider": self.name, "error": str(e)}
