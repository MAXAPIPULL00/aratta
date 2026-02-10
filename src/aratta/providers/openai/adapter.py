"""
OpenAI provider adapter.

Supports Responses API (primary) and Chat Completions API (legacy).
GPT-5.x, Codex, O-series reasoning models, built-in tools.
"""

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from aratta.config import ProviderConfig
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
from aratta.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    name = "openai"
    display_name = "OpenAI"

    def __init__(self, config: ProviderConfig, *, use_responses_api: bool = True):
        super().__init__(config)
        self.use_responses_api = use_responses_api

    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key or ''}",
            "Content-Type": "application/json",
            "User-Agent": "Aratta/0.1.0",
        }

    def get_models(self) -> list[ModelCapabilities]:
        return [
            ModelCapabilities("gpt-4.1", "openai", "GPT-4.1", True, True, True, True, False, 1_000_000, 32768, 2.0, 8.0, ["chat", "code"]),
            ModelCapabilities("gpt-4.1-mini", "openai", "GPT-4.1 Mini", True, True, True, True, False, 1_000_000, 32768, 0.4, 1.6, ["chat", "fast"]),
            ModelCapabilities("gpt-4.1-nano", "openai", "GPT-4.1 Nano", True, True, True, True, False, 1_000_000, 32768, 0.1, 0.4, ["fast"]),
            ModelCapabilities("o3", "openai", "O3", True, True, True, False, True, 200000, 100000, 2.0, 8.0, ["reasoning"]),
        ]

    def convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict]]:
        instructions = None
        converted = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                instructions = msg.content if isinstance(msg.content, str) else " ".join(c.text for c in msg.content if c.text)
                continue
            m: dict[str, Any] = {"role": msg.role.value}
            if isinstance(msg.content, str):
                m["content"] = msg.content
            else:
                parts = []
                for b in msg.content:
                    if b.type.value == "text" and b.text:
                        parts.append({"type": "text", "text": b.text})
                    elif b.type.value == "image" and b.image_url:
                        parts.append({"type": "image_url", "image_url": {"url": b.image_url}})
                m["content"] = parts or msg.content
            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            converted.append(m)
        return instructions, converted

    def convert_tools(self, tools: list[Tool]) -> list[dict]:
        return [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}} for t in tools]

    async def chat(self, request: ChatRequest, *, reasoning_effort: str | None = None) -> ChatResponse:
        start = time.time()
        instructions, msgs = self.convert_messages(request.messages)

        if self.use_responses_api:
            return await self._responses_chat(request, instructions, msgs, reasoning_effort, start)
        return await self._completions_chat(request, instructions, msgs, reasoning_effort, start)

    async def _responses_chat(self, request: ChatRequest, instructions: str | None, msgs: list[dict], reasoning_effort: str | None, start: float) -> ChatResponse:
        payload: dict[str, Any] = {"model": request.model}
        if len(msgs) == 1 and msgs[0]["role"] == "user" and isinstance(msgs[0]["content"], str):
            payload["input"] = msgs[0]["content"]
        else:
            payload["input"] = msgs
        if instructions:
            payload["instructions"] = instructions
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}
        if request.max_tokens:
            payload["max_output_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = [{"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters, "strict": True} for t in request.tools]
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice

        resp = await self.client.post("/responses", json=payload, headers=self._get_headers())
        if resp.status_code >= 400:
            self._handle_error(resp)
        data = resp.json()
        latency = (time.time() - start) * 1000

        content = data.get("output_text", "")
        tool_calls = []
        thinking = []
        for item in data.get("output", []):
            if item.get("type") == "function_call":
                tool_calls.append(ToolCall(item.get("call_id", item.get("id", "")), item.get("name", ""), json.loads(item.get("arguments", "{}"))))
            elif item.get("type") == "reasoning":
                thinking.append(ThinkingBlock(item.get("content", ""), item.get("encrypted_content")))

        finish = FinishReason.STOP
        if data.get("status") == "incomplete":
            reason = data.get("incomplete_details", {}).get("reason")
            finish = FinishReason.LENGTH if reason == "max_output_tokens" else FinishReason.CONTENT_FILTER if reason == "content_filter" else FinishReason.STOP
        elif tool_calls:
            finish = FinishReason.TOOL_CALLS

        u = data.get("usage", {})
        return ChatResponse(
            id=data.get("id", ""), content=content,
            tool_calls=tool_calls or None, thinking=thinking or None,
            model=data.get("model", request.model), provider="openai", finish_reason=finish,
            usage=Usage(u.get("input_tokens", 0), u.get("output_tokens", 0), u.get("total_tokens", 0),
                        reasoning_tokens=u.get("output_tokens_details", {}).get("reasoning_tokens"),
                        cache_read_tokens=u.get("input_tokens_details", {}).get("cached_tokens")),
            lineage=Lineage("openai", data.get("model", request.model), request_id=data.get("id"), latency_ms=latency),
        )

    async def _completions_chat(self, request: ChatRequest, instructions: str | None, msgs: list[dict], reasoning_effort: str | None, start: float) -> ChatResponse:
        if instructions:
            msgs.insert(0, {"role": "system", "content": instructions})
        payload: dict[str, Any] = {"model": request.model, "messages": msgs}
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        elif request.temperature != 0.7:
            payload["temperature"] = request.temperature
        if request.max_tokens:
            payload["max_completion_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = self.convert_tools(request.tools)
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice

        resp = await self.client.post("/chat/completions", json=payload, headers=self._get_headers())
        if resp.status_code >= 400:
            self._handle_error(resp)
        data = resp.json()
        latency = (time.time() - start) * 1000

        choice = data["choices"][0]
        msg = choice["message"]
        tool_calls = None
        if "tool_calls" in msg:
            tool_calls = [ToolCall(tc["id"], tc["function"]["name"], json.loads(tc["function"]["arguments"])) for tc in msg["tool_calls"]]

        fr_map = {"stop": FinishReason.STOP, "tool_calls": FinishReason.TOOL_CALLS, "length": FinishReason.LENGTH, "content_filter": FinishReason.CONTENT_FILTER}
        u = data.get("usage", {})
        return ChatResponse(
            id=data.get("id", ""), content=msg.get("content", "") or "",
            tool_calls=tool_calls, model=data.get("model", request.model), provider="openai",
            finish_reason=fr_map.get(choice.get("finish_reason", "stop"), FinishReason.STOP),
            usage=Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0), u.get("total_tokens", 0)),
            lineage=Lineage("openai", data.get("model", request.model), request_id=data.get("id"), latency_ms=latency),
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        _, msgs = self.convert_messages(request.messages)
        payload: dict[str, Any] = {"model": request.model, "messages": msgs, "stream": True}
        if request.tools:
            payload["tools"] = self.convert_tools(request.tools)
        async with self.client.stream("POST", "/chat/completions", json=payload, headers=self._get_headers()) as resp:
            if resp.status_code >= 400:
                self._handle_error(resp)
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    raw = line[6:]
                    if raw == "[DONE]":
                        break
                    yield f"data: {raw}\n\n"
        yield "data: [DONE]\n\n"

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        payload = {"model": request.model, "input": request.input}
        if request.dimensions:
            payload["dimensions"] = request.dimensions
        resp = await self.client.post("/embeddings", json=payload, headers=self._get_headers())
        if resp.status_code >= 400:
            self._handle_error(resp)
        data = resp.json()
        embeddings = [Embedding(item["embedding"], item["index"]) for item in data["data"]]
        u = data.get("usage", {})
        return EmbeddingResponse(embeddings, data["model"], "openai",
                                 Usage(u.get("prompt_tokens", 0), 0, u.get("total_tokens", 0)))
