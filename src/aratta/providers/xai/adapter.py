"""
xAI (Grok) provider adapter.

Supports agentic server-side tools (web_search, x_search, code_execution,
collections_search), encrypted thinking, and OpenAI-compatible chat API.
"""

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

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
    Tool,
    ToolCall,
    Usage,
)
from aratta.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class XAIProvider(BaseProvider):
    name = "xai"
    display_name = "xAI (Grok)"

    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key or ''}",
            "Content-Type": "application/json",
            "User-Agent": "Aratta/0.1.0",
        }

    def get_models(self) -> list[ModelCapabilities]:
        return [
            ModelCapabilities("grok-4", "xai", "Grok 4", True, True, True, True, True, 131072, 16384, categories=["reasoning", "agentic"]),
            ModelCapabilities("grok-4-fast", "xai", "Grok 4 Fast", True, True, True, True, True, 131072, 16384, categories=["agentic", "fast"]),
            ModelCapabilities("grok-4-1-fast", "xai", "Grok 4.1 Fast", True, True, True, True, True, 131072, 16384, categories=["agentic", "research"]),
        ]

    def convert_messages(self, messages: list[Message]) -> list[dict]:
        converted = []
        for msg in messages:
            m: dict[str, Any] = {"role": msg.role.value}
            if isinstance(msg.content, str):
                m["content"] = msg.content
            else:
                parts = []
                for b in msg.content:
                    if b.type.value == "text" and b.text:
                        parts.append({"type": "input_text", "text": b.text})
                    elif b.type.value == "image" and b.image_url:
                        parts.append({"type": "input_image", "image_url": b.image_url, "detail": "high"})
                m["content"] = parts or msg.content
            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            converted.append(m)
        return converted

    def convert_tools(self, tools: list[Tool]) -> list[dict]:
        return [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}} for t in tools]

    async def chat(
        self,
        request: ChatRequest,
        *,
        builtin_tools: list[str] | None = None,
        collection_ids: list[str] | None = None,
    ) -> ChatResponse:
        start = time.time()
        messages = self.convert_messages(request.messages)
        payload: dict[str, Any] = {"model": request.model, "messages": messages}
        if request.temperature != 0.7:
            payload["temperature"] = request.temperature
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        tools: list[dict] = []
        for bt in (builtin_tools or []):
            tools.append({"type": bt})
        if collection_ids:
            tools.append({"type": "collections_search", "collection_ids": collection_ids})
        if request.tools:
            tools.extend(self.convert_tools(request.tools))
        if tools:
            payload["tools"] = tools
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
            tool_calls = [ToolCall(tc["id"], tc["function"]["name"], json.loads(tc["function"]["arguments"]))
                          for tc in msg["tool_calls"] if tc.get("type") == "function"]
            if not tool_calls:
                tool_calls = None

        fr_map = {"stop": FinishReason.STOP, "tool_calls": FinishReason.TOOL_CALLS, "length": FinishReason.LENGTH}
        u = data.get("usage", {})
        return ChatResponse(
            id=data.get("id", ""), content=msg.get("content", "") or "",
            tool_calls=tool_calls, model=data.get("model", request.model), provider="xai",
            finish_reason=fr_map.get(choice.get("finish_reason", "stop"), FinishReason.STOP),
            usage=Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0), u.get("total_tokens", 0),
                        reasoning_tokens=u.get("reasoning_tokens")),
            lineage=Lineage("xai", data.get("model", request.model), request_id=data.get("id"), latency_ms=latency),
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        messages = self.convert_messages(request.messages)
        payload: dict[str, Any] = {"model": request.model, "messages": messages, "stream": True}
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
        resp = await self.client.post("/embeddings", json=payload, headers=self._get_headers())
        if resp.status_code >= 400:
            self._handle_error(resp)
        data = resp.json()
        embeddings = [Embedding(item["embedding"], item["index"]) for item in data["data"]]
        u = data.get("usage", {})
        return EmbeddingResponse(embeddings, data["model"], "xai", Usage(u.get("prompt_tokens", 0), 0, u.get("total_tokens", 0)))
