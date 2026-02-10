"""
Local provider adapter — Ollama, vLLM, llama.cpp.

All three expose an OpenAI-compatible /v1/chat/completions endpoint,
so one adapter covers them all. The differences are:

    Ollama:    base_url = http://localhost:11434  (uses /api/ for native, /v1/ for compat)
    vLLM:     base_url = http://localhost:8000    (native OpenAI compat)
    llama.cpp: base_url = http://localhost:8080   (native OpenAI compat)

No API key required. No data leaves your machine.
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


class LocalProvider(BaseProvider):
    """
    Adapter for local LLM servers that speak the OpenAI-compatible API.

    Works out of the box with Ollama, vLLM, and llama.cpp server.
    """

    name = "local"
    display_name = "Local (Ollama / vLLM / llama.cpp)"

    def _get_headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "User-Agent": "Aratta/0.1.0"}

    @property
    def _is_ollama(self) -> bool:
        return "11434" in self.config.base_url or "ollama" in self.config.name.lower()

    @property
    def _chat_path(self) -> str:
        """Ollama uses /v1/chat/completions for OpenAI compat mode."""
        return "/v1/chat/completions"

    def get_models(self) -> list[ModelCapabilities]:
        # Local models are dynamic — user pulls what they want.
        # Return the configured default as a known model.
        return [
            ModelCapabilities(
                model_id=self.config.default_model,
                provider=self.config.name,
                display_name=f"Local: {self.config.default_model}",
                supports_tools=True,
                supports_vision=False,
                supports_streaming=True,
                supports_json_mode=True,
                supports_thinking=False,
                context_window=8192,
                categories=["local", "sovereign"],
            ),
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
                        parts.append({"type": "text", "text": b.text})
                    elif b.type.value == "image" and b.image_base64:
                        parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b.image_base64}"}})
                    elif b.type.value == "image" and b.image_url:
                        parts.append({"type": "image_url", "image_url": {"url": b.image_url}})
                m["content"] = parts or msg.content
            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            converted.append(m)
        return converted

    def convert_tools(self, tools: list[Tool]) -> list[dict]:
        return [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in tools
        ]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        start = time.time()
        messages = self.convert_messages(request.messages)

        payload: dict[str, Any] = {"model": request.model, "messages": messages}
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = self.convert_tools(request.tools)
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice
        if request.stop:
            payload["stop"] = request.stop

        resp = await self.client.post(self._chat_path, json=payload, headers=self._get_headers())
        if resp.status_code >= 400:
            self._handle_error(resp)

        data = resp.json()
        latency = (time.time() - start) * 1000

        choice = data["choices"][0]
        msg = choice["message"]

        tool_calls = None
        if "tool_calls" in msg and msg["tool_calls"]:
            tool_calls = []
            for tc in msg["tool_calls"]:
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                tool_calls.append(ToolCall(tc.get("id", ""), tc["function"]["name"], args))

        fr_map = {"stop": FinishReason.STOP, "tool_calls": FinishReason.TOOL_CALLS, "length": FinishReason.LENGTH}
        u = data.get("usage", {})

        return ChatResponse(
            id=data.get("id", ""),
            content=msg.get("content", "") or "",
            tool_calls=tool_calls,
            model=data.get("model", request.model),
            provider=self.config.name,
            finish_reason=fr_map.get(choice.get("finish_reason", "stop"), FinishReason.STOP),
            usage=Usage(
                u.get("prompt_tokens", 0),
                u.get("completion_tokens", 0),
                u.get("total_tokens", 0),
            ),
            lineage=Lineage(self.config.name, data.get("model", request.model), latency_ms=latency),
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        messages = self.convert_messages(request.messages)
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": True,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = self.convert_tools(request.tools)

        async with self.client.stream("POST", self._chat_path, json=payload, headers=self._get_headers()) as resp:
            if resp.status_code >= 400:
                self._handle_error(resp)
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    raw = line[6:]
                    if raw.strip() == "[DONE]":
                        break
                    yield f"data: {raw}\n\n"
        yield "data: [DONE]\n\n"

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Embeddings via /v1/embeddings (supported by Ollama and vLLM)."""
        payload: dict[str, Any] = {"model": request.model, "input": request.input}

        resp = await self.client.post("/v1/embeddings", json=payload, headers=self._get_headers())
        if resp.status_code >= 400:
            self._handle_error(resp)

        data = resp.json()
        embeddings = [Embedding(item["embedding"], item["index"]) for item in data["data"]]
        u = data.get("usage", {})

        return EmbeddingResponse(
            embeddings, data.get("model", request.model), self.config.name,
            Usage(u.get("prompt_tokens", 0), 0, u.get("total_tokens", 0)),
        )

    async def health_check(self) -> dict[str, Any]:
        """Check if the local server is reachable."""
        try:
            # Ollama native endpoint
            if self._is_ollama:
                resp = await self.client.get("/api/tags")
            else:
                resp = await self.client.get("/v1/models")

            return {
                "status": "healthy" if resp.status_code == 200 else "degraded",
                "provider": self.config.name,
                "latency_ms": resp.elapsed.total_seconds() * 1000,
            }
        except Exception as e:
            return {"status": "unhealthy", "provider": self.config.name, "error": str(e)}
