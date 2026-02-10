"""
Google (Gemini) provider adapter.

Supports Gemini 3 and 2.5 series with thinking levels, function calling,
vision, streaming, and embeddings.
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
    Embedding,
    EmbeddingRequest,
    EmbeddingResponse,
    FinishReason,
    Lineage,
    Message,
    ModelCapabilities,
    Role,
    Tool,
    ToolCall,
    Usage,
)
from aratta.providers.base import BaseProvider, ProviderError

logger = logging.getLogger(__name__)

FINISH_MAP = {
    "STOP": FinishReason.STOP,
    "MAX_TOKENS": FinishReason.LENGTH,
    "SAFETY": FinishReason.CONTENT_FILTER,
}

API_VERSION = "v1beta"


class GoogleProvider(BaseProvider):
    name = "google"
    display_name = "Google (Gemini)"

    def _get_headers(self, request: ChatRequest | None = None) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": self.config.api_key or "",
            "User-Agent": "Aratta/0.1.0",
        }

    def get_models(self) -> list[ModelCapabilities]:
        return [
            ModelCapabilities("gemini-3-pro-preview", "google", "Gemini 3 Pro", True, True, True, True, True, 1_000_000, 64000, 2.0, 12.0, ["chat", "reasoning"]),
            ModelCapabilities("gemini-3-flash-preview", "google", "Gemini 3 Flash", True, True, True, True, True, 1_000_000, 64000, 0.5, 3.0, ["chat", "fast"]),
            ModelCapabilities("gemini-2.5-pro", "google", "Gemini 2.5 Pro", True, True, True, True, True, 1_000_000, 64000, 1.25, 5.0, ["chat", "reasoning"]),
            ModelCapabilities("gemini-2.5-flash", "google", "Gemini 2.5 Flash", True, True, True, True, True, 1_000_000, 64000, 0.15, 0.6, ["chat", "code"]),
            ModelCapabilities("gemini-2.5-flash-lite", "google", "Gemini 2.5 Flash-Lite", True, True, True, True, False, 1_000_000, 64000, 0.075, 0.3, ["fast", "cheap"]),
        ]

    def convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict]]:
        system_instruction = None
        contents = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_instruction = msg.content if isinstance(msg.content, str) else "\n".join(c.text for c in msg.content if c.type == ContentType.TEXT and c.text)
                continue
            role = "user" if msg.role in (Role.USER, Role.TOOL) else "model"
            parts = []
            if isinstance(msg.content, str):
                parts.append({"text": msg.content})
            else:
                for b in msg.content:
                    if b.type == ContentType.TEXT:
                        parts.append({"text": b.text})
                    elif b.type == ContentType.IMAGE and b.image_base64:
                        parts.append({"inlineData": {"mimeType": "image/jpeg", "data": b.image_base64}})
                    elif b.type == ContentType.TOOL_RESULT:
                        parts.append({"functionResponse": {"name": b.tool_name or "unknown", "response": b.tool_result}})
                    elif b.type == ContentType.TOOL_USE:
                        parts.append({"functionCall": {"name": b.tool_name, "args": b.tool_input or {}}})
            if parts:
                contents.append({"role": role, "parts": parts})
        return system_instruction, contents

    def convert_tools(self, tools: list[Tool]) -> list[dict]:
        return [{"functionDeclarations": [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in tools]}]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        start = time.time()
        sys_instr, contents = self.convert_messages(request.messages)
        body: dict[str, Any] = {"contents": contents, "generationConfig": {}}
        if sys_instr:
            body["systemInstruction"] = {"parts": [{"text": sys_instr}]}
        gc = body["generationConfig"]
        if request.max_tokens:
            gc["maxOutputTokens"] = request.max_tokens
        if request.temperature is not None:
            gc["temperature"] = request.temperature
        if request.thinking_enabled:
            level = "high" if request.thinking_budget > 8192 else "medium" if "flash" in request.model else "low"
            gc["thinkingConfig"] = {"thinkingLevel": level} if "gemini-3" in request.model else {"thinkingBudget": max(1024, request.thinking_budget)}
        if request.tools:
            body["tools"] = self.convert_tools(request.tools)

        url = f"/{API_VERSION}/models/{request.model}:generateContent"
        resp = await self.client.post(url, json=body, headers=self._get_headers(request))
        if resp.status_code != 200:
            self._handle_error(resp)
        return self._normalize(resp.json(), request.model, (time.time() - start) * 1000)

    def _normalize(self, data: dict, model: str, latency: float) -> ChatResponse:
        candidates = data.get("candidates", [])
        if not candidates:
            raise ProviderError("No candidates in response", self.name)
        parts = candidates[0].get("content", {}).get("parts", [])
        text = ""
        tool_calls: list[ToolCall] = []
        for p in parts:
            if "text" in p:
                text += p["text"]
            elif "functionCall" in p:
                fc = p["functionCall"]
                tool_calls.append(ToolCall(f"call_{uuid.uuid4().hex[:12]}", fc.get("name", ""), fc.get("args", {})))
            elif "executableCode" in p:
                text += f"\n```python\n{p['executableCode'].get('code', '')}\n```\n"
            elif "codeExecutionResult" in p:
                text += f"\n[Output]\n{p['codeExecutionResult'].get('output', '')}\n"

        finish = FINISH_MAP.get(candidates[0].get("finishReason", "STOP"), FinishReason.STOP)
        if tool_calls:
            finish = FinishReason.TOOL_CALLS
        u = data.get("usageMetadata", {})
        return ChatResponse(
            id=f"gemini_{uuid.uuid4().hex[:12]}", content=text,
            tool_calls=tool_calls or None, model=model, provider="google", finish_reason=finish,
            usage=Usage(u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0), u.get("totalTokenCount", 0),
                        cache_read_tokens=u.get("cachedContentTokenCount")),
            lineage=Lineage("google", data.get("modelVersion", model), latency_ms=latency),
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        sys_instr, contents = self.convert_messages(request.messages)
        body: dict[str, Any] = {"contents": contents, "generationConfig": {}}
        if sys_instr:
            body["systemInstruction"] = {"parts": [{"text": sys_instr}]}
        if request.max_tokens:
            body["generationConfig"]["maxOutputTokens"] = request.max_tokens
        if request.tools:
            body["tools"] = self.convert_tools(request.tools)
        url = f"/{API_VERSION}/models/{request.model}:streamGenerateContent?alt=sse"
        async with self.client.stream("POST", url, json=body, headers=self._get_headers(request)) as resp:
            if resp.status_code != 200:
                self._handle_error(resp)
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    raw = line[6:]
                    if raw == "[DONE]":
                        break
                    try:
                        evt = json.loads(raw)
                        parts = evt.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                        for p in parts:
                            if "text" in p:
                                yield f'data: {json.dumps({"type": "content", "content": p["text"]})}\n\n'
                    except (json.JSONDecodeError, IndexError):
                        continue
        yield "data: [DONE]\n\n"

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        inputs = request.input if isinstance(request.input, list) else [request.input]
        body = {"requests": [{"model": f"models/{request.model}", "content": {"parts": [{"text": t}]}} for t in inputs]}
        url = f"/{API_VERSION}/models/{request.model}:batchEmbedContents"
        resp = await self.client.post(url, json=body, headers=self._get_headers())
        if resp.status_code != 200:
            self._handle_error(resp)
        data = resp.json()
        embeddings = [Embedding(e.get("values", []), i) for i, e in enumerate(data.get("embeddings", []))]
        est = sum(len(t) for t in inputs) // 4
        return EmbeddingResponse(embeddings, request.model, "google", Usage(est, 0, est))
