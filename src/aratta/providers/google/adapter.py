"""
Google (Gemini) provider adapter.

Uses the official google-genai SDK for communication with the Gemini API.
Supports Gemini 3.1, 3, and 2.5 series with thinking levels, function calling,
vision, streaming, and embeddings.
"""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

try:
    from google import genai
    _HAS_GENAI_SDK = True
except ImportError:
    genai = None  # type: ignore[assignment]
    _HAS_GENAI_SDK = False

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


class GoogleProvider(BaseProvider):
    """Google (Gemini) provider using the official google-genai SDK."""

    name = "google"
    display_name = "Google (Gemini)"

    def __init__(self, config):
        if not _HAS_GENAI_SDK:
            raise ImportError(
                "google-genai is required for the Google provider. "
                "Install it with: pip install aratta[google]"
            )
        super().__init__(config)
        self._sdk_client = genai.Client(api_key=config.api_key)

    # ------------------------------------------------------------------
    # Message conversion  (SCRI → google-genai SDK format)
    # ------------------------------------------------------------------

    def convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict]]:
        """Convert SCRI Messages to google-genai content dicts.

        Returns (system_instruction, contents) where SYSTEM messages are
        extracted into the system_instruction string and all other messages
        become Content dicts with role "user" or "model".
        """
        system_instruction = None
        contents: list[dict] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_instruction = (
                    msg.content
                    if isinstance(msg.content, str)
                    else "\n".join(
                        c.text
                        for c in msg.content
                        if c.type == ContentType.TEXT and c.text
                    )
                )
                continue

            role = "user" if msg.role in (Role.USER, Role.TOOL) else "model"
            parts: list[dict] = []

            if isinstance(msg.content, str):
                parts.append({"text": msg.content})
            else:
                for b in msg.content:
                    if b.type == ContentType.TEXT:
                        parts.append({"text": b.text})
                    elif b.type == ContentType.IMAGE and b.image_base64:
                        parts.append({
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": b.image_base64,
                            }
                        })
                    elif b.type == ContentType.TOOL_RESULT:
                        parts.append({
                            "functionResponse": {
                                "name": b.tool_name or "unknown",
                                "response": b.tool_result,
                            }
                        })
                    elif b.type == ContentType.TOOL_USE:
                        parts.append({
                            "functionCall": {
                                "name": b.tool_name,
                                "args": b.tool_input or {},
                            }
                        })

            if parts:
                contents.append({"role": role, "parts": parts})

        return system_instruction, contents

    # ------------------------------------------------------------------
    # Tool conversion  (SCRI → google-genai SDK format)
    # ------------------------------------------------------------------

    def convert_tools(self, tools: list[Tool]) -> list[dict]:
        """Convert SCRI Tools to google-genai FunctionDeclaration dicts."""
        return [
            {
                "functionDeclarations": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                    for t in tools
                ]
            }
        ]

    # ------------------------------------------------------------------
    # Generation config builder
    # ------------------------------------------------------------------

    def _build_generation_config(self, request: ChatRequest) -> dict[str, Any]:
        """Build the GenerateContentConfig dict for the SDK."""
        config: dict[str, Any] = {}

        if request.max_tokens:
            config["max_output_tokens"] = request.max_tokens
        if request.temperature is not None:
            config["temperature"] = request.temperature

        if request.thinking_enabled:
            if "gemini-3" in request.model and "2.5" not in request.model:
                # Gemini 3.x uses thinkingLevel
                level = (
                    "high" if request.thinking_budget > 8192
                    else "medium" if "flash" in request.model
                    else "low"
                )
                config["thinking_config"] = {"thinking_level": level}
            else:
                # Gemini 2.5 uses thinkingBudget
                config["thinking_config"] = {
                    "thinking_budget": max(1024, request.thinking_budget)
                }

        if request.tools:
            config["tools"] = self.convert_tools(request.tools)

        return config

    # ------------------------------------------------------------------
    # Chat  (non-streaming)
    # ------------------------------------------------------------------

    async def chat(self, request: ChatRequest) -> ChatResponse:
        start = time.time()
        sys_instr, contents = self.convert_messages(request.messages)
        config = self._build_generation_config(request)

        if sys_instr:
            config["system_instruction"] = sys_instr

        try:
            response = await self._sdk_client.aio.models.generate_content(
                model=request.model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            raise ProviderError(str(exc), self.name) from exc

        latency = (time.time() - start) * 1000
        return self._normalize(response, request.model, latency)

    # ------------------------------------------------------------------
    # Response normalization  (google-genai SDK → SCRI ChatResponse)
    # ------------------------------------------------------------------

    def _normalize(self, response: Any, model: str, latency: float) -> ChatResponse:
        """Map a google-genai SDK response to a SCRI ChatResponse."""
        # The SDK response has .candidates, .usage_metadata, .model_version
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            raise ProviderError("No candidates in response", self.name)

        candidate = candidates[0]
        content_obj = getattr(candidate, "content", None)
        parts = getattr(content_obj, "parts", None) or []

        text = ""
        tool_calls: list[ToolCall] = []

        for p in parts:
            # SDK Part objects have .text, .function_call, .executable_code, etc.
            p_text = getattr(p, "text", None)
            p_fc = getattr(p, "function_call", None)
            p_exec = getattr(p, "executable_code", None)
            p_result = getattr(p, "code_execution_result", None)

            if p_text is not None:
                text += p_text
            elif p_fc is not None:
                fc_name = getattr(p_fc, "name", "") or ""
                fc_args = getattr(p_fc, "args", {}) or {}
                tool_calls.append(
                    ToolCall(f"call_{uuid.uuid4().hex[:12]}", fc_name, fc_args)
                )
            elif p_exec is not None:
                code = getattr(p_exec, "code", "") or ""
                text += f"\n```python\n{code}\n```\n"
            elif p_result is not None:
                output = getattr(p_result, "output", "") or ""
                text += f"\n[Output]\n{output}\n"

        # Finish reason
        raw_finish = getattr(candidate, "finish_reason", None)
        finish_str = str(raw_finish) if raw_finish else "STOP"
        # SDK may return enum-like values; normalize to string key
        for key in FINISH_MAP:
            if key in finish_str.upper():
                finish = FINISH_MAP[key]
                break
        else:
            finish = FinishReason.STOP

        if tool_calls:
            finish = FinishReason.TOOL_CALLS

        # Usage
        u = getattr(response, "usage_metadata", None)
        usage = Usage(
            input_tokens=getattr(u, "prompt_token_count", 0) or 0,
            output_tokens=getattr(u, "candidates_token_count", 0) or 0,
            total_tokens=getattr(u, "total_token_count", 0) or 0,
            cache_read_tokens=getattr(u, "cached_content_token_count", None),
        ) if u else Usage(input_tokens=0, output_tokens=0, total_tokens=0)

        model_version = getattr(response, "model_version", model) or model

        return ChatResponse(
            id=f"gemini_{uuid.uuid4().hex[:12]}",
            content=text,
            tool_calls=tool_calls or None,
            model=model,
            provider="google",
            finish_reason=finish,
            usage=usage,
            lineage=Lineage(
                provider="google",
                model=model_version,
                latency_ms=latency,
            ),
        )

    # ------------------------------------------------------------------
    # Chat  (streaming)
    # ------------------------------------------------------------------

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        sys_instr, contents = self.convert_messages(request.messages)
        config = self._build_generation_config(request)

        if sys_instr:
            config["system_instruction"] = sys_instr

        try:
            async for chunk in await self._sdk_client.aio.models.generate_content_stream(
                model=request.model,
                contents=contents,
                config=config,
            ):
                candidates = getattr(chunk, "candidates", None) or []
                if not candidates:
                    continue
                parts = getattr(
                    getattr(candidates[0], "content", None), "parts", None
                ) or []
                for p in parts:
                    p_text = getattr(p, "text", None)
                    if p_text:
                        yield f'data: {json.dumps({"type": "content", "content": p_text})}\n\n'
        except Exception as exc:
            raise ProviderError(str(exc), self.name) from exc

        yield "data: [DONE]\n\n"

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        inputs = request.input if isinstance(request.input, list) else [request.input]

        try:
            response = await self._sdk_client.aio.models.embed_content(
                model=request.model,
                contents=inputs,
            )
        except Exception as exc:
            raise ProviderError(str(exc), self.name) from exc

        # SDK returns response.embeddings — list of embedding objects
        raw_embeddings = getattr(response, "embeddings", None) or []
        embeddings = [
            Embedding(
                getattr(e, "values", []) if hasattr(e, "values") else e,
                i,
            )
            for i, e in enumerate(raw_embeddings)
        ]

        est = sum(len(t) for t in inputs) // 4
        return EmbeddingResponse(
            embeddings, request.model, "google", Usage(est, 0, est)
        )

    # ------------------------------------------------------------------
    # Model registry
    # ------------------------------------------------------------------

    def get_models(self) -> list[ModelCapabilities]:
        return [
            # --- Existing models (preserved) ---
            ModelCapabilities(
                model_id="gemini-3-pro-preview",
                provider="google",
                display_name="Gemini 3 Pro",
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_thinking=True,
                context_window=1_000_000,
                max_output_tokens=64000,
                input_cost_per_million=2.0,
                output_cost_per_million=12.0,
                categories=["chat", "reasoning"],
            ),
            ModelCapabilities(
                model_id="gemini-3-flash-preview",
                provider="google",
                display_name="Gemini 3 Flash",
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_thinking=True,
                context_window=1_000_000,
                max_output_tokens=64000,
                input_cost_per_million=0.5,
                output_cost_per_million=3.0,
                categories=["chat", "fast"],
            ),
            ModelCapabilities(
                model_id="gemini-2.5-pro",
                provider="google",
                display_name="Gemini 2.5 Pro",
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_thinking=True,
                context_window=1_000_000,
                max_output_tokens=64000,
                input_cost_per_million=1.25,
                output_cost_per_million=5.0,
                categories=["chat", "reasoning"],
            ),
            ModelCapabilities(
                model_id="gemini-2.5-flash",
                provider="google",
                display_name="Gemini 2.5 Flash",
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_thinking=True,
                context_window=1_000_000,
                max_output_tokens=64000,
                input_cost_per_million=0.15,
                output_cost_per_million=0.6,
                categories=["chat", "code"],
            ),
            ModelCapabilities(
                model_id="gemini-2.5-flash-lite",
                provider="google",
                display_name="Gemini 2.5 Flash-Lite",
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_thinking=False,
                context_window=1_000_000,
                max_output_tokens=64000,
                input_cost_per_million=0.075,
                output_cost_per_million=0.3,
                categories=["fast", "cheap"],
            ),
            # --- New models ---
            ModelCapabilities(
                model_id="gemini-3.1-pro-preview",
                provider="google",
                display_name="Gemini 3.1 Pro",
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_thinking=True,
                context_window=1_000_000,
                max_output_tokens=65_000,
                input_cost_per_million=2.0,
                output_cost_per_million=12.0,
                categories=["reasoning", "agentic"],
            ),
            ModelCapabilities(
                model_id="gemini-3.1-pro-preview-customtools",
                provider="google",
                display_name="Gemini 3.1 Pro Custom Tools",
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_thinking=True,
                context_window=1_000_000,
                max_output_tokens=65_000,
                input_cost_per_million=2.0,
                output_cost_per_million=12.0,
                categories=["agentic", "tools"],
            ),
            ModelCapabilities(
                model_id="gemini-3-pro-image-preview",
                provider="google",
                display_name="Gemini 3 Pro Image",
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_thinking=False,
                context_window=1_000_000,
                max_output_tokens=64000,
                input_cost_per_million=2.0,
                output_cost_per_million=12.0,
                categories=["image_generation"],
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
