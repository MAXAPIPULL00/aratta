# Providers

Aratta ships with five provider adapters. Each handles the translation between
SCRI and a provider's native API — so your system speaks one language and
invokes any of them as interchangeable services.

## Local (Ollama / vLLM / llama.cpp)

The foundation. No API key, no data leaves your machine. This is the default
provider — cloud is the fallback.

All three servers expose an OpenAI-compatible `/v1/chat/completions` endpoint,
so one adapter covers them all.

| Feature | Ollama | vLLM | llama.cpp |
|---------|--------|------|-----------|
| Chat | Yes | Yes | Yes |
| Streaming | Yes | Yes | Yes |
| Tool calling | Yes (model-dependent) | Yes | Yes |
| Embeddings | Yes | Yes | No |
| Vision | Model-dependent | Model-dependent | No |
| Default port | 11434 | 8000 | 8080 |

```bash
# Ollama quickstart
ollama pull llama3.1:8b
aratta init    # select ollama
aratta serve
```

## Anthropic (Claude)

Supports Claude 4.5 series with extended thinking, prompt caching, tool calling,
and vision.

| Feature | Supported |
|---------|-----------|
| Chat | Yes |
| Streaming | Yes (SSE) |
| Tool calling | Yes (native) |
| Vision | Yes (base64 + URL) |
| Extended thinking | Yes (budget_tokens) |
| Prompt caching | Yes (automatic) |
| Embeddings | No (use OpenAI) |

Auth: `ANTHROPIC_API_KEY` environment variable.

## OpenAI (GPT)

Uses the official `openai` Python SDK. Supports both the Responses API
(default) and Chat Completions API (legacy). 7 models: GPT-5.2 series,
GPT-4.1 series, and O-series reasoning models.

| Feature | Supported |
|---------|-----------|
| Chat | Yes |
| Streaming | Yes |
| Tool calling | Yes (function calling) |
| Vision | Yes |
| Reasoning | Yes (reasoning_effort param) |
| Embeddings | Yes (text-embedding-3) |

Auth: `OPENAI_API_KEY` environment variable.

## Google (Gemini)

Uses the official `google-genai` SDK. 8 models across Gemini 3.1, 3, and
2.5 series — including Gemini 3.1 Pro (reasoning/agentic), custom tools
variant, and Image Pro for image generation.

| Feature | Supported |
|---------|-----------|
| Chat | Yes |
| Streaming | Yes (SSE) |
| Tool calling | Yes (functionDeclarations) |
| Vision | Yes (inlineData) |
| Thinking | Yes (thinkingLevel / thinkingBudget) |
| Image generation | Yes (gemini-3-pro-image-preview) |
| Embeddings | Yes (embed_content) |

Auth: `GOOGLE_API_KEY` environment variable.

## xAI (Grok)

Uses the official `xai_sdk` package with the Responses API (gRPC transport).
4 models with agentic server-side tools (web search, X search, code execution,
collections search). Supports conversation chaining via `previous_response_id`
and server-side message persistence via `store_messages`.

| Feature | Supported |
|---------|-----------|
| Chat | Yes |
| Streaming | Yes |
| Tool calling | Yes (function + server-side) |
| Vision | Yes |
| Web search | Yes (builtin tool) |
| X/Twitter search | Yes (builtin tool) |
| Code execution | Yes (builtin tool) |
| Conversation chaining | Yes (previous_response_id) |
| Encrypted thinking | Yes (grok-4, grok-4-1-fast-reasoning) |
| Embeddings | No |

Auth: `XAI_API_KEY` environment variable.

## Adding a Custom Provider

Inherit from `BaseProvider` and implement the abstract methods:

```python
from aratta.providers.base import BaseProvider
from aratta.config import ProviderConfig

class MyProvider(BaseProvider):
    name = "my_provider"
    display_name = "My Provider"

    async def chat(self, request):
        # Translate SCRI request -> your API -> SCRI response
        ...

    async def chat_stream(self, request):
        # Yield SSE chunks
        ...

    async def embed(self, request):
        ...

    def get_models(self):
        return [...]

    def convert_messages(self, messages):
        ...

    def convert_tools(self, tools):
        ...
```

Then register it in `server.py`'s `_get_provider()` function.
