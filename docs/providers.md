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
| Chat | ✅ | ✅ | ✅ |
| Streaming | ✅ | ✅ | ✅ |
| Tool calling | ✅ (model-dependent) | ✅ | ✅ |
| Embeddings | ✅ | ✅ | ❌ |
| Vision | Model-dependent | Model-dependent | ❌ |
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
| Chat | ✅ |
| Streaming | ✅ (SSE) |
| Tool calling | ✅ (native) |
| Vision | ✅ (base64 + URL) |
| Extended thinking | ✅ (budget_tokens) |
| Prompt caching | ✅ (automatic) |
| Embeddings | ❌ (use OpenAI) |

Auth: `ANTHROPIC_API_KEY` environment variable.

## OpenAI (GPT)

Supports both the Responses API (default) and Chat Completions API (legacy).
GPT-4.1 series, O-series reasoning models.

| Feature | Supported |
|---------|-----------|
| Chat | ✅ |
| Streaming | ✅ |
| Tool calling | ✅ (function calling) |
| Vision | ✅ |
| Reasoning | ✅ (reasoning_effort param) |
| Embeddings | ✅ (text-embedding-3) |

Auth: `OPENAI_API_KEY` environment variable.

## Google (Gemini)

Supports Gemini 3 and 2.5 series with thinking levels, function calling,
code execution results, and embeddings.

| Feature | Supported |
|---------|-----------|
| Chat | ✅ |
| Streaming | ✅ (SSE) |
| Tool calling | ✅ (functionDeclarations) |
| Vision | ✅ (inlineData) |
| Thinking | ✅ (thinkingLevel / thinkingBudget) |
| Embeddings | ✅ (batchEmbedContents) |

Auth: `GOOGLE_API_KEY` environment variable.

## xAI (Grok)

OpenAI-compatible API with agentic server-side tools (web search, X search,
code execution, collections search).

| Feature | Supported |
|---------|-----------|
| Chat | ✅ |
| Streaming | ✅ |
| Tool calling | ✅ (function + server-side) |
| Vision | ✅ |
| Web search | ✅ (builtin tool) |
| X/Twitter search | ✅ (builtin tool) |
| Embeddings | ✅ |

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
        # Translate SCRI request → your API → SCRI response
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
