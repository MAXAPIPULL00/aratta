# OpenAI Provider — SCRI ↔ GPT Mapping

How Aratta translates between OpenAI's API and SCRI, the language your
system speaks. Every field below shows the SCRI side (what your code sees)
and the OpenAI side (what the adapter handles for you).

---

## SDK

The OpenAI adapter uses the official `openai` Python SDK. Install with:

```bash
pip install aratta[openai-sdk]
```

The adapter creates an `openai.AsyncOpenAI` client and supports both the
Responses API (default) and Chat Completions API (legacy) via the
`use_responses_api` flag.

---

## Models

7 models across three generations:

### GPT-5.2 Series

| Alias | Model ID | Context | Pricing (in/out per MTok) | Use Case |
|-------|----------|---------|---------------------------|----------|
| `gpt` | gpt-5.2 | 1M | $1.75 / $14 | Flagship chat + reasoning |
| — | gpt-5.2-pro | 1M | $3 / $18 | Maximum capability, reasoning |
| — | gpt-5.2-codex | 1M | $1.75 / $14 | Code generation |

### GPT-4.1 Series

| Alias | Model ID | Context | Pricing (in/out per MTok) | Use Case |
|-------|----------|---------|---------------------------|----------|
| — | gpt-4.1 | 1M | $2 / $8 | Previous gen flagship |
| `gpt-mini` | gpt-4.1-mini | 1M | $0.40 / $1.60 | Fast, cost-effective |
| — | gpt-4.1-nano | 1M | $0.10 / $0.40 | Ultra fast |

### O-Series

| Alias | Model ID | Context | Pricing (in/out per MTok) | Use Case |
|-------|----------|---------|---------------------------|----------|
| `o3` | o3 | 200k | $2 / $8 | Deep reasoning |

Aliases are configurable in `~/.aratta/config.toml`.

---

## API Endpoints

| OpenAI SDK Method | Adapter Method | Notes |
|-------------------|----------------|-------|
| `client.responses.create()` | `chat()` | Primary API (recommended) |
| `client.chat.completions.create()` | `chat()` | Legacy API (`use_responses_api=False`) |
| `client.chat.completions.create(stream=True)` | `chat_stream()` | SSE streaming |
| `client.embeddings.create()` | `embed()` | Embeddings |

---

## Request Parameters

### Responses API

| OpenAI | SCRI Type | Notes |
|--------|-----------|-------|
| `model` | `ChatRequest.model` | Direct |
| `input` | `ChatRequest.messages` | Can be string or array |
| `instructions` | System message | Extracted from messages |
| `reasoning.effort` | Reasoning effort | none/low/medium/high/xhigh |
| `max_output_tokens` | `ChatRequest.max_tokens` | Renamed |
| `temperature` | `ChatRequest.temperature` | Direct |
| `tools` | `ChatRequest.tools` | Function + built-in |
| `tool_choice` | `ChatRequest.tool_choice` | Direct |

### Chat Completions API

| OpenAI | SCRI Type | Notes |
|--------|-----------|-------|
| `model` | `ChatRequest.model` | Direct |
| `messages` | `ChatRequest.messages` | Direct |
| `max_completion_tokens` | `ChatRequest.max_tokens` | Renamed |
| `temperature` | `ChatRequest.temperature` | Direct |
| `tools` | `ChatRequest.tools` | Direct |
| `tool_choice` | `ChatRequest.tool_choice` | Direct |
| `reasoning_effort` | Reasoning effort | For GPT-5.x/o-series |

---

## Response Mapping

### Responses API

| OpenAI | SCRI | Notes |
|--------|------|-------|
| `response.id` | `ChatResponse.id` | Response ID |
| `response.output_text` | `ChatResponse.content` | Helper field |
| `output[type=message]` | `ChatResponse.content` | Message item |
| `output[type=function_call]` | `ChatResponse.tool_calls` | Tool call item |
| `output[type=reasoning]` | `ChatResponse.thinking` | Reasoning block |
| `response.status` | `ChatResponse.finish_reason` | Mapped |
| `usage.input_tokens` | `Usage.input_tokens` | Direct |
| `usage.output_tokens` | `Usage.output_tokens` | Direct |
| `usage.input_tokens_details.cached_tokens` | `Usage.cache_read_tokens` | Renamed |
| `usage.output_tokens_details.reasoning_tokens` | `Usage.reasoning_tokens` | Direct |

### Chat Completions

| OpenAI | SCRI | Notes |
|--------|------|-------|
| `choices[0].message.content` | `ChatResponse.content` | Direct |
| `choices[0].message.tool_calls` | `ChatResponse.tool_calls` | Direct |
| `choices[0].finish_reason` | `ChatResponse.finish_reason` | Mapped |
| `usage.prompt_tokens` | `Usage.input_tokens` | Renamed |
| `usage.completion_tokens` | `Usage.output_tokens` | Renamed |

---

## Finish Reason Mapping

| OpenAI | SCRI |
|--------|------|
| `stop` / `completed` | `FinishReason.STOP` |
| `tool_calls` | `FinishReason.TOOL_CALLS` |
| `length` / `max_output_tokens` | `FinishReason.LENGTH` |
| `content_filter` | `FinishReason.CONTENT_FILTER` |

---

## Message Role Mapping

| OpenAI | SCRI |
|--------|------|
| `system` | `Role.SYSTEM` |
| `user` | `Role.USER` |
| `assistant` | `Role.ASSISTANT` |
| `tool` | `Role.TOOL` |

---

## Tool Format Mapping

### Responses API (Internally Tagged)

```json
// OpenAI
{"type": "function", "name": "get_weather", "description": "...", "parameters": {...}, "strict": true}

// SCRI
Tool(name="get_weather", description="...", parameters={...})
```

### Chat Completions (Externally Tagged)

```json
// OpenAI
{"type": "function", "function": {"name": "get_weather", "description": "...", "parameters": {...}}}

// SCRI (same)
Tool(name="get_weather", description="...", parameters={...})
```

---

## Built-in Tools

| OpenAI | Description |
|--------|-------------|
| `web_search` | Web search |
| `file_search` | Vector store search |
| `code_interpreter` | Python sandbox (1-64GB containers) |
| `computer_use` | Computer control |

---

## Reasoning Effort

| Level | Use Case |
|-------|----------|
| `none` | No reasoning |
| `low` | Quick tasks |
| `medium` | Balanced |
| `high` | Deep analysis |
| `xhigh` | Maximum depth |

For o-series models, reasoning is always enabled.

---

## Embedding Mapping

| OpenAI SDK | SCRI |
|------------|------|
| `response.data[].embedding` | `Embedding.embedding` |
| `response.data[].index` | `Embedding.index` |
| `response.usage.prompt_tokens` | `Usage.input_tokens` |
| `response.usage.total_tokens` | `Usage.total_tokens` |

---

## Key Differences from Anthropic

| Feature | OpenAI | Anthropic |
|---------|--------|-----------|
| SDK | `openai` | `anthropic` |
| Thinking blocks | `reasoning` item | `thinking` block |
| Effort control | `reasoning.effort` | `thinking.budget_tokens` |
| Built-in tools | Native types | Beta headers |
| Prompt caching | Automatic | `cache_control` |
| System prompt | `instructions` field | First message |
| State management | `previous_response_id` | Manual |

---

## See Also

- [Architecture](../architecture.md) — sovereignty layer design
- [Providers](../providers.md) — all providers + writing your own
