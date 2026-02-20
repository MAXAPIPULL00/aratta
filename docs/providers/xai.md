# xAI Provider — SCRI ↔ Grok Mapping

How Aratta translates between xAI's Grok API and SCRI, the language your
system speaks. Every field below shows the SCRI side (what your code sees)
and the Grok side (what the adapter handles for you).

---

## SDK

The xAI adapter uses the official `xai_sdk` package with gRPC transport
via the Responses API. Install with:

```bash
pip install aratta[xai]
```

The adapter creates an `xai_sdk.Client` and uses the
`client.chat.create()` + `conversation.sample()` pattern for chat
completions.

---

## Models

| Alias | xAI Model | Context | Pricing (in/out per MTok) | Use Case |
|-------|-----------|---------|---------------------------|----------|
| `grok` | grok-4-1-fast | 2M | $2 / $10 | Default Grok alias, agentic/research |
| `grok-reason` | grok-4-1-fast-reasoning | 2M | $2 / $10 | Reasoning with thinking traces |
| (configurable) | grok-4 | 256k | $3 / $15 | Deep reasoning |
| (configurable) | grok-4-fast | 131k | $2 / $10 | Fast agentic |

Aliases are configurable in `~/.aratta/config.toml`.

---

## API Pattern

The adapter uses the xAI SDK's Responses API pattern instead of the
legacy Chat Completions REST API:

```python
# SDK pattern (what the adapter does internally)
client = xai_sdk.Client(api_key="...")
conversation = client.chat.create(
    model="grok-4-1-fast",
    messages=[...],
    tools=[...],
    previous_response_id="resp_abc123",  # conversation chaining
    store=True,                          # server-side persistence
)
response = await conversation.sample()
```

| Feature | Description |
|---------|-------------|
| `client.chat.create()` | Creates a conversation with model, messages, tools |
| `conversation.sample()` | Produces a response (async) |
| `conversation.stream()` | Produces streaming chunks (async iterator) |

---

## Conversation Chaining

The Responses API supports stateful conversations via `previous_response_id`.
Pass it in request metadata:

```python
ChatRequest(
    messages=[...],
    model="grok-4-1-fast",
    metadata={"previous_response_id": "resp_abc123"},
)
```

The adapter passes this to the SDK, which chains the conversation
server-side without resending the full message history.

---

## Server-Side Message Persistence

Enable `store_messages` in metadata to persist messages on xAI's servers:

```python
ChatRequest(
    messages=[...],
    model="grok-4-1-fast",
    metadata={"store_messages": True},
)
```

The adapter sets `store=True` on the SDK call, enabling server-side
message storage for later retrieval or conversation continuation.

---

## Request Parameters

| xAI SDK | SCRI Type | Notes |
|---------|-----------|-------|
| `model` | `ChatRequest.model` | Direct |
| `messages` | `ChatRequest.messages` | Converted via `convert_messages()` |
| `tools` | `ChatRequest.tools` + builtin | Merged user + server-side tools |
| `max_tokens` | `ChatRequest.max_tokens` | Direct |
| `temperature` | `ChatRequest.temperature` | Direct |
| `previous_response_id` | `ChatRequest.metadata["previous_response_id"]` | Conversation chaining |
| `store` | `ChatRequest.metadata["store_messages"]` | Server-side persistence |
| `use_encrypted_content` | `ChatRequest.thinking_enabled` | Encrypted thinking traces |

---

## Response Mapping

| xAI SDK | SCRI | Notes |
|---------|------|-------|
| `response.id` | `ChatResponse.id` | Response ID |
| `response.content` | `ChatResponse.content` | Text content (joined if list) |
| `response.tool_calls` | `ChatResponse.tool_calls` | Client-side function calls |
| `response.finish_reason` | `ChatResponse.finish_reason` | Mapped to FinishReason enum |
| `response.model` | `ChatResponse.model` | Direct |
| `response.usage.prompt_tokens` | `Usage.input_tokens` | Renamed |
| `response.usage.completion_tokens` | `Usage.output_tokens` | Renamed |
| `response.usage.reasoning_tokens` | `Usage.reasoning_tokens` | Direct |
| `response.thinking` / `encrypted_content` | `ChatResponse.thinking` | ThinkingBlock list |

---

## Finish Reason Mapping

| xAI | SCRI |
|-----|------|
| `stop` | `FinishReason.STOP` |
| `tool_calls` | `FinishReason.TOOL_CALLS` |
| `length` | `FinishReason.LENGTH` |
| `content_filter` | `FinishReason.CONTENT_FILTER` |

---

## Message Role Mapping

| xAI | SCRI |
|-----|------|
| `system` | `Role.SYSTEM` |
| `user` | `Role.USER` |
| `assistant` | `Role.ASSISTANT` |
| `tool` | `Role.TOOL` |

---

## Encrypted Thinking Traces

Grok 4 and grok-4-1-fast-reasoning support thinking traces via encrypted
content. When `thinking_enabled=True`, the adapter sets
`use_encrypted_content=True` on the SDK call. Thinking traces are mapped
to SCRI `ThinkingBlock` objects with optional `signature` fields.

```python
# Response thinking blocks
response.thinking  # → [ThinkingBlock(thinking="...", signature="...")]
```

---

## Server-Side Tools

| xAI Tool | Cost per 1K calls | Description |
|----------|--------------------|-------------|
| `web_search` | $5.00 | Web search + page browsing |
| `x_search` | $5.00 | X/Twitter search (user, keyword, semantic) |
| `code_execution` | $5.00 | Python sandbox |
| `collections_search` | $2.50 | Document collection search |

Server-side tools are passed via the `builtin_tools` parameter on `chat()`:

```python
await provider.chat(
    request,
    builtin_tools=["web_search", "x_search"],
    collection_ids=["col_abc123"],
)
```

The adapter merges these with any user-defined function tools via
`_build_tools()`.

### Search Modes

| Mode | Use Case |
|------|----------|
| `keyword` | Exact terms, IDs, numbers |
| `semantic` | Conceptual matching |
| `hybrid` | Best of both (recommended) |

### X Search Modes

| Mode | Use Case |
|------|----------|
| `Latest` | Most recent posts |
| `Top` | Most engaged posts |

---

## Tool Call Types

| xAI Type | Meaning | Client Action |
|----------|---------|---------------|
| `function` | Client-side tool | Execute locally |
| `web_search_call` | Server-side web search | None (server handles) |
| `x_search_call` | Server-side X search | None |
| `code_interpreter_call` | Server-side code exec | None |
| `file_search_call` | Server-side collection search | None |

---

## Collections API

| Operation | xAI Endpoint |
|-----------|--------------|
| Create collection | `POST /v1/collections` |
| List collections | `GET /v1/collections` |
| Upload document | `POST /v1/collections/:id/documents` |
| Search documents | `POST /v1/documents/search` |

---

## Citations

Grok provides structured citations:

```python
# All citations
response.citations  # ['https://x.com/...', 'https://example.com/...']

# Inline citations in content
# "Text with citation [[1]](https://source.com)"
```

---

## Self-Healing Research Provider

Grok is the default research provider for Aratta's self-healing cycle.
When a provider's API changes, the heal worker uses Grok's web search
to find current API documentation and changelogs. This is configurable —
the research provider falls back through xai → openai → google → anthropic.

---

## Key Differences from Other Providers

| Feature | xAI | Anthropic | OpenAI |
|---------|-----|-----------|--------|
| SDK | `xai_sdk` (gRPC) | `anthropic` | `openai` |
| API pattern | Responses API | Messages API | Responses API |
| X/Twitter search | Native | Not available | Not available |
| Collections | Native API | Not available | Vector stores |
| Web search | `web_search` | Not available | `web_search` |
| Thinking | `encrypted_content` | Explicit `thinking` blocks | `reasoning` items |
| Conversation chaining | `previous_response_id` | Manual | `previous_response_id` |
| Citations | All + inline | Not structured | Not structured |

---

## See Also

- [Architecture](../architecture.md) — sovereignty layer design
- [Providers](../providers.md) — all providers + writing your own
