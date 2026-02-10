# xAI Provider — SCRI ↔ Grok Mapping

How Aratta translates between xAI's Grok API and SCRI, the language your
system speaks. Every field below shows the SCRI side (what your code sees)
and the Grok side (what the adapter handles for you).

---

## Models

| Alias | xAI Model | Use Case |
|-------|-----------|----------|
| `grok` | grok-4-1-fast | Default Grok alias |
| (configurable) | grok-4 | Deep reasoning |
| (configurable) | grok-4-fast | Fast agentic |
| (configurable) | grok-2-vision | Image analysis |

Aliases are configurable in `~/.aratta/config.toml`.

---

## API Endpoints

| xAI | Adapter Method | Notes |
|-----|----------------|-------|
| POST /v1/chat/completions | `chat()` | Primary chat endpoint |
| POST /v1/chat/completions (stream) | `chat_stream()` | With verbose_streaming |
| POST /v1/embeddings | `embed()` | Embeddings |

---

## Request Parameters

| xAI | SCRI Type | Notes |
|-----|-----------|-------|
| `model` | `ChatRequest.model` | Direct |
| `messages` | `ChatRequest.messages` | Direct |
| `tools` | Server-side tools | Built-in tools |
| `max_tokens` | `ChatRequest.max_tokens` | Direct |
| `temperature` | `ChatRequest.temperature` | Direct |
| `top_p` | `ChatRequest.top_p` | Direct |

---

## Response Mapping

| xAI | SCRI | Notes |
|-----|------|-------|
| `id` | `ChatResponse.id` | Response ID |
| `choices[0].message.content` | `ChatResponse.content` | Direct |
| `choices[0].message.tool_calls` | `ChatResponse.tool_calls` | Client-side only |
| `choices[0].finish_reason` | `ChatResponse.finish_reason` | Mapped |
| `model` | `ChatResponse.model` | Direct |
| `usage.prompt_tokens` | `Usage.input_tokens` | Renamed |
| `usage.completion_tokens` | `Usage.output_tokens` | Renamed |
| `usage.reasoning_tokens` | `Usage.reasoning_tokens` | Direct |
| `usage.cached_prompt_text_tokens` | `Usage.cache_read_tokens` | Renamed |

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
| `system` / `developer` | `Role.SYSTEM` |
| `user` | `Role.USER` |
| `assistant` | `Role.ASSISTANT` |
| `tool` | `Role.TOOL` |

---

## Server-Side Tools

| xAI Tool | Description |
|----------|-------------|
| `web_search` | Web search + page browsing |
| `x_search` | X/Twitter search (user, keyword, semantic) |
| `code_execution` | Python sandbox |
| `collections_search` | Document collection search |
| `attachment_search` | Auto-enabled with file uploads |

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

## Collections API

| Operation | xAI Endpoint |
|-----------|--------------|
| Create collection | `POST /v1/collections` |
| List collections | `GET /v1/collections` |
| Upload document | `POST /v1/collections/:id/documents` |
| Search documents | `POST /v1/documents/search` |

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
| X/Twitter search | Native | Not available | Not available |
| Collections | Native API | Not available | Vector stores |
| Web search | `web_search` + `browse_page` | Not available | `web_search` |
| Thinking | `encrypted_content` | Explicit `thinking` blocks | `reasoning` items |
| Citations | All + inline | Not structured | Not structured |

---

## Pricing (Tool Invocations)

| Tool | Cost per 1K calls |
|------|-------------------|
| web_search | $10.00 |
| x_search | $10.00 |
| code_execution | Variable (compute) |
| collections_search | $2.50 |
| attachment_search | $10.00 |

---

## See Also

- [Architecture](../architecture.md) — sovereignty layer design
- [Providers](../providers.md) — all providers + writing your own
