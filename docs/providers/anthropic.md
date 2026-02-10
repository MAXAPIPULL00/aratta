# Anthropic Provider — SCRI ↔ Claude Mapping

How Aratta translates between Anthropic's Claude API and SCRI, the language
your system speaks. Every field below shows the SCRI side (what your code
sees) and the Anthropic side (what the adapter handles for you).

---

## Models

| Alias | Anthropic Model ID | Context | Max Output |
|-------|-------------------|---------|------------|
| `reason` | claude-opus-4-5-20251101 | 200K | 64K |
| `code` | claude-sonnet-4-5-20250929 | 200K (1M beta) | 64K |
| (configurable) | claude-haiku-4-5-20251001 | 200K | 64K |

Aliases are configurable in `~/.aratta/config.toml`.

---

## Message Roles

| SCRI Role | Anthropic Role | Notes |
|-----------|----------------|-------|
| `Role.SYSTEM` | N/A (separate `system` field) | Anthropic uses top-level system |
| `Role.USER` | `"user"` | Direct mapping |
| `Role.ASSISTANT` | `"assistant"` | Direct mapping |
| `Role.TOOL` | `"user"` with tool_result content | Special content block |

---

## Content Types

| SCRI Type | Anthropic Type | Description |
|-----------|----------------|-------------|
| `ContentType.TEXT` | `"text"` | Plain text content |
| `ContentType.IMAGE` | `"image"` | Base64 or URL images |
| `ContentType.TOOL_USE` | `"tool_use"` | Model requesting tool execution |
| `ContentType.TOOL_RESULT` | `"tool_result"` | Tool execution result |
| `ContentType.THINKING` | `"thinking"` | Extended thinking block |

---

## Finish Reasons

| SCRI Reason | Anthropic Reason | When |
|-------------|------------------|------|
| `FinishReason.STOP` | `"end_turn"` | Natural completion |
| `FinishReason.STOP` | `"stop_sequence"` | Hit stop sequence |
| `FinishReason.TOOL_CALLS` | `"tool_use"` | Model wants tools |
| `FinishReason.LENGTH` | `"max_tokens"` | Hit token limit |
| `FinishReason.CONTENT_FILTER` | N/A | Not used by Anthropic |

---

## Extended Thinking

### Configuration

| SCRI | Anthropic | Notes |
|------|-----------|-------|
| `thinking_enabled=True` | `thinking.type: "enabled"` | Enable thinking |
| `thinking_budget` | `thinking.budget_tokens` | Min 1024 tokens |

### Request Mapping
```python
# SCRI request
ChatRequest(thinking_enabled=True, thinking_budget=10000)

# → Anthropic request
{"thinking": {"type": "enabled", "budget_tokens": 10000}}
```

### Response Mapping
```python
# Anthropic response
{"content": [
    {"type": "thinking", "thinking": "...", "signature": "..."},
    {"type": "text", "text": "..."}
]}

# → SCRI response
ChatResponse(thinking=[ThinkingBlock(thinking="...", signature="...")], content="...")
```

### Rules
- Temperature fixed at 1.0 when enabled
- Min budget: 1,024 tokens
- Max budget: max_tokens - 1
- Thinking blocks must be preserved in tool use loops
- Signatures verify authenticity

---

## Effort Control (Opus 4.5 Only)

| Level | Anthropic Level | Use Case |
|-------|-----------------|----------|
| `HIGH` | `"high"` | Maximum reasoning (default) |
| `MEDIUM` | `"medium"` | Balanced |
| `LOW` | `"low"` | Minimal reasoning |

```python
# → Anthropic request
{"output_config": {"effort": "medium"}}
```

---

## Prompt Caching

### TTL Options

| TTL | Duration | Write Cost | Read Cost |
|-----|----------|------------|-----------|
| Ephemeral 5m | 5 minutes | 1.25x | 0.1x |
| Ephemeral 1h | 1 hour | 2x | 0.1x |

### Minimum Cacheable Tokens

| Model | Min Tokens |
|-------|------------|
| Opus 4.5, Haiku 4.5 | 4,096 |
| Sonnet 4.5, Sonnet 4 | 1,024 |

### Usage Fields
| SCRI Field | Anthropic Field |
|------------|-----------------|
| `Usage.cache_read_tokens` | `cache_read_input_tokens` |
| `Usage.cache_write_tokens` | `cache_creation_input_tokens` |

---

## Tool Use

### Tool Definition

| SCRI | Anthropic | Notes |
|------|-----------|-------|
| `Tool.name` | `name` | Tool identifier |
| `Tool.description` | `description` | What tool does |
| `Tool.parameters` | `input_schema` | JSON Schema format |

### Tool Choice

| SCRI | Anthropic | Behavior |
|------|-----------|----------|
| `"auto"` | `{"type": "auto"}` | Model decides |
| `"none"` | `{"type": "none"}` | No tools |
| `"any"` | `{"type": "any"}` | Must use a tool |
| `{"tool": "name"}` | `{"type": "tool", "name": "..."}` | Force specific |

### Tool Call Response

```python
# Anthropic response
{"content": [{"type": "tool_use", "id": "toolu_01...", "name": "get_weather", "input": {"location": "NYC"}}]}

# → SCRI response
ChatResponse(tool_calls=[ToolCall(id="toolu_01...", name="get_weather", arguments={"location": "NYC"})])
```

### Tool Result

```python
# SCRI message
Message(role=Role.USER, content=[Content(type=ContentType.TOOL_RESULT, tool_use_id="toolu_01...", tool_result="15°C, sunny")])

# → Anthropic message
{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01...", "content": "15°C, sunny"}]}
```

---

## Streaming Events

| Anthropic Event | Description |
|-----------------|-------------|
| `message_start` | Message metadata |
| `content_block_start` | Block type |
| `content_block_delta` | Content chunk (text, thinking, tool input) |
| `content_block_stop` | Block complete |
| `message_delta` | Usage, stop_reason |
| `message_stop` | Message complete |

### Delta Types

| Anthropic Delta | Content |
|-----------------|---------|
| `text_delta` | Text content |
| `thinking_delta` | Reasoning content |
| `input_json_delta` | Tool parameters |
| `signature_delta` | Thinking verification |

---

## Vision

| Limit | Value |
|-------|-------|
| Max images/request | 100 |
| Max file size (API) | 5MB |
| Max dimensions | 8000x8000 px |
| Token calculation | `(width × height) / 750` |

```python
# SCRI content
Content(type=ContentType.IMAGE, image_base64="...")

# → Anthropic content
{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
```

---

## Required Headers

```
x-api-key: $ANTHROPIC_API_KEY
anthropic-version: 2023-06-01
content-type: application/json
```

### Feature-Specific Beta Headers

| Feature | Beta Header |
|---------|-------------|
| Extended Thinking | `extended-thinking-2025-01-24` |
| Interleaved Thinking | `interleaved-thinking-2025-05-14` |
| Effort Control | `effort-2025-11-24` |
| Context Management | `context-management-2025-06-27` |
| 1M Context | `context-1m-2025-08-07` |
| Code Execution | `code-execution-2025-08-25` |
| Fine-grain Streaming | `fine-grained-tool-streaming-2025-05-14` |
| Files API | `files-api-2025-04-14` |

---

## Server-Side Tools (Anthropic-Provided)

| Tool | Purpose |
|------|---------|
| `bash` | Execute shell commands |
| `text_editor` | File operations |
| `computer` | Mouse/keyboard control |
| `code_execution` | Python sandbox (5GB RAM, pandas/numpy/matplotlib) |
| `memory_tool` | Persistent memory (view, create, replace, delete) |

---

## See Also

- [Architecture](../architecture.md) — sovereignty layer design
- [Providers](../providers.md) — all providers + writing your own
