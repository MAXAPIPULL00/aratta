# Google Provider — SCRI ↔ Gemini Mapping

How Aratta translates between Google's Gemini API and SCRI, the language
your system speaks. Every field below shows the SCRI side (what your code
sees) and the Gemini side (what the adapter handles for you).

---

## SDK

The Google adapter uses the official `google-genai` SDK. Install with:

```bash
pip install aratta[google]
```

The adapter creates a `google.genai.Client` and uses
`client.aio.models.generate_content()` for async chat completions.

---

## Models

8 models across three generations:

### Gemini 3.1

| Alias | Gemini Model ID | Context | Output | Pricing (in/out per MTok) | Notes |
|-------|-----------------|---------|--------|---------------------------|-------|
| `gemini-pro` / `gemini-3.1` | gemini-3.1-pro-preview | 1M | 65k | $2 / $12 | Latest Pro, reasoning + agentic |
| — | gemini-3.1-pro-preview-customtools | 1M | 65k | $2 / $12 | Custom tool use optimized |

### Gemini 3

| Alias | Gemini Model ID | Context | Output | Pricing (in/out per MTok) | Notes |
|-------|-----------------|---------|--------|---------------------------|-------|
| — | gemini-3-pro-preview | 1M | 64k | $2 / $12 | Pro reasoning |
| `fast` | gemini-3-flash-preview | 1M | 64k | $0.50 / $3 | Speed optimized (default fast) |
| `image` | gemini-3-pro-image-preview | 1M | 64k | $2 / $12 | Image generation |

### Gemini 2.5

| Alias | Gemini Model ID | Context | Output | Pricing (in/out per MTok) | Notes |
|-------|-----------------|---------|--------|---------------------------|-------|
| — | gemini-2.5-pro | 1M | 64k | $1.25 / $5 | Previous gen pro |
| — | gemini-2.5-flash | 1M | 64k | $0.15 / $0.60 | Previous gen flash |
| `cheap` | gemini-2.5-flash-lite | 1M | 64k | $0.075 / $0.30 | Lowest cost |

Aliases are configurable in `~/.aratta/config.toml`.

---

## API Pattern

```python
# SDK pattern (what the adapter does internally)
client = genai.Client(api_key="...")
response = await client.aio.models.generate_content(
    model="gemini-3.1-pro-preview",
    contents=[...],
    config={
        "system_instruction": "...",
        "temperature": 0.7,
        "tools": [...],
        "thinking_config": {"thinking_level": "high"},
    },
)
```

---

## Message Roles

| SCRI Role | Gemini Role | Notes |
|-----------|-------------|-------|
| `Role.SYSTEM` | N/A (uses `system_instruction`) | Separate config field |
| `Role.USER` | `"user"` | Direct mapping |
| `Role.ASSISTANT` | `"model"` | Gemini uses "model" |
| `Role.TOOL` | `"user"` with functionResponse | Tool results from user |

---

## Content Types

| SCRI Type | Gemini Type | Description |
|-----------|-------------|-------------|
| `ContentType.TEXT` | `text` | Plain text content |
| `ContentType.IMAGE` | `inlineData` / `fileData` | Images |
| `ContentType.TOOL_USE` | `functionCall` | Model requesting tool |
| `ContentType.TOOL_RESULT` | `functionResponse` | Tool result |

---

## Finish Reasons

| SCRI Reason | Gemini Reason | When |
|-------------|---------------|------|
| `FinishReason.STOP` | `STOP` | Natural completion |
| `FinishReason.LENGTH` | `MAX_TOKENS` | Hit token limit |
| `FinishReason.CONTENT_FILTER` | `SAFETY` | Safety filter |
| `FinishReason.CONTENT_FILTER` | `RECITATION` | Copyright filter |
| `FinishReason.TOOL_CALLS` | N/A (inferred) | Model wants tools |

---

## Thinking Configuration

### Gemini 3 / 3.1 (thinking_level)

| Budget Range | Gemini Config | Notes |
|-------------|---------------|-------|
| ≤ 1024 tokens | `thinkingLevel: "low"` | Minimal thinking |
| ≤ 8192 tokens | `thinkingLevel: "medium"` | Flash only |
| > 8192 tokens | `thinkingLevel: "high"` | Maximum reasoning |

### Gemini 2.5 (thinking_budget)

Direct token budget mapping: `thinkingBudget: N`

### Request Mapping
```python
# SCRI request
ChatRequest(thinking_enabled=True, thinking_budget=10000)

# → Gemini 3.x config
{"thinking_config": {"thinking_level": "high"}}

# → Gemini 2.5 config
{"thinking_config": {"thinking_budget": 10000}}
```

---

## Tool Use

### Tool Definition

| SCRI | Gemini | Notes |
|------|--------|-------|
| `Tool.name` | `name` | Tool identifier |
| `Tool.description` | `description` | What tool does |
| `Tool.parameters` | `parameters` | JSON Schema |

### Conversion
```python
# SCRI tool
Tool(name="get_weather", description="Get weather", parameters={...})

# → Gemini tool (via SDK)
{"functionDeclarations": [{"name": "get_weather", "description": "Get weather", "parameters": {...}}]}
```

### Tool Call Response

```python
# SDK response part
part.function_call  # FunctionCall(name="get_weather", args={"location": "NYC"})

# → SCRI response
ChatResponse(tool_calls=[ToolCall(id="call_...", name="get_weather", arguments={"location": "NYC"})])
```

### Tool Result

```python
# SCRI message
Message(role=Role.TOOL, content=[Content(type=ContentType.TOOL_RESULT, tool_name="get_weather", tool_result={"temp": "15C"})])

# → Gemini content
{"role": "user", "parts": [{"functionResponse": {"name": "get_weather", "response": {"temp": "15C"}}}]}
```

---

## Built-in Tools (Google-Provided)

| Tool | Gemini Config | Description |
|------|---------------|-------------|
| Google Search | `{"googleSearch": {}}` | Web search grounding |
| Code Execution | `{"codeExecution": {}}` | Python sandbox |
| URL Context | `{"urlContext": {}}` | Read web pages |

---

## System Instruction

Gemini uses a separate config field for system messages. The adapter
extracts SYSTEM role messages and passes them as `system_instruction`:

```python
# SCRI messages
[Message(role=Role.SYSTEM, content="You are helpful"), Message(role=Role.USER, content="Hello")]

# → SDK call
config={"system_instruction": "You are helpful"}
contents=[{"role": "user", "parts": [{"text": "Hello"}]}]
```

---

## Thought Signatures

Gemini 3 requires preserving thought signatures in multi-turn tool use.
The adapter handles this automatically. When importing conversations from
other models, use the bypass signature:
```json
{"thoughtSignature": "context_engineering_is_the_way_to_go"}
```

---

## Usage Mapping

| SCRI Field | Gemini SDK Field |
|------------|------------------|
| `Usage.input_tokens` | `usage_metadata.prompt_token_count` |
| `Usage.output_tokens` | `usage_metadata.candidates_token_count` |
| `Usage.total_tokens` | `usage_metadata.total_token_count` |
| `Usage.cache_read_tokens` | `usage_metadata.cached_content_token_count` |

---

## Image Content

### Base64
```python
# SCRI: Content(type=ContentType.IMAGE, image_base64="...")
# → Gemini: {"inlineData": {"mimeType": "image/jpeg", "data": "..."}}
```

### URL
```python
# SCRI: Content(type=ContentType.IMAGE, image_url="gs://bucket/image.jpg")
# → Gemini: {"fileData": {"fileUri": "gs://bucket/image.jpg"}}
```

---

## Structured Output

| Feature | Gemini Config |
|---------|---------------|
| JSON mode | `responseMimeType: "application/json"` |
| Schema | `responseJsonSchema: {...}` |

---

## Embeddings

```python
# SDK pattern
response = await client.aio.models.embed_content(
    model="gemini-embedding-001",
    contents=["text1", "text2"],
)
# response.embeddings → list of embedding objects with .values
```

---

## See Also

- [Architecture](../architecture.md) — sovereignty layer design
- [Providers](../providers.md) — all providers + writing your own
