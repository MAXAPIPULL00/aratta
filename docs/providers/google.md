# Google Provider — SCRI ↔ Gemini Mapping

How Aratta translates between Google's Gemini API and SCRI, the language
your system speaks. Every field below shows the SCRI side (what your code
sees) and the Gemini side (what the adapter handles for you).

---

## Models

| Alias | Gemini Model ID | Notes |
|-------|-----------------|-------|
| `fast` | gemini-3-flash-preview | Speed optimized (default fast) |
| `cheap` | gemini-2.5-flash-lite | Cost optimized |
| (configurable) | gemini-3-pro-preview | Highest capability |

Aliases are configurable in `~/.aratta/config.toml`.

---

## Message Roles

| SCRI Role | Gemini Role | Notes |
|-----------|-------------|-------|
| `Role.SYSTEM` | N/A (uses `systemInstruction`) | Separate field |
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

### Gemini 3 (thinking_level)

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

# → Gemini 3 request
{"generationConfig": {"thinkingConfig": {"thinkingLevel": "high"}}}

# → Gemini 2.5 request
{"generationConfig": {"thinkingConfig": {"thinkingBudget": 10000}}}
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

# → Gemini tool
{"tools": [{"functionDeclarations": [{"name": "get_weather", "description": "Get weather", "parameters": {...}}]}]}
```

### Tool Call Response

```python
# Gemini response
{"candidates": [{"content": {"parts": [{"functionCall": {"name": "get_weather", "args": {"location": "NYC"}}, "thoughtSignature": "<sig>"}]}}]}

# → SCRI response
ChatResponse(tool_calls=[ToolCall(id="call_...", name="get_weather", arguments={"location": "NYC"})])
```

### Tool Result

```python
# SCRI message
Message(role=Role.TOOL, content=[Content(type=ContentType.TOOL_RESULT, tool_name="get_weather", tool_result={"temp": "15C"})])

# → Gemini message
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

Gemini uses a separate field for system messages:

```python
# SCRI messages
[Message(role=Role.SYSTEM, content="You are helpful"), Message(role=Role.USER, content="Hello")]

# → Gemini request
{"systemInstruction": {"parts": [{"text": "You are helpful"}]}, "contents": [{"role": "user", "parts": [{"text": "Hello"}]}]}
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

| SCRI Field | Gemini Field |
|------------|--------------|
| `Usage.input_tokens` | `usageMetadata.promptTokenCount` |
| `Usage.output_tokens` | `usageMetadata.candidatesTokenCount` |
| `Usage.total_tokens` | `usageMetadata.totalTokenCount` |
| `Usage.cache_read_tokens` | `usageMetadata.cachedContentTokenCount` |

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
# SCRI request
EmbeddingRequest(input=["text1", "text2"], model="gemini-embedding-001")

# → Gemini batch request
{"requests": [
    {"model": "models/gemini-embedding-001", "content": {"parts": [{"text": "text1"}]}},
    {"model": "models/gemini-embedding-001", "content": {"parts": [{"text": "text2"}]}}
]}
```

---

## API Endpoints

| Operation | Gemini Endpoint |
|-----------|-----------------|
| Chat | `POST /v1beta/models/{model}:generateContent` |
| Stream | `POST /v1beta/models/{model}:streamGenerateContent?alt=sse` |
| Count Tokens | `POST /v1beta/models/{model}:countTokens` |
| Embed (batch) | `POST /v1beta/models/{model}:batchEmbedContents` |

---

## See Also

- [Architecture](../architecture.md) — sovereignty layer design
- [Providers](../providers.md) — all providers + writing your own
