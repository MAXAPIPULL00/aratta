# Model Aliases

Aliases let you use human-friendly names instead of memorizing provider-specific
model IDs. They're defined in `config.py` and can be overridden in
`~/.aratta/config.toml`.

## Built-in Aliases

### Use-case aliases

These are the ones you'll type most often:

| Alias | Resolves To | Provider | Use Case |
|-------|-------------|----------|----------|
| `local` | `ollama:llama3.1:8b` | Ollama | Default local model |
| `sovereign` | `ollama:llama3.1:8b` | Ollama | Same as local — nothing leaves your machine |
| `fast` | `google:gemini-3-flash-preview` | Google | Fastest cloud response |
| `reason` | `anthropic:claude-opus-4-5-20251101` | Anthropic | Deep reasoning |
| `code` | `anthropic:claude-sonnet-4-5-20250929` | Anthropic | Code generation |
| `cheap` | `google:gemini-2.5-flash-lite` | Google | Lowest cost per token |

### Provider shortcuts

| Alias | Resolves To | Provider |
|-------|-------------|----------|
| `opus` | `claude-opus-4-5-20251101` | Anthropic |
| `sonnet` | `claude-sonnet-4-5-20250929` | Anthropic |
| `haiku` | `claude-haiku-4-5-20251001` | Anthropic |
| `gpt` | `gpt-4.1` | OpenAI |
| `gpt-mini` | `gpt-4.1-mini` | OpenAI |
| `o3` | `o3` | OpenAI |
| `gemini` | `gemini-3-flash-preview` | Google |
| `gemini-pro` | `gemini-3-pro-preview` | Google |
| `grok` | `grok-4-1-fast` | xAI |

### Embedding aliases

| Alias | Resolves To | Provider |
|-------|-------------|----------|
| `embed` | `text-embedding-3-large` | OpenAI |
| `embed-small` | `text-embedding-3-small` | OpenAI |

## Resolution Order

When you pass a model string, Aratta resolves it in this order:

1. **Alias lookup** — check `model_aliases` dict
2. **Explicit `provider:model`** — e.g. `anthropic:claude-opus-4-5-20251101`
3. **Infer from name** — `claude*` → anthropic, `gpt*` → openai, `gemini*` → google, `grok*` → xai, `llama*`/`mistral*`/`qwen*` → ollama
4. **Default provider** — falls back to `default_provider` (ollama by default)

## Custom Aliases

Add to `~/.aratta/config.toml`:

```toml
[aliases]
my-model = "ollama:deepseek-coder-v2:16b"
work = "anthropic:claude-sonnet-4-5-20250929"
```

Or pass directly in API calls:

```json
{"model": "my-model", "messages": [...]}
```
