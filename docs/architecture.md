# Architecture

Aratta is a sovereignty layer. It sits between your system and every AI
provider, inverting the dependency relationship. Instead of your code being a
client of Anthropic, OpenAI, Google, or xAI — they become interchangeable
services your system invokes.

Your local models are the foundation. Cloud providers are commoditized resources
you call when a task requires specific capabilities. The resilience infrastructure
makes this sovereignty real rather than theoretical.

## The Stack

```
┌─────────────────────────────────────────────┐
│                  CLI / Server                │  aratta init, aratta serve
├─────────────────────────────────────────────┤
│               Config + Aliases               │  ~/.aratta/config.toml
├─────────────────────────────────────────────┤
│              Resilience Layer                │
│  Health Monitor → Heal Worker → Reload Mgr  │
│  Circuit Breaker    Metrics                 │
├──────┬──────┬──────┬──────┬─────────────────┤
│Local │Anthr.│OpenAI│Google│  xAI            │  Provider adapters
├──────┴──────┴──────┴──────┴─────────────────┤
│              SCRI (the language)              │  ChatRequest, ChatResponse, etc.
└─────────────────────────────────────────────┘
```

The local provider sits at the same level as cloud providers, but it's the
default. Cloud is the fallback.

## SCRI — The Language (`aratta.core.types`)

SCRI is the language your system speaks. These types define it:

- `ChatRequest` / `ChatResponse` — the universal chat format
- `Message` — role + content (text or structured blocks)
- `Tool` / `ToolCall` — provider-agnostic tool definitions and invocations
- `Usage` — token counts with optional cache/reasoning breakdowns
- `Lineage` — provenance tracking (which provider, model, latency, request ID)
- `EmbeddingRequest` / `EmbeddingResponse` — unified embeddings

Your code writes SCRI. Responses come back as SCRI. Provider-specific formats
never leak into your application logic.

## Provider Adapters (`aratta.providers.*`)

Each adapter translates between SCRI and a provider's native API. They inherit
from `BaseProvider` and implement:

| Method | Purpose |
|--------|---------|
| `chat()` | SCRI request → provider API → SCRI response |
| `chat_stream()` | SCRI request → provider SSE → unified SSE |
| `embed()` | SCRI embedding request → provider → SCRI response |
| `convert_messages()` | SCRI messages → provider message format |
| `convert_tools()` | SCRI tools → provider tool schema |
| `get_models()` | Provider models → SCRI `ModelCapabilities` |
| `health_check()` | Ping the provider |

The local adapter covers Ollama, vLLM, and llama.cpp — all three speak the
OpenAI-compatible API, so one adapter handles them.

For field-by-field mapping details per provider, see:
- [providers/anthropic.md](providers/anthropic.md)
- [providers/openai.md](providers/openai.md)
- [providers/google.md](providers/google.md)
- [providers/xai.md](providers/xai.md)

## Resilience Layer (`aratta.resilience`)

The resilience layer makes sovereignty real rather than theoretical. Without it,
you're just a wrapper that breaks when providers change.

### Health Monitor (`health.py`)

Continuous error tracking with pattern detection. Every provider error is
classified:

- Schema mismatches, unknown fields, deprecated fields
- Tool schema changes, streaming format changes
- Transient errors (rate limits, timeouts, connection resets) — tracked but
  don't trigger healing

Configurable thresholds: how many errors in what time window before healing
kicks in. Cooldown periods prevent heal loops. Pluggable callbacks for custom
alerting.

### Heal Worker (`heal_worker.py`)

Three-phase self-healing cycle. This is the core of sovereignty.

```
Phase 1: DIAGNOSE (local model)
    → Analyzes the error, classifies the failure
    → Formulates search queries for current API docs

Phase 2: RESEARCH (cloud provider — Grok web search by default)
    → Escalates to a search-capable provider
    → Finds current API docs, changelogs, breaking changes
    → Falls back: xai → openai → google → anthropic

Phase 3: FIX (local model)
    → Takes research + adapter source code
    → Generates fix proposal with confidence score
    → High confidence → auto-apply. Low confidence → human queue.
```

The local model is the brain. Cloud search is the eyes. Cloud providers don't
make decisions — they fetch what the local model asked for.

### Reload Manager (`reload_manager.py`)

Hot-reloads provider adapters after fixes:

- Backs up the current adapter before any change
- Applies the code patch
- Reloads the Python module in-memory (no restart)
- Runs a verification health check
- Commits on success, auto-rollbacks on failure
- Maintains bounded version history (10 versions per provider)
- Human approval queue for low-confidence fixes

### Circuit Breaker (`circuit_breaker.py`)

Three-state circuit breaker per provider:

```
CLOSED → normal operation, requests pass through
OPEN   → provider is down, fail fast (no timeouts)
HALF_OPEN → testing recovery with probe requests
```

Configurable failure thresholds, recovery timeouts, and success thresholds
for closing. Force open for maintenance, force close after manual fix, reset
to initial state — all through the API.

### Metrics (`metrics.py`)

Thread-safe counters, gauges, and histograms:

- `provider_errors_total` — errors by provider and type
- `circuit_opens_total` — circuit breaker activations
- `heal_requests_total` — heal cycles triggered
- `open_circuits` — currently open circuits
- `heal_duration_seconds` — heal cycle timing

Exportable as JSON. The `/api/v1/metrics` endpoint exposes these.

## Provider Fallback

If the primary provider for a model isn't available, Aratta walks the priority
list. Local providers have priority 0 (highest). Cloud providers are ranked by
configuration. The fallback is automatic and transparent — your application
doesn't know or care which provider handled the request.

## Agent Framework (`aratta.agents`)

Provider-agnostic ReAct agent loop:

- `Agent` — the main class. Takes a model alias, runs the reason-act-observe
  loop until the task is done or max iterations hit.
- `AgentContext` — manages conversation history, tool results, and state
  across iterations.
- `AgentConfig` — model, max iterations, temperature, tool permissions.
- Executor — runs tool calls with audit logging.
- Sandbox — AST-validated code execution in subprocess isolation.
- Permissions — four tiers (restricted/standard/elevated/system) controlling
  what tools an agent can invoke.

Switch the model alias and the same agent uses a different provider. No code
changes.

## Tool Registry (`aratta.tools`)

Register tools once in a universal format, export to any provider's schema:

```python
from aratta.tools.registry import ToolDef, get_registry

registry = get_registry()
registry.register(ToolDef(
    name="get_weather",
    description="Get current weather",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}}
))

# Export for any provider
anthropic_tools = registry.export_for_provider("anthropic")  # input_schema
openai_tools = registry.export_for_provider("openai")        # function.parameters
google_tools = registry.export_for_provider("google")        # functionDeclarations
```

Import from provider formats too — `from_anthropic()`, `from_openai()`,
`from_google()`. The registry handles the translation both ways.

## Configuration (`aratta.config`)

`~/.aratta/config.toml` with sections for server, providers, aliases, behaviour,
and healing. Environment variables override TOML values. API keys are always
read from environment.

The `[healing]` section controls the self-healing stack:

```toml
[healing]
enabled = true
auto_apply = false        # require human approval
heal_model = "local"      # local model diagnoses and fixes
error_threshold = 3       # errors before healing triggers
cooldown_seconds = 600    # minimum time between heal cycles
```

See the [README](../README.md) for the full config reference.