# Aratta

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-82%20passing-green.svg)](tests/)

*The land that traded with empires but was never conquered.*

---

## Why

You got rate-limited again. Or your API key got revoked. Or they changed
their message format and your pipeline broke at 2am. Or you watched your
entire system go dark because one provider had an outage.

You built on their platform. You followed their docs. You used their
SDK. And now you depend on them completely — their pricing, their
uptime, their rules, their format, their permission.

That's not infrastructure. That's a leash.

Aratta takes it off.

## What Aratta Is

Aratta is a sovereignty layer. It sits between your application and every
AI provider — local and cloud — and inverts the power relationship.

Your local models are the foundation. Cloud providers — Claude, GPT,
Gemini, Grok — become callable services your system invokes when a task
requires specific capabilities. They're interchangeable. One goes down,
another picks up. One changes their API, the system self-heals. You
don't depend on any of them. They work for you.

```
            ┌───────────────────┐
            │  Your App (SCRI)  │  ← you own this
            └─────────┬─────────┘
                      │
                ┌───────────┐
                │  Aratta   │  ← sovereignty layer
                └─────┬─────┘
         ┌───┬───┬────┴────┬───┐
         ▼   ▼   ▼         ▼   ▼
      Ollama Claude GPT Gemini Grok
       local  ─── cloud services ───
```

## SCRI

SCRI is the language your system speaks. One set of types for messages,
tool calls, responses, usage, and streaming — regardless of which
provider is on the other end. Your code speaks SCRI. Every provider
adapter translates to and from SCRI. Provider-specific structures
never leak into your application logic.

```python
from aratta.core.types import ChatRequest, Message, Role

request = ChatRequest(
    messages=[Message(role=Role.USER, content="Explain quantum computing")],
    model="local",     # your foundation
    # model="reason",  # or invoke Claude when you need it
    # model="gpt",     # or GPT — same code, same response shape
)
```

The response comes back in SCRI regardless of which provider handled it.
Same fields, same types, same structure. Your application logic is
decoupled from every provider's implementation details. You write it once.

### What SCRI replaces

Every provider does everything differently:

| Concept | Anthropic | OpenAI | Google | xAI |
|---------|-----------|--------|--------|-----|
| Tool calls | `tool_use` block | `function_call` | `functionCall` | `function` |
| Tool defs | `input_schema` | `function.parameters` | `functionDeclarations` | `function.parameters` |
| Finish reason | `stop_reason` | `finish_reason` | `finishReason` | `finish_reason` |
| Token usage | `usage.input_tokens` | `usage.prompt_tokens` | `usageMetadata.promptTokenCount` | `usage.prompt_tokens` |
| Streaming | `content_block_delta` | `choices[0].delta` | `candidates[0]` | OpenAI-compat |
| Thinking | `thinking` block | `reasoning` output | `thinkingConfig` | encrypted |
| Auth | `x-api-key` | `Bearer` token | `x-goog-api-key` | `Bearer` token |

SCRI: `Message`, `ToolCall`, `Usage`, `FinishReason`. One language. Every provider.

## Quick Start

```bash
git clone https://github.com/MAXAPIPULL00/aratta.git
cd aratta
pip install -e .
aratta init                   # pick providers, set API keys, configure local
aratta serve                  # starts on :8084
```

The `init` wizard walks you through setup — which providers to enable,
API keys, and local model configuration. Ollama, vLLM, and llama.cpp
are supported as local backends. Local is the default. Cloud is optional.

> **Note:** Requires Python 3.11 or newer. PyPI package coming soon —
> for now, install from source as shown above.

### Use it

```python
import httpx

# Local model — your foundation
resp = httpx.post("http://localhost:8084/api/v1/chat", json={
    "messages": [{"role": "user", "content": "Hello"}],
    "model": "local",
})

# Need deep reasoning? Invoke a cloud provider
resp = httpx.post("http://localhost:8084/api/v1/chat", json={
    "messages": [{"role": "user", "content": "Analyze this contract"}],
    "model": "reason",
})

# Need something else? Same interface, different provider
resp = httpx.post("http://localhost:8084/api/v1/chat", json={
    "messages": [{"role": "user", "content": "Generate test cases"}],
    "model": "gpt",
})

# Response is always SCRI — same shape, same types, regardless of provider
```

## Model Aliases

Route by capability, not by provider model ID:

| Alias | Default | Provider |
|-------|---------|----------|
| `local` | llama3.1:8b | Ollama |
| `sovereign` | llama3.1:8b | Ollama |
| `fast` | gemini-3-flash-preview | Google |
| `reason` | claude-opus-4-5-20251101 | Anthropic |
| `code` | claude-sonnet-4-5-20250929 | Anthropic |
| `cheap` | gemini-2.5-flash-lite | Google |
| `gpt` | gpt-4.1 | OpenAI |
| `grok` | grok-4-1-fast | xAI |
| `embed` | text-embedding-3-large | OpenAI |

Aliases are configurable in `~/.aratta/config.toml`. Point `reason` at
your local 70B if you want. Point `fast` at GPT. Your routing. Your rules.

Full reference: [docs/model-aliases.md](docs/model-aliases.md)

## What Makes the Sovereignty Real

The sovereignty isn't a metaphor. It's enforced by infrastructure.

### Self-Healing

When a cloud provider changes their API — and they will — Aratta
doesn't just break and wait for you to fix it. It heals itself.

The heal cycle is a three-phase process:

1. **Diagnose** — your local model analyzes the error. It classifies
   the failure (schema change? new field? deprecated endpoint?),
   determines if it's transient or structural, and formulates search
   queries to find the current documentation.

2. **Research** — Aratta escalates to a search-capable cloud provider
   (Grok with web search by default) to find the actual current API
   docs and changelogs. The cloud provider is a tool here — it fetches
   what the local model asked for. It doesn't make decisions.

3. **Fix** — your local model takes the research findings plus the
   current adapter source code and generates a fix proposal with a
   confidence score. High-confidence fixes can auto-apply. Low-confidence
   fixes go to a human approval queue.

The local model is the brain. Cloud search is the eyes. That's
sovereignty in practice — you use trillion-dollar platforms as tools,
not as decision makers.

```
Error detected
    │
    ▼
┌──────────────────┐
│  Health Monitor   │  tracks errors, classifies types, detects patterns
└────────┬─────────┘
         │ threshold reached (3 errors / 5 min)
         ▼
┌──────────────────┐
│  Phase 1: LOCAL  │  your model diagnoses the error
│  (diagnose)      │  formulates search queries
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Phase 2: CLOUD  │  Grok searches the web for current API docs
│  (research)      │  falls back through xai → openai → google → anthropic
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Phase 3: LOCAL  │  your model generates the fix
│  (fix)           │  confidence-scored proposal
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Reload Manager  │  backup → apply → reload → verify → commit/rollback
│                  │  version history, human approval queue
└──────────────────┘
```

Every fix is backed up before application. Every fix is verified with a
health check after application. Failed verifications auto-rollback. You
can review pending fixes, approve or reject them, and rollback to any
previous version through the API.

### Circuit Breakers

If a cloud provider fails, your system doesn't. The circuit breaker
opens, requests fail fast instead of timing out, and half-open probes
test recovery automatically. When the provider comes back, the circuit
closes and traffic resumes.

Force circuits open for maintenance. Force them closed after a manual
fix. Reset to initial state. All through the API.

### Health Monitoring

Continuous error tracking with pattern detection. Errors are classified
by type — schema mismatches, unknown fields, deprecated fields, tool
schema changes, streaming format changes. Transient errors (rate limits,
timeouts, connection resets) are tracked but don't trigger healing.
Structural errors do.

Configurable thresholds, time windows, and cooldown periods. Pluggable
callbacks for custom alerting.

### Provider Fallback

If your primary provider for a model isn't available, Aratta walks the
priority list and routes to the next available provider. Local providers
have priority 0 (highest). Cloud providers are ranked by your
configuration. The fallback is automatic and transparent.

### Local-First

`default_provider = "ollama"`. That's not a suggestion. Your foundation
runs on your hardware. Cloud is the fallback, not the default.

## API

### Core

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe |
| `/api/v1/chat` | POST | Chat completion — any provider, SCRI in and out |
| `/api/v1/chat/stream` | POST | Streaming chat (SSE) |
| `/api/v1/embed` | POST | Embeddings |
| `/api/v1/models` | GET | Available models and aliases |
| `/api/v1/health` | GET | Per-provider health + circuit breaker states |

### Self-Healing

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/healing/status` | GET | Self-healing system status |
| `/api/v1/healing/pause/{provider}` | POST | Pause healing for a provider |
| `/api/v1/healing/resume/{provider}` | POST | Resume healing |
| `/api/v1/fixes/pending` | GET | Fixes awaiting human approval |
| `/api/v1/fixes/{provider}/approve` | POST | Approve and apply a pending fix |
| `/api/v1/fixes/{provider}/reject` | POST | Reject a pending fix |
| `/api/v1/fixes/{provider}/history` | GET | Version history for a provider's adapter |
| `/api/v1/fixes/{provider}/rollback/{version}` | POST | Rollback to a specific version |

### Circuit Breakers

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/circuit/{provider}/open` | POST | Force circuit open (maintenance) |
| `/api/v1/circuit/{provider}/close` | POST | Force circuit closed (manual fix) |
| `/api/v1/circuit/{provider}/reset` | POST | Reset to initial state |

### Observability

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/metrics` | GET | Resilience metrics (errors, circuit opens, heal requests) |
| `/api/v1/dashboard` | GET | Full system dashboard — providers, health, healing, metrics |

## Agent Framework

Aratta includes a provider-agnostic ReAct agent loop:

```python
from aratta.agents import Agent, AgentConfig, AgentContext

ctx = AgentContext.create(aratta_client)
agent = Agent(config=AgentConfig(model="local"), context=ctx)
result = await agent.run("Research this topic and summarize")
```

The agent reasons, calls tools, observes results, and iterates. Switch
the model alias and the same agent uses a different provider. No code
changes.

Includes sandboxed code execution (AST-validated, subprocess-isolated),
a permission system (restricted/standard/elevated/system tiers), and a
tool executor with audit logging.

Details: [docs/agents.md](docs/agents.md)

## Testing

```bash
pytest                        # Run all 82 tests
pytest --cov=aratta           # Run with coverage
pytest tests/test_resilience/ # Run a specific suite
```

The test suite covers:

| Suite | Tests | What it covers |
|-------|-------|---------------|
| `test_core/` | Type system | SCRI types, serialization, validation |
| `test_providers/` | All 5 adapters | Anthropic, OpenAI, Google, xAI, Local message/tool conversion |
| `test_resilience/` | Circuit breaker | State transitions, thresholds, half-open recovery |
| `test_agents/` | Agent loop | ReAct iterations, tool execution, exit conditions |
| `test_tools/` | Tool registry | Registration, format export, duplicate handling |
| `test_server.py` | API endpoints | Health, chat, models, error responses |

## Environment Variables

Copy `.env.example` to `.env` before running `aratta init`. API keys are
the only required configuration — and only for the providers you enable.

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | If using Claude | Anthropic API key (`sk-ant-...`) |
| `OPENAI_API_KEY` | If using GPT | OpenAI API key (`sk-...`) |
| `GOOGLE_API_KEY` | If using Gemini | Google AI API key |
| `XAI_API_KEY` | If using Grok | xAI API key (`xai-...`) |
| `ARATTA_PORT` | No | Server port (default: `8084`) |
| `ARATTA_HOST` | No | Bind address (default: `0.0.0.0`) |
| `LOG_LEVEL` | No | Logging level (default: `INFO`) |

Local models (Ollama, vLLM, llama.cpp) don't need API keys. If you're
running local-only, you don't need a `.env` file at all.

## Configuration

`~/.aratta/config.toml`:

```toml
[server]
host = "0.0.0.0"
port = 8084

[local.ollama]
base_url = "http://localhost:11434"
default_model = "llama3.1:8b"

[providers.anthropic]
enabled = true

[aliases]
reason = "anthropic:claude-opus-4-5-20251101"
fast = "ollama:llama3.1:8b"

[behaviour]
default_provider = "ollama"
prefer_local = true
enable_fallback = true

[healing]
enabled = true
auto_apply = false
heal_model = "local"
error_threshold = 3
cooldown_seconds = 600
```

Environment variables override TOML. API keys are read from env
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `XAI_API_KEY`).

## How It Compares

| Feature | Aratta | LiteLLM | OpenRouter | Portkey |
|---------|--------|---------|------------|---------|
| Multi-provider routing | Yes | Yes | Yes | Yes |
| Local-first (Ollama/vLLM) | Foundation | Supported | No | No |
| Self-healing adapters | Yes | No | No | No |
| Circuit breakers | Yes | No | No | Yes |
| Health monitoring | Per-provider | Basic | No | Yes |
| Provider fallback | Automatic | Configurable | Automatic | Configurable |
| Agent framework | Built-in ReAct | No | No | No |
| Universal type system | SCRI | Partial | No | No |
| Human approval queue | Yes (fixes) | No | No | No |
| Adapter hot-reload | Yes | No | No | No |
| Self-hosted | Yes | Yes | No (proxy) | Cloud + gateway |
| Vendor lock-in | None | Low | Medium | Medium |

Aratta's differentiator is **self-healing** — when a provider changes
their API, the system diagnoses the break with a local model, researches
the fix via web search, and generates a confidence-scored patch. LiteLLM
is the closest alternative for multi-provider routing, but it relies on
manual updates when providers change. Aratta treats that as a solved
problem.

## Project Structure

```
src/aratta/
├── core/               SCRI type system — the language
├── providers/
│   ├── local/          Ollama, vLLM, llama.cpp (the foundation)
│   ├── anthropic/      Claude adapter
│   ├── openai/         GPT adapter
│   ├── google/         Gemini adapter
│   └── xai/            Grok adapter
├── resilience/
│   ├── health.py       Health monitor — error tracking, pattern detection
│   ├── heal_worker.py  Self-healing — diagnose → research → fix
│   ├── reload_manager.py  Hot-reload — backup, apply, verify, rollback
│   ├── circuit_breaker.py  Circuit breaker — fail fast, auto-recover
│   └── metrics.py      Counters, gauges, histograms
├── agents/             ReAct agent loop, executor, sandbox, permissions
├── tools/              Tool registry + provider format translation
├── config.py           Provider config, model aliases, healing config
├── server.py           FastAPI — all endpoints above
└── cli.py              CLI — init wizard, serve, health, models
```

## Development

```bash
git clone https://github.com/MAXAPIPULL00/aratta.git
cd aratta
python -m venv .venv
.venv/Scripts/activate      # Windows
# source .venv/bin/activate # Linux/macOS
pip install -e ".[dev]"
pytest                      # 82 tests
ruff check src/ tests/      # clean
```

## Docs

- [Architecture](docs/architecture.md) — sovereignty layer design
- [Providers](docs/providers.md) — supported providers + writing your own
- [Model Aliases](docs/model-aliases.md) — routing by capability
- [Agent Framework](docs/agents.md) — ReAct agents across providers

## Built With AI

This project was built with significant AI assistance (Claude). The
architecture, code, and documentation were developed through human-AI
collaboration. We're upfront about this — and we think it validates the
thesis: if AI tools are powerful enough to build their own sovereignty
layer, they should be powerful enough to use it.

## License

MIT — see [LICENSE](LICENSE).
