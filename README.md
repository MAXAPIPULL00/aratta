# Aratta

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-82%20passing-green.svg)](tests/)

> **What it is:** Aratta is an adapter layer that normalises all major AI provider APIs (Anthropic, OpenAI, Google, xAI) into one standard interface called SCRI. You write your code once against SCRI. Aratta handles every provider's different format behind the scenes.

---

## The Problem It Solves

If you are learning fast and shipping a lot of projects, you end up talking to four different AI APIs that all do the same things completely differently.

| What you want to do | Anthropic | OpenAI | Google | xAI |
|---|---|---|---|---|
| Pass a tool definition | `input_schema` | `function.parameters` | `functionDeclarations` | `function.parameters` |
| Check why it stopped | `stop_reason` | `finish_reason` | `finishReason` | `finish_reason` |
| Count tokens used | `usage.input_tokens` | `usage.prompt_tokens` | `usageMetadata.promptTokenCount` | `usage.prompt_tokens` |
| Handle streaming | `content_block_delta` | `choices[0].delta` | `candidates[0]` | OpenAI-compat |
| Pass auth | `x-api-key` header | `Bearer` token | `x-goog-api-key` header | `Bearer` token |

Multiply that across four providers and everything you build has four different code paths for the same logic. Aratta collapses all of that into one.

## How It Works

Aratta runs as a local server. Your code talks to Aratta using **SCRI** (one standard format). Aratta translates to and from each provider's native format. The translation is completely invisible to your application.

```
┌─────────────────┐
│   Your App      │  ← speaks SCRI, nothing else
└────────┬────────┘
         │
    ┌────┴────┐
    │  Aratta │  ← translates to/from every provider
    └────┬────┘
    ┌────┴──────────────────────┐
    │                           │
  Ollama    Claude    GPT    Gemini    Grok
 (local)                   (cloud, optional)
```

**Local is the default.** Your Ollama instance runs everything. Cloud providers are there when you need a specific capability — same interface, zero code changes.

## Quick Start

```bash
git clone https://github.com/MAXAPIPULL00/aratta.git
cd aratta
pip install -e .
aratta init    # walks you through which providers to enable + API keys
aratta serve   # starts on localhost:8084
```

Requires Python 3.11+. PyPI package coming soon.

## Usage

Once Aratta is running, you call it the same way regardless of which provider handles the request:

```python
import httpx

# Use your local model (Ollama — the default)
resp = httpx.post("http://localhost:8084/api/v1/chat", json={
    "messages": [{"role": "user", "content": "Hello"}],
    "model": "local",
})

# Use Claude — same call, same response shape
resp = httpx.post("http://localhost:8084/api/v1/chat", json={
    "messages": [{"role": "user", "content": "Analyse this contract"}],
    "model": "reason",
})

# Use GPT — still the same
resp = httpx.post("http://localhost:8084/api/v1/chat", json={
    "messages": [{"role": "user", "content": "Generate test cases"}],
    "model": "gpt",
})

# The response is always SCRI — same fields, same types, every time
```

You can also use the Python types directly:

```python
from aratta.core.types import ChatRequest, Message, Role

request = ChatRequest(
    messages=[Message(role=Role.USER, content="Explain quantum computing")],
    model="local",   # swap to "reason", "gpt", "gemini" — code stays the same
)
```

## SCRI — The Standard Format

SCRI is the type system Aratta uses as its single language. Every provider adapter translates to and from SCRI. Provider-specific structures never reach your application code.

The core types are: `Message`, `ChatRequest`, `ChatResponse`, `ToolCall`, `Usage`, `FinishReason`.

Every response comes back with the same fields regardless of which provider answered it — `content`, `model`, `provider`, `finish_reason`, `usage`, and a `lineage` block that tells you exactly where the response came from.

## Model Aliases

Route by what you need, not by provider model IDs:

| Alias | Capability | Default |
|---|---|---|
| `local` | General, runs on your hardware | Ollama llama3.1:8b |
| `sovereign` | Same as local | Ollama llama3.1:8b |
| `fast` | Low latency | Gemini Flash |
| `reason` | Deep reasoning | Claude Opus |
| `code` | Code generation | Claude Sonnet |
| `cheap` | Minimal cost | Gemini Flash Lite |
| `gpt` | OpenAI | GPT latest |
| `gpt-mini` | OpenAI, smaller | GPT Mini |
| `gemini-pro` | Google | Gemini Pro |
| `image` | Multimodal | Gemini Pro Image |
| `grok` | xAI | Grok Fast |
| `grok-reason` | xAI reasoning | Grok Reasoning |
| `embed` | Embeddings | OpenAI text-embedding-3-large |

All aliases are configurable in `~/.aratta/config.toml`. Point `reason` at a local 70B if you want. Point `fast` at GPT. Your routing, your rules.

Full reference: [docs/model-aliases.md](docs/model-aliases.md)

## Supported Providers

| Provider | Chat | Streaming | Tools | Vision | Thinking | Embeddings |
|---|---|---|---|---|---|---|
| Ollama (local) | ✓ | ✓ | ✓ | ✓ | — | — |
| vLLM (local) | ✓ | ✓ | ✓ | — | — | — |
| llama.cpp (local) | ✓ | ✓ | — | — | — | — |
| Anthropic (Claude) | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| OpenAI (GPT) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Google (Gemini) | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| xAI (Grok) | ✓ | ✓ | ✓ | — | ✓ | — |

## API

### Core Endpoints

| Endpoint | Method | What it does |
|---|---|---|
| `/health` | GET | Liveness check |
| `/api/v1/chat` | POST | Chat completion (any provider, SCRI in and out) |
| `/api/v1/chat/stream` | POST | Streaming chat via SSE |
| `/api/v1/embed` | POST | Embeddings |
| `/api/v1/models` | GET | List available models and aliases |
| `/api/v1/health` | GET | Per-provider health and circuit breaker state |

### Self-Healing Endpoints

| Endpoint | Method | What it does |
|---|---|---|
| `/api/v1/healing/status` | GET | Current healing system status |
| `/api/v1/healing/pause/{provider}` | POST | Pause healing for a provider |
| `/api/v1/healing/resume/{provider}` | POST | Resume healing |
| `/api/v1/fixes/pending` | GET | Fixes waiting for human approval |
| `/api/v1/fixes/{provider}/approve` | POST | Approve and apply a fix |
| `/api/v1/fixes/{provider}/reject` | POST | Reject a fix |
| `/api/v1/fixes/{provider}/history` | GET | Adapter version history |
| `/api/v1/fixes/{provider}/rollback/{version}` | POST | Roll back to a previous version |

### Circuit Breaker Endpoints

| Endpoint | Method | What it does |
|---|---|---|
| `/api/v1/circuit/{provider}/open` | POST | Force open (maintenance mode) |
| `/api/v1/circuit/{provider}/close` | POST | Force close after manual fix |
| `/api/v1/circuit/{provider}/reset` | POST | Reset to initial state |

### Observability

| Endpoint | Method | What it does |
|---|---|---|
| `/api/v1/metrics` | GET | Error counts, circuit opens, heal requests |
| `/api/v1/dashboard` | GET | Full system view — providers, health, healing, metrics |

## Resilience Features

These are built in and on by default. You don't need to configure anything for basic use.

**Provider fallback.** If a provider is unavailable, Aratta walks the priority list and routes to the next one. Transparent to your application. Local providers are highest priority (0). Cloud providers rank by your config.

**Circuit breakers.** If a cloud provider starts failing, the circuit opens and requests fail fast instead of timing out. Half-open probes test recovery automatically. When it comes back, the circuit closes.

**Health monitoring.** Errors are tracked continuously and classified by type — schema mismatches, unknown fields, deprecated fields, tool schema changes, streaming format changes. Transient errors (rate limits, timeouts) are tracked but don't trigger escalation. Structural errors do.

**Self-healing.** When a cloud provider changes their API format, Aratta detects the structural error, runs a three-phase heal cycle, and generates a fix:

1. **Diagnose** — your local model classifies the failure and formulates search queries
2. **Research** — Grok searches the web for current docs (with fallback through xAI → OpenAI → Google → Anthropic)
3. **Fix** — your local model generates a patch with a confidence score

High-confidence fixes can auto-apply. Low-confidence fixes go to a human approval queue (the `/api/v1/fixes` endpoints). Every fix is backed up, verified after application, and auto-rolled back if the health check fails.

This is configurable in `~/.aratta/config.toml` under `[healing]`.

## Agent Framework

Aratta includes a provider-agnostic ReAct agent loop:

```python
from aratta.agents import Agent, AgentConfig, AgentContext

ctx = AgentContext.create(aratta_client)
agent = Agent(config=AgentConfig(model="local"), context=ctx)
result = await agent.run("Research this topic and summarise")
```

Switch the model alias and the same agent runs on a different provider. Includes sandboxed code execution (AST-validated, subprocess-isolated), a permission tier system (restricted / standard / elevated / system), and an audited tool executor.

Full docs: [docs/agents.md](docs/agents.md)

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
fast   = "ollama:llama3.1:8b"

[behaviour]
default_provider = "ollama"
prefer_local     = true
enable_fallback  = true

[healing]
enabled           = true
auto_apply        = false
heal_model        = "local"
error_threshold   = 3
cooldown_seconds  = 600
```

Environment variables override TOML values. API keys are always read from environment:

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Only if using Claude | `sk-ant-...` |
| `OPENAI_API_KEY` | Only if using GPT | `sk-...` |
| `GOOGLE_API_KEY` | Only if using Gemini | Google AI key |
| `XAI_API_KEY` | Only if using Grok | `xai-...` |
| `ARATTA_PORT` | No | Default: `8084` |
| `ARATTA_HOST` | No | Default: `0.0.0.0` |
| `LOG_LEVEL` | No | Default: `INFO` |

Running local-only? You don't need a `.env` file at all.

## Project Structure

```
src/aratta/
├── core/              SCRI type system — Message, ChatRequest, ChatResponse, ToolCall, Usage
├── providers/
│   ├── local/         Ollama, vLLM, llama.cpp
│   ├── anthropic/     Claude adapter
│   ├── openai/        GPT adapter
│   ├── google/        Gemini adapter
│   └── xai/           Grok adapter
├── resilience/
│   ├── health.py      Error tracking and pattern classification
│   ├── heal_worker.py Diagnose → research → fix cycle
│   ├── reload_manager.py Backup, apply, verify, rollback
│   ├── circuit_breaker.py Fail fast and auto-recover
│   └── metrics.py     Counters, gauges, histograms
├── agents/            ReAct agent loop, sandboxed executor, permission tiers
├── tools/             Tool registry and per-provider format translation
├── config.py          Provider config, model aliases, healing config
├── server.py          FastAPI — all endpoints
└── cli.py             CLI — init wizard, serve, health, models
```

## Testing

```bash
pytest                      # all 82 tests
pytest --cov=aratta         # with coverage
pytest tests/test_resilience/  # specific suite
```

| Suite | What it covers |
|---|---|
| `test_core/` | SCRI types, serialisation, validation |
| `test_providers/` | All 5 adapters — message and tool conversion for each provider |
| `test_resilience/` | Circuit breaker state transitions, thresholds, half-open recovery |
| `test_agents/` | ReAct loop iterations, tool execution, exit conditions |
| `test_tools/` | Tool registry, format export, duplicate handling |
| `test_server.py` | API endpoints, health checks, error responses |

## Development

```bash
git clone https://github.com/MAXAPIPULL00/aratta.git
cd aratta
python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -e ".[dev]"
pytest
ruff check src/ tests/
```

## How Aratta Compares

| Feature | Aratta | LiteLLM | OpenRouter | Portkey |
|---|---|---|---|---|
| Multi-provider routing | Yes | Yes | Yes | Yes |
| Local-first (Ollama/vLLM) | Default | Supported | No | No |
| Standard type system | SCRI | Partial | No | No |
| Self-healing adapters | Yes | No | No | No |
| Circuit breakers | Yes | No | No | Yes |
| Per-provider health monitoring | Yes | Basic | No | Yes |
| Automatic provider fallback | Yes | Configurable | Automatic | Configurable |
| Built-in agent framework | Yes | No | No | No |
| Human approval queue for fixes | Yes | No | No | No |
| Adapter hot-reload | Yes | No | No | No |
| Self-hosted | Yes | Yes | No | Cloud only |
| Vendor lock-in | None | Low | Medium | Medium |

The main differentiator is the unified type system and self-healing. LiteLLM is the closest alternative for routing but relies on manual updates when providers change their APIs.

## Docs

- [Architecture](docs/architecture.md)
- [Providers](docs/providers.md)
- [Model Aliases](docs/model-aliases.md)
- [Agent Framework](docs/agents.md)

## License

MIT — see [LICENSE](LICENSE).
