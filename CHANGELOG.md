# Changelog

All notable changes to Aratta will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-02-10

### Added

- Initial open-source release.
- Sovereignty architecture — local models are the foundation, cloud providers are callable services.
- SCRI type system — `ChatRequest`, `ChatResponse`, `Tool`, `ToolCall`, `Usage`, `Lineage`. One language for all providers.
- `aratta init` interactive setup wizard with health checks.
- `aratta serve` — FastAPI server on `:8084` with chat, streaming, embeddings, models, health endpoints.
- Provider adapters: Local (Ollama/vLLM/llama.cpp), Anthropic, OpenAI, Google, xAI.
- Model alias system (`local`, `fast`, `reason`, `code`, `cheap`) with automatic provider routing.
- Tool registry — define tools in SCRI once, invoke through any provider.
- Circuit breaker with half-open recovery and per-provider failure tracking.
- Health monitor with pluggable heal callbacks.
- Resilience metrics (counters, gauges, histograms).
- Provider-agnostic ReAct agent framework with sandboxed code execution.
- Agent permission system (restricted/standard/elevated/system levels).
- AST-validated Python sandbox with import allowlisting.
- Configuration via `~/.aratta/config.toml` with environment variable overrides.
- Extended thinking support for Anthropic, OpenAI (reasoning_effort), and Google (thinkingLevel).
- Streaming support across all providers via SSE.
- 82 tests covering types, all provider adapters, circuit breaker, tool registry, and server endpoints.
