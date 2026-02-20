"""
Aratta API server — FastAPI application with provider routing and self-healing.

Core endpoints:
    POST /api/v1/chat          Unified chat completion
    POST /api/v1/chat/stream   Streaming chat completion
    POST /api/v1/embed         Embeddings
    GET  /api/v1/models        List available models
    GET  /api/v1/health        Provider health status
    GET  /health               Liveness probe

Self-healing endpoints:
    GET  /api/v1/healing/status          Self-healing system status
    POST /api/v1/healing/pause/{prov}    Pause healing for a provider
    POST /api/v1/healing/resume/{prov}   Resume healing
    GET  /api/v1/fixes/pending           Pending fixes awaiting approval
    POST /api/v1/fixes/{prov}/approve    Approve a pending fix
    POST /api/v1/fixes/{prov}/reject     Reject a pending fix
    GET  /api/v1/fixes/{prov}/history    Version history
    POST /api/v1/fixes/{prov}/rollback/{ver}  Rollback to version

Circuit breaker endpoints:
    POST /api/v1/circuit/{prov}/open     Force circuit open
    POST /api/v1/circuit/{prov}/close    Force circuit closed
    POST /api/v1/circuit/{prov}/reset    Reset circuit

Observability:
    GET  /api/v1/metrics       Resilience metrics
    GET  /api/v1/dashboard     Full system dashboard
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from aratta.config import ArattaConfig, load_config
from aratta.core.types import ChatRequest, Message, Role, Tool
from aratta.providers.base import BaseProvider, ProviderError, RateLimitError
from aratta.resilience.circuit_breaker import CircuitBreaker
from aratta.resilience.heal_worker import HealWorker
from aratta.resilience.health import HealthMonitor
from aratta.resilience.metrics import get_metrics
from aratta.resilience.reload_manager import ReloadManager

logger = logging.getLogger("aratta.server")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_providers: dict[str, BaseProvider] = {}
_provider_lock: threading.Lock = threading.Lock()
_config: ArattaConfig | None = None
_circuit_breaker: CircuitBreaker | None = None
_health_monitor: HealthMonitor | None = None
_reload_manager: ReloadManager | None = None
_heal_worker: HealWorker | None = None
_metrics = get_metrics()


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

def _get_provider(name: str) -> BaseProvider:
    """Get or lazily create a provider adapter (thread-safe)."""
    if name in _providers:
        return _providers[name]

    with _provider_lock:
        # Double-check after acquiring lock
        if name in _providers:
            return _providers[name]

        cfg = _config.get_provider(name) if _config else None
        if not cfg:
            raise HTTPException(404, f"Provider '{name}' not configured")

        if name == "anthropic":
            from aratta.providers.anthropic import AnthropicProvider
            _providers[name] = AnthropicProvider(cfg)
        elif name == "openai":
            from aratta.providers.openai import OpenAIProvider
            _providers[name] = OpenAIProvider(cfg)
        elif name == "google":
            from aratta.providers.google import GoogleProvider
            _providers[name] = GoogleProvider(cfg)
        elif name == "xai":
            from aratta.providers.xai import XAIProvider
            _providers[name] = XAIProvider(cfg)
        elif name in ("ollama", "vllm", "llamacpp"):
            from aratta.providers.local import LocalProvider
            _providers[name] = LocalProvider(cfg)
        else:
            raise HTTPException(400, f"No adapter for provider '{name}'")

        return _providers[name]


def _get_provider_with_fallback(model_str: str) -> tuple[BaseProvider, str, str]:
    """
    Resolve model → provider with fallback.

    Returns (provider_instance, provider_name, model_id).
    Tries the primary provider first; if unavailable and fallback is enabled,
    walks the priority list.
    """
    provider_name, model_id = _config.resolve_model(model_str)

    try:
        provider = _get_provider(provider_name)
        return provider, provider_name, model_id
    except Exception as e:
        logger.warning(f"Primary provider {provider_name} failed: {e}")
        if not _config.enable_fallback:
            raise

    # Fallback: try other available providers in priority order
    for fallback_name in _config.get_available_providers():
        if fallback_name == provider_name:
            continue
        try:
            fallback = _get_provider(fallback_name)
            fallback_cfg = _config.get_provider(fallback_name)
            fallback_model = fallback_cfg.default_model if fallback_cfg else model_id
            logger.warning(f"Provider {provider_name} unavailable, falling back to {fallback_name}")
            return fallback, fallback_name, fallback_model
        except Exception as e:
            logger.debug(f"Fallback provider {fallback_name} also unavailable: {e}")
            continue

    raise HTTPException(503, f"No available providers (primary: {provider_name})")


# ---------------------------------------------------------------------------
# Self-healing callback
# ---------------------------------------------------------------------------

async def _on_heal_request(provider: str, error: Any, recent_errors: list[Any]) -> None:
    """Called by HealthMonitor when healing threshold is reached."""
    if not _heal_worker or not _reload_manager:
        logger.warning(f"Heal requested for {provider} but heal worker not available")
        return

    logger.info(f"Self-healing triggered for {provider}")

    # Get adapter source for context
    adapter_source = _heal_worker.get_adapter_source(provider)

    # Ask the local model to diagnose
    recent = [{"type": e.error_type.value, "message": e.error_message[:200]} for e in recent_errors]
    proposal = await _heal_worker.diagnose(
        provider=provider,
        model=error.model,
        error_type=error.error_type.value,
        error_message=error.error_message,
        recent_errors=recent,
        adapter_source=adapter_source,
    )

    # Verification callback
    async def verify(prov: str) -> bool:
        try:
            p = _get_provider(prov)
            health = await p.health_check()
            return health.get("status") == "healthy"
        except Exception as e:
            logger.warning(f"Heal verification failed for {prov}: {e}")
            return False

    # Apply via ReloadManager
    result = await _reload_manager.apply_fix(provider, proposal, verify_callback=verify)

    if result.success:
        logger.info(f"Self-heal succeeded for {provider} v{result.version}")
        await _health_monitor.handle_heal_complete(provider, True)
    else:
        logger.warning(f"Self-heal result for {provider}: {result.message}")
        if "Queued" not in result.message:
            await _health_monitor.handle_heal_complete(provider, False)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _config, _circuit_breaker, _health_monitor, _reload_manager, _heal_worker

    _config = load_config()
    logger.info(f"Aratta starting — default provider: {_config.default_provider}")
    logger.info(f"Available providers: {_config.get_available_providers()}")

    # Circuit breaker
    if _config.circuit_breaker_enabled:
        _circuit_breaker = CircuitBreaker(
            failure_threshold=_config.circuit_failure_threshold,
            recovery_timeout_seconds=_config.circuit_recovery_seconds,
        )

    # Self-healing stack
    if _config.self_healing_enabled:
        _health_monitor = HealthMonitor()
        _health_monitor.ERROR_THRESHOLD = _config.error_threshold
        _health_monitor.WINDOW_SECONDS = _config.error_window_seconds
        _health_monitor.COOLDOWN_SECONDS = _config.heal_cooldown_seconds

        _reload_manager = ReloadManager(
            auto_apply=_config.auto_apply_fixes,
            auto_apply_threshold=0.85,
        )

        _heal_worker = HealWorker(
            get_provider_fn=_get_provider,
            resolve_model_fn=_config.resolve_model,
            heal_model=_config.heal_model,
            research_model=_config.research_model,
            research_web_search=True,
            cloud_providers=_config.get_available_providers(),
        )

        # Wire the heal callback: when HealthMonitor detects enough errors,
        # it fires this callback which runs the full diagnose→research→fix cycle
        _health_monitor.on_heal_request(_on_heal_request)

        logger.info(
            f"Self-healing enabled — heal model: {_config.heal_model}, "
            f"research model: grok (web search)"
        )

    logger.info(f"Aratta ready on {_config.host}:{_config.port}")
    yield

    # Shutdown
    for p in _providers.values():
        await p.close()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="Aratta",
        version="0.1.0",
        description="A sovereignty layer for AI",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    # ==================================================================
    # Health
    # ==================================================================

    @app.get("/health")
    async def liveness():
        return {"status": "ok"}

    @app.get("/api/v1/health")
    async def provider_health():
        results = {}
        for name in _config.get_available_providers():
            try:
                provider = _get_provider(name)
                results[name] = await provider.health_check()
            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}
        circuits = _circuit_breaker.get_all_states() if _circuit_breaker else {}
        return {"providers": results, "circuits": circuits}

    # ==================================================================
    # Models
    # ==================================================================

    @app.get("/api/v1/models")
    async def list_models():
        models = []
        for name in _config.get_available_providers():
            try:
                provider = _get_provider(name)
                models.extend([m.to_dict() for m in provider.get_models()])
            except Exception as e:
                logger.warning(f"Provider {name} models unavailable: {e}")
        return {"models": models, "aliases": _config.model_aliases}

    # ==================================================================
    # Chat
    # ==================================================================

    @app.post("/api/v1/chat")
    async def chat(request: Request):
        body = await request.json()
        model_str = body.get("model", "local")

        # Resolve with fallback
        provider, provider_name, model_id = _get_provider_with_fallback(model_str)

        # Circuit breaker check
        if _circuit_breaker and not _circuit_breaker.can_execute(provider_name):
            recovery = _circuit_breaker.get_recovery_time(provider_name)
            raise HTTPException(503, f"Provider {provider_name} circuit open. Retry in {recovery}s")

        messages = [Message(role=Role(m["role"]), content=m["content"]) for m in body.get("messages", [])]
        tools = [Tool(**t) for t in body.get("tools", [])] if body.get("tools") else None
        thinking = body.get("thinking", {}) if isinstance(body.get("thinking"), dict) else {}

        req = ChatRequest(
            messages=messages, model=model_id,
            temperature=body.get("temperature", 0.7),
            max_tokens=body.get("max_tokens"),
            tools=tools, tool_choice=body.get("tool_choice"),
            thinking_enabled=thinking.get("enabled", False),
            thinking_budget=thinking.get("budget_tokens", 10000),
        )

        try:
            response = await provider.chat(req)
            if _circuit_breaker:
                _circuit_breaker.record_success(provider_name)
            if _health_monitor:
                _health_monitor.record_success(provider_name)
            return response.to_dict()

        except RateLimitError as e:
            if _health_monitor:
                await _health_monitor.record_error(provider_name, model_id, e)
            raise HTTPException(429, str(e)) from None

        except ProviderError as e:
            if _circuit_breaker:
                _circuit_breaker.record_failure(provider_name, e)
            if _health_monitor:
                await _health_monitor.record_error(provider_name, model_id, e)
            raise HTTPException(e.status_code or 502, e.message) from None

        except Exception as e:
            logger.error(f"Chat request failed: {e}", exc_info=True)
            if _health_monitor:
                await _health_monitor.record_error(provider_name, model_id or "unknown", e)
            raise HTTPException(500, str(e)) from None

    @app.post("/api/v1/chat/stream")
    async def chat_stream(request: Request):
        body = await request.json()
        model_str = body.get("model", "local")
        provider, provider_name, model_id = _get_provider_with_fallback(model_str)
        messages = [Message(role=Role(m["role"]), content=m["content"]) for m in body.get("messages", [])]
        req = ChatRequest(messages=messages, model=model_id, stream=True, max_tokens=body.get("max_tokens"))

        async def generate():
            async for chunk in provider.chat_stream(req):
                yield chunk

        return EventSourceResponse(generate())

    # ==================================================================
    # Embeddings
    # ==================================================================

    @app.post("/api/v1/embed")
    async def embed(request: Request):
        body = await request.json()
        from aratta.core.types import EmbeddingRequest
        model_str = body.get("model", "embed")
        provider_name, model_id = _config.resolve_model(model_str)
        provider = _get_provider(provider_name)
        req = EmbeddingRequest(input=body["input"], model=model_id)
        response = await provider.embed(req)
        return response.to_dict()

    # ==================================================================
    # Self-healing status
    # ==================================================================

    @app.get("/api/v1/healing/status")
    async def healing_status():
        status: dict[str, Any] = {
            "enabled": _config.self_healing_enabled,
            "heal_model": _config.heal_model,
            "research_model": _config.research_model,
            "circuit_breaker_enabled": _config.circuit_breaker_enabled,
        }
        if _health_monitor:
            status["health"] = _health_monitor.get_summary()
        if _circuit_breaker:
            status["circuits"] = {
                "states": _circuit_breaker.get_all_states(),
                "open": _circuit_breaker.get_open_circuits(),
            }
        if _reload_manager:
            status["reload_manager"] = _reload_manager.get_status()
        return status

    @app.post("/api/v1/healing/pause/{provider}")
    async def pause_healing(provider: str):
        if not _health_monitor:
            raise HTTPException(503, "Self-healing not enabled")
        _health_monitor.healing_in_progress.discard(provider)
        # Add to a paused set (extend HealthMonitor if needed)
        return {"status": "paused", "provider": provider}

    @app.post("/api/v1/healing/resume/{provider}")
    async def resume_healing(provider: str):
        if not _health_monitor:
            raise HTTPException(503, "Self-healing not enabled")
        return {"status": "resumed", "provider": provider}

    # ==================================================================
    # Circuit breaker control
    # ==================================================================

    @app.post("/api/v1/circuit/{provider}/open")
    async def force_circuit_open(provider: str):
        if not _circuit_breaker:
            raise HTTPException(503, "Circuit breaker not enabled")
        _circuit_breaker.force_open(provider)
        return {"status": "opened", "provider": provider}

    @app.post("/api/v1/circuit/{provider}/close")
    async def force_circuit_close(provider: str):
        if not _circuit_breaker:
            raise HTTPException(503, "Circuit breaker not enabled")
        _circuit_breaker.force_close(provider)
        return {"status": "closed", "provider": provider}

    @app.post("/api/v1/circuit/{provider}/reset")
    async def reset_circuit(provider: str):
        if not _circuit_breaker:
            raise HTTPException(503, "Circuit breaker not enabled")
        _circuit_breaker.reset(provider)
        return {"status": "reset", "provider": provider}

    # ==================================================================
    # Fix management (human override)
    # ==================================================================

    @app.get("/api/v1/fixes/pending")
    async def pending_fixes():
        if not _reload_manager:
            raise HTTPException(503, "Reload manager not enabled")
        pending = _reload_manager.get_pending_fixes()
        return {
            "pending_fixes": [
                {
                    "provider": p,
                    "fix_type": f.get("fix_type"),
                    "confidence": f.get("confidence"),
                    "change_summary": f.get("change_summary"),
                    "analysis": f.get("analysis", ""),
                    "research_summary": f.get("research_summary", "")[:500],
                }
                for p, f in pending.items()
            ]
        }

    @app.post("/api/v1/fixes/{provider}/approve")
    async def approve_fix(provider: str):
        if not _reload_manager:
            raise HTTPException(503, "Reload manager not enabled")

        async def verify(prov: str) -> bool:
            try:
                p = _get_provider(prov)
                h = await p.health_check()
                return h.get("status") == "healthy"
            except Exception as e:
                logger.warning(f"Post-fix verification failed for {prov}: {e}")
                return False

        result = await _reload_manager.approve_fix(provider, verify_callback=verify)
        return {
            "success": result.success, "provider": result.provider,
            "version": result.version, "message": result.message,
            "code_changed": result.code_changed,
            "verification_passed": result.verification_passed,
        }

    @app.post("/api/v1/fixes/{provider}/reject")
    async def reject_fix(provider: str, reason: str = ""):
        if not _reload_manager:
            raise HTTPException(503, "Reload manager not enabled")
        if not _reload_manager.reject_fix(provider, reason):
            raise HTTPException(404, f"No pending fix for {provider}")
        return {"status": "rejected", "provider": provider, "reason": reason}

    @app.get("/api/v1/fixes/{provider}/history")
    async def fix_history(provider: str):
        if not _reload_manager:
            raise HTTPException(503, "Reload manager not enabled")
        return {"provider": provider, "versions": _reload_manager.get_version_history(provider)}

    @app.post("/api/v1/fixes/{provider}/rollback/{version}")
    async def rollback(provider: str, version: int):
        if not _reload_manager:
            raise HTTPException(503, "Reload manager not enabled")
        if not await _reload_manager.rollback_to_version(provider, version):
            raise HTTPException(400, f"Rollback failed for {provider}")
        return {"status": "rolled_back", "provider": provider, "version": version}

    # ==================================================================
    # Observability
    # ==================================================================

    @app.get("/api/v1/metrics")
    async def metrics():
        return _metrics.get_summary()

    @app.get("/api/v1/dashboard")
    async def dashboard():
        data: dict[str, Any] = {
            "system": {
                "service": "aratta", "version": "0.1.0",
                "self_healing_enabled": _config.self_healing_enabled,
                "heal_model": _config.heal_model,
                "research_model": _config.research_model,
                "circuit_breaker_enabled": _config.circuit_breaker_enabled,
            },
            "providers": [],
            "metrics": _metrics.get_summary(),
        }

        for name in _config.get_available_providers():
            prov_status: dict[str, Any] = {"name": name, "circuit_state": "closed"}
            if _circuit_breaker:
                states = _circuit_breaker.get_all_states()
                if name in states:
                    prov_status["circuit_state"] = states[name].get("state", "closed")
            if _health_monitor:
                prov_status["consecutive_failures"] = _health_monitor.consecutive_failures.get(name, 0)
                prov_status["healing"] = name in _health_monitor.healing_in_progress
            if _reload_manager:
                prov_status["pending_fix"] = name in _reload_manager.get_pending_fixes()
                prov_status["current_version"] = _reload_manager.current_version.get(name, 0)
            data["providers"].append(prov_status)

        if _health_monitor:
            data["health"] = _health_monitor.get_summary()

        return data

    return app
