"""
Aratta configuration.

Provider settings, model aliases, and runtime options.
Reads from ~/.aratta/config.toml with environment variable overrides.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ARATTA_HOME = Path(os.getenv("ARATTA_HOME", Path.home() / ".aratta"))
CONFIG_PATH = ARATTA_HOME / "config.toml"
ENV_PATH = ARATTA_HOME / ".env"


# ---------------------------------------------------------------------------
# Provider priority — lower number = preferred
# ---------------------------------------------------------------------------

class ProviderPriority(Enum):
    LOCAL = 0       # Sovereign — never leaves your machine
    PRIMARY = 1     # First cloud choice
    SECONDARY = 2
    TERTIARY = 3
    FALLBACK = 4


# ---------------------------------------------------------------------------
# Single provider config
# ---------------------------------------------------------------------------

@dataclass
class ProviderConfig:
    """Configuration for one AI provider."""

    name: str
    base_url: str
    api_key_env: str | None
    default_model: str
    priority: int
    timeout: float = 30.0
    max_retries: int = 3
    enabled: bool = True

    @property
    def api_key(self) -> str | None:
        if self.api_key_env:
            return os.getenv(self.api_key_env)
        return None

    @property
    def is_available(self) -> bool:
        if not self.enabled:
            return False
        if self.api_key_env:
            return bool(self.api_key)
        return True  # local providers don't need keys


# ---------------------------------------------------------------------------
# Default provider definitions
# ---------------------------------------------------------------------------

DEFAULT_CLOUD_PROVIDERS: dict[str, ProviderConfig] = {
    "anthropic": ProviderConfig(
        name="anthropic",
        base_url="https://api.anthropic.com",
        api_key_env="ANTHROPIC_API_KEY",
        default_model="claude-sonnet-4-5-20250929",
        priority=ProviderPriority.PRIMARY.value,
    ),
    "openai": ProviderConfig(
        name="openai",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4.1",
        priority=ProviderPriority.SECONDARY.value,
    ),
    "google": ProviderConfig(
        name="google",
        base_url="https://generativelanguage.googleapis.com",
        api_key_env="GOOGLE_API_KEY",
        default_model="gemini-3-flash-preview",
        priority=ProviderPriority.TERTIARY.value,
    ),
    "xai": ProviderConfig(
        name="xai",
        base_url="https://api.x.ai/v1",
        api_key_env="XAI_API_KEY",
        default_model="grok-4-1-fast",
        priority=ProviderPriority.FALLBACK.value,
    ),
}

DEFAULT_LOCAL_PROVIDERS: dict[str, ProviderConfig] = {
    "ollama": ProviderConfig(
        name="ollama",
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        api_key_env=None,
        default_model="llama3.1:8b",
        priority=ProviderPriority.LOCAL.value,
    ),
    "vllm": ProviderConfig(
        name="vllm",
        base_url=os.getenv("VLLM_URL", "http://localhost:8000"),
        api_key_env=None,
        default_model="meta-llama/Llama-3.1-8B-Instruct",
        priority=ProviderPriority.LOCAL.value,
        enabled=False,  # opt-in
    ),
    "llamacpp": ProviderConfig(
        name="llamacpp",
        base_url=os.getenv("LLAMACPP_URL", "http://localhost:8080"),
        api_key_env=None,
        default_model="default",
        priority=ProviderPriority.LOCAL.value,
        enabled=False,  # opt-in
    ),
}


# ---------------------------------------------------------------------------
# Model aliases — human-friendly names → provider:model
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ALIASES: dict[str, str] = {
    # ── Use-case aliases (the ones people actually type) ──────────────
    "fast": "google:gemini-3-flash-preview",
    "reason": "anthropic:claude-opus-4-5-20251101",
    "code": "anthropic:claude-sonnet-4-5-20250929",
    "cheap": "google:gemini-2.5-flash-lite",
    "local": "ollama:llama3.1:8b",
    "sovereign": "ollama:llama3.1:8b",

    # ── Anthropic ─────────────────────────────────────────────────────
    "opus": "anthropic:claude-opus-4-5-20251101",
    "sonnet": "anthropic:claude-sonnet-4-5-20250929",
    "haiku": "anthropic:claude-haiku-4-5-20251001",

    # ── OpenAI ────────────────────────────────────────────────────────
    "gpt": "openai:gpt-4.1",
    "gpt-mini": "openai:gpt-4.1-mini",
    "o3": "openai:o3",

    # ── Google ────────────────────────────────────────────────────────
    "gemini": "google:gemini-3-flash-preview",
    "gemini-pro": "google:gemini-3-pro-preview",

    # ── xAI ───────────────────────────────────────────────────────────
    "grok": "xai:grok-4-1-fast",

    # ── Embeddings ────────────────────────────────────────────────────
    "embed": "openai:text-embedding-3-large",
    "embed-small": "openai:text-embedding-3-small",
}


# ---------------------------------------------------------------------------
# Main config
# ---------------------------------------------------------------------------

@dataclass
class ArattaConfig:
    """Top-level Aratta configuration."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8084

    # Providers
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    local_providers: dict[str, ProviderConfig] = field(default_factory=dict)

    # Aliases
    model_aliases: dict[str, str] = field(default_factory=dict)

    # Behaviour
    default_provider: str = "ollama"  # local first
    enable_fallback: bool = True
    prefer_local: bool = True  # route to local when possible

    # Resilience
    circuit_breaker_enabled: bool = True
    circuit_failure_threshold: int = 5
    circuit_recovery_seconds: int = 60

    # Self-healing
    self_healing_enabled: bool = True
    auto_apply_fixes: bool = False
    heal_model: str = "local"  # model alias used for self-diagnosis
    error_threshold: int = 3
    error_window_seconds: int = 300
    heal_cooldown_seconds: int = 600

    # Streaming
    stream_timeout: float = 60.0

    # Agent defaults
    agent_default_model: str = "local"
    agent_max_iterations: int = 25
    agent_timeout_seconds: float = 300.0

    # ── Resolver ──────────────────────────────────────────────────────

    def resolve_model(self, alias: str) -> tuple[str, str]:
        """
        Resolve a model string to (provider, model_id).

        Accepts:
            "fast"                      → alias lookup
            "anthropic:claude-opus-..."  → explicit provider:model
            "claude-opus-..."            → infer provider from model name
            "llama3.1:8b"               → infer local provider
        """
        # 1. Check aliases
        if alias in self.model_aliases:
            resolved = self.model_aliases[alias]
            if ":" in resolved:
                parts = resolved.split(":", 1)
                return (parts[0], parts[1])
            return (self.default_provider, resolved)

        # 2. Explicit provider:model
        if ":" in alias:
            parts = alias.split(":", 1)
            return (parts[0], parts[1])

        # 3. Infer provider from model name
        lower = alias.lower()
        if "claude" in lower:
            return ("anthropic", alias)
        if any(x in lower for x in ("gpt", "o1", "o3", "o4", "codex")):
            return ("openai", alias)
        if "gemini" in lower:
            return ("google", alias)
        if "grok" in lower:
            return ("xai", alias)
        if any(x in lower for x in ("llama", "mistral", "qwen", "phi", "deepseek")):
            return ("ollama", alias)

        # 4. Fall back to default
        return (self.default_provider, alias)

    def get_provider(self, name: str) -> ProviderConfig | None:
        if name in self.providers:
            return self.providers[name]
        if name in self.local_providers:
            return self.local_providers[name]
        return None

    def get_available_providers(self) -> list[str]:
        available = []
        for name, cfg in {**self.local_providers, **self.providers}.items():
            if cfg.is_available:
                available.append(name)
        return sorted(available, key=lambda n: (self.get_provider(n) or ProviderConfig(
            name=n, base_url="", api_key_env=None, default_model="", priority=99
        )).priority)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _apply_toml(config: ArattaConfig, data: dict[str, Any]) -> None:
    """Overlay TOML data onto an ArattaConfig."""
    server = data.get("server", {})
    if "host" in server:
        config.host = server["host"]
    if "port" in server:
        config.port = int(server["port"])

    # Provider overrides
    for section_key, target in [("providers", config.providers), ("local", config.local_providers)]:
        for name, overrides in data.get(section_key, {}).items():
            if name in target:
                cfg = target[name]
                if "base_url" in overrides:
                    cfg.base_url = overrides["base_url"]
                if "default_model" in overrides:
                    cfg.default_model = overrides["default_model"]
                if "enabled" in overrides:
                    cfg.enabled = overrides["enabled"]
                if "timeout" in overrides:
                    cfg.timeout = float(overrides["timeout"])
                if "priority" in overrides:
                    cfg.priority = int(overrides["priority"])

    # Alias overrides
    for alias, target_model in data.get("aliases", {}).items():
        config.model_aliases[alias] = target_model

    # Behaviour
    behaviour = data.get("behaviour", data.get("behavior", {}))
    if "default_provider" in behaviour:
        config.default_provider = behaviour["default_provider"]
    if "prefer_local" in behaviour:
        config.prefer_local = behaviour["prefer_local"]
    if "enable_fallback" in behaviour:
        config.enable_fallback = behaviour["enable_fallback"]

    # Self-healing
    healing = data.get("healing", data.get("self_healing", {}))
    if "enabled" in healing:
        config.self_healing_enabled = healing["enabled"]
    if "auto_apply" in healing:
        config.auto_apply_fixes = healing["auto_apply"]
    if "heal_model" in healing:
        config.heal_model = healing["heal_model"]
    if "error_threshold" in healing:
        config.error_threshold = int(healing["error_threshold"])
    if "cooldown_seconds" in healing:
        config.heal_cooldown_seconds = int(healing["cooldown_seconds"])


def load_config(config_path: Path | None = None) -> ArattaConfig:
    """
    Build config from defaults → TOML file → environment variables.

    Precedence (highest wins):
        1. Environment variables
        2. ~/.aratta/config.toml
        3. Built-in defaults
    """
    config = ArattaConfig(
        providers={k: ProviderConfig(**{f.name: getattr(v, f.name) for f in v.__dataclass_fields__.values()})
                   for k, v in DEFAULT_CLOUD_PROVIDERS.items()},
        local_providers={k: ProviderConfig(**{f.name: getattr(v, f.name) for f in v.__dataclass_fields__.values()})
                         for k, v in DEFAULT_LOCAL_PROVIDERS.items()},
        model_aliases=dict(DEFAULT_MODEL_ALIASES),
    )

    # Load TOML if it exists
    path = config_path or CONFIG_PATH
    if path.exists():
        with open(path, "rb") as f:
            toml_data = tomllib.load(f)
        _apply_toml(config, toml_data)

    # Env overrides
    if os.getenv("ARATTA_HOST"):
        config.host = os.getenv("ARATTA_HOST")  # type: ignore[assignment]
    if os.getenv("ARATTA_PORT"):
        config.port = int(os.getenv("ARATTA_PORT"))  # type: ignore[arg-type]

    return config


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_config: ArattaConfig | None = None


def get_config() -> ArattaConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config
