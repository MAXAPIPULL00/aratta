"""
Health monitor — tracks provider errors and triggers healing callbacks.

Pluggable notification: register a callback via `on_heal_request` to
integrate with any alerting system.
"""

import hashlib
import logging
import re
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger("aratta.health")


class ErrorType(Enum):
    SCHEMA_MISMATCH = "schema_mismatch"
    UNKNOWN_FIELD = "unknown_field"
    DEPRECATED_FIELD = "deprecated_field"
    STREAMING_FORMAT = "streaming_format"
    TOOL_SCHEMA = "tool_schema"
    CONNECTION_ERROR = "connection_error"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTH_ERROR = "auth_error"
    UNKNOWN = "unknown"

    @classmethod
    def from_error(cls, error: Exception) -> "ErrorType":
        s = str(error).lower()
        if "timeout" in s:
            return cls.TIMEOUT
        if any(x in s for x in ("connection", "refused", "reset")):
            return cls.CONNECTION_ERROR
        if "429" in s or "rate" in s:
            return cls.RATE_LIMIT
        if "401" in s or "unauthorized" in s:
            return cls.AUTH_ERROR
        if any(x in s for x in ("schema", "validation")):
            return cls.SCHEMA_MISMATCH
        if "tool" in s or "function" in s:
            return cls.TOOL_SCHEMA
        return cls.UNKNOWN


HEALABLE = {ErrorType.SCHEMA_MISMATCH, ErrorType.UNKNOWN_FIELD, ErrorType.DEPRECATED_FIELD, ErrorType.STREAMING_FORMAT, ErrorType.TOOL_SCHEMA}
TRANSIENT = {ErrorType.RATE_LIMIT, ErrorType.CONNECTION_ERROR, ErrorType.TIMEOUT}


@dataclass
class AdapterError:
    provider: str
    model: str
    error_type: ErrorType
    error_message: str
    timestamp: datetime
    consecutive_failures: int

    @property
    def signature(self) -> str:
        normalized = re.sub(r'\d+', 'N', self.error_message.lower())
        return hashlib.sha256(f"{self.provider}:{self.error_type.value}:{normalized}".encode()).hexdigest()[:16]


# Type for heal-request callbacks
HealCallback = Callable[[str, "AdapterError", list["AdapterError"]], Awaitable[None]]


class HealthMonitor:
    """Monitors provider health and fires pluggable heal callbacks."""

    ERROR_THRESHOLD = 3
    WINDOW_SECONDS = 300
    COOLDOWN_SECONDS = 600
    MAX_HISTORY = 100

    def __init__(self):
        self.error_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.MAX_HISTORY))
        self.consecutive_failures: dict[str, int] = defaultdict(int)
        self.healing_in_progress: set[str] = set()
        self.last_heal_request: dict[str, datetime] = {}
        self._heal_callbacks: list[HealCallback] = []

    def on_heal_request(self, callback: HealCallback):
        """Register a callback invoked when healing is needed."""
        self._heal_callbacks.append(callback)

    async def record_error(self, provider: str, model: str, error: Exception) -> bool:
        self.consecutive_failures[provider] += 1
        etype = ErrorType.from_error(error)
        entry = AdapterError(provider, model, etype, str(error), datetime.now(UTC), self.consecutive_failures[provider])
        self.error_history[provider].append(entry)

        if self._should_heal(entry):
            await self._fire_heal(entry)
            return True
        return False

    def record_success(self, provider: str):
        self.consecutive_failures[provider] = 0

    def _should_heal(self, err: AdapterError) -> bool:
        p = err.provider
        if p in self.healing_in_progress:
            return False
        last = self.last_heal_request.get(p)
        if last and (datetime.now(UTC) - last).total_seconds() < self.COOLDOWN_SECONDS:
            return False
        if err.error_type in TRANSIENT:
            return False
        recent = [e for e in self.error_history[p] if e.timestamp > datetime.now(UTC) - timedelta(seconds=self.WINDOW_SECONDS)]
        return len(recent) >= self.ERROR_THRESHOLD

    async def _fire_heal(self, err: AdapterError):
        self.healing_in_progress.add(err.provider)
        self.last_heal_request[err.provider] = datetime.now(UTC)
        recent = list(self.error_history[err.provider])[-5:]
        for cb in self._heal_callbacks:
            try:
                await cb(err.provider, err, recent)
            except Exception as e:
                logger.error(f"Heal callback error: {e}")

    async def handle_heal_complete(self, provider: str, success: bool):
        self.healing_in_progress.discard(provider)
        if success:
            self.error_history[provider].clear()
            self.consecutive_failures[provider] = 0

    def get_summary(self) -> dict[str, Any]:
        cutoff = datetime.now(UTC) - timedelta(seconds=self.WINDOW_SECONDS)
        return {
            "providers": {
                p: {
                    "recent_errors": len([e for e in self.error_history[p] if e.timestamp > cutoff]),
                    "consecutive_failures": self.consecutive_failures[p],
                    "healing": p in self.healing_in_progress,
                }
                for p in self.error_history
            }
        }
