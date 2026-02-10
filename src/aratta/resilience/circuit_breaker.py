"""
Circuit breaker — prevents cascading failures when providers are down.

States:
    CLOSED    → Normal operation, requests pass through
    OPEN      → Provider is down, fail fast
    HALF_OPEN → Testing if provider recovered
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

logger = logging.getLogger("aratta.circuit_breaker")


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerState:
    provider: str
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    consecutive_failures: int = 0
    last_failure: datetime | None = None
    last_failure_error: str | None = None
    success_count: int = 0
    last_success: datetime | None = None
    opened_at: datetime | None = None
    last_state_change: datetime = field(default_factory=lambda: datetime.now(UTC))
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 60
    success_threshold: int = 3
    _half_open_successes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider, "state": self.state.value,
            "failure_count": self.failure_count, "consecutive_failures": self.consecutive_failures,
            "success_count": self.success_count,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
        }


class CircuitBreakerError(Exception):
    def __init__(self, provider: str, recovery_in_seconds: int):
        self.provider = provider
        self.recovery_in_seconds = recovery_in_seconds
        super().__init__(f"Circuit breaker OPEN for {provider}. Try again in {recovery_in_seconds}s")


class CircuitBreaker:
    """Circuit breaker for AI provider calls."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout_seconds: int = 60, success_threshold: int = 3):
        self.circuits: dict[str, CircuitBreakerState] = {}
        self.default_failure_threshold = failure_threshold
        self.default_recovery_timeout = recovery_timeout_seconds
        self.default_success_threshold = success_threshold

    def _get(self, provider: str) -> CircuitBreakerState:
        if provider not in self.circuits:
            self.circuits[provider] = CircuitBreakerState(
                provider=provider, failure_threshold=self.default_failure_threshold,
                recovery_timeout_seconds=self.default_recovery_timeout, success_threshold=self.default_success_threshold)
        return self.circuits[provider]

    def can_execute(self, provider: str) -> bool:
        s = self._get(provider)
        if s.state == CircuitState.CLOSED:
            return True
        if s.state == CircuitState.OPEN and s.opened_at:
            elapsed = (datetime.now(UTC) - s.opened_at).total_seconds()
            if elapsed >= s.recovery_timeout_seconds:
                self._transition(s, CircuitState.HALF_OPEN)
                return True
            return False
        return s.state == CircuitState.HALF_OPEN

    def get_recovery_time(self, provider: str) -> int:
        s = self._get(provider)
        if s.state != CircuitState.OPEN or not s.opened_at:
            return 0
        return max(0, int(s.recovery_timeout_seconds - (datetime.now(UTC) - s.opened_at).total_seconds()))

    def record_success(self, provider: str):
        s = self._get(provider)
        s.success_count += 1
        s.last_success = datetime.now(UTC)
        s.consecutive_failures = 0
        if s.state == CircuitState.HALF_OPEN:
            s._half_open_successes += 1
            if s._half_open_successes >= s.success_threshold:
                self._transition(s, CircuitState.CLOSED)
                s._half_open_successes = 0
        elif s.state == CircuitState.CLOSED:
            s.failure_count = 0

    def record_failure(self, provider: str, error: Exception) -> bool:
        s = self._get(provider)
        s.failure_count += 1
        s.consecutive_failures += 1
        s.last_failure = datetime.now(UTC)
        s.last_failure_error = str(error)
        should_heal = False
        if s.state == CircuitState.HALF_OPEN:
            self._transition(s, CircuitState.OPEN)
            s._half_open_successes = 0
        elif s.state == CircuitState.CLOSED and s.failure_count >= s.failure_threshold:
            self._transition(s, CircuitState.OPEN)
            should_heal = True
        return should_heal

    def _transition(self, s: CircuitBreakerState, new: CircuitState):
        s.state = new
        s.last_state_change = datetime.now(UTC)
        if new == CircuitState.OPEN:
            s.opened_at = datetime.now(UTC)
        elif new == CircuitState.CLOSED:
            s.opened_at = None
            s.failure_count = 0
        logger.info(f"Circuit {s.provider}: -> {new.value}")

    def force_open(self, provider: str):
        self._transition(self._get(provider), CircuitState.OPEN)

    def force_close(self, provider: str):
        self._transition(self._get(provider), CircuitState.CLOSED)

    def reset(self, provider: str):
        self.circuits.pop(provider, None)

    def get_all_states(self) -> dict[str, dict[str, Any]]:
        return {p: s.to_dict() for p, s in self.circuits.items()}

    def get_open_circuits(self) -> list[str]:
        return [p for p, s in self.circuits.items() if s.state == CircuitState.OPEN]
