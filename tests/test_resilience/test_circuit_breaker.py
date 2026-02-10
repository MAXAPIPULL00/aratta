"""Tests for the circuit breaker pattern."""

from aratta.resilience.circuit_breaker import CircuitBreaker, CircuitState


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker()
        assert cb.can_execute("test") is True

    def test_stays_closed_under_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure("test", Exception("err"))
        cb.record_failure("test", Exception("err"))
        assert cb.can_execute("test") is True

    def test_opens_at_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure("test", Exception("err"))
        assert cb.can_execute("test") is False

    def test_success_resets_failures(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure("test", Exception("err"))
        cb.record_failure("test", Exception("err"))
        cb.record_success("test")
        # After success, failure count resets — need 3 more to open
        cb.record_failure("test", Exception("err"))
        cb.record_failure("test", Exception("err"))
        assert cb.can_execute("test") is True

    def test_record_failure_returns_should_heal(self):
        cb = CircuitBreaker(failure_threshold=2)
        assert cb.record_failure("test", Exception("err")) is False
        assert cb.record_failure("test", Exception("err")) is True

    def test_recovery_time(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_seconds=60)
        cb.record_failure("test", Exception("err"))
        recovery = cb.get_recovery_time("test")
        assert 55 <= recovery <= 60

    def test_half_open_success_closes(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_seconds=0, success_threshold=2)
        cb.record_failure("test", Exception("err"))
        # Recovery timeout is 0, so it should transition to half-open
        assert cb.can_execute("test") is True  # triggers half-open
        cb.record_success("test")
        cb.record_success("test")
        # Should be closed now
        state = cb.circuits["test"]
        assert state.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_seconds=0)
        cb.record_failure("test", Exception("err"))
        cb.can_execute("test")  # triggers half-open
        cb.record_failure("test", Exception("err again"))
        state = cb.circuits["test"]
        assert state.state == CircuitState.OPEN

    def test_force_open(self):
        cb = CircuitBreaker()
        cb.force_open("test")
        assert cb.can_execute("test") is False

    def test_force_close(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure("test", Exception("err"))
        cb.force_close("test")
        assert cb.can_execute("test") is True

    def test_reset(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure("test", Exception("err"))
        cb.reset("test")
        # reset removes the circuit; can_execute lazily re-creates it as closed
        assert cb.can_execute("test") is True
        assert cb.circuits["test"].state == CircuitState.CLOSED
        assert cb.circuits["test"].failure_count == 0

    def test_get_all_states(self):
        cb = CircuitBreaker()
        cb.record_success("a")
        cb.record_failure("b", Exception("err"))
        states = cb.get_all_states()
        assert "a" in states
        assert "b" in states
        assert states["a"]["state"] == "closed"

    def test_independent_providers(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure("a", Exception("err"))
        cb.record_failure("a", Exception("err"))
        assert cb.can_execute("a") is False
        assert cb.can_execute("b") is True
