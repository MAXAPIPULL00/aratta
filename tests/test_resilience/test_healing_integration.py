"""Tests for health monitor and heal worker integration."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from aratta.resilience.health import HealthMonitor


class TestHealthMonitor:
    @pytest.fixture
    def monitor(self):
        m = HealthMonitor()
        # Lower threshold for testing
        m.ERROR_THRESHOLD = 2
        m.COOLDOWN_SECONDS = 0
        return m

    async def test_single_error_does_not_fire_callback(self, monitor):
        cb = AsyncMock()
        monitor.on_heal_request(cb)

        await monitor.record_error("test_provider", "test-model", Exception("Something broke"))

        cb.assert_not_called()

    async def test_threshold_errors_fires_callback(self, monitor):
        cb = AsyncMock()
        monitor.on_heal_request(cb)

        for _ in range(2):
            await monitor.record_error("test_provider", "test-model", Exception("Something broke"))

        cb.assert_called_once()
        args = cb.call_args[0]
        assert args[0] == "test_provider"

    async def test_transient_errors_ignored(self, monitor):
        cb = AsyncMock()
        monitor.on_heal_request(cb)

        from aratta.providers.base import RateLimitError

        for _ in range(5):
            await monitor.record_error(
                "test_provider", "test-model",
                RateLimitError("Rate limited", "test_provider", 429)
            )

        cb.assert_not_called()

    async def test_heal_complete_resets_errors(self, monitor):
        cb = AsyncMock()
        monitor.on_heal_request(cb)

        for _ in range(2):
            await monitor.record_error("test_provider", "test-model", Exception("Broke"))

        assert cb.call_count == 1

        await monitor.handle_heal_complete("test_provider", success=True)

        # After successful heal, errors should be cleared
        cb.reset_mock()
        await monitor.record_error("test_provider", "test-model", Exception("Broke again"))
        cb.assert_not_called()  # Only 1 error, threshold is 2

    async def test_different_providers_tracked_separately(self, monitor):
        cb = AsyncMock()
        monitor.on_heal_request(cb)

        await monitor.record_error("provider_a", "model-a", Exception("Broke"))
        await monitor.record_error("provider_b", "model-b", Exception("Broke"))

        cb.assert_not_called()  # Neither hit threshold of 2


class TestHealWorkerCategorization:
    async def test_auth_error_categorized(self):
        from aratta.resilience.heal_worker import HealWorker

        worker = HealWorker(
            get_provider_fn=MagicMock(side_effect=Exception("auth key invalid")),
            resolve_model_fn=MagicMock(return_value=("ollama", "llama3.1:8b")),
        )

        result = await worker.diagnose(
            provider="test",
            model="test-model",
            error_type="auth_error",
            error_message="auth key invalid",
        )

        assert result["fix_type"] in ("auth_error", "transient_error", "heal_error", "no_fix_needed")
        assert result["confidence"] >= 0.0

    async def test_transient_error_detected(self):
        from aratta.resilience.heal_worker import HealWorker

        worker = HealWorker(
            get_provider_fn=MagicMock(side_effect=Exception("connection timeout")),
            resolve_model_fn=MagicMock(return_value=("ollama", "llama3.1:8b")),
        )

        result = await worker.diagnose(
            provider="test",
            model="test-model",
            error_type="timeout",
            error_message="connection timeout",
        )

        assert result["fix_type"] in ("transient_error", "heal_error", "no_fix_needed")
