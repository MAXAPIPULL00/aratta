"""Tests for thread-safe provider creation."""

import threading

import pytest


class TestProviderConcurrency:
    def test_get_provider_returns_same_instance(self):
        """_get_provider should return the same provider instance for the same name."""
        from aratta.config import ArattaConfig, ProviderConfig

        # Set up a minimal config
        import aratta.server as server_mod

        original_config = server_mod._config
        original_providers = server_mod._providers.copy()

        try:
            server_mod._config = ArattaConfig(
                local_providers={
                    "ollama": ProviderConfig(
                        name="ollama",
                        base_url="http://localhost:11434",
                        api_key_env=None,
                        default_model="llama3.1:8b",
                        priority=0,
                    )
                },
                providers={},
                model_aliases={},
            )
            server_mod._providers = {}

            p1 = server_mod._get_provider("ollama")
            p2 = server_mod._get_provider("ollama")
            assert p1 is p2
        finally:
            server_mod._config = original_config
            server_mod._providers = original_providers

    def test_concurrent_get_provider_is_safe(self):
        """Multiple threads calling _get_provider should not raise or create duplicates."""
        from aratta.config import ArattaConfig, ProviderConfig

        import aratta.server as server_mod

        original_config = server_mod._config
        original_providers = server_mod._providers.copy()

        try:
            server_mod._config = ArattaConfig(
                local_providers={
                    "ollama": ProviderConfig(
                        name="ollama",
                        base_url="http://localhost:11434",
                        api_key_env=None,
                        default_model="llama3.1:8b",
                        priority=0,
                    )
                },
                providers={},
                model_aliases={},
            )
            server_mod._providers = {}

            results = []
            errors = []

            def get_provider():
                try:
                    p = server_mod._get_provider("ollama")
                    results.append(id(p))
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=get_provider) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

            assert len(errors) == 0, f"Errors during concurrent access: {errors}"
            # All threads should get the same provider instance
            assert len(set(results)) == 1, "Different provider instances created concurrently"
        finally:
            server_mod._config = original_config
            server_mod._providers = original_providers
