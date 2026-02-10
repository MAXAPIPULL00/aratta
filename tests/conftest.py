"""Shared fixtures for the Aratta test suite."""

import pytest

from aratta.config import ArattaConfig, ProviderConfig, ProviderPriority


@pytest.fixture
def provider_config():
    """A minimal provider config for testing."""
    return ProviderConfig(
        name="test",
        base_url="http://localhost:9999",
        api_key_env=None,
        default_model="test-model",
        priority=ProviderPriority.LOCAL.value,
    )


@pytest.fixture
def aratta_config():
    """A minimal ArattaConfig with one local provider."""
    cfg = ArattaConfig()
    cfg.local_providers["test"] = ProviderConfig(
        name="test",
        base_url="http://localhost:9999",
        api_key_env=None,
        default_model="test-model",
        priority=ProviderPriority.LOCAL.value,
    )
    return cfg
