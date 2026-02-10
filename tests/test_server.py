"""Integration tests for the FastAPI server endpoints."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from aratta.config import ArattaConfig, ProviderConfig, ProviderPriority
from aratta.server import create_app


@pytest.fixture
def config():
    cfg = ArattaConfig()
    cfg.local_providers["ollama"] = ProviderConfig(
        name="ollama", base_url="http://localhost:11434",
        api_key_env=None, default_model="llama3.1:8b",
        priority=ProviderPriority.LOCAL.value,
    )
    return cfg


@pytest.fixture
def client(config):
    with patch("aratta.server.load_config", return_value=config):
        app = create_app()
        with TestClient(app) as c:
            yield c


class TestLiveness:
    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestModelsEndpoint:
    def test_models_returns_list(self, client):
        resp = client.get("/api/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert "aliases" in data
