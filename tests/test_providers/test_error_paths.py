"""Tests for provider error handling paths."""

from unittest.mock import MagicMock

import httpx
import pytest

from aratta.providers.base import (
    AuthenticationError,
    BaseProvider,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
)


class DummyProvider(BaseProvider):
    """Minimal concrete provider for testing error paths."""

    name = "dummy"
    display_name = "Dummy Provider"

    async def chat(self, messages, model=None, **kwargs):
        return {}

    async def chat_stream(self, messages, model=None, **kwargs):
        yield {}

    async def embed(self, texts, model=None, **kwargs):
        return []

    async def get_models(self):
        return []

    def convert_messages(self, messages):
        return messages

    def convert_tools(self, tools):
        return tools


@pytest.fixture
def provider():
    from aratta.config import ProviderConfig

    cfg = ProviderConfig(
        name="dummy",
        base_url="http://localhost:9999",
        api_key_env=None,
        default_model="test-model",
        priority=0,
    )
    return DummyProvider(cfg)


def _make_response(status_code: int, json_body: dict | None = None, text: str = "") -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = text or str(json_body)
    if json_body is not None:
        resp.json.return_value = json_body
    else:
        resp.json.side_effect = ValueError("No JSON")
    return resp


class TestHandleError:
    def test_401_raises_authentication_error(self, provider):
        resp = _make_response(401)
        with pytest.raises(AuthenticationError) as exc:
            provider._handle_error(resp)
        assert exc.value.status_code == 401

    def test_429_raises_rate_limit_error(self, provider):
        resp = _make_response(429)
        with pytest.raises(RateLimitError) as exc:
            provider._handle_error(resp)
        assert exc.value.status_code == 429

    def test_404_raises_model_not_found_error(self, provider):
        resp = _make_response(404)
        with pytest.raises(ModelNotFoundError) as exc:
            provider._handle_error(resp)
        assert exc.value.status_code == 404

    def test_500_with_json_body(self, provider):
        resp = _make_response(500, json_body={"error": {"message": "Internal server error"}})
        with pytest.raises(ProviderError) as exc:
            provider._handle_error(resp)
        assert "Internal server error" in str(exc.value)
        assert exc.value.status_code == 500

    def test_500_without_json_body(self, provider):
        resp = _make_response(500, json_body=None, text="Something went wrong")
        with pytest.raises(ProviderError) as exc:
            provider._handle_error(resp)
        assert "Something went wrong" in str(exc.value)

    def test_200_does_not_raise(self, provider):
        resp = _make_response(200)
        # _handle_error only triggers on 400+, 200 should be a no-op
        provider._handle_error(resp)  # Should not raise


class TestProviderErrorAttributes:
    def test_error_has_provider_name(self):
        err = ProviderError("test error", "my_provider", 500)
        assert err.provider == "my_provider"
        assert err.status_code == 500
        assert "my_provider" in str(err)

    def test_rate_limit_inherits_provider_error(self):
        err = RateLimitError("rate limited", "openai", 429)
        assert isinstance(err, ProviderError)
        assert err.status_code == 429

    def test_auth_error_inherits_provider_error(self):
        err = AuthenticationError("bad key", "anthropic", 401)
        assert isinstance(err, ProviderError)
