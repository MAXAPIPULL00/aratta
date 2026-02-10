"""Tests for the agent types and basic agent construction."""

from aratta.agents.types import AgentConfig, AgentMessage, LoopResult, ToolResult


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.model == "local"
        assert cfg.max_iterations == 10
        assert cfg.temperature == 0.7

    def test_custom(self):
        cfg = AgentConfig(model="opus", max_iterations=25, enable_thinking=True)
        assert cfg.model == "opus"
        assert cfg.enable_thinking is True


class TestAgentMessage:
    def test_to_dict(self):
        msg = AgentMessage(role="user", content="hello")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert "timestamp" in d

    def test_round_trip(self):
        msg = AgentMessage(role="assistant", content="hi", thinking="Let me think")
        restored = AgentMessage.from_dict(msg.to_dict())
        assert restored.thinking == "Let me think"


class TestToolResult:
    def test_success(self):
        r = ToolResult(call_id="c1", tool_name="search", success=True, output="found it")
        d = r.to_dict()
        assert d["success"] is True
        assert d["error"] is None

    def test_failure(self):
        r = ToolResult(call_id="c2", tool_name="search", success=False, output=None, error="not found")
        assert r.error == "not found"


class TestLoopResult:
    def test_to_dict(self):
        lr = LoopResult(success=True, content="done", iterations=3, tool_calls=[])
        d = lr.to_dict()
        assert d["success"] is True
        assert d["iterations"] == 3
