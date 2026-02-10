"""Tests for core types — serialization, construction, round-trips."""

from aratta.core.types import (
    ChatRequest,
    ChatResponse,
    Content,
    ContentType,
    Embedding,
    EmbeddingResponse,
    Message,
    ModelCapabilities,
    Role,
    ThinkingBlock,
    Tool,
    ToolCall,
    Usage,
)


class TestMessage:
    def test_text_message_to_dict(self):
        msg = Message(role=Role.USER, content="hello")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "hello"

    def test_content_blocks_to_dict(self):
        blocks = [Content(type=ContentType.TEXT, text="hi"), Content(type=ContentType.IMAGE, image_url="http://img.png")]
        msg = Message(role=Role.USER, content=blocks)
        d = msg.to_dict()
        assert isinstance(d["content"], list)
        assert d["content"][0]["type"] == "text"
        assert d["content"][1]["type"] == "image"

    def test_from_dict_text(self):
        msg = Message.from_dict({"role": "assistant", "content": "hi"})
        assert msg.role == Role.ASSISTANT
        assert msg.content == "hi"

    def test_from_dict_blocks(self):
        msg = Message.from_dict({"role": "user", "content": [{"type": "text", "text": "hello"}]})
        assert isinstance(msg.content, list)
        assert msg.content[0].type == ContentType.TEXT

    def test_round_trip(self):
        original = Message(role=Role.SYSTEM, content="you are helpful")
        restored = Message.from_dict(original.to_dict())
        assert restored.role == original.role
        assert restored.content == original.content


class TestTool:
    def test_to_dict(self):
        t = Tool(name="search", description="Search the web", parameters={"type": "object", "properties": {"q": {"type": "string"}}})
        d = t.to_dict()
        assert d["name"] == "search"
        assert "properties" in d["parameters"]

    def test_from_dict(self):
        t = Tool.from_dict({"name": "calc", "description": "Calculate", "parameters": {}})
        assert t.name == "calc"


class TestToolCall:
    def test_to_dict(self):
        tc = ToolCall(id="call_1", name="search", arguments={"q": "test"})
        d = tc.to_dict()
        assert d["id"] == "call_1"
        assert d["arguments"]["q"] == "test"


class TestUsage:
    def test_basic(self):
        u = Usage(input_tokens=100, output_tokens=50, total_tokens=150)
        d = u.to_dict()
        assert d["total_tokens"] == 150
        assert "cache_read_tokens" not in d

    def test_with_cache(self):
        u = Usage(input_tokens=100, output_tokens=50, total_tokens=150, cache_read_tokens=20)
        d = u.to_dict()
        assert d["cache_read_tokens"] == 20


class TestChatRequest:
    def test_defaults(self):
        req = ChatRequest(messages=[Message(role=Role.USER, content="hi")])
        assert req.model == "local"
        assert req.temperature == 0.7
        assert req.stream is False

    def test_to_dict(self):
        req = ChatRequest(messages=[Message(role=Role.USER, content="hi")], model="opus")
        d = req.to_dict()
        assert d["model"] == "opus"
        assert len(d["messages"]) == 1

    def test_thinking_in_dict(self):
        req = ChatRequest(messages=[], thinking_enabled=True, thinking_budget=5000)
        d = req.to_dict()
        assert d["thinking"]["enabled"] is True
        assert d["thinking"]["budget_tokens"] == 5000


class TestChatResponse:
    def test_to_dict(self):
        resp = ChatResponse(id="msg_1", content="hello", model="test", provider="test")
        d = resp.to_dict()
        assert d["id"] == "msg_1"
        assert d["finish_reason"] == "stop"
        assert "tool_calls" not in d

    def test_with_tool_calls(self):
        tc = ToolCall(id="c1", name="search", arguments={"q": "test"})
        resp = ChatResponse(id="msg_2", content="", model="test", provider="test", tool_calls=[tc])
        d = resp.to_dict()
        assert len(d["tool_calls"]) == 1


class TestThinkingBlock:
    def test_to_dict(self):
        tb = ThinkingBlock(thinking="Let me think...", signature="sig123")
        d = tb.to_dict()
        assert d["type"] == "thinking"
        assert d["signature"] == "sig123"


class TestModelCapabilities:
    def test_to_dict(self):
        mc = ModelCapabilities(
            model_id="test-model", provider="test", display_name="Test",
            supports_tools=True, context_window=128000,
        )
        d = mc.to_dict()
        assert d["model_id"] == "test-model"
        assert d["supports_tools"] is True


class TestEmbedding:
    def test_response_to_dict(self):
        emb = EmbeddingResponse(
            embeddings=[Embedding(embedding=[0.1, 0.2], index=0)],
            model="embed", provider="openai",
            usage=Usage(input_tokens=10, output_tokens=0, total_tokens=10),
        )
        d = emb.to_dict()
        assert len(d["embeddings"]) == 1
        assert d["embeddings"][0]["index"] == 0
