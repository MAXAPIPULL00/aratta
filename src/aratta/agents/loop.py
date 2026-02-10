"""AgentLoop — ReAct (Reason-Act-Observe) execution cycle."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from .types import AgentMessage, AgentState, LoopResult, ToolResult

if TYPE_CHECKING:
    from .agent import Agent
    from .context import AgentContext

logger = logging.getLogger("aratta.loop")


class AgentLoop:
    def __init__(self, agent: Agent, context: AgentContext):
        self.agent = agent
        self.context = context
        self.tool_calls_made: list[dict[str, Any]] = []
        self.start_time: float | None = None

    async def run(self) -> LoopResult:
        self.start_time = time.time()
        self.agent.set_state(AgentState.REASONING)
        try:
            while self.agent.iterations < self.agent.config.max_iterations:
                if time.time() - self.start_time > self.agent.config.timeout_seconds:
                    return self._result(False, "Agent timed out", "timeout")
                self.agent.iterations += 1
                self.agent.set_state(AgentState.REASONING)
                resp = await self._reason()
                if resp.get("error"):
                    self.agent.metrics["errors"] += 1
                    if self.agent.config.retry_on_error and self.agent.iterations < self.agent.config.max_iterations:
                        continue
                    return self._result(False, resp["error"], "error", error=resp["error"])
                tool_calls = resp.get("tool_calls", [])
                if not tool_calls:
                    self.agent.set_state(AgentState.COMPLETED)
                    return self._result(True, resp.get("content", ""), "completed", reasoning=resp.get("thinking"))
                # Check confirmation
                needs = self._needs_confirm(tool_calls)
                if needs:
                    self.agent.set_state(AgentState.WAITING)
                    return self._result(False, f"Tool '{needs}' requires confirmation", "paused")
                self.agent.set_state(AgentState.EXECUTING)
                results = await self._act(tool_calls)
                self._observe(tool_calls, results)
            return self._result(False, "Max iterations reached", "max_iterations")
        except asyncio.CancelledError:
            self.agent.set_state(AgentState.FAILED)
            return self._result(False, "Cancelled", "cancelled")
        except Exception as e:
            self.agent.set_state(AgentState.FAILED)
            return self._result(False, str(e), "error", error=str(e))

    async def _reason(self) -> dict[str, Any]:
        if not self.context or not self.context.aratta:
            return {"error": "No AI client available"}
        try:
            messages = [{"role": "system", "content": self.agent.system_prompt}]
            for m in self.agent.messages:
                messages.append({"role": "user" if m.role == "tool" else m.role, "content": m.content})
            req: dict[str, Any] = {"messages": messages, "model": self.agent.config.model,
                                    "temperature": self.agent.config.temperature, "max_tokens": self.agent.config.max_tokens}
            if self.agent.tools:
                req["tools"] = [t.model_dump() if hasattr(t, "model_dump") else (t.to_dict() if hasattr(t, "to_dict") else {"name": getattr(t, "name", str(t)), "description": getattr(t, "description", ""), "parameters": getattr(t, "parameters", {})}) for t in self.agent.tools]
            if self.agent.config.enable_thinking:
                req["thinking"] = {"enabled": True, "budget_tokens": self.agent.config.thinking_budget}
            start = time.time()
            response = await self.context.aratta.chat(req) if asyncio.iscoroutinefunction(getattr(self.context.aratta, "chat", None)) else self.context.aratta.chat(req)
            self.agent.metrics["reasoning_time_ms"] += (time.time() - start) * 1000
            content = response.content if hasattr(response, "content") else response.get("content", "") if isinstance(response, dict) else str(response)
            tc = response.tool_calls if hasattr(response, "tool_calls") else response.get("tool_calls", []) if isinstance(response, dict) else []
            tool_calls = [{"id": getattr(c, "id", f"call_{i}"), "name": getattr(c, "name", c.get("name", "") if isinstance(c, dict) else ""), "arguments": getattr(c, "arguments", c.get("arguments", {}) if isinstance(c, dict) else {})} for i, c in enumerate(tc or [])]
            self.agent.add_message(AgentMessage(role="assistant", content=content, tool_calls=tool_calls or None))
            return {"content": content, "tool_calls": tool_calls}
        except Exception as e:
            return {"error": str(e)}

    def _needs_confirm(self, calls: list[dict]) -> str | None:
        confirm = set(self.agent.config.require_confirmation)
        for c in calls:
            if c.get("name") in confirm:
                return c["name"]
        return None

    async def _act(self, tool_calls: list[dict]) -> list[ToolResult]:
        results = []
        for tc in tool_calls:
            name, args = tc.get("name", ""), tc.get("arguments", {})
            for cb in self.agent._on_tool_call:
                try:
                    cb(name, args)
                except Exception:
                    pass
            start = time.time()
            if self.context and self.context.tool_executor:
                r = await self.context.tool_executor.execute(name, args)
            else:
                r = ToolResult(call_id=tc.get("id", ""), tool_name=name, success=False, output=None, error="No executor")
            ms = (time.time() - start) * 1000
            self.agent.metrics["tool_calls"] += 1
            self.agent.metrics["execution_time_ms"] += ms
            results.append(r)
            self.tool_calls_made.append({"name": name, "arguments": args, "success": r.success, "output": r.output, "error": r.error})
        return results

    def _observe(self, calls: list[dict], results: list[ToolResult]):
        for _tc, r in zip(calls, results, strict=False):
            content = f"Tool '{r.tool_name}' {'succeeded' if r.success else 'failed'}:\n{r.output if r.success else r.error}"
            self.agent.add_message(AgentMessage(role="tool", content=content, tool_results=[r.to_dict()]))

    def _result(self, success: bool, content: str, reason: str, reasoning=None, error=None) -> LoopResult:
        ms = (time.time() - self.start_time) * 1000 if self.start_time else 0
        return LoopResult(success, content, self.agent.iterations, self.tool_calls_made, reasoning, error, reason, ms, self.agent.metrics.copy())
