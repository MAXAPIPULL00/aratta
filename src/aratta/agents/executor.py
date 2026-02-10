"""Tool executor — runs tools on behalf of agents with permissions and audit."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .permissions import check_permission
from .sandbox import get_sandbox
from .types import ToolResult

if TYPE_CHECKING:
    from .context import AgentContext

logger = logging.getLogger("aratta.executor")


class ToolExecutor:
    DEFAULT_TIMEOUT = 30
    MAX_AUDIT = 1000

    def __init__(self, context: AgentContext = None):
        self.context = context
        self.sandbox = get_sandbox()
        self._impls: dict[str, Callable[..., Awaitable[Any]]] = {}
        self._audit: deque = deque(maxlen=self.MAX_AUDIT)
        self._register_builtins()

    def _register_builtins(self):
        self._impls["python_sandbox"] = self._exec_sandbox
        self._impls["execute_code"] = self._exec_sandbox
        self._impls["get_time"] = self._exec_time
        self._impls["list_tools"] = self._exec_list_tools

    def register_tool(self, name: str, impl: Callable[..., Awaitable[Any]]):
        self._impls[name] = impl

    async def execute(self, tool_name: str, arguments: dict[str, Any], timeout: int = None) -> ToolResult:
        start = time.time()
        timeout = timeout or self.DEFAULT_TIMEOUT
        if not check_permission(tool_name, {}):
            return ToolResult(call_id="", tool_name=tool_name, success=False, output=None, error=f"Permission denied: {tool_name}")
        impl = self._impls.get(tool_name)
        if not impl:
            return ToolResult(call_id="", tool_name=tool_name, success=False, output=None, error=f"Unknown tool: {tool_name}")
        try:
            result = await asyncio.wait_for(impl(arguments), timeout=timeout)
            ms = (time.time() - start) * 1000
            if isinstance(result, ToolResult):
                return result
            if isinstance(result, dict) and "success" in result:
                return ToolResult("", tool_name, result.get("success", True), result.get("output", result), result.get("error"), ms)
            return ToolResult("", tool_name, True, result, execution_time_ms=ms)
        except TimeoutError:
            return ToolResult("", tool_name, False, None, f"Timed out after {timeout}s", (time.time() - start) * 1000)
        except Exception as e:
            return ToolResult("", tool_name, False, None, str(e), (time.time() - start) * 1000)

    async def _exec_sandbox(self, args: dict) -> Any:
        r = await self.sandbox.execute(args.get("code", ""), timeout=args.get("timeout", 30))
        return {"success": r.success, "output": r.stdout if r.success else None, "error": r.error or r.blocked_reason}

    async def _exec_time(self, args: dict) -> Any:
        now = datetime.now(UTC)
        return {"success": True, "output": {"iso": now.isoformat(), "timestamp": now.timestamp()}}

    async def _exec_list_tools(self, args: dict) -> Any:
        return {"success": True, "output": {"tools": list(self._impls.keys())}}
