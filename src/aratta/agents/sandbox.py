"""Secure sandbox — AST-validated, subprocess-isolated code execution."""

import ast
import asyncio
import hashlib
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("aratta.sandbox")

BLOCKED_IMPORTS: set[str] = {
    "os", "sys", "subprocess", "shutil", "pathlib", "glob", "tempfile",
    "socket", "requests", "urllib", "http", "aiohttp", "httpx", "asyncio",
    "importlib", "imp", "builtins", "__builtins__", "io",
    "signal", "multiprocessing", "threading", "concurrent",
    "ctypes", "cffi", "pickle", "marshal",
}

ALLOWED_IMPORTS: set[str] = {
    "math", "statistics", "random", "decimal", "fractions",
    "datetime", "time", "calendar", "json", "csv", "re", "string",
    "collections", "itertools", "functools", "operator",
    "typing", "dataclasses", "enum", "copy", "pprint", "textwrap",
}

BLOCKED_CALLS: set[str] = {"eval", "exec", "compile", "__import__", "open", "input", "breakpoint", "globals", "locals"}
BLOCKED_ATTRS: set[str] = {"__class__", "__bases__", "__mro__", "__subclasses__", "__builtins__", "__globals__", "__code__"}


@dataclass
class ExecutionResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int = -1
    error: str | None = None
    blocked_reason: str | None = None
    execution_time_ms: float = 0


class CodeValidator:
    def validate(self, code: str) -> tuple[bool, str | None]:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import) else [(node.module or "").split(".")[0]]
                for m in names:
                    if m in BLOCKED_IMPORTS:
                        return False, f"Blocked import: {m}"
                    if m not in ALLOWED_IMPORTS:
                        return False, f"Import not in allowlist: {m}"
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
                return False, f"Blocked call: {node.func.id}"
            elif isinstance(node, ast.Attribute) and node.attr in BLOCKED_ATTRS:
                return False, f"Blocked attribute: {node.attr}"
        return True, None


class SecureSandbox:
    DEFAULT_TIMEOUT = 30
    MAX_OUTPUT = 100_000

    def __init__(self, sandbox_dir: str = None, timeout: int = DEFAULT_TIMEOUT):
        self.sandbox_dir = Path(sandbox_dir) if sandbox_dir else Path(tempfile.gettempdir()) / "aratta_sandbox"
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.validator = CodeValidator()

    async def execute(self, code: str, timeout: int = None) -> ExecutionResult:
        timeout = timeout or self.timeout
        start = time.time()
        ok, reason = self.validator.validate(code)
        if not ok:
            return ExecutionResult(success=False, blocked_reason=reason)
        code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]
        tmp = self.sandbox_dir / f"exec_{code_hash}.py"
        try:
            tmp.write_text(code, encoding="utf-8")
            env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
            if os.name == "nt":
                env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
                env["TEMP"] = env["TMP"] = str(self.sandbox_dir)
            proc = await asyncio.create_subprocess_exec("python", str(tmp), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env, cwd=str(self.sandbox_dir))
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                ms = (time.time() - start) * 1000
                out = stdout.decode("utf-8", errors="replace")[:self.MAX_OUTPUT]
                err = stderr.decode("utf-8", errors="replace")[:self.MAX_OUTPUT]
                return ExecutionResult(success=proc.returncode == 0, stdout=out, stderr=err, return_code=proc.returncode, error=err if proc.returncode != 0 else None, execution_time_ms=ms)
            except TimeoutError:
                proc.kill()
                await proc.wait()
                return ExecutionResult(success=False, error=f"Timed out after {timeout}s", execution_time_ms=(time.time() - start) * 1000)
        except Exception as e:
            return ExecutionResult(success=False, error=str(e), execution_time_ms=(time.time() - start) * 1000)
        finally:
            tmp.unlink(missing_ok=True)


_sandbox: SecureSandbox | None = None


def get_sandbox() -> SecureSandbox:
    global _sandbox
    if _sandbox is None:
        _sandbox = SecureSandbox()
    return _sandbox
