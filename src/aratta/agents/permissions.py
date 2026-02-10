"""Permission system for tool execution."""

from enum import Enum
from typing import Any


class PermissionLevel(Enum):
    RESTRICTED = "restricted"
    STANDARD = "standard"
    ELEVATED = "elevated"
    SYSTEM = "system"


PERMISSION_MAP: dict[str, PermissionLevel] = {
    "read_file": PermissionLevel.RESTRICTED,
    "list_files": PermissionLevel.RESTRICTED,
    "list_tools": PermissionLevel.RESTRICTED,
    "get_time": PermissionLevel.RESTRICTED,
    "search_web": PermissionLevel.RESTRICTED,
    "write_file": PermissionLevel.STANDARD,
    "create_file": PermissionLevel.STANDARD,
    "send_message": PermissionLevel.STANDARD,
    "python_sandbox": PermissionLevel.ELEVATED,
    "execute_code": PermissionLevel.ELEVATED,
    "shell_execute": PermissionLevel.ELEVATED,
    "delete_file": PermissionLevel.ELEVATED,
}

BLOCKED_TOOLS: set[str] = {"delete_all", "format_disk", "rm_rf", "shutdown_system"}


def check_permission(tool_name: str, context: dict[str, Any] | None = None) -> bool:
    context = context or {}
    if tool_name in BLOCKED_TOOLS:
        return False
    blocked = set(context.get("blocked_tools", []))
    if tool_name in blocked:
        return False
    if context.get("safe_mode"):
        return PERMISSION_MAP.get(tool_name, PermissionLevel.STANDARD) == PermissionLevel.RESTRICTED
    level = PERMISSION_MAP.get(tool_name, PermissionLevel.STANDARD)
    if level in (PermissionLevel.RESTRICTED, PermissionLevel.STANDARD):
        return True
    if level == PermissionLevel.ELEVATED:
        return tool_name in set(context.get("elevated_tools", []))
    return level != PermissionLevel.SYSTEM
