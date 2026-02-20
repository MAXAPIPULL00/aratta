"""
Reload manager — hot-reloads provider adapters after self-healing fixes.

Workflow:
    Error detected → HealWorker generates fix → ReloadManager applies → Verify → Commit/Rollback

Capabilities:
    - Version tracking with bounded history
    - Backup before every change
    - Hot-reload provider modules in-memory
    - Auto-rollback on verification failure
    - Human approval queue for low-confidence fixes
"""

from __future__ import annotations

import importlib
import json
import logging
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger("aratta.reload_manager")

BACKUP_DIR = Path("data/adapter_backups")
MAX_VERSIONS = 10
AUTO_APPLY_CONFIDENCE = 0.85


class FixStatus(Enum):
    PENDING = "pending"
    APPLIED = "applied"
    VERIFIED = "verified"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"


@dataclass
class AdapterVersion:
    version: int
    provider: str
    timestamp: str
    backup_path: Path
    change_summary: str
    fix_proposal: dict[str, Any] | None = None
    status: FixStatus = FixStatus.APPLIED
    verification_result: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version, "provider": self.provider,
            "timestamp": self.timestamp, "backup_path": str(self.backup_path),
            "change_summary": self.change_summary, "fix_proposal": self.fix_proposal,
            "status": self.status.value, "verification_result": self.verification_result,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AdapterVersion:
        return cls(
            version=data["version"], provider=data["provider"],
            timestamp=data["timestamp"], backup_path=Path(data["backup_path"]),
            change_summary=data["change_summary"], fix_proposal=data.get("fix_proposal"),
            status=FixStatus(data.get("status", "applied")),
            verification_result=data.get("verification_result"),
        )


@dataclass
class FixApplication:
    success: bool
    provider: str
    version: int
    message: str
    code_changed: bool = False
    requires_restart: bool = False
    verification_passed: bool | None = None


class ReloadManager:
    """
    Manages hot-reloading of provider adapters.

    Backs up → applies fix → reloads module → verifies → commits or rolls back.
    Thread-safe for async operation.
    """

    # Maps provider name → Python module path (relative to aratta package)
    PROVIDER_MODULES = {
        "anthropic": "aratta.providers.anthropic.adapter",
        "openai": "aratta.providers.openai.adapter",
        "google": "aratta.providers.google.adapter",
        "xai": "aratta.providers.xai.adapter",
        "ollama": "aratta.providers.local.adapter",
        "vllm": "aratta.providers.local.adapter",
        "llamacpp": "aratta.providers.local.adapter",
    }

    def __init__(
        self,
        auto_apply: bool = False,
        auto_apply_threshold: float = AUTO_APPLY_CONFIDENCE,
        backup_dir: Path = BACKUP_DIR,
    ):
        self.auto_apply = auto_apply
        self.auto_apply_threshold = auto_apply_threshold
        self.backup_dir = backup_dir
        self.versions: dict[str, list[AdapterVersion]] = {}
        self.current_version: dict[str, int] = {}
        self.pending_fixes: dict[str, dict[str, Any]] = {}
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._load_history()

    def _load_history(self):
        history_file = self.backup_dir / "version_history.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    data = json.load(f)
                for provider, versions in data.get("versions", {}).items():
                    self.versions[provider] = [AdapterVersion.from_dict(v) for v in versions]
                self.current_version = data.get("current_version", {})
            except Exception as e:
                logger.warning(f"Failed to load version history: {e}")

    def _save_history(self):
        history_file = self.backup_dir / "version_history.json"
        try:
            data = {
                "versions": {p: [v.to_dict() for v in vs] for p, vs in self.versions.items()},
                "current_version": self.current_version,
            }
            with open(history_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save version history: {e}")

    def _get_adapter_path(self, provider: str) -> Path | None:
        """Resolve the file path for a provider's adapter module."""
        module_name = self.PROVIDER_MODULES.get(provider)
        if not module_name:
            return None
        module = sys.modules.get(module_name)
        if module and hasattr(module, "__file__") and module.__file__:
            return Path(module.__file__)
        # Fallback: try to find it via importlib
        try:
            spec = importlib.util.find_spec(module_name)
            if spec and spec.origin:
                return Path(spec.origin)
        except (ModuleNotFoundError, ValueError):
            pass
        return None

    def _backup_adapter(self, provider: str, change_summary: str) -> AdapterVersion:
        adapter_path = self._get_adapter_path(provider)
        if not adapter_path or not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found for {provider}")

        current = self.current_version.get(provider, 0)
        new_version = current + 1
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / provider / f"v{new_version}_{ts}.py"
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(adapter_path, backup_path)

        version = AdapterVersion(
            version=new_version, provider=provider,
            timestamp=datetime.now(UTC).isoformat(),
            backup_path=backup_path, change_summary=change_summary,
            status=FixStatus.PENDING,
        )
        if provider not in self.versions:
            self.versions[provider] = []
        self.versions[provider].append(version)

        # Trim old versions
        if len(self.versions[provider]) > MAX_VERSIONS:
            oldest = self.versions[provider].pop(0)
            if oldest.backup_path.exists():
                oldest.backup_path.unlink()

        logger.info(f"Backed up {provider} adapter to v{new_version}")
        return version

    async def apply_fix(
        self,
        provider: str,
        fix_proposal: dict[str, Any],
        verify_callback: Callable | None = None,
    ) -> FixApplication:
        """
        Apply a fix proposal.

        If confidence is below threshold and auto_apply is off, queues for human review.
        """
        fix_type = fix_proposal.get("fix_type", "unknown")
        confidence = fix_proposal.get("confidence", 0)
        change_summary = fix_proposal.get("change_summary", "Unknown change")

        logger.info(f"Applying fix for {provider}: {fix_type} (confidence: {confidence})")

        # Queue for review if below threshold
        if not self.auto_apply and confidence < self.auto_apply_threshold:
            self.pending_fixes[provider] = fix_proposal
            return FixApplication(
                success=False, provider=provider,
                version=self.current_version.get(provider, 0),
                message=f"Queued for review (confidence {confidence:.2f} < {self.auto_apply_threshold})",
            )

        try:
            version = self._backup_adapter(provider, change_summary)
            version.fix_proposal = fix_proposal
            code_changed = False

            if fix_type in ("code_patch", "workaround"):
                fix_code = fix_proposal.get("fix_code")
                if fix_code:
                    code_changed = await self._apply_code_patch(provider, fix_code)
            elif fix_type == "no_fix_needed":
                version.status = FixStatus.VERIFIED
                version.verification_result = "No fix needed"
                self._save_history()
                return FixApplication(True, provider, version.version, "No fix needed")

            if code_changed:
                await self._reload_module(provider)

            # Verify
            verified = True
            if verify_callback:
                try:
                    verified = await verify_callback(provider)
                except Exception as e:
                    logger.error(f"Verification failed: {e}")
                    verified = False

            if verified:
                version.status = FixStatus.VERIFIED
                version.verification_result = "Passed"
                self.current_version[provider] = version.version
                self._save_history()
                return FixApplication(True, provider, version.version, "Fix applied and verified",
                                      code_changed=code_changed, verification_passed=True)
            else:
                await self._rollback(provider, version.version - 1)
                version.status = FixStatus.ROLLED_BACK
                version.verification_result = "Failed, rolled back"
                self._save_history()
                return FixApplication(False, provider, version.version - 1,
                                      "Verification failed, rolled back", verification_passed=False)

        except Exception as e:
            logger.error(f"Fix application failed: {e}", exc_info=True)
            return FixApplication(False, provider, self.current_version.get(provider, 0),
                                  f"Failed: {e}")

    async def _apply_code_patch(self, provider: str, fix_code: str) -> bool:
        """Apply a code patch. Conservative — logs proposal, returns False by design.

        Automatic code patching is intentionally disabled for safety.
        Patches are logged for human review via the /api/v1/fixes endpoint.
        """
        adapter_path = self._get_adapter_path(provider)
        if not adapter_path or not adapter_path.exists():
            logger.warning(f"Cannot propose code patch: adapter path not found for {provider}")
            return False
        logger.info(
            f"Code patch proposed for {provider} (requires human review via API):\n"
            f"{fix_code[:500]}..."
        )
        return False

    async def _reload_module(self, provider: str):
        module_name = self.PROVIDER_MODULES.get(provider)
        if not module_name:
            return
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
            logger.info(f"Reloaded module: {module_name}")

    async def _rollback(self, provider: str, to_version: int):
        if provider not in self.versions:
            raise ValueError(f"No version history for {provider}")
        target = next((v for v in self.versions[provider] if v.version == to_version), None)
        if not target:
            raise ValueError(f"Version {to_version} not found for {provider}")
        if not target.backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {target.backup_path}")

        adapter_path = self._get_adapter_path(provider)
        if adapter_path:
            shutil.copy2(target.backup_path, adapter_path)
            await self._reload_module(provider)
        self.current_version[provider] = to_version
        self._save_history()
        logger.info(f"Rolled back {provider} to v{to_version}")

    async def rollback_to_version(self, provider: str, version: int) -> bool:
        try:
            await self._rollback(provider, version)
            return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def get_version_history(self, provider: str) -> list[dict[str, Any]]:
        return [v.to_dict() for v in self.versions.get(provider, [])]

    def get_pending_fixes(self) -> dict[str, dict[str, Any]]:
        return dict(self.pending_fixes)

    async def approve_fix(self, provider: str, verify_callback: Callable | None = None) -> FixApplication:
        if provider not in self.pending_fixes:
            return FixApplication(False, provider, self.current_version.get(provider, 0),
                                  "No pending fix")
        fix = self.pending_fixes.pop(provider)
        old = self.auto_apply
        self.auto_apply = True
        try:
            return await self.apply_fix(provider, fix, verify_callback)
        finally:
            self.auto_apply = old

    def reject_fix(self, provider: str, reason: str = "") -> bool:
        if provider not in self.pending_fixes:
            return False
        self.pending_fixes.pop(provider)
        logger.info(f"Rejected fix for {provider}: {reason}")
        if provider in self.versions and self.versions[provider]:
            latest = self.versions[provider][-1]
            latest.status = FixStatus.REJECTED
            latest.verification_result = f"Rejected: {reason}"
            self._save_history()
        return True

    def get_status(self) -> dict[str, Any]:
        return {
            "auto_apply": self.auto_apply,
            "auto_apply_threshold": self.auto_apply_threshold,
            "current_versions": self.current_version,
            "pending_fixes_count": len(self.pending_fixes),
            "pending_fixes": list(self.pending_fixes.keys()),
            "version_counts": {p: len(vs) for p, vs in self.versions.items()},
        }
