"""
Heal worker — self-healing through Aratta's own provider stack.

The heal flow is a two-phase process that demonstrates sovereignty in action:

Phase 1 — DIAGNOSE (local model):
    The local model analyzes the error, identifies what broke, and formulates
    a search query to find current API documentation or changelogs.

Phase 2 — RESEARCH (cloud provider with search):
    Escalates to a search-capable provider (Grok with web/X search, or any
    provider with content retrieval) to find the actual current API docs.
    The cloud provider is a tool — it fetches what the local model asked for.

Phase 3 — FIX (local model):
    The local model takes the research findings + the adapter source code
    and generates the actual code fix. The local model makes the decision.
    Cloud was just the eyes. Local is the brain.

This is what sovereignty means in practice: cloud providers are callable
services you invoke when a task requires specific capabilities. They don't
make decisions. Your local model does.

In SCRI constellation: errors → Mycelium → Librarian (Grok collections) → fix
In standalone Aratta: errors → local diagnose → cloud search → local fix
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any

logger = logging.getLogger("aratta.heal_worker")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DIAGNOSE_PROMPT = """\
You are analyzing an adapter failure in Aratta, a sovereignty layer for AI providers.

Given the error details below, determine:
1. Is this a transient issue (rate limit, timeout) or a real API/schema change?
2. If it looks like an API change, what specific thing changed?
3. What search query would find the current API documentation or changelog?

Respond in this exact JSON format:
{
    "is_transient": false,
    "diagnosis": "Brief analysis of what broke",
    "search_queries": ["query to find current API docs", "query for changelog"],
    "what_to_look_for": "Specific thing the search should find"
}"""

FIX_PROMPT = """\
You are generating a code fix for an Aratta provider adapter.

You have:
1. The original error and diagnosis
2. Research findings from current API documentation
3. The current adapter source code

Generate a fix. Be conservative — only change what's necessary.

Respond in this exact JSON format:
{
    "fix_type": "code_patch | config_change | workaround | no_fix_needed",
    "confidence": 0.0,
    "change_summary": "One-line description",
    "fix_code": "The corrected function or code block (only if fix_type is code_patch/workaround)",
    "reasoning": "Why this fix addresses the issue"
}"""


class HealWorker:
    """
    Self-healing worker using Aratta's own provider stack.

    Local model diagnoses → cloud provider researches → local model fixes.

    Args:
        get_provider_fn: Returns a provider adapter by name.
        resolve_model_fn: Resolves alias → (provider_name, model_id).
        heal_model: Model alias for diagnosis and fix generation (default: "local").
        research_model: Model alias for web search / doc retrieval (default: "grok").
            Falls back to any available cloud provider if the primary isn't configured.
        research_web_search: Whether to enable web search on the research model.
    """

    def __init__(
        self,
        get_provider_fn=None,
        resolve_model_fn=None,
        heal_model: str = "local",
        research_model: str = "grok",
        research_web_search: bool = True,
        cloud_providers: list[str] | None = None,
    ):
        self._get_provider = get_provider_fn
        self._resolve_model = resolve_model_fn
        self.heal_model = heal_model
        self.research_model = research_model
        self.research_web_search = research_web_search
        self.cloud_providers = cloud_providers or ["xai", "openai", "google", "anthropic"]

    async def diagnose(
        self,
        provider: str,
        model: str,
        error_type: str,
        error_message: str,
        recent_errors: list[dict[str, Any]] | None = None,
        adapter_source: str | None = None,
    ) -> dict[str, Any]:
        """
        Full heal cycle: diagnose → research → fix.

        Returns a dict compatible with ReloadManager.apply_fix().
        """
        try:
            # Phase 1: Local model diagnoses the error
            diagnosis = await self._phase_diagnose(
                provider, model, error_type, error_message, recent_errors or [],
            )

            if diagnosis.get("is_transient", False):
                logger.info(f"Diagnosis: transient issue for {provider}, skipping fix")
                return {
                    "fix_type": "no_fix_needed",
                    "confidence": 0.8,
                    "change_summary": f"Transient: {diagnosis.get('diagnosis', 'temporary failure')}",
                    "analysis": diagnosis.get("diagnosis", ""),
                    "reasoning": "Transient errors resolve on their own",
                }

            # Phase 2: Cloud provider researches current docs
            research = await self._phase_research(
                provider, diagnosis.get("search_queries", []),
                diagnosis.get("what_to_look_for", ""),
            )

            # Phase 3: Local model generates the fix
            fix = await self._phase_fix(
                provider, model, error_type, error_message,
                diagnosis, research, adapter_source,
            )

            logger.info(
                f"Heal cycle complete for {provider}: "
                f"{fix.get('fix_type')} (confidence: {fix.get('confidence', 0):.2f})"
            )
            return fix

        except Exception as e:
            logger.error(f"Heal worker failed for {provider}: {e}", exc_info=True)
            # Categorize the error instead of masking as "no_fix_needed"
            error_str = str(e).lower()
            if "auth" in error_str or "401" in error_str or "key" in error_str:
                fix_type = "auth_error"
            elif "timeout" in error_str or "connect" in error_str or "temporary" in error_str:
                fix_type = "transient_error"
            else:
                fix_type = "heal_error"
            return {
                "fix_type": fix_type,
                "confidence": 0.0,
                "change_summary": f"Heal cycle failed: {e}",
                "analysis": str(e),
                "reasoning": f"Could not complete heal cycle ({fix_type})",
            }

    # -----------------------------------------------------------------------
    # Phase 1: Diagnose (local model)
    # -----------------------------------------------------------------------

    async def _phase_diagnose(
        self,
        provider: str,
        model: str,
        error_type: str,
        error_message: str,
        recent_errors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Local model analyzes the error and formulates search queries."""
        prompt_parts = [
            "## Adapter Failure Report",
            f"Provider: {provider}",
            f"Model: {model}",
            f"Error type: {error_type}",
            f"Error message: {error_message}",
        ]
        if recent_errors:
            prompt_parts.append("\n## Recent Error History")
            for i, err in enumerate(recent_errors[-5:], 1):
                prompt_parts.append(
                    f"{i}. [{err.get('type', 'unknown')}] {err.get('message', '')[:200]}"
                )

        user_prompt = "\n".join(prompt_parts)

        response = await self._call_model(
            self.heal_model, DIAGNOSE_PROMPT, user_prompt,
        )
        return self._parse_json(response, fallback={
            "is_transient": False,
            "diagnosis": response[:500],
            "search_queries": [f"{provider} API changelog latest"],
            "what_to_look_for": "API changes",
        })

    # -----------------------------------------------------------------------
    # Phase 2: Research (cloud provider with search)
    # -----------------------------------------------------------------------

    async def _phase_research(
        self,
        provider: str,
        search_queries: list[str],
        what_to_look_for: str,
    ) -> str:
        """
        Use a search-capable cloud provider to find current API docs.

        Tries the configured research_model first (default: grok with web search).
        Falls back to any available cloud provider.
        """
        if not search_queries:
            search_queries = [f"{provider} API documentation latest changes"]

        research_prompt = (
            f"Search for the latest {provider} API documentation and recent changes.\n\n"
            f"Specifically look for: {what_to_look_for}\n\n"
            f"Search queries to try:\n"
            + "\n".join(f"- {q}" for q in search_queries)
            + "\n\nReturn a summary of what you find about recent API changes, "
            f"new fields, deprecated fields, or format changes for the {provider} API. "
            f"Include specific details about request/response schemas if available."
        )

        system = (
            "You are a research assistant finding current API documentation. "
            "Search the web for the most recent information and summarize your findings. "
            "Focus on API changes, schema updates, and breaking changes."
        )

        # Try research model (grok with web search)
        try:
            result = await self._call_model(
                self.research_model, system, research_prompt,
                web_search=self.research_web_search,
            )
            if result and len(result.strip()) > 50:
                logger.info(f"Research phase: got findings from {self.research_model}")
                return result
        except Exception as e:
            logger.warning(f"Research model {self.research_model} failed: {e}")

        # Fallback: try other cloud providers
        for cloud_name in self.cloud_providers:
            if cloud_name == self.research_model.split(":")[0]:
                continue
            try:
                result = await self._call_model(
                    cloud_name, system, research_prompt, web_search=True,
                )
                if result and len(result.strip()) > 50:
                    logger.info(f"Research phase: got findings from {cloud_name} (fallback)")
                    return result
            except Exception as e:
                logger.debug(f"Research fallback {cloud_name} failed: {e}")
                continue

        logger.warning("Research phase: no cloud provider available, proceeding without docs")
        return "No current documentation found. Fix based on error analysis only."

    # -----------------------------------------------------------------------
    # Phase 3: Fix (local model)
    # -----------------------------------------------------------------------

    async def _phase_fix(
        self,
        provider: str,
        model: str,
        error_type: str,
        error_message: str,
        diagnosis: dict[str, Any],
        research: str,
        adapter_source: str | None,
    ) -> dict[str, Any]:
        """Local model generates the fix using diagnosis + research findings."""
        prompt_parts = [
            "## Error",
            f"Provider: {provider}, Model: {model}",
            f"Type: {error_type}",
            f"Message: {error_message}",
            "\n## Diagnosis",
            diagnosis.get("diagnosis", "Unknown"),
            "\n## Research Findings (current API docs)",
            research[:6000],
        ]

        if adapter_source:
            truncated = adapter_source[:6000]
            if len(adapter_source) > 6000:
                truncated += "\n\n... (truncated)"
            prompt_parts.append(f"\n## Current Adapter Source\n```python\n{truncated}\n```")

        user_prompt = "\n".join(prompt_parts)

        response = await self._call_model(self.heal_model, FIX_PROMPT, user_prompt)
        result = self._parse_json(response, fallback={
            "fix_type": "no_fix_needed",
            "confidence": 0.1,
            "change_summary": "Could not parse fix response",
            "reasoning": response[:500],
        })
        result["provider"] = provider
        result["analysis"] = diagnosis.get("diagnosis", "")
        result["research_summary"] = research[:1000]
        return result

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    async def _call_model(
        self,
        model_alias: str,
        system_prompt: str,
        user_prompt: str,
        web_search: bool = False,
    ) -> str:
        """Call a model through Aratta's provider system."""
        if not self._get_provider or not self._resolve_model:
            raise RuntimeError("Heal worker not connected to provider system")

        from aratta.core.types import ChatRequest, Message, Role

        provider_name, model_id = self._resolve_model(model_alias)
        provider = self._get_provider(provider_name)

        request = ChatRequest(
            messages=[
                Message(role=Role.SYSTEM, content=system_prompt),
                Message(role=Role.USER, content=user_prompt),
            ],
            model=model_id,
            temperature=0.3,
            max_tokens=3000,
        )

        # Pass web_search hint in metadata for providers that support it
        if web_search:
            request.metadata["web_search"] = True

        response = await provider.chat(request)
        return response.content if hasattr(response, "content") else str(response)

    def _parse_json(self, text: str, fallback: dict[str, Any]) -> dict[str, Any]:
        """Extract JSON from model response, handling markdown code blocks."""
        cleaned = text.strip()
        if "```json" in cleaned:
            start = cleaned.index("```json") + 7
            end = cleaned.index("```", start)
            cleaned = cleaned[start:end].strip()
        elif "```" in cleaned:
            start = cleaned.index("```") + 3
            end = cleaned.index("```", start)
            cleaned = cleaned[start:end].strip()

        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            return fallback

    def get_adapter_source(self, provider: str) -> str | None:
        """Read the current source code of a provider's adapter."""
        if not self._get_provider:
            return None
        try:
            adapter = self._get_provider(provider)
            return inspect.getsource(type(adapter))
        except Exception:
            return None
