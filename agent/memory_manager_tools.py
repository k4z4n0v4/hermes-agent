"""Memory-provider tool-schema plumbing: the gate that decides whether external memory-provider
tools are exposed on an agent's tool surface, and the injection of their schemas."""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def normalize_tool_schema(schema: Any) -> Optional[dict[str, Any]]:
    """Return a bare function-tool dict with a resolvable top-level ``name``, else None.

    Providers should return ``{"name", "description", "parameters"}`` but some return the
    wrapped OpenAI form; wrapping that twice yields a nameless ``function`` and strict
    providers (DeepSeek) reject the ENTIRE request, so both shapes are normalized here.
    """
    if not isinstance(schema, dict):
        return None
    if schema.get("type") == "function" and isinstance(schema.get("function"), dict):
        schema = schema["function"]
    name = schema.get("name", "")
    return schema if name and isinstance(name, str) else None


def memory_provider_tools_enabled(
    enabled_toolsets: Optional[list[str]],
    disabled_toolsets: Optional[list[str]] = None,
    *,
    memory_tool_present: bool = False,
    has_provider: bool = False,
) -> bool:
    """Return whether external memory-provider tools should be exposed.

    The gate honours several signals, in order of precedence:

    1. ``disabled_toolsets`` (hard kill) — if ``"memory"`` is explicitly
       listed, provider tools are suppressed regardless of other settings.
       This preserves the documented global-suppression contract
       (configuration.md §disabled_toolsets).
    2. ``memory_tool_present`` — a memory tool is already on the surface,
       so skip redundant re-injection.
    3. A *truly empty* ``enabled_toolsets`` (e.g. ``platform_toolsets:
       telegram: []``) suppresses injection, per #5544's low-latency
       contract — an empty platform toolset must not leak memory tools.
    4. ``enabled_toolsets`` (opt-in) — if ``"memory"`` is present (directly
       or via a composite toolset), provider tools are exposed.
    5. **Provider configured** — if a memory provider is active
       (``has_provider=True``) and memory was *not* hard-disabled, provider
       tools are exposed even when ``"memory"`` is absent from a *populated*
       ``enabled_toolsets``.  This fixes issue #46108: removing the legacy
       ``memory`` toolset from a platform's toolset list should not silently
       suppress external provider tools (Honcho, Mnemosyne, etc.) that the
       user explicitly configured via ``memory.provider``.

    ``enabled_toolsets is None`` means "no filter" (backward compat) and
    always allows injection unless hard-disabled.
    """
    # 1. Hard kill takes precedence.
    if disabled_toolsets and "memory" in disabled_toolsets:
        return False
    if memory_tool_present:
        return True
    # 2. No filter → allow (backward compat).
    if enabled_toolsets is None:
        return True
    # 3. Truly empty list (e.g. ``platform_toolsets: telegram: []``) →
    #    suppress.  Preserves the #5544 low-latency contract (empty platform
    #    toolsets must not leak memory tools).  #46108 only relaxes the
    #    *populated*-but-no-memory case below, not this hard-empty case.
    if not enabled_toolsets:
        return False
    # 4. "memory" in enabled toolsets → allow.
    if "memory" in enabled_toolsets:
        return True

    # 5. Composite toolset that resolves to "memory" → allow.
    try:
        from toolsets import resolve_toolset

        if any("memory" in resolve_toolset(name) for name in enabled_toolsets):
            return True
    except Exception:
        logger.debug("Failed to resolve enabled toolsets for memory-provider tools", exc_info=True)

    # 6. Provider configured but memory toolset simply not included → allow.
    #    (the #46108 fix)
    if has_provider:
        return True

    return False


def _tool_name(tool: Any) -> Any:
    return tool.get("function", {}).get("name") if isinstance(tool, dict) else None


def memory_provider_tools_exposed(agent: Any) -> bool:
    """Whether external memory-provider tools are exposed on ``agent``.

    Same gate as ``inject_memory_provider_tools`` so a provider's ``system_prompt_block()``
    never advertises tools absent from the tool surface.
    """
    tools = getattr(agent, "tools", None)
    present = isinstance(tools, (list, tuple)) and any(_tool_name(t) == "memory" for t in tools)
    enabled, disabled = getattr(agent, "enabled_toolsets", None), getattr(agent, "disabled_toolsets", None)
    return memory_provider_tools_enabled(
        enabled,
        disabled,
        memory_tool_present=present,
        has_provider=bool(getattr(getattr(agent, "_memory_manager", None), "providers", None)),
    )


def inject_memory_provider_tools(agent: Any) -> int:
    """Append external memory-provider tool schemas to an agent tool surface; return count added."""
    memory_manager = getattr(agent, "_memory_manager", None)
    tools = getattr(agent, "tools", None)
    if not memory_manager or tools is None:
        return 0

    if not memory_provider_tools_exposed(agent):
        # Say so once: a silent 0 leaves the provider looking "half on" with no clue which
        # config key (platform_toolsets / disabled_toolsets) gated it.
        # See #81014.
        _providers = [p for p in getattr(memory_manager, "providers", None) or []
                      if getattr(p, "name", "") != "builtin"]
        if _providers:
            logger.info(
                "Memory provider(s) %s configured but the 'memory' toolset is "
                "gated off for this session (platform_toolsets / "
                "agent.disabled_toolsets) — provider tools and system-prompt "
                "block are both withheld.",
                [getattr(p, "name", type(p).__name__) for p in _providers],
            )
        return 0

    get_schemas = getattr(memory_manager, "get_all_tool_schemas", None)
    if not callable(get_schemas):
        return 0

    if getattr(agent, "valid_tool_names", None) is None:
        agent.valid_tool_names = set()
    existing_tool_names = {_tool_name(tool) for tool in tools if isinstance(tool, dict)}
    added = 0
    for raw_schema in get_schemas():
        schema = normalize_tool_schema(raw_schema)
        if schema is None:
            logger.warning(
                "Memory provider returned a tool schema with no resolvable "
                "name; skipping to avoid poisoning the request (%r)", raw_schema,
            )
        elif schema["name"] not in existing_tool_names:
            tools.append({"type": "function", "function": schema})
            agent.valid_tool_names.add(schema["name"])
            existing_tool_names.add(schema["name"])
            added += 1
    return added
