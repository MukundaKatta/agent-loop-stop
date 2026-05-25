"""agent-loop-stop: composable stop conditions for LLM agent loops.

Public API:
    after_n_turns(n) -> StopCondition
    cost_exceeds(limit) -> StopCondition
    response_contains(pattern) -> StopCondition
    last_tool_was(name) -> StopCondition
    custom(fn) -> StopCondition
    any_of(*conditions) -> StopCondition
    all_of(*conditions) -> StopCondition
    negate(condition) -> StopCondition
    always() / never() -> StopCondition
    check_all(state, conditions) -> StopResult
    StopCondition   — base class
    StopResult      — diagnostic result
"""

from .core import (
    StopCondition,
    StopResult,
    after_n_turns,
    all_of,
    always,
    any_of,
    check_all,
    cost_exceeds,
    custom,
    last_tool_was,
    negate,
    never,
    response_contains,
)

__all__ = [
    "StopCondition",
    "StopResult",
    "after_n_turns",
    "cost_exceeds",
    "response_contains",
    "last_tool_was",
    "custom",
    "any_of",
    "all_of",
    "negate",
    "always",
    "never",
    "check_all",
]
__version__ = "0.1.0"
