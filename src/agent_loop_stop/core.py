"""Composable stop conditions for LLM agent loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# StopCondition base
# ---------------------------------------------------------------------------

class StopCondition:
    """Base class for agent loop stop conditions.

    Subclass and implement ``check(state) -> bool``, or use the provided
    factory functions to compose conditions.

    Usage::

        stopper = any_of(
            after_n_turns(20),
            cost_exceeds(2.00),
            response_contains("FINAL ANSWER"),
        )

        for turn in agent_loop():
            state = {"turn": turn, "response": response.text, "cost_usd": 0.05 * turn}
            if stopper.check(state):
                break
    """

    def check(self, state: dict[str, Any]) -> bool:
        """Return True if the loop should stop."""
        raise NotImplementedError

    # Composition operators
    def __or__(self, other: "StopCondition") -> "StopCondition":
        return _AnyOf([self, other])

    def __and__(self, other: "StopCondition") -> "StopCondition":
        return _AllOf([self, other])

    def __invert__(self) -> "StopCondition":
        return _Not(self)


# ---------------------------------------------------------------------------
# Composite conditions
# ---------------------------------------------------------------------------

class _AnyOf(StopCondition):
    def __init__(self, conditions: list[StopCondition]) -> None:
        self._conditions = conditions

    def check(self, state: dict[str, Any]) -> bool:
        return any(c.check(state) for c in self._conditions)


class _AllOf(StopCondition):
    def __init__(self, conditions: list[StopCondition]) -> None:
        self._conditions = conditions

    def check(self, state: dict[str, Any]) -> bool:
        return all(c.check(state) for c in self._conditions)


class _Not(StopCondition):
    def __init__(self, condition: StopCondition) -> None:
        self._condition = condition

    def check(self, state: dict[str, Any]) -> bool:
        return not self._condition.check(state)


# ---------------------------------------------------------------------------
# Concrete conditions
# ---------------------------------------------------------------------------

class _AfterNTurns(StopCondition):
    def __init__(self, n: int, key: str) -> None:
        self._n = n
        self._key = key

    def check(self, state: dict[str, Any]) -> bool:
        val = state.get(self._key)
        if val is None:
            return False
        try:
            return int(val) >= self._n
        except (TypeError, ValueError):
            return False


class _CostExceeds(StopCondition):
    def __init__(self, limit: float, keys: tuple[str, ...]) -> None:
        self._limit = limit
        self._keys = keys

    def check(self, state: dict[str, Any]) -> bool:
        for key in self._keys:
            val = state.get(key)
            if val is not None:
                try:
                    return float(val) > self._limit
                except (TypeError, ValueError):
                    pass
        return False


class _ResponseContains(StopCondition):
    def __init__(self, pattern: str, key: str, case_sensitive: bool) -> None:
        self._key = key
        self._pattern = pattern if case_sensitive else pattern.lower()
        self._case_sensitive = case_sensitive

    def check(self, state: dict[str, Any]) -> bool:
        text = state.get(self._key)
        if text is None:
            return False
        text = str(text)
        if not self._case_sensitive:
            text = text.lower()
        return self._pattern in text


class _LastToolWas(StopCondition):
    def __init__(self, name: str, key: str) -> None:
        self._name = name
        self._key = key

    def check(self, state: dict[str, Any]) -> bool:
        val = state.get(self._key)
        return str(val) == self._name if val is not None else False


class _CustomCondition(StopCondition):
    def __init__(self, fn: Callable[[dict[str, Any]], bool]) -> None:
        self._fn = fn

    def check(self, state: dict[str, Any]) -> bool:
        return bool(self._fn(state))


class _Always(StopCondition):
    def check(self, state: dict[str, Any]) -> bool:
        return True


class _Never(StopCondition):
    def check(self, state: dict[str, Any]) -> bool:
        return False


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def after_n_turns(n: int, *, key: str = "turn") -> StopCondition:
    """Stop when state[key] >= n.

    Args:
        n: turn threshold.
        key: state dict key holding the current turn number. Default "turn".
    """
    return _AfterNTurns(n, key)


def cost_exceeds(
    limit: float,
    *,
    keys: tuple[str, ...] = ("cost_usd", "cost", "total_cost"),
) -> StopCondition:
    """Stop when the cumulative cost in state exceeds limit.

    Args:
        limit: USD cost threshold.
        keys: state dict keys to check for cost value.
    """
    return _CostExceeds(limit, keys)


def response_contains(
    pattern: str,
    *,
    key: str = "response",
    case_sensitive: bool = False,
) -> StopCondition:
    """Stop when state[key] contains pattern.

    Args:
        pattern: substring to search for.
        key: state dict key holding the response text. Default "response".
        case_sensitive: if False (default), match is case-insensitive.
    """
    return _ResponseContains(pattern, key, case_sensitive)


def last_tool_was(name: str, *, key: str = "last_tool") -> StopCondition:
    """Stop when state[key] equals name.

    Args:
        name: expected tool name.
        key: state dict key. Default "last_tool".
    """
    return _LastToolWas(name, key)


def custom(fn: Callable[[dict[str, Any]], bool]) -> StopCondition:
    """Stop when fn(state) returns True."""
    return _CustomCondition(fn)


def always() -> StopCondition:
    """Always stop (useful for testing)."""
    return _Always()


def never() -> StopCondition:
    """Never stop (useful as a no-op placeholder)."""
    return _Never()


def any_of(*conditions: StopCondition) -> StopCondition:
    """Stop when any condition is True (logical OR)."""
    return _AnyOf(list(conditions))


def all_of(*conditions: StopCondition) -> StopCondition:
    """Stop when all conditions are True (logical AND)."""
    return _AllOf(list(conditions))


def negate(condition: StopCondition) -> StopCondition:
    """Invert a stop condition."""
    return _Not(condition)


# ---------------------------------------------------------------------------
# StopResult (for diagnostic use)
# ---------------------------------------------------------------------------

@dataclass
class StopResult:
    """Diagnostic result from checking all conditions individually."""

    stopped: bool
    triggered: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.stopped


def check_all(
    state: dict[str, Any],
    conditions: dict[str, StopCondition],
) -> StopResult:
    """Check named conditions individually and return a StopResult.

    Args:
        state: the current agent loop state dict.
        conditions: mapping of name → StopCondition.

    Returns:
        StopResult with stopped=True if any condition triggered, and
        triggered listing which condition names fired.
    """
    triggered: list[str] = []
    for name, cond in conditions.items():
        if cond.check(state):
            triggered.append(name)
    return StopResult(stopped=bool(triggered), triggered=triggered)
