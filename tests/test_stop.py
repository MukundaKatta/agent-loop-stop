"""Tests for agent-loop-stop."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import pytest
from agent_loop_stop import (
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


# ---------------------------------------------------------------------------
# after_n_turns
# ---------------------------------------------------------------------------

def test_after_n_turns_not_reached():
    c = after_n_turns(10)
    assert c.check({"turn": 9}) is False


def test_after_n_turns_at_limit():
    c = after_n_turns(10)
    assert c.check({"turn": 10}) is True


def test_after_n_turns_past_limit():
    c = after_n_turns(10)
    assert c.check({"turn": 15}) is True


def test_after_n_turns_missing_key():
    c = after_n_turns(10)
    assert c.check({}) is False


def test_after_n_turns_custom_key():
    c = after_n_turns(5, key="iteration")
    assert c.check({"iteration": 5}) is True
    assert c.check({"iteration": 4}) is False


# ---------------------------------------------------------------------------
# cost_exceeds
# ---------------------------------------------------------------------------

def test_cost_exceeds_not_reached():
    c = cost_exceeds(1.0)
    assert c.check({"cost_usd": 0.99}) is False


def test_cost_exceeds_at_limit():
    c = cost_exceeds(1.0)
    # strictly greater than
    assert c.check({"cost_usd": 1.0}) is False


def test_cost_exceeds_over():
    c = cost_exceeds(1.0)
    assert c.check({"cost_usd": 1.01}) is True


def test_cost_exceeds_fallback_key():
    c = cost_exceeds(0.5)
    assert c.check({"cost": 0.6}) is True


def test_cost_exceeds_missing_key():
    c = cost_exceeds(1.0)
    assert c.check({}) is False


def test_cost_exceeds_total_cost_key():
    c = cost_exceeds(2.0)
    assert c.check({"total_cost": 2.5}) is True


# ---------------------------------------------------------------------------
# response_contains
# ---------------------------------------------------------------------------

def test_response_contains_match():
    c = response_contains("FINAL ANSWER")
    assert c.check({"response": "Here is the FINAL ANSWER to your question."}) is True


def test_response_contains_no_match():
    c = response_contains("FINAL ANSWER")
    assert c.check({"response": "Still working..."}) is False


def test_response_contains_case_insensitive():
    c = response_contains("final answer")
    assert c.check({"response": "FINAL ANSWER: done"}) is True


def test_response_contains_case_sensitive():
    c = response_contains("FINAL", case_sensitive=True)
    assert c.check({"response": "final"}) is False
    assert c.check({"response": "FINAL"}) is True


def test_response_contains_custom_key():
    c = response_contains("done", key="text")
    assert c.check({"text": "All done"}) is True


def test_response_contains_missing_key():
    c = response_contains("done")
    assert c.check({}) is False


# ---------------------------------------------------------------------------
# last_tool_was
# ---------------------------------------------------------------------------

def test_last_tool_was_match():
    c = last_tool_was("write_summary")
    assert c.check({"last_tool": "write_summary"}) is True


def test_last_tool_was_no_match():
    c = last_tool_was("write_summary")
    assert c.check({"last_tool": "web_search"}) is False


def test_last_tool_was_missing():
    c = last_tool_was("write_summary")
    assert c.check({}) is False


def test_last_tool_was_custom_key():
    c = last_tool_was("done_tool", key="tool")
    assert c.check({"tool": "done_tool"}) is True


# ---------------------------------------------------------------------------
# always / never
# ---------------------------------------------------------------------------

def test_always_returns_true():
    assert always().check({}) is True
    assert always().check({"turn": 99}) is True


def test_never_returns_false():
    assert never().check({}) is False
    assert never().check({"turn": 99}) is False


# ---------------------------------------------------------------------------
# custom
# ---------------------------------------------------------------------------

def test_custom_fn():
    c = custom(lambda s: s.get("done") is True)
    assert c.check({"done": True}) is True
    assert c.check({"done": False}) is False
    assert c.check({}) is False


def test_custom_fn_with_complex_logic():
    c = custom(lambda s: s.get("turn", 0) > 5 and s.get("cost_usd", 0) > 0.5)
    assert c.check({"turn": 6, "cost_usd": 0.6}) is True
    assert c.check({"turn": 4, "cost_usd": 0.6}) is False


# ---------------------------------------------------------------------------
# any_of
# ---------------------------------------------------------------------------

def test_any_of_first_true():
    c = any_of(after_n_turns(5), cost_exceeds(1.0))
    assert c.check({"turn": 5, "cost_usd": 0.1}) is True


def test_any_of_second_true():
    c = any_of(after_n_turns(10), cost_exceeds(0.5))
    assert c.check({"turn": 3, "cost_usd": 0.6}) is True


def test_any_of_none_true():
    c = any_of(after_n_turns(10), cost_exceeds(1.0))
    assert c.check({"turn": 3, "cost_usd": 0.3}) is False


def test_any_of_all_true():
    c = any_of(always(), always())
    assert c.check({}) is True


# ---------------------------------------------------------------------------
# all_of
# ---------------------------------------------------------------------------

def test_all_of_both_true():
    c = all_of(after_n_turns(5), cost_exceeds(0.5))
    assert c.check({"turn": 6, "cost_usd": 0.6}) is True


def test_all_of_first_false():
    c = all_of(after_n_turns(10), cost_exceeds(0.5))
    assert c.check({"turn": 3, "cost_usd": 0.6}) is False


def test_all_of_second_false():
    c = all_of(after_n_turns(5), cost_exceeds(1.0))
    assert c.check({"turn": 6, "cost_usd": 0.3}) is False


def test_all_of_none_true():
    c = all_of(never(), never())
    assert c.check({}) is False


# ---------------------------------------------------------------------------
# negate
# ---------------------------------------------------------------------------

def test_negate_true_becomes_false():
    c = negate(always())
    assert c.check({}) is False


def test_negate_false_becomes_true():
    c = negate(never())
    assert c.check({}) is True


def test_negate_after_n_turns():
    c = negate(after_n_turns(10))
    assert c.check({"turn": 5}) is True
    assert c.check({"turn": 10}) is False


# ---------------------------------------------------------------------------
# Operator overloads
# ---------------------------------------------------------------------------

def test_or_operator():
    c = after_n_turns(5) | cost_exceeds(1.0)
    assert c.check({"turn": 5}) is True
    assert c.check({"cost_usd": 1.5}) is True
    assert c.check({"turn": 3, "cost_usd": 0.3}) is False


def test_and_operator():
    c = after_n_turns(5) & cost_exceeds(0.5)
    assert c.check({"turn": 5, "cost_usd": 0.6}) is True
    assert c.check({"turn": 5, "cost_usd": 0.1}) is False


def test_invert_operator():
    c = ~after_n_turns(10)
    assert c.check({"turn": 5}) is True
    assert c.check({"turn": 10}) is False


# ---------------------------------------------------------------------------
# check_all
# ---------------------------------------------------------------------------

def test_check_all_none_triggered():
    result = check_all(
        {"turn": 3},
        {"turns": after_n_turns(10), "cost": cost_exceeds(1.0)},
    )
    assert result.stopped is False
    assert result.triggered == []


def test_check_all_one_triggered():
    result = check_all(
        {"turn": 10},
        {"turns": after_n_turns(10), "cost": cost_exceeds(1.0)},
    )
    assert result.stopped is True
    assert "turns" in result.triggered


def test_check_all_multiple_triggered():
    result = check_all(
        {"turn": 15, "cost_usd": 2.0},
        {"turns": after_n_turns(10), "cost": cost_exceeds(1.0)},
    )
    assert result.stopped is True
    assert set(result.triggered) == {"turns", "cost"}


def test_stop_result_bool():
    assert bool(StopResult(stopped=True)) is True
    assert bool(StopResult(stopped=False)) is False
