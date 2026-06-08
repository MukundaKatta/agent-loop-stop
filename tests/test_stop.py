"""Tests for agent-loop-stop.

These tests use only the Python standard library (``unittest``) so they
run with no third-party dependencies::

    python3 -m unittest discover -s tests

They are also discoverable by ``pytest`` (which understands ``unittest``
TestCase classes), so the same file works under both runners.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from agent_loop_stop import (  # noqa: E402
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


class AfterNTurnsTests(unittest.TestCase):
    def test_not_reached(self):
        c = after_n_turns(10)
        self.assertFalse(c.check({"turn": 9}))

    def test_at_limit(self):
        c = after_n_turns(10)
        self.assertTrue(c.check({"turn": 10}))

    def test_past_limit(self):
        c = after_n_turns(10)
        self.assertTrue(c.check({"turn": 15}))

    def test_missing_key(self):
        c = after_n_turns(10)
        self.assertFalse(c.check({}))

    def test_custom_key(self):
        c = after_n_turns(5, key="iteration")
        self.assertTrue(c.check({"iteration": 5}))
        self.assertFalse(c.check({"iteration": 4}))

    def test_string_turn_is_coerced(self):
        # Numeric strings should be coerced to int.
        c = after_n_turns(10)
        self.assertTrue(c.check({"turn": "12"}))
        self.assertFalse(c.check({"turn": "8"}))

    def test_non_numeric_turn_is_safe(self):
        # A garbage value must not raise; it should simply not trigger.
        c = after_n_turns(10)
        self.assertFalse(c.check({"turn": "not-a-number"}))


class CostExceedsTests(unittest.TestCase):
    def test_not_reached(self):
        c = cost_exceeds(1.0)
        self.assertFalse(c.check({"cost_usd": 0.99}))

    def test_at_limit_is_strictly_greater(self):
        c = cost_exceeds(1.0)
        self.assertFalse(c.check({"cost_usd": 1.0}))

    def test_over(self):
        c = cost_exceeds(1.0)
        self.assertTrue(c.check({"cost_usd": 1.01}))

    def test_fallback_key_cost(self):
        c = cost_exceeds(0.5)
        self.assertTrue(c.check({"cost": 0.6}))

    def test_total_cost_key(self):
        c = cost_exceeds(2.0)
        self.assertTrue(c.check({"total_cost": 2.5}))

    def test_missing_key(self):
        c = cost_exceeds(1.0)
        self.assertFalse(c.check({}))

    def test_first_present_key_wins(self):
        # cost_usd is checked before total_cost; the first present key is
        # authoritative even if a later key would exceed the limit.
        c = cost_exceeds(1.0)
        self.assertFalse(c.check({"cost_usd": 0.1, "total_cost": 99.0}))

    def test_invalid_value_falls_through(self):
        # A non-numeric value on the first key should be skipped so a
        # valid later key can still trigger.
        c = cost_exceeds(0.5)
        self.assertTrue(c.check({"cost_usd": "oops", "total_cost": 0.9}))

    def test_custom_keys(self):
        c = cost_exceeds(1.0, keys=("spend",))
        self.assertTrue(c.check({"spend": 1.5}))
        self.assertFalse(c.check({"cost_usd": 99.0}))  # not in custom keys


class ResponseContainsTests(unittest.TestCase):
    def test_match(self):
        c = response_contains("FINAL ANSWER")
        self.assertTrue(
            c.check({"response": "Here is the FINAL ANSWER to your question."})
        )

    def test_no_match(self):
        c = response_contains("FINAL ANSWER")
        self.assertFalse(c.check({"response": "Still working..."}))

    def test_case_insensitive_default(self):
        c = response_contains("final answer")
        self.assertTrue(c.check({"response": "FINAL ANSWER: done"}))

    def test_case_sensitive(self):
        c = response_contains("FINAL", case_sensitive=True)
        self.assertFalse(c.check({"response": "final"}))
        self.assertTrue(c.check({"response": "FINAL"}))

    def test_custom_key(self):
        c = response_contains("done", key="text")
        self.assertTrue(c.check({"text": "All done"}))

    def test_missing_key(self):
        c = response_contains("done")
        self.assertFalse(c.check({}))

    def test_non_string_value_is_stringified(self):
        # A non-string response should be coerced to str before matching.
        c = response_contains("42")
        self.assertTrue(c.check({"response": 1422}))


class LastToolWasTests(unittest.TestCase):
    def test_match(self):
        c = last_tool_was("write_summary")
        self.assertTrue(c.check({"last_tool": "write_summary"}))

    def test_no_match(self):
        c = last_tool_was("write_summary")
        self.assertFalse(c.check({"last_tool": "web_search"}))

    def test_missing(self):
        c = last_tool_was("write_summary")
        self.assertFalse(c.check({}))

    def test_custom_key(self):
        c = last_tool_was("done_tool", key="tool")
        self.assertTrue(c.check({"tool": "done_tool"}))


class AlwaysNeverTests(unittest.TestCase):
    def test_always_returns_true(self):
        self.assertTrue(always().check({}))
        self.assertTrue(always().check({"turn": 99}))

    def test_never_returns_false(self):
        self.assertFalse(never().check({}))
        self.assertFalse(never().check({"turn": 99}))


class CustomTests(unittest.TestCase):
    def test_simple(self):
        c = custom(lambda s: s.get("done") is True)
        self.assertTrue(c.check({"done": True}))
        self.assertFalse(c.check({"done": False}))
        self.assertFalse(c.check({}))

    def test_complex_logic(self):
        c = custom(lambda s: s.get("turn", 0) > 5 and s.get("cost_usd", 0) > 0.5)
        self.assertTrue(c.check({"turn": 6, "cost_usd": 0.6}))
        self.assertFalse(c.check({"turn": 4, "cost_usd": 0.6}))

    def test_result_is_coerced_to_bool(self):
        # A truthy non-bool return value should become True.
        c = custom(lambda s: s.get("items", []))
        result = c.check({"items": [1, 2, 3]})
        self.assertIs(result, True)
        self.assertIs(c.check({"items": []}), False)


class AnyOfTests(unittest.TestCase):
    def test_first_true(self):
        c = any_of(after_n_turns(5), cost_exceeds(1.0))
        self.assertTrue(c.check({"turn": 5, "cost_usd": 0.1}))

    def test_second_true(self):
        c = any_of(after_n_turns(10), cost_exceeds(0.5))
        self.assertTrue(c.check({"turn": 3, "cost_usd": 0.6}))

    def test_none_true(self):
        c = any_of(after_n_turns(10), cost_exceeds(1.0))
        self.assertFalse(c.check({"turn": 3, "cost_usd": 0.3}))

    def test_empty_any_of_is_false(self):
        # any() over an empty iterable is False.
        self.assertFalse(any_of().check({}))


class AllOfTests(unittest.TestCase):
    def test_both_true(self):
        c = all_of(after_n_turns(5), cost_exceeds(0.5))
        self.assertTrue(c.check({"turn": 6, "cost_usd": 0.6}))

    def test_first_false(self):
        c = all_of(after_n_turns(10), cost_exceeds(0.5))
        self.assertFalse(c.check({"turn": 3, "cost_usd": 0.6}))

    def test_second_false(self):
        c = all_of(after_n_turns(5), cost_exceeds(1.0))
        self.assertFalse(c.check({"turn": 6, "cost_usd": 0.3}))

    def test_none_true(self):
        c = all_of(never(), never())
        self.assertFalse(c.check({}))

    def test_empty_all_of_is_true(self):
        # all() over an empty iterable is True.
        self.assertTrue(all_of().check({}))


class NegateTests(unittest.TestCase):
    def test_true_becomes_false(self):
        self.assertFalse(negate(always()).check({}))

    def test_false_becomes_true(self):
        self.assertTrue(negate(never()).check({}))

    def test_after_n_turns(self):
        c = negate(after_n_turns(10))
        self.assertTrue(c.check({"turn": 5}))
        self.assertFalse(c.check({"turn": 10}))


class OperatorOverloadTests(unittest.TestCase):
    def test_or(self):
        c = after_n_turns(5) | cost_exceeds(1.0)
        self.assertTrue(c.check({"turn": 5}))
        self.assertTrue(c.check({"cost_usd": 1.5}))
        self.assertFalse(c.check({"turn": 3, "cost_usd": 0.3}))

    def test_and(self):
        c = after_n_turns(5) & cost_exceeds(0.5)
        self.assertTrue(c.check({"turn": 5, "cost_usd": 0.6}))
        self.assertFalse(c.check({"turn": 5, "cost_usd": 0.1}))

    def test_invert(self):
        c = ~after_n_turns(10)
        self.assertTrue(c.check({"turn": 5}))
        self.assertFalse(c.check({"turn": 10}))

    def test_chained_operators(self):
        # (turns OR cost) AND NOT(error in response)
        c = (after_n_turns(5) | cost_exceeds(1.0)) & ~response_contains("error")
        self.assertTrue(c.check({"turn": 6, "response": "all good"}))
        self.assertFalse(c.check({"turn": 6, "response": "fatal error"}))
        self.assertFalse(c.check({"turn": 1, "response": "all good"}))


class CheckAllTests(unittest.TestCase):
    def test_none_triggered(self):
        result = check_all(
            {"turn": 3},
            {"turns": after_n_turns(10), "cost": cost_exceeds(1.0)},
        )
        self.assertFalse(result.stopped)
        self.assertEqual(result.triggered, [])

    def test_one_triggered(self):
        result = check_all(
            {"turn": 10},
            {"turns": after_n_turns(10), "cost": cost_exceeds(1.0)},
        )
        self.assertTrue(result.stopped)
        self.assertIn("turns", result.triggered)

    def test_multiple_triggered(self):
        result = check_all(
            {"turn": 15, "cost_usd": 2.0},
            {"turns": after_n_turns(10), "cost": cost_exceeds(1.0)},
        )
        self.assertTrue(result.stopped)
        self.assertEqual(set(result.triggered), {"turns", "cost"})

    def test_triggered_order_follows_dict(self):
        # Insertion order of the conditions dict is preserved.
        result = check_all(
            {"turn": 99, "cost_usd": 99.0},
            {"a": cost_exceeds(1.0), "b": after_n_turns(10)},
        )
        self.assertEqual(result.triggered, ["a", "b"])

    def test_empty_conditions(self):
        result = check_all({}, {})
        self.assertFalse(result.stopped)
        self.assertEqual(result.triggered, [])


class StopResultTests(unittest.TestCase):
    def test_bool(self):
        self.assertTrue(bool(StopResult(stopped=True)))
        self.assertFalse(bool(StopResult(stopped=False)))

    def test_default_triggered_is_empty_list(self):
        self.assertEqual(StopResult(stopped=False).triggered, [])

    def test_distinct_instances_do_not_share_list(self):
        # field(default_factory=list) must give each instance its own list.
        a = StopResult(stopped=True)
        b = StopResult(stopped=True)
        a.triggered.append("x")
        self.assertEqual(b.triggered, [])


class StopConditionBaseTests(unittest.TestCase):
    def test_base_check_raises(self):
        with self.assertRaises(NotImplementedError):
            StopCondition().check({})

    def test_subclassing(self):
        class TokenBudgetStop(StopCondition):
            def check(self, state):
                return state.get("tokens_used", 0) > 4000

        c = TokenBudgetStop()
        self.assertTrue(c.check({"tokens_used": 5000}))
        self.assertFalse(c.check({"tokens_used": 100}))
        # Subclasses inherit the composition operators.
        combined = c | after_n_turns(3)
        self.assertTrue(combined.check({"turn": 3, "tokens_used": 0}))


if __name__ == "__main__":
    unittest.main()
