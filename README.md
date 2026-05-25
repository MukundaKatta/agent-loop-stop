# agent-loop-stop

Composable stop conditions for LLM agent loops.

Zero dependencies. Python 3.10+. MIT.

## Install

```bash
pip install agent-loop-stop
```

## Usage

```python
from agent_loop_stop import any_of, after_n_turns, cost_exceeds, response_contains

stopper = any_of(
    after_n_turns(20),
    cost_exceeds(2.00),
    response_contains("FINAL ANSWER"),
)

for turn in range(1, 100):
    response = call_llm(messages)
    state = {
        "turn": turn,
        "cost_usd": running_cost,
        "response": response.text,
    }
    if stopper.check(state):
        break
```

## Built-in conditions

```python
after_n_turns(20)                          # turn >= 20
cost_exceeds(2.00)                         # cost_usd > 2.00
response_contains("FINAL ANSWER")         # case-insensitive substring match
last_tool_was("write_summary")            # last_tool == name
custom(lambda s: s["error_count"] > 3)   # any callable
always()                                  # always True
never()                                   # always False
```

## Compose with operators

```python
# OR: stop when either fires
c = after_n_turns(20) | cost_exceeds(1.00)

# AND: stop when both fire
c = after_n_turns(10) & cost_exceeds(0.50)

# NOT: invert
c = ~response_contains("error")
```

Or use functions:

```python
any_of(after_n_turns(20), cost_exceeds(2.00), response_contains("done"))
all_of(after_n_turns(5), cost_exceeds(0.25))
negate(response_contains("continue"))
```

## Diagnostic check (which conditions fired?)

```python
from agent_loop_stop import check_all

result = check_all(
    state,
    {
        "turn_limit": after_n_turns(20),
        "cost_limit": cost_exceeds(2.00),
        "done_signal": response_contains("FINAL ANSWER"),
    },
)
if result.stopped:
    print(f"Stopped by: {result.triggered}")  # ["turn_limit"]
```

## Custom condition

```python
c = custom(lambda s: len(s.get("tool_calls", [])) > 50)
```

Or subclass:

```python
class TokenBudgetStop(StopCondition):
    def check(self, state):
        return state.get("tokens_used", 0) > 4000
```

## State dict

Pass whatever you want in the state dict. Built-in conditions read these keys:

| Condition | Default key | Override with |
|-----------|------------|---------------|
| `after_n_turns` | `turn` | `key=` |
| `cost_exceeds` | `cost_usd`, `cost`, `total_cost` | `keys=` |
| `response_contains` | `response` | `key=` |
| `last_tool_was` | `last_tool` | `key=` |

## License

MIT
