# Five silent failures in a working computer-use agent

Found by auditing `solara_agent.py`, a functioning voice-driven browser agent
that had been in use. Every bug was confirmed by *executing the original code*,
not by reading it — see `scripts/verify_legacy_bugs.py`.

```
CONFIRMED BUG  triple_click performs no action at all
CONFIRMED BUG  click with button="right" performs a LEFT click
CONFIRMED BUG  malformed coordinate silently clicks x=0
CONFIRMED BUG  unknown scroll direction issues wheel(0, 0)
CONFIRMED BUG  unknown action warns and returns as if successful

5/5 bugs confirmed present in the original code
```

**They are all the same bug.** Not "the agent crashed" — *the agent reported
success while doing nothing, or the wrong thing*. The model's only feedback is
the screenshot it receives next turn. An action that silently does nothing is
indistinguishable from one that worked, so the model burns turns retrying
something that never ran, and the task failure is attributed to its reasoning.

---

## 1. `triple_click` was a no-op

`triple_click` appeared in the tuple guarding the mouse-action branch, but had no
inner branch of its own. It fell through the entire `if/elif` chain.

```python
elif fname in ("click", "click_at", "double_click", "triple_click",
               "middle_click", "right_click", "move"):
    if fname in ("click", "click_at"):   ...
    elif fname == "double_click":        ...
    elif ... right_click ...:            ...
    elif fname == "middle_click":        ...
    elif fname == "move":                ...
    # triple_click: matched the outer guard, matched no inner branch
```

No action, no warning, no error. Any task requiring a paragraph selection failed,
and looked like the model didn't understand selection.

**Fix:** `page.mouse.click(x, y, click_count=3)` — verified against
`playwright.sync_api.Mouse.click`, which accepts `click_count`.

## 2. `click` with `button="right"` performed a *left* click

```python
if fname in ("click", "click_at"):
    page.mouse.click(actual_x, actual_y)        # <-- catches it here, no button
...
elif fname in ("click", "click_at") and args.get("button") == "right" or fname == "right_click":
    page.mouse.click(actual_x, actual_y, button="right")
```

Two defects stacked. The first branch catches `click` unconditionally, so the
button-aware branch is unreachable for it. And the condition parses as
`(A and B) or C` — `right_click` only worked by falling through to `C`.

A context-menu task would left-click, dismiss nothing, screenshot an unchanged
page, and report success.

**Fix:** explicit dispatch where the action name sets the button, falling back to
`args["button"]`, defaulting to `left`.

## 3. A malformed coordinate silently clicked the origin

```python
def denormalize_x(x, width):
    try:
        return int(float(x) / 1000 * width)
    except Exception:
        return 0        # <-- clicks the top-left corner
```

The worst of the five, because `(0, 0)` is a *plausible* click. It lands
somewhere real, often on a logo or nav element, and produces a page change the
model then has to explain. There is no error anywhere in the trace.

**Fix:** raise `CoordinateError`. The model receives an error it can act on.

## 4. An unknown scroll direction scrolled nothing

```python
delta_x, delta_y = 0, 0
if direction == "down":    delta_y = mag
elif direction == "up":    delta_y = -mag
elif direction == "right": delta_x = mag
elif direction == "left":  delta_x = -mag
page.mouse.wheel(delta_x, delta_y)      # wheel(0, 0) for anything else
```

**Fix:** raise on any direction not in the table.

## 5. Unknown actions reported success

```python
else:
    print(f"Warning: Custom or unhandled function {fname}")
```

A `print` to stdout, then the action was recorded as completed. In a headless
service nobody reads stdout. Every model capability beyond the implemented set
degraded to a silent no-op.

**Fix:** raise `UnhandledActionError`.

---

## A sixth, structural

`solara_env.py` constructed its API client at import time, so the module could
not be imported without a live API key. That is why the agent had **no tests** —
not neglect, an import-time dependency that made testing impossible. Making the
client lazy is what unblocked all 50 tests in this repo.

Worth stating plainly: the absence of tests was a *symptom*. The cause was a
design decision three lines long.

---

## Why this generalises

A benchmark harness has no privileged view of itself. If it drops actions
silently, it reports a low success rate with total confidence and no anomalies.
The demo in `scripts/demo_offline.py` shows this directly: the buggy executor
scores **0% with zero faults flagged**, four clean `fail_model` verdicts. Three
were its own bugs.

Every silent fallback here — `return 0`, `wheel(0, 0)`, `print` and continue —
was written to make the harness *robust*. Each one converted a loud failure into
a quiet wrong answer.

That is the argument for classifying every action rather than only the outcome.
