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

---

# Five more, in the harness built to catch them

Wiring up the first real model backend produced five further faults — in the
evaluation harness itself. Each would have been reported as a model result.

None were found by reading code. All five came from reading action traces.

## 6. An empty reply that meant "budget too small"

The local model always reasons before answering; neither `enable_thinking:
false` nor a `/no_think` suffix stops it. At `max_tokens: 100` the reasoning
consumed the whole budget and the reply came back `content: ""` with
`finish_reason: "length"`.

An empty string is indistinguishable from a model with nothing to say. Scored
naively it is a model failure — of a model that was mid-sentence.

**Fix:** `HARNESS_TOKEN_BUDGET`, and a floor on `max_tokens` that refuses to
construct the backend below it. Filed *ambiguous*, not a definite harness fault:
too small a budget is the harness's fault, a model that reasons forever is the
model's, and one truncated reply cannot distinguish them.

## 7. A correct answer in the wrong shape

Asked to click the same button, the model replied:

```
{"action": "click", "x": 600, "y": 480}      constrained decoding
<click x="600" y="480"/>                     unconstrained
```

Same grounding. Same pixel. A strict JSON parser scores the second as a failure
for a model that located the target perfectly.

**Fix:** the parser tries several readings and records which one was needed.
"Required the fallback path" is a finding about a backend; "produced garbage" is
a different finding, and they must not be the same number.

## 8. An invented action, charged to the harness

If a model requests an action the executor does not implement, the executor
raises `UnhandledActionError` → `HARNESS_UNIMPLEMENTED` → the run is
**contaminated**. Correct when the harness is genuinely missing a capability.

But the model is *given* its vocabulary in the system prompt. A model inventing
a verb outside that list is making a model error — and routing it through the
executor would have inflated the exact contamination number this repo reports.

**Fix:** the parser rejects out-of-vocabulary actions as `MODEL_UNPARSEABLE`
before they reach the executor. Tolerant about format, strict about vocabulary.

## 9. Scoring the end state, when the model does not stop

The worst of the five, and the one that produced a wrong published number.

```
click        {"x":500,"y":480}                  focuses the field
type_text_at {"x":500,"y":480,"text":"solara"}  TASK COMPLETE, turn 2
triple_click {"x":500,"y":480}
type_text_at {"x":500,"y":480,"text":"solara"}
click        {"x":500,"y":480}
press_key    {"key":"a"}                        overwrites its own answer
```

The criterion was evaluated once, at the end of the run. So *never solved it*
and *solved it then undid it* produced the same result. The model did the task
on turn 2; the harness recorded `turn_limit` and the suite reported **83.3%**.
Scored continuously, the same behaviour is **91.7%**.

This is not a rare edge case. In a full run, **10 of 12 solved tasks kept
acting after success** — so under end-state scoring, every one of those scores
is decided by whatever the model happened to do on a turn it should never have
taken. The flailing is not harmless. It is a coin flip on the result.

**Fix:** evaluate after every action; record `satisfied_at_turn` and
`still_satisfied_at_end` separately. Reaching the goal is a pass; leaving it
again is a real but *different* failure, reported as `regressed`.

## 10. The obvious fix, wrong in the same shape

The natural repair for #9 is to end the episode the moment the criterion passes.
It scores correctly. It is still wrong.

Stopping at success removes the model's opportunity to declare itself finished,
so every episode terminates by harness intervention — and "solved it" becomes
indistinguishable from "solved it and knew it". A visible error traded for an
invisible one.

Measured properly, that signal is not incidental: the model solved 11 of 12
tasks and recognised completion in **2** of them. It stopped correctly on the
*impossible* task — it is better at knowing it cannot proceed than at knowing it
is done.

**Fix:** let the episode run to its natural end and record `stopped_by`.

---

## What the second five have in common with the first five

The original bugs were written to make an agent robust. These were written to
make a benchmark clean — one number per run, evaluated once, strictly parsed,
stopping promptly.

Both sets convert a loud failure into a quiet wrong answer, and every one of
them lands on the *model's* side of the ledger. A harness has no privileged view
of itself, and the default direction of its errors is not neutral: it flatters
the instrument and blames the subject.

The only defence found here that actually worked was instrumenting every action
and reading the traces.
