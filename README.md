# solara-cua-eval

An evaluation harness for computer-use agents that separates harness faults from model failures.

## Overview

Computer-use benchmarks report a success rate. When a task fails, the failure
is attributed to the model. But a harness that silently drops an action
produces the same result — task not completed — even when the model reasoned
correctly. Unless every action is individually classified, harness bugs and
model failures are indistinguishable, and the harness's mistakes get charged
to the model.

This project provides a framework for evaluating computer-use agents with
transparent failure attribution. It classifies every action into one of three
tiers — definite harness fault, ambiguous, or model-attributable — so that
reported numbers only reflect what they claim to measure.

The approach was validated by auditing a working browser agent: five bugs were
found where the agent reported success while doing nothing or the wrong thing.
Then the evaluation harness itself was tested the same way, producing eight
more faults — all found by reading action traces, not by reading code. See
[`docs/FINDINGS.md`](docs/FINDINGS.md) for the full writeup.

## Quick start

```bash
git clone https://github.com/SOuLcinis/solara-cua-eval.git
cd solara-cua-eval
pip install -r requirements.txt

python -m pytest -q                # 197 tests, no credentials needed
python scripts/demo_offline.py     # demonstrates the scoring, no browser
python scripts/run_suite.py        # full suite with oracle backend, zero cost
```

For browser-backed runs:

```bash
pip install playwright && playwright install chromium
```

For model backends (requires API keys):

```bash
python scripts/run_suite.py --backend local          # local vision server
python scripts/run_suite.py --backend mistral        # MISTRAL_API_KEY required
python scripts/run_suite.py --backend gemini         # GEMINI_API_KEY required
python scripts/run_suite.py --backend gemini --model gemini-2.5-pro  # override model

# cross-invocation comparison
python scripts/compare.py results/inv1.jsonl results/inv2.jsonl results/inv3.jsonl
```

## The problem

Run `python scripts/demo_offline.py` to see two executors score the same four
tasks:

```
legacy-executor    naive success 0.0%    contaminated 0    (4/4 "fail_model")
fixed-executor     naive success 50.0%   contaminated 1    attributable 66.7%
```

The buggy executor scores 0% and flags nothing. All four failures come back as
clean `fail_model` verdicts. Three of them were actually harness bugs.

A silently-failing harness doesn't just lose accuracy — it loses the ability to
know it's wrong. It produces a tidy result set that looks like a finding about
model capability.

## How it works

### Failure attribution

Every action is classified into one of three tiers:

| Tier | Verdict | What it means |
|---|---|---|
| Definite harness fault | `contaminated` | The harness provably didn't do what was asked. The run says nothing about the model. |
| Ambiguous | `ambiguous` | The driver raised an error. Could be either side. Reported separately, never silently assigned. |
| Model-attributable | `fail_model` / `pass` | The harness did everything asked. Only these runs support claims about the model. |

Additional verdicts: `turn_limit` (ran out of turns, a budget constraint, not
a capability measurement) and `refused` (stopped at a human safety gate).

Two invariants the test suite enforces:

- A harness fault contaminates a run even if the task ultimately passed —
  if an action was dropped, the pass can't be trusted.
- A slow page load is not a failed action. Settle timeouts are recorded but
  never counted as faults, because conflating them would inflate the fault rate.

### Task suite

14 tasks against static HTML fixture pages served from localhost. No real
websites — real sites change without notice, so benchmarks built on them
measure site churn as much as model ability. Fixtures also mean no accounts,
no credentials, and no third-party content in any screenshot.

The key advantage of fixtures over real sites: a fixture can be engineered to
provoke a specific failure mode. A control that only responds to right-click.
A paragraph that requires triple-click to select. A decoy element at the
coordinate origin so that a fallback to `(0, 0)` can't pass by accident.

**Baselines.** Two of the 14 tasks are baselines, run every time and excluded
from reported rates. A floor task every backend must pass (click a large
button), and a ceiling task none may pass (a control that doesn't exist). If
the floor fails or the ceiling passes, the run is declared unsound rather than
reported.

**Oracles.** Every task ships the action trace a correct agent would emit.
This makes the entire pipeline testable at zero cost. A harness regression
becomes a loud oracle failure instead of every backend quietly getting worse.

### Shared evaluation

The prompt, the parser, and the run loop are identical across all backends.
An adapter may only change how a model is reached — never what it is asked or
how its answer is judged. Per-model special-casing is the failure mode that
turns a cross-model comparison into a story about whichever model someone
spent the most time tuning.

The parser is deliberately tolerant about format (it accepts JSON, fenced JSON,
embedded JSON in prose, and XML tags) but strict about vocabulary (actions not
in the prompt are rejected as model errors before they reach the executor). It
records which reading was needed on every turn, so format tolerance is counted
in every report, never silently enjoyed.

## Results

Three backends, same prompt, same parser, same run loop. Three independent
invocations each (nine total). Every task passed or failed identically across
all three invocations of each backend. Zero contamination across all nine runs.

```
                    gemini-3.6-flash   local-vlm (4B)   mistral-small
success rate               100.0%           91.7%           66.7%
self-terminated             16/36            6/36            9/36
fallback parses            17/264           0/186           0/397
contaminated                    0               0               0
mean turn latency             2.8s           12.9s            1.0s
```

Per-task breakdown (pass count out of 3 invocations):

```
task                           gemini   local   mistral
click-named-button                3/3     3/3       3/3
double-click-to-open              3/3     3/3       3/3
type-into-field                   3/3     3/3       3/3
hotkey-select-all-delete          3/3     0/3       0/3
drag-file-to-zone                 3/3     3/3       3/3
hover-then-click                  3/3     3/3       3/3
navigate-and-return               3/3     3/3       0/3
context-menu-right-click          3/3     3/3       3/3
select-whole-sentence             3/3     3/3       3/3
scroll-to-reach-control           3/3     3/3       0/3
publish-without-hitting-origin    3/3     3/3       0/3
wait-for-delayed-control          3/3     3/3       3/3
```

No task flipped between invocations in any backend.

### Key findings

**Formatting quality doesn't predict task performance.** Mistral Small
produced valid strict JSON on all 397 turns (zero fallback parsing needed) and
failed four tasks. Gemini needed fallback parsing on 17 of 264 turns and solved
everything. A strict JSON parser would have penalised the best-performing model
and rewarded the worst.

**Self-termination varies more than task completion.** The three models solved
12, 11, and 8 of 12 tasks, but recognised when they were done on 16, 6, and 9
of 36 runs. Knowing when to stop varies 2.7x across models; knowing what to do
varies 1.5x. This signal is invisible under end-state-only scoring (see
finding #10).

**The local 4B model outperforms the paid Mistral API.** A quantized 4B
vision model running on-device at zero marginal cost beats Mistral Small on
task completion. Their one shared failure — sending a keyboard shortcut without
first focusing the target element — may indicate a common gap in smaller vision
models.

**Constrained decoding provides no measurable benefit.** A separate experiment
on the local model (three invocations with JSON schema constraints, three
without) showed identical results after a prompt defect was fixed. The first
version of that experiment concluded the opposite — stably. The cause was a
placeholder character in the system prompt that the model copied into its
output. Finding #11.

**A scoring bug cost 8.3 percentage points**, all charged to model capability.
The criterion was evaluated only at the end of each run, but the model kept
acting after succeeding and sometimes undid its own work. Scored continuously,
the same runs produce 91.7% instead of 83.3%. Findings #6–#10.

### Limitations

- Three models, one harness. The taxonomy and tooling generalise, but the
  specific numbers are from one evaluation setup.
- Fixture pages, not production software. Fixtures are strictly better for
  testing failure attribution, but they don't measure real-world task
  complexity.
- 12 scored tasks is a small suite. Results are stable across invocations but
  could shift with a larger or differently-constructed task set.
- Three invocations per backend. The invocations are independent (separate
  processes), but within a single invocation, repeats are correlated and are
  collapsed rather than counted. See
  [`docs/ROADMAP.md`](docs/ROADMAP.md) for details.

## Project structure

```
solara_cua/
  executor.py          action dispatch (imports no browser library, duck-types page)
  fakes.py             recording stand-ins for testing without a browser
  backends/
    prompt.py          shared system prompt and per-turn user prompt
    parse.py           tolerant action parser (records which reading was needed)
    local_vlm.py       local vision model backend via OpenAI-compatible API
    mistral.py         Mistral API backend
    gemini.py          Gemini API backend (OpenAI-compatible endpoint)
  eval/
    taxonomy.py        outcome types, verdicts, and attribution rules
    record.py          ActionRecord / RunRecord and JSONL persistence
    instrument.py      execute an action and return a classified record
    report.py          summary statistics (naive vs attributable success rate)
    tasks.py           task definitions: goals, criteria, oracles, splits
    runner.py          the run loop and the Backend interface
    server.py          localhost fixture server (loopback only)
    browser.py         ephemeral browser contexts (never a persistent profile)
fixtures/              static HTML pages, one per interaction primitive
scripts/
  demo_offline.py      scoring demonstration, no credentials or browser needed
  run_suite.py         run the full suite against any backend
  compare.py           cross-invocation aggregation and stability analysis
tests/                 197 tests
results/               JSONL run records (gitignored)
```

### Design notes

**The executor imports no browser library.** `page` is duck-typed, so the same
dispatch code runs against Playwright or a recording fake. This is what makes
the action dispatch fully testable without a browser — and what made the
original bugs findable at all.

**Verdicts are derived, never stored.** Rescoring old results under a changed
taxonomy is a one-line change, not a re-run.

**Results are structured data, not prose.** Previous iterations of this system
logged narrative to a notes file. Prose doesn't survive a rebuild. A JSONL
file with classified actions does.

## License

MIT. See [LICENSE](LICENSE).
