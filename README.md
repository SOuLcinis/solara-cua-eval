# solara-cua-eval

An evaluation harness for computer-use agents that **separates harness faults from model failures**.

## The claim

Computer-use benchmarks report a success rate. When a task fails, the failure is
attributed to the model.

But a harness that silently drops an action produces the *same observable* — task
not completed — while the model reasoned correctly the whole time. Unless every
action is classified, those two are indistinguishable, and harness bugs get
charged to model reasoning.

This is not hypothetical. It was found by auditing a working computer-use agent
and discovering five ways it reported success while doing nothing or the wrong
thing. See [`docs/FINDINGS.md`](docs/FINDINGS.md).

## The part that should worry you

Run `python scripts/demo_offline.py`. Two executors, the same four scripted
tasks, scored identically:

```
legacy-executor    naive success 0.0%    contaminated 0    (4/4 "fail_model")
fixed-executor     naive success 50.0%   contaminated 1    attributable 66.7%
```

The buggy harness scores **0%** and flags **nothing**. All four failures come
back as clean, confident `fail_model` verdicts. Three of them were harness bugs.

That is the actual danger. A silently-failing harness doesn't merely lose
accuracy — it loses the ability to know it is wrong, and it produces a tidy
result set that reads as a model capability finding.

## The taxonomy

Three tiers, because honest attribution matters more than a clean number:

| Tier | Verdict | Meaning |
|---|---|---|
| Definite harness fault | `contaminated` | The executor provably didn't do what was asked. The run says nothing about the model. |
| Ambiguous | `ambiguous` | The driver raised. Could be either side. Reported separately, never silently assigned. |
| Model-attributable | `fail_model` / `pass` | The harness did everything asked. Only these support claims about the model. |

Plus `turn_limit` (budget, not capability) and `refused` (stopped at a human
safety gate).

Two rules the tests lock down:

- **A harness fault contaminates a run even if the task passed.** If an action
  was dropped, the pass can't be trusted either.
- **A slow page is not a failed action.** Settle timeouts are recorded but never
  counted as faults; conflating them inflates the fault rate.

## The suite

14 tasks against static fixture pages served from localhost — never a real
website. Real sites change without notice, so a benchmark built on them measures
site churn as much as model ability. Fixtures also mean no accounts, no
credentials, and no third-party content in any screenshot.

The payoff is bigger than reproducibility: a fixture can be **engineered to
provoke a specific failure mode**. Every confirmed bug in `FINDINGS.md` has a
page built to catch it — a control that only answers to right-click, a paragraph
that needs a triple-click, a decoy occupying the exact origin so a coordinate
that falls back to `(0, 0)` cannot pass by accident.

Two of the tasks are baselines, run every time and excluded from the reported
rates: a floor every backend must pass, and a ceiling — a control that does not
exist — that none may. If the floor fails or the ceiling passes, the run is
declared unsound instead of reported.

Every task also ships an **oracle**: the action trace a perfect model would emit.
That makes the whole pipeline verifiable for zero API spend, and turns a harness
regression into a loud failure rather than every backend quietly getting worse at
once.

```
$ python scripts/run_suite.py
suite v1  backend=scripted-oracle  tasks=14  repeat=1
...
baseline check passed (floor reached, ceiling held)
naive success rate  100.0%    contaminated 0
```

## First real result

One model so far: a 4B vision model running locally on llama.cpp. No key, no
cost, and no screenshot leaves the machine. Two conditions — with and without
constrained JSON decoding — three independent invocations each.

```
                          constrained    unconstrained
success rate                    91.7%            91.7%     (3/3 invocations each)
actions performed cleanly    204 / 204        204 / 204
malformed replies                    0                0
contaminated runs                    0                0
mean turn latency                12.9s            13.0s
```

**Constrained decoding turns out to buy nothing** — which is only a trustworthy
statement because of what it took to get there.

The first version of this experiment said the opposite, stably, across three
invocations: unconstrained produced 15 malformed replies, exhausted the token
budget through a whole run, and failed a baseline. The conclusion — *this model
needs grammar constraints* — was reproducible and wrong. Nine of those malformed
replies were the model copying a `-` placeholder out of a table in **my system
prompt**.

Note what that fault did *not* touch: the success rate was 91.7% throughout. It
would have produced a false claim about what the model **needs**, and checking
the headline number would never have caught it. Finding #11.

**Two more results the success rate cannot carry:**

- **It solves tasks it cannot tell it has solved.** 11 of 12 scored tasks pass;
  the model recognises completion in **2**. On the rest it acts until the turn
  limit. It *does* stop correctly on the impossible task, every invocation — it
  is better at knowing it cannot proceed than at knowing it is done.
- **An earlier scoring bug cost 8.3 points**, all charged to the model, because
  the criterion was evaluated only at the end of a run and the model kept acting
  after succeeding — findings #6–#10.

Honest limits: one model, one harness, fixture pages rather than production
software, and 12 scored tasks is a small suite. The three invocations per
condition are independent (separate processes) — repeats *within* one
invocation are not, and are collapsed rather than counted; see
[`docs/ROADMAP.md`](docs/ROADMAP.md).

## Status — read this before believing any number

- ✅ Executor, taxonomy, instrumentation, scoring, reporting — **162 tests passing**
- ✅ Five real bugs found in a working agent, each confirmed by executing it
- ✅ Fixture suite + oracle replay green end to end in a real browser
- ✅ **Five more faults found in this harness**, by running a real model against
  it and reading the traces — findings #6–#10
- ✅ One live backend: a local vision model, 12 scored tasks
- ⬜ **A second backend.** Everything above is one model against one harness, so
  nothing here is yet a *cross-model* claim. The adapter interface is done and a
  second implementation is a small file; it is waiting on API budget, not on
  code.

## Layout

```
solara_cua/
  executor.py          action dispatch; imports no browser library, duck-types `page`
  fakes.py             FakePage / BrokenPage — recording stand-ins
  eval/
    taxonomy.py        outcomes, verdicts, and the attribution rules
    record.py          ActionRecord / RunRecord, JSONL persistence
    instrument.py      execute an action, return a classified record
    report.py          naive vs attributable success rate
    tasks.py           the suite: goals, criteria, oracles, splits
    runner.py          the run loop and the Backend interface
    server.py          localhost fixture server, loopback only
    browser.py         ephemeral contexts — never a persistent profile
fixtures/              static pages, one per primitive and per failure mode
scripts/
  demo_offline.py      deterministic demonstration, no credentials needed
  run_suite.py         run the suite; defaults to the oracle backend
tests/                 129 tests
results/               JSONL run records (gitignored except .gitkeep)
```

## Design notes

**`executor.py` imports no browser library.** `page` is duck-typed, so the same
code runs against Playwright or a recording fake. That is what makes dispatch
testable without a browser — and it is why the bugs were findable at all.

**Verdicts are derived, never stored.** Rescoring old results under a changed
taxonomy is a one-line change, not a re-run.

**Results are JSONL, not prose.** Three previous builds of this system left
nothing behind because they logged narrative to a notes file. Prose doesn't
survive a rebuild. A results file with numbers in it does.

## Running

```bash
python -m pytest -q               # 129 tests, no credentials required
python scripts/demo_offline.py    # scoring demonstration, no browser
python scripts/run_suite.py       # full suite, oracle backend, zero API cost
```

The browser-backed test skips itself with a reason when Playwright is absent, so
the suite stays runnable on a machine with nothing installed. For the real thing:

```bash
pip install playwright && playwright install chromium
```

## Provenance

Extracted from a working voice-driven computer-use agent in the Solara hive,
which continues to import `solara_cua.executor` in production.
