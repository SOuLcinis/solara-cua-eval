# Roadmap — cross-model computer-use suite

Written as a handoff document. A fresh session should be able to read this file
and continue without re-deriving anything.

---

## 1. Privacy architecture — decided, non-negotiable

The production agent launches Chromium with `launch_persistent_context(
user_data_dir=~/solara_live/.browser_profile)`. That profile holds live
logged-in cookies for a real Google account. The agent loop base64-encodes a
full screenshot into every API request.

**Therefore: the eval suite never uses that profile, and never browses a site
anyone is logged into.**

Four rules:

1. **Ephemeral context only.** Every eval run gets a fresh throwaway profile
   directory, deleted afterward. Never `PROFILE_DIR`.
2. **Local fixtures only** (see §2). No public internet targets, so there is no
   account, no session, and no third-party page content to leak.
3. **Results record structure, never content.** A run record holds action names,
   arguments, outcomes and verdicts. It never stores page text, DOM dumps, or
   screenshots.
4. **Screenshots stay local.** They are sent to the model API because the task
   requires it, but are written only to `artifacts/` — gitignored, never
   committed. With fixtures, they contain only synthetic content anyway.

Verification before any public push: `git grep` for personal identifiers and
confirm `results/` and `artifacts/` are ignored. Already passing as of the
initial commit.

---

## 2. Build the target, don't borrow it

The instinct is to benchmark against real websites. That is the wrong call, and
this is the single most important design decision here.

**Test against a local static fixture site**, served by `python -m http.server`
on localhost.

Why:

- **Reproducibility.** Real sites change without notice. A benchmark whose
  numbers move because a site shipped a redesign is measuring the wrong thing.
  Anyone re-running this in a year must get comparable numbers.
- **Isolating the variable.** The point is to compare *models*. Everything else
  — page, harness, criteria — must be held constant. A live site injects
  uncontrolled variance into the one thing being measured.
- **Targeted probes.** Fixtures can be engineered to elicit specific failure
  modes: a control that only responds to right-click, a paragraph that requires
  triple-click to select, a modal that steals focus, an element below the fold,
  a control that appears only after a delay. Every bug in `FINDINGS.md` becomes
  a page that provokes it on purpose.
- **Zero PII, zero credentials, zero cost, zero rate limits, works offline.**

Cost: fixtures are less "real" than production software. State that limitation
explicitly in the writeup rather than pretending otherwise. The claim being made
is about *attribution of failures*, not about absolute real-world capability —
and fixtures are strictly better for that claim.

---

## 3. How a research team would actually run this

Ten practices worth copying, roughly in order of how much they protect you:

**1. Pre-register the claim.** Before running anything, write down what is being
measured and what result would falsify it. Otherwise the analysis drifts toward
whatever the data happened to show.

> Claim: *a meaningful fraction of runs that a naive success-rate benchmark
> scores as model failures are actually harness faults.*
> Falsified if: contamination rate is ~0 across all backends.

**2. Task taxonomy before task list.** Enumerate the interaction primitives
first (click variants, typing, scrolling, drag, keyboard, waiting, navigation),
then write tasks covering each. A hand-picked task list silently over-samples
whatever was easy to think of.

**3. Hold the harness constant.** Same executor, same prompt scaffold, same turn
limit, same viewport for every backend. Any per-model special-casing invalidates
the comparison — and that special-casing is exactly the thing that quietly
creeps in when one model underperforms.

**4. Mechanical grading.** Success criteria inspect page state programmatically.
Never a model judging another model's work, and never a human eyeballing it.
Every criterion is a function returning a bool.

**5. Sanity baselines.** Include one task every backend must pass (a single
click on a large button) and one nothing should pass (an element that does not
exist). If the floor task fails or the ceiling task passes, the plumbing is
broken and every other number is noise. Run these first, every time.

**6. N and variance.** Models are stochastic. Run each task ≥3 times per
backend, report the spread, never a single-run number. A single run is an
anecdote with a percent sign on it.

**7. Dev vs held-out split.** Develop against a subset. Keep the rest sealed
until the harness is final. Otherwise the suite gets tuned until the numbers
look good, which is overfitting with extra steps.

**8. Version the suite.** `SUITE_VERSION` in every run record. Results from
different suite versions are not comparable and must never be pooled.

**9. Replayable logs.** Store the full action trace so any run can be
re-examined without re-spending API budget.

**10. Report cost and latency beside accuracy.** A local 4B that gets 60% at
zero marginal cost and a frontier model that gets 75% are different products,
not a ranking. This matters especially given the local-model goal.

---

## 4. Goals — deliberately resume-scoped

**In scope:**

- G1. A reproducible, self-contained computer-use benchmark: local fixtures, no
  credentials, no external dependencies, runs offline.
- G2. Cross-model results across ≥3 backends, at least one of them local.
- G3. The failure taxonomy applied to real runs, producing the headline number:
  *what fraction of apparent model failures are actually harness faults.*
- G4. A public repo and a written finding that a stranger can evaluate in two
  minutes.

**Explicitly out of scope, for now:**

- Improving Solara's production agent beyond what the suite needs.
- Any new hardware, any new local models.
- Anything requiring the credentialed browser profile.
- Hive expansion, dashboards, sandbox Phase 1.
- Real-website testing (revisit only after G1–G4 land).

If a task doesn't serve G1–G4, it waits.

---

## 5. Phases

**Phase 1 — fixtures and tasks (offline, no API spend)** — ✅ **done**
14 fixture pages, 11 interaction primitives, 4 failure-mode pages (one per
confirmed bug), 2 baselines. `tasks.py` holds goal, JS criterion, oracle trace
and split for each. `run_suite.py` runs them against a localhost server in an
ephemeral browser context. Oracle replay is green end to end and deterministic
across repeats. **Cost: zero.**

**Phase 2 — backend adapters** — 🟡 **local done; paid backends deferred**
One interface, several implementations. The adapter converts model output into
the executor's action vocabulary; everything downstream stays identical. Models
emit different function-call formats, and normalising them is where bias creeps
in — so the prompt and the parser are *shared*, and an adapter may change only
how a model is reached.

- ✅ **Local vision model** (`--backend local`). Zero cost, no key, nothing
  leaves the machine. Also `--backend local-free` for the same model without
  constrained decoding, which isolates formatting from capability.
- ⏸️ **Gemini / Mistral** — deferred. Not a technical blocker: the adapter
  interface is done and a second implementation is a small file. Revisit when
  paid API budget is available.

Building this backend produced five harness faults, all of which would have been
published as model results. They are written up as findings #6–#10 in
`docs/FINDINGS.md` — the harness turning out to have the disease it was built to
diagnose is the strongest evidence in the repo, not an embarrassment to bury.

**Phase 3 — run and analyse**
Baselines first. Then the full suite, ≥3 runs per task per backend. Produce
per-backend summaries and the cross-model comparison. The interesting result:
which failure modes are model-specific versus harness-wide.

**Correction to practice #6, from the first real runs.** "Run each task ≥3
times" is not sufficient, and following it literally here produced a number that
looks far more robust than it is.

Three repeats of the full suite in one invocation returned identical outcomes
*and* identical solve turns on all 14 tasks — apparent perfect stability. But
the same task run in a *separately launched* invocation produced a different
action sequence. Repeats inside one process re-read one trajectory against a
warm server; they are one observation reported three times, not three samples.

A spread of zero is exactly what a correlated sample looks like, and it is
indistinguishable from a genuinely robust result unless you know how the
repeats were drawn. Real variance requires separate invocations, and any
reported spread must say which kind it is.

**Phase 4 — write up and publish**
Short, honest, falsifiable. Lead with the finding, state the limitations
(fixtures aren't production software; N is small; one harness), and flip the
repo public.

---

## 6. Current state

- ✅ Executor, taxonomy, instrumentation, scoring, reporting — 129 tests passing
- ✅ Five bugs found and confirmed by executing the original code
- ✅ Offline demo runs with no key and no browser
- ✅ Private repo, SSH auth working, local and remote reconciled
- ✅ Phase 1 — fixtures, tasks, oracles, runner; suite green in a real browser
- ⬜ Phase 2 — backend adapters, next
- ⚠️ No live model runs. Every number so far is a demonstration of the scoring,
  not an empirical result about any model.

**Where Phase 2 plugs in.** `eval/runner.py` defines `Backend` with one method
that matters, `next_action(observation)`, and `ScriptedBackend` implements it by
replaying oracles. A model adapter subclasses `Backend`, sets
`needs_screenshot = True`, and translates that model's function-call format into
the executor's action vocabulary. Nothing else in the run loop may change —
turn limit, viewport, settle policy and criteria stay identical across backends,
because per-model special-casing is what invalidates the comparison.
