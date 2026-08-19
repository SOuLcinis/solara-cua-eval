#!/usr/bin/env python3
"""Run the fixture suite against a backend and score it.

    python scripts/run_suite.py                    # oracle replay, no model, no cost
    python scripts/run_suite.py --repeat 3         # variance across repeated runs
    python scripts/run_suite.py --split dev        # dev subset only
    python scripts/run_suite.py --headed           # watch it happen

The default backend is the scripted oracle: it replays each task's known-correct
action trace. That exercises the fixtures, the server, the executor, the
instrumentation and the scoring end to end without a model or an API key, and it
is the check that must pass before any model number is worth reading.

Baselines are validated on every invocation. If the floor task fails or the
ceiling task passes, the run is declared unsound and the numbers are reported as
untrustworthy rather than printed as though they meant something.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solara_cua.eval import tasks as task_module  # noqa: E402
from solara_cua.eval.browser import ephemeral_page, playwright_available  # noqa: E402
from solara_cua.eval.record import write_jsonl  # noqa: E402
from solara_cua.eval.report import compare_backends, format_summary  # noqa: E402
from solara_cua.eval.runner import MAX_TURNS, ScriptedBackend, run_task  # noqa: E402
from solara_cua.eval.server import serve_fixtures  # noqa: E402
from solara_cua.eval.tasks import KIND_CEILING, KIND_FLOOR, SUITE_VERSION  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def check_baselines(runs):
    """Return the list of baseline violations. Empty means the plumbing is sound."""
    problems = []
    for run in runs:
        task = task_module.BY_ID[run.task_id]
        if task.kind == KIND_FLOOR and not run.passed:
            problems.append(
                f"FLOOR FAILED   {run.backend} / {task.id} -- verdict {run.verdict.value}. "
                "The harness is broken; every other number on this run is noise."
            )
        if task.kind == KIND_CEILING and run.passed:
            problems.append(
                f"CEILING PASSED {run.backend} / {task.id} -- an impossible task "
                "was scored as a success. The criterion is wrong."
            )
    return problems


def build_backend(choice, model=None):
    """Resolve a backend name. Fails loudly rather than falling back to a stub.

    A backend that silently degrades to something cheaper would produce a full
    results file attributed to a model that never ran -- the most expensive
    silent failure available here.
    """
    if choice == "oracle":
        return ScriptedBackend()

    if choice in ("local", "local-free"):
        from solara_cua.backends.local_vlm import LocalVLMBackend, server_reachable

        if not server_reachable():
            sys.exit(
                "No local vision server on http://127.0.0.1:8080.\n"
                "Start llama-server with a multimodal model and an mmproj, then retry."
            )
        kw = {"constrained": choice == "local"}
        if model:
            kw["model"] = model
        return LocalVLMBackend(**kw)

    if choice == "mistral":
        from solara_cua.backends.mistral import MistralBackend
        kw = {}
        if model:
            kw["model"] = model
        return MistralBackend(**kw)

    if choice == "gemini":
        from solara_cua.backends.gemini import GeminiBackend
        kw = {}
        if model:
            kw["model"] = model
        return GeminiBackend(**kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeat", type=int, default=1,
                    help="runs per task; models are stochastic, so >1 for any model backend")
    ap.add_argument("--split", choices=("dev", "heldout"), default=None,
                    help="restrict to one split (default: the whole suite)")
    ap.add_argument("--task", default=None, help="run a single task by id")
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    ap.add_argument("--headed", action="store_true", help="show the browser")
    ap.add_argument("--backend", default="oracle",
                    choices=("oracle", "local", "local-free",
                             "mistral", "gemini"),
                    help="oracle replays known-correct traces (free); "
                         "local drives the on-device vision model; "
                         "local-free is the same model without constrained decoding; "
                         "mistral and gemini call their respective cloud APIs")
    ap.add_argument("--model", default=None,
                    help="override the default model ID for the chosen backend")
    ap.add_argument("--out", default=None,
                    help="results file (default: results/<backend>.jsonl)")
    ap.add_argument("--artifacts", action="store_true",
                    help="save per-turn screenshots to artifacts/ (gitignored)")
    args = ap.parse_args()

    if not playwright_available():
        sys.exit(
            "Playwright with a Chromium build is required.\n"
            "  pip install playwright && playwright install chromium"
        )

    selected = (
        (task_module.BY_ID[args.task],) if args.task
        else task_module.select(split=args.split)
    )
    if not selected:
        sys.exit("no tasks selected")

    gaps = task_module.uncovered_primitives()
    if gaps:
        print(f"note: primitives with no task: {', '.join(gaps)}\n")

    backend = build_backend(args.backend, model=args.model)
    artifacts_dir = (REPO / "artifacts") if args.artifacts else None

    runs = []
    print(f"suite v{SUITE_VERSION}  backend={backend.name}  "
          f"tasks={len(selected)}  repeat={args.repeat}\n")

    with serve_fixtures() as base_url:
        with ephemeral_page(headless=not args.headed) as page:
            for rep in range(args.repeat):
                for task in selected:
                    run = run_task(task, backend, page, base_url,
                                   max_turns=args.max_turns,
                                   artifacts_dir=artifacts_dir)
                    runs.append(run)
                    flag = "ok  " if run.passed else "FAIL"
                    rep_tag = f" [rep {rep + 1}]" if args.repeat > 1 else ""
                    print(f"  {flag} {task.id:<32} {run.verdict.value}{rep_tag}")

    print()
    problems = check_baselines(runs)
    if problems:
        print("BASELINE CHECK FAILED")
        print("-" * 58)
        for p in problems:
            print(f"  {p}")
        print("\nResults below are NOT trustworthy.\n")
    elif any(task_module.BY_ID[r.task_id].kind in (KIND_FLOOR, KIND_CEILING)
             for r in runs):
        print("baseline check passed (floor reached, ceiling held)\n")
    else:
        # Saying "passed" when nothing was checked is the same silent-success
        # pattern this repo documents -- a confident all-clear from a check that
        # never ran.
        print("baseline check SKIPPED — no baseline task in this selection\n")

    out = Path(args.out) if args.out else REPO / "results" / f"{backend.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("")  # a suite run replaces its own file rather than appending
    write_jsonl(runs, out)

    # Baselines are excluded from the reported rates. They measure the
    # instrument, not the model: the ceiling task is designed to be unpassable,
    # so counting it as a failure would understate every backend by a fixed
    # amount and make the number mean something other than what it says. They
    # are still written to the results file -- the check has to be auditable.
    scored = [r for r in runs
              if task_module.BY_ID[r.task_id].kind not in (KIND_FLOOR, KIND_CEILING)]
    n_baseline = len(runs) - len(scored)

    rows = [r.to_dict() for r in scored]
    print(f"({n_baseline} baseline runs excluded from the rates below; "
          "they check the harness, not the model)\n")
    for name, summary in compare_backends(rows).items():
        print(format_summary(summary, backend=name))
        print()

    print(f"results written to {out}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
