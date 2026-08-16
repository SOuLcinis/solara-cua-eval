#!/usr/bin/env python3
"""Compare backends across independent invocations.

    python scripts/compare.py results/inv*.jsonl

One results file is treated as one sample. See solara_cua/eval/compare.py for
why that matters more than it sounds: repeats inside a single invocation are
correlated, and reporting their spread manufactures a robustness that is not
there.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solara_cua.eval.compare import (  # noqa: E402
    MIN_INVOCATIONS,
    baseline_violations,
    collapse_invocation,
    scored_only,
    split_by_backend,
    success_rate,
    task_stability,
    unstable_tasks,
)
from solara_cua.eval.report import summarize  # noqa: E402


def load(paths):
    """Return ({backend: [invocation_rows, ...]}, warnings)."""
    by_backend, warnings = defaultdict(list), []

    for path in paths:
        text = Path(path).read_text()
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not rows:
            warnings.append(f"{Path(path).name}: empty, skipped")
            continue

        for backend, group in split_by_backend(rows).items():
            group, warning = collapse_invocation(group, label=Path(path).name)
            if warning:
                warnings.append(warning)
            by_backend[backend].append(group)

    return dict(by_backend), warnings


def _pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="one results file per invocation")
    args = ap.parse_args()

    by_backend, warnings = load(args.results)
    if not by_backend:
        sys.exit("no results loaded")

    for w in warnings:
        print(f"note: {w}")
    if warnings:
        print()

    for backend, invocations in sorted(by_backend.items()):
        _print_backend(backend, invocations)

    _print_per_task(by_backend)


def _print_backend(backend, invocations):
    n = len(invocations)
    rates = [r for r in (success_rate(rows) for rows in invocations) if r is not None]
    flat = [r for rows in invocations for r in rows]
    summary = summarize(scored_only(flat))
    violations = [v for rows in invocations for v in baseline_violations(rows)]

    print(backend)
    print("=" * len(backend))
    print(f"  invocations              {n}"
          + ("" if n >= MIN_INVOCATIONS
             else f"   <- fewer than {MIN_INVOCATIONS}; treat as provisional"))

    if rates:
        print(f"  success rate             {_pct(sum(rates) / len(rates))}"
              f"   across invocations: {_pct(min(rates))} - {_pct(max(rates))}")
        if n > 1 and min(rates) == max(rates):
            print("                           (identical across independent "
                  "invocations -- stable, not merely unsampled)")

    print(f"  contaminated             {summary['contaminated']}")
    print(f"  regressed                {summary['regressed']}")
    print(f"  self-terminated          {summary['self_terminated']} of {summary['runs']}")

    if summary.get("mean_turn_latency_s") is not None:
        print(f"  mean turn latency        {summary['mean_turn_latency_s']:.1f}s"
              f"   ({summary['completion_tokens']} completion tokens)")

    if summary.get("parse_forms"):
        forms = summary["parse_forms"]
        total = sum(forms.values())
        fallback = total - forms.get("json", 0)
        print(f"  replies needing fallback {fallback} of {total}"
              f"   {dict(sorted(forms.items(), key=lambda kv: -kv[1]))}")

    print(f"  baseline violations      {len(violations)}"
          + ("" if not violations else f"   {violations}"))
    print()


def _print_per_task(by_backend):
    backends = sorted(by_backend)
    print("per-task pass counts across invocations")
    print("-" * (34 + 20 * len(backends)))
    print(f"{'task':<34}" + "".join(f"{b[:18]:>20}" for b in backends))

    reference = next(iter(by_backend.values()))
    for task_id, _p, _n in task_stability(reference):
        cells = ""
        for backend in backends:
            counts = {t: (p, n) for t, p, n in task_stability(by_backend[backend])}
            passes, present = counts.get(task_id, (0, 0))
            cells += f"{f'{passes}/{present}' if present else '-':>20}"
        print(f"{task_id:<34}{cells}")

    print()
    found = False
    for backend in backends:
        for task_id, passes, present in unstable_tasks(by_backend[backend]):
            if not found:
                print("unstable across invocations (the interesting rows):")
                found = True
            print(f"  {task_id} / {backend}: {passes}/{present}")
    if not found:
        print("no task flipped between invocations")


if __name__ == "__main__":
    main()
