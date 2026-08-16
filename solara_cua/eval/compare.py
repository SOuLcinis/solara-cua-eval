"""Aggregate results across independent invocations.

ONE FILE IS ONE SAMPLE, and that is the entire reason this module is separate
from `report.py`.

Running the suite with `--repeat 3` and reporting the spread looks like sound
practice and is not. Three repeats inside one invocation returned identical
outcomes AND identical solve turns on all 14 tasks -- a spread of zero, which
reads as a robust result. It is a correlated sample: the repeats re-read one
trajectory against a warm server. The same task in a separately launched run took
a different action sequence.

A spread of zero is exactly what a correlated sample looks like, and it is
indistinguishable from a genuinely stable result unless you know how the repeats
were drawn. So in-file repeats are collapsed to their first occurrence rather
than counted, spread is computed across files, and a backend with too few
invocations is labelled provisional instead of quietly reported.
"""
from collections import Counter, defaultdict

from solara_cua.eval.tasks import BY_ID, KIND_CEILING, KIND_FLOOR

BASELINE_KINDS = (KIND_FLOOR, KIND_CEILING)
MIN_INVOCATIONS = 3


def first_per_task(rows):
    """Keep the first run of each task, preserving order."""
    seen, kept = set(), []
    for row in rows:
        if row["task_id"] not in seen:
            seen.add(row["task_id"])
            kept.append(row)
    return kept


def split_by_backend(rows):
    out = defaultdict(list)
    for row in rows:
        out[row["backend"]].append(row)
    return dict(out)


def collapse_invocation(rows, label=""):
    """One invocation's rows, with correlated in-file repeats dropped.

    Returns (rows, warning_or_None). Dropping rather than counting is the point:
    keeping the extras inflates every total with re-reads of one trajectory,
    which is the error this module exists to prevent.
    """
    repeated = [t for t, n in Counter(r["task_id"] for r in rows).items() if n > 1]
    if not repeated:
        return rows, None

    kept = first_per_task(rows)
    return kept, (
        f"{label}: {len(repeated)} task(s) run more than once in a single "
        f"invocation. Correlated, not independent -- keeping the first of each "
        f"({len(rows)} rows -> {len(kept)})."
    )


def scored_only(rows):
    """Drop baselines. They measure the instrument, not the model."""
    return [r for r in rows
            if r["task_id"] in BY_ID and BY_ID[r["task_id"]].kind not in BASELINE_KINDS]


def baseline_violations(rows):
    """Floor failures and ceiling passes. Either one invalidates the invocation."""
    problems = []
    for row in rows:
        task = BY_ID.get(row["task_id"])
        if task is None:
            continue
        if task.kind == KIND_FLOOR and not row["passed"]:
            problems.append(f"floor failed ({row['task_id']})")
        if task.kind == KIND_CEILING and row["passed"]:
            problems.append(f"ceiling passed ({row['task_id']})")
    return problems


def success_rate(rows):
    """Pass rate over scored (non-baseline) runs, or None if there are none."""
    scored = scored_only(rows)
    return (sum(bool(r["passed"]) for r in scored) / len(scored)) if scored else None


def task_stability(invocations):
    """(task_id, passes, present) per task, across invocations.

    A task that flips between invocations is the interesting row: it is the only
    place a single-invocation number is actively misleading.
    """
    out = []
    for task_id in BY_ID:
        present = sum(1 for rows in invocations
                      if any(r["task_id"] == task_id for r in rows))
        passes = sum(1 for rows in invocations
                     if any(r["task_id"] == task_id and r["passed"] for r in rows))
        if present:
            out.append((task_id, passes, present))
    return out


def unstable_tasks(invocations):
    """Tasks that did not agree with themselves across invocations."""
    return [(t, p, n) for t, p, n in task_stability(invocations) if 0 < p < n]
