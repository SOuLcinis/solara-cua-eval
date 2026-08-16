"""Tests for cross-invocation aggregation.

The thing being defended here is a claim about sampling, not arithmetic: that
repeats inside one invocation are collapsed rather than counted, and that a
spread computed over correlated samples never gets reported as if it were real.
"""
from solara_cua.eval.compare import (
    baseline_violations,
    collapse_invocation,
    first_per_task,
    scored_only,
    split_by_backend,
    success_rate,
    task_stability,
    unstable_tasks,
)


def _row(task_id, passed=True, backend="local-vlm"):
    return {"task_id": task_id, "backend": backend, "passed": passed}


def test_in_file_repeats_are_dropped_not_counted():
    """Keeping them would inflate every total with re-reads of one trajectory."""
    rows = [_row("click-named-button"), _row("click-named-button", passed=False),
            _row("type-into-field")]
    kept, warning = collapse_invocation(rows, label="inv1.jsonl")

    assert len(kept) == 2
    assert [r["task_id"] for r in kept] == ["click-named-button", "type-into-field"]
    assert warning and "Correlated" in warning


def test_no_warning_when_each_task_appears_once():
    rows = [_row("click-named-button"), _row("type-into-field")]
    kept, warning = collapse_invocation(rows)
    assert kept == rows
    assert warning is None


def test_first_per_task_keeps_the_first_and_preserves_order():
    rows = [_row("b"), _row("a"), _row("b", passed=False)]
    assert [r["task_id"] for r in first_per_task(rows)] == ["b", "a"]


def test_baselines_are_excluded_from_the_scored_set():
    rows = [_row("floor-single-click"), _row("ceiling-absent-control", passed=False),
            _row("click-named-button")]
    assert [r["task_id"] for r in scored_only(rows)] == ["click-named-button"]


def test_unknown_task_ids_are_ignored_rather_than_scored():
    """A results file from a different suite version must not silently count."""
    assert scored_only([_row("a-task-that-no-longer-exists")]) == []


def test_success_rate_ignores_baselines():
    rows = [_row("floor-single-click"), _row("click-named-button"),
            _row("type-into-field", passed=False)]
    assert success_rate(rows) == 0.5


def test_success_rate_is_none_without_scored_runs():
    assert success_rate([_row("floor-single-click")]) is None


def test_baseline_violations_catch_both_directions():
    assert baseline_violations([_row("floor-single-click", passed=False)])
    assert baseline_violations([_row("ceiling-absent-control", passed=True)])
    assert baseline_violations([_row("floor-single-click", passed=True),
                                _row("ceiling-absent-control", passed=False)]) == []


def test_split_by_backend():
    rows = [_row("a", backend="x"), _row("b", backend="y")]
    assert set(split_by_backend(rows)) == {"x", "y"}


def test_a_task_that_flips_between_invocations_is_surfaced():
    """The only place a single-invocation number is actively misleading."""
    invocations = [[_row("click-named-button", passed=True)],
                   [_row("click-named-button", passed=False)],
                   [_row("click-named-button", passed=True)]]
    assert unstable_tasks(invocations) == [("click-named-button", 2, 3)]


def test_a_consistent_task_is_not_flagged():
    invocations = [[_row("click-named-button")], [_row("click-named-button")]]
    assert unstable_tasks(invocations) == []
    assert ("click-named-button", 2, 2) in task_stability(invocations)


def test_tasks_absent_from_the_results_are_not_reported():
    stability = task_stability([[_row("click-named-button")]])
    assert all(present > 0 for _t, _p, present in stability)
    assert [t for t, _p, _n in stability] == ["click-named-button"]
