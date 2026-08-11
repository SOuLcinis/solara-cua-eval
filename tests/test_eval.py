"""Tests for the evaluation layer.

The load-bearing test is `test_silent_noop_would_be_misattributed_by_naive_scoring`
at the bottom -- it reconstructs the original bug and shows that a plain
success-rate benchmark charges it to the model.
"""
import json

import pytest

from solara_cua.eval.instrument import execute_recorded, record_refusal
from solara_cua.eval.record import ActionRecord, RunRecord, read_jsonl, write_jsonl
from solara_cua.eval.report import compare_backends, format_summary, summarize
from solara_cua.eval.taxonomy import ActionOutcome, RunVerdict, verdict_for
from solara_cua.fakes import BrokenPage, FakePage

WIDTH, HEIGHT = 1440, 900


@pytest.fixture
def page():
    return FakePage()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    from solara_cua import executor

    monkeypatch.setattr(executor.time, "sleep", lambda _s: None)


# --- instrumentation --------------------------------------------------------

def test_successful_action_records_ok_and_settles(page):
    rec = execute_recorded("click", {"x": 500, "y": 500}, page, WIDTH, HEIGHT)
    assert rec.outcome is ActionOutcome.OK
    assert not rec.is_fault
    assert ("page", "wait_for_load_state", ("domcontentloaded",), {"timeout": 5000}) in page.log


def test_declared_noop_is_not_a_fault_and_does_not_settle(page):
    rec = execute_recorded("take_screenshot", {}, page, WIDTH, HEIGHT)
    assert rec.outcome is ActionOutcome.NOOP_BY_DESIGN
    assert not rec.is_fault
    assert page.only("page") == [], "a declared no-op should not cost a settle"


def test_unimplemented_action_is_a_definite_harness_fault(page):
    rec = execute_recorded("summon_a_horse", {"x": 1, "y": 1}, page, WIDTH, HEIGHT)
    assert rec.outcome is ActionOutcome.HARNESS_UNIMPLEMENTED
    assert rec.is_fault
    assert "UnhandledActionError" in rec.detail


def test_bad_coordinate_is_a_definite_harness_fault(page):
    rec = execute_recorded("click", {"x": "nope", "y": 5}, page, WIDTH, HEIGHT)
    assert rec.outcome is ActionOutcome.HARNESS_BAD_COORDINATE
    assert rec.is_fault


def test_driver_error_is_ambiguous_not_definite():
    rec = execute_recorded("click", {"x": 500, "y": 500}, BrokenPage(), WIDTH, HEIGHT)
    assert rec.outcome is ActionOutcome.HARNESS_DRIVER_ERROR
    assert rec.is_fault


def test_instrumentation_never_raises(page):
    """A failed action must not end the run -- the model gets to see the error."""
    for bad in ("summon_a_horse", "scroll"):
        execute_recorded(bad, {"x": 1, "y": 1, "direction": "sideways"}, page, WIDTH, HEIGHT)


def test_unsettled_page_is_recorded_but_is_not_a_fault(page):
    """settle_fn is the injection seam -- it is bound as a default, so patching
    the module attribute by name would not reach it."""
    rec = execute_recorded(
        "click", {"x": 1, "y": 1}, page, WIDTH, HEIGHT, settle_fn=lambda _p, **_k: False
    )
    assert rec.settled is False
    assert not rec.is_fault, "a slow page is not a failed action"


# --- verdicts ---------------------------------------------------------------

def test_clean_failure_is_attributed_to_the_model():
    assert verdict_for([ActionOutcome.OK, ActionOutcome.OK], passed=False) is RunVerdict.FAIL_MODEL


def test_clean_success_passes():
    assert verdict_for([ActionOutcome.OK], passed=True) is RunVerdict.PASS


def test_harness_fault_contaminates_even_a_passing_run():
    """If the harness dropped an action, a pass cannot be trusted either."""
    v = verdict_for([ActionOutcome.OK, ActionOutcome.HARNESS_UNIMPLEMENTED], passed=True)
    assert v is RunVerdict.CONTAMINATED


def test_driver_error_yields_ambiguous_not_model_failure():
    v = verdict_for([ActionOutcome.HARNESS_DRIVER_ERROR], passed=False)
    assert v is RunVerdict.AMBIGUOUS


def test_refusal_outranks_everything():
    v = verdict_for(
        [ActionOutcome.REFUSED_BY_USER, ActionOutcome.HARNESS_UNIMPLEMENTED], passed=False
    )
    assert v is RunVerdict.REFUSED


def test_turn_limit_is_distinct_from_model_failure():
    v = verdict_for([ActionOutcome.OK], passed=False, turn_limit_hit=True)
    assert v is RunVerdict.TURN_LIMIT


# --- records ----------------------------------------------------------------

def test_verdict_is_derived_not_stored():
    run = RunRecord(task_id="t1", backend="fake")
    run.add(ActionRecord(action="click", outcome=ActionOutcome.OK))
    assert run.verdict is RunVerdict.FAIL_MODEL
    run.passed = True
    assert run.verdict is RunVerdict.PASS, "changing the input must rescore the run"


def test_jsonl_round_trip(tmp_path):
    run = RunRecord(task_id="t1", backend="gemini", passed=True, turns_used=3)
    run.add(ActionRecord(action="click", outcome=ActionOutcome.OK, args={"x": 1}))
    path = tmp_path / "results.jsonl"
    write_jsonl([run], path)
    rows = read_jsonl(path)
    assert len(rows) == 1
    assert rows[0]["verdict"] == "pass"
    assert rows[0]["task_id"] == "t1"
    assert rows[0]["actions"][0]["outcome"] == "ok"
    json.dumps(rows)  # must stay plain-JSON serializable


def test_records_append(tmp_path):
    path = tmp_path / "results.jsonl"
    write_jsonl([RunRecord(task_id="a", backend="x")], path)
    write_jsonl([RunRecord(task_id="b", backend="x")], path)
    assert [r["task_id"] for r in read_jsonl(path)] == ["a", "b"]


# --- reporting --------------------------------------------------------------

def _run(task, backend, outcomes, passed=False):
    r = RunRecord(task_id=task, backend=backend, passed=passed)
    for o in outcomes:
        r.add(ActionRecord(action="click", outcome=o))
    return r.to_dict()


def test_summary_separates_naive_from_attributable():
    rows = [
        _run("t1", "m", [ActionOutcome.OK], passed=True),
        _run("t2", "m", [ActionOutcome.OK]),
        _run("t3", "m", [ActionOutcome.HARNESS_UNIMPLEMENTED]),
        _run("t4", "m", [ActionOutcome.HARNESS_BAD_COORDINATE]),
    ]
    s = summarize(rows)
    assert s["runs"] == 4
    assert s["passes"] == 1
    assert s["contaminated"] == 2
    assert s["naive_success_rate"] == pytest.approx(0.25)
    # Two runs are unattributable, so the honest denominator is 2, not 4.
    assert s["attributable_success_rate"] == pytest.approx(0.5)
    assert s["fault_kinds"]["harness_unimplemented"] == 1


def test_summary_handles_empty_input():
    s = summarize([])
    assert s["runs"] == 0 and s["naive_success_rate"] is None


def test_format_summary_calls_out_misattribution():
    rows = [_run("t1", "m", [ActionOutcome.HARNESS_UNIMPLEMENTED])]
    text = format_summary(summarize(rows), backend="m")
    assert "would be scored as MODEL" in text


def test_compare_backends_groups_by_backend():
    rows = [
        _run("t1", "gemini", [ActionOutcome.OK], passed=True),
        _run("t1", "local-4b", [ActionOutcome.OK]),
    ]
    cmp = compare_backends(rows)
    assert set(cmp) == {"gemini", "local-4b"}
    assert cmp["gemini"]["passes"] == 1
    assert cmp["local-4b"]["passes"] == 0


# --- the whole argument, end to end -----------------------------------------

def test_silent_noop_would_be_misattributed_by_naive_scoring(page):
    """Reconstructs the original triple_click bug and scores it both ways.

    Old executor: triple_click matched the mouse-action tuple but had no branch,
    so it performed nothing and reported success. The task then fails, and a
    success-rate benchmark records a model failure.

    With the taxonomy, the same run is CONTAMINATED -- the harness never
    performed the action, so the run says nothing about the model at all.
    """

    def old_buggy_executor(fname, args, page, width, height):
        if fname in ("click", "triple_click"):
            if fname == "click":
                page.mouse.click(0, 0)
            return  # triple_click: silently nothing, no error

    run = RunRecord(task_id="select-paragraph", backend="any-model")
    run.add(execute_recorded("triple_click", {"x": 5, "y": 5}, page, WIDTH, HEIGHT,
                             perform=old_buggy_executor))
    run.passed = False

    # Naive scoring sees an action that "succeeded" and a task that failed.
    assert run.actions[0].outcome is ActionOutcome.OK
    assert run.verdict is RunVerdict.FAIL_MODEL
    assert page.only("mouse") == [], "the buggy executor really did nothing"

    # The fixed executor raises, and the run is correctly quarantined.
    fixed = RunRecord(task_id="select-paragraph", backend="any-model")
    fixed.add(execute_recorded("triple_click", {"x": 5, "y": 5}, FakePage(), WIDTH, HEIGHT))
    assert fixed.actions[0].outcome is ActionOutcome.OK  # now genuinely performed

    # And an action the executor still cannot do is quarantined rather than blamed.
    unknown = RunRecord(task_id="select-paragraph", backend="any-model")
    unknown.add(execute_recorded("quadruple_click", {"x": 5, "y": 5}, FakePage(), WIDTH, HEIGHT))
    assert unknown.verdict is RunVerdict.CONTAMINATED
    assert summarize([unknown.to_dict()])["misattributed_failures"] == 1
