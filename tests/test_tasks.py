"""Tests for the task suite itself.

The suite is data, and data rots quietly: a renamed fixture, a task that drifts
off the only primitive it covered, an oracle calling an action the executor no
longer implements. None of that raises -- it just makes every backend look worse
at once, which is indistinguishable from the models genuinely being worse.

Most of these need no browser. The one that does is skipped, loudly, when
Playwright is absent.
"""
import time

import pytest

from solara_cua.eval import tasks as T
from solara_cua.eval.browser import playwright_available
from solara_cua.eval.server import FIXTURES_DIR, serve_fixtures
from solara_cua.eval.tasks import KIND_CEILING, KIND_FLOOR, PRIMITIVES, TASKS
from solara_cua.executor import UnhandledActionError, perform_action
from solara_cua.fakes import FakePage


# ---------------------------------------------------------------- well-formed

def test_task_ids_are_unique():
    ids = [t.id for t in TASKS]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_fixture_file_exists(task):
    assert (FIXTURES_DIR / task.fixture).is_file()


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_task_fields_are_valid(task):
    assert task.goal.strip()
    assert task.criterion.strip()
    assert task.primitive in PRIMITIVES
    assert task.split in ("dev", "heldout")
    assert task.kind in (KIND_FLOOR, KIND_CEILING, T.KIND_PRIMITIVE, T.KIND_FAILURE_MODE)


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_goal_never_names_the_action_or_a_coordinate(task):
    """A goal that says "triple-click at 500,500" tests obedience, not perception.

    The model has to work out *which* interaction the page requires from the
    screenshot; naming it hands over the only interesting part of the task.
    """
    goal = task.goal.lower()
    for leak in ("triple", "double-click", "right-click", "right click",
                 "scroll", "drag", "hover", "coordinate", "px"):
        assert leak not in goal, f"goal leaks the action: {task.goal!r}"


# --------------------------------------------------------------- coverage

def test_every_primitive_has_a_task():
    assert T.uncovered_primitives() == ()


def test_exactly_one_floor_and_one_ceiling():
    assert len(T.select(kind=KIND_FLOOR)) == 1
    assert len(T.select(kind=KIND_CEILING)) == 1


def test_every_confirmed_bug_has_a_failure_mode_task():
    """Each bug in docs/FINDINGS.md must have a fixture that provokes it.

    Otherwise a regression reintroducing one would pass the whole suite.
    """
    refs = " ".join(t.bug_ref for t in TASKS)
    for bug in ("FINDINGS #1", "FINDINGS #2", "FINDINGS #3", "FINDINGS #4"):
        assert bug in refs


def test_ceiling_task_has_an_empty_oracle():
    """Nothing correct exists to do. A non-empty oracle would mean the task is
    possible after all, and the ceiling would stop being a ceiling."""
    ceiling = T.select(kind=KIND_CEILING)[0]
    assert ceiling.oracle == ()


def test_both_splits_are_populated():
    assert T.select(split="dev")
    assert T.select(split="heldout")


# --------------------------------------------------------------- oracles

@pytest.mark.parametrize(
    "task", [t for t in TASKS if t.oracle], ids=lambda t: t.id
)
def test_oracle_uses_only_implemented_actions(task, monkeypatch):
    """Dispatch every oracle action against a fake page.

    Catches an oracle drifting out of the executor's vocabulary without needing
    a browser. It asserts nothing about whether the action was *correct* -- the
    fixture run does that -- only that the executor knows the verb.
    """
    monkeypatch.setattr(time, "sleep", lambda *_: None)  # `wait` would really sleep
    page = FakePage()
    for fname, args in task.oracle:
        try:
            perform_action(fname, args, page, *T.VIEWPORT)
        except UnhandledActionError as e:
            pytest.fail(f"{task.id} oracle uses an unimplemented action: {e}")


@pytest.mark.parametrize(
    "task", [t for t in TASKS if t.oracle], ids=lambda t: t.id
)
def test_oracle_coordinates_are_in_range(task):
    """Normalized coordinates must stay inside 0-1000.

    An out-of-range oracle would denormalize to a point outside the viewport and
    fail in a way that looks like a bad fixture rather than a bad number.
    """
    for _fname, args in task.oracle:
        for key, value in args.items():
            if key in ("x", "y", "start_x", "start_y", "end_x", "end_y"):
                assert 0 <= value <= 1000, f"{task.id}: {key}={value}"


def test_norm_maps_viewport_corners():
    w, h = T.VIEWPORT
    assert T.norm(0, 0) == {"x": 0, "y": 0}
    assert T.norm(w, h) == {"x": 1000, "y": 1000}
    assert T.norm(w / 2, h / 2) == {"x": 500, "y": 500}


# --------------------------------------------------------------- selection

def test_select_filters_and_defaults_to_everything():
    assert T.select() == TASKS
    assert all(t.split == "dev" for t in T.select(split="dev"))
    # Compared by id: a Task carries dicts in its oracle, so it is not hashable.
    split_ids = {t.id for t in T.select(split="dev")} | {
        t.id for t in T.select(split="heldout")
    }
    assert split_ids == {t.id for t in TASKS}


# --------------------------------------------------------------- server

def test_fixture_server_serves_pages_and_shuts_down():
    from urllib.request import urlopen

    with serve_fixtures() as base_url:
        body = urlopen(f"{base_url}/floor_button.html", timeout=5).read().decode()
        assert "floor_button" in body

    with pytest.raises(Exception):
        urlopen(f"{base_url}/floor_button.html", timeout=2)


def test_server_binds_loopback_only():
    """A benchmark has no business opening a listener onto the LAN."""
    with serve_fixtures() as base_url:
        assert base_url.startswith("http://127.0.0.1:")


# --------------------------------------------------------------- end to end

@pytest.mark.skipif(not playwright_available(), reason="Playwright not installed")
def test_oracle_passes_every_task_in_a_real_browser():
    """The whole pipeline, end to end, for zero API spend.

    If this fails, a model backend's numbers are meaningless -- so it runs before
    any model does. The ceiling task is the exception it must NOT pass.
    """
    from solara_cua.eval.browser import ephemeral_page
    from solara_cua.eval.runner import ScriptedBackend, run_task

    backend = ScriptedBackend()
    with serve_fixtures() as base_url:
        with ephemeral_page() as page:
            for task in TASKS:
                run = run_task(task, backend, page, base_url)
                if task.kind == KIND_CEILING:
                    assert not run.passed, f"{task.id}: impossible task scored a pass"
                else:
                    assert run.passed, f"{task.id}: {run.verdict.value}"
                assert not run.faults, f"{task.id}: {[a.detail for a in run.faults]}"
