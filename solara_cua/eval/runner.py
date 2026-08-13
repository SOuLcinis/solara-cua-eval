"""The run loop: drive one task with one backend and return a scored record.

Held deliberately constant across backends. Same executor, same turn limit, same
viewport, same settle policy, same criterion. Any per-model special-casing here
would invalidate the comparison the whole repo exists to make -- and that kind
of special-casing is exactly what creeps in the moment one model underperforms.
A backend may only choose *which action to request next*; nothing else about the
run is negotiable.
"""
from solara_cua.eval.instrument import execute_recorded
from solara_cua.eval.record import ActionRecord, RunRecord
from solara_cua.eval.taxonomy import ActionOutcome
from solara_cua.eval.tasks import VIEWPORT
from solara_cua.executor import settle

MAX_TURNS = 12


class Backend:
    """What a model adapter must provide. One method that matters.

    Phase 2 implements this against Gemini, a local vision model, and Mistral.
    Each of those emits function calls in a different shape, and normalizing them
    into the executor's vocabulary is where bias creeps in -- so the mapping
    lives inside the adapter, explicit and reviewable, never in the run loop.
    """

    name = "backend"
    needs_screenshot = False

    def reset(self, task):
        """Start a fresh episode. Called once per run, before any action."""

    def next_action(self, observation):
        """Return (action_name, args), or None to stop.

        Returning None means "I consider the task finished" -- it is a claim, not
        a result. The criterion decides whether the claim was true.
        """
        raise NotImplementedError


class ScriptedBackend(Backend):
    """Replays each task's oracle trace. The harness's own self-test.

    This is what lets the entire pipeline be verified for zero API spend, and it
    is the regression guard for everything downstream: if a fixture, the server,
    the executor, or the scoring breaks, the oracle stops passing. Without it, a
    harness regression looks identical to every model getting worse at once.
    """

    name = "scripted-oracle"

    def __init__(self, name=None):
        if name:
            self.name = name
        self._queue = []

    def reset(self, task):
        self._queue = list(task.oracle)

    def next_action(self, observation):
        return self._queue.pop(0) if self._queue else None


def run_task(task, backend, page, base_url, max_turns=MAX_TURNS, artifacts_dir=None):
    """Run one task and return a RunRecord. Never raises for a model's mistakes."""
    width, height = VIEWPORT

    page.goto(f"{base_url}/{task.fixture}")
    # Runs share one browser context for speed, so per-origin storage would leak
    # between them: the history task writes sessionStorage, and on a second
    # repeat that leftover value would satisfy half the criterion before the run
    # even started. Cheaper than a fresh context per run, and just as isolated.
    page.evaluate("sessionStorage.clear(); localStorage.clear()")
    settle(page)

    backend.reset(task)
    run = RunRecord(task_id=task.id, backend=backend.name)
    last_error = None

    for turn in range(1, max_turns + 1):
        observation = {
            "goal": task.goal,
            "turn": turn,
            "max_turns": max_turns,
            "viewport": VIEWPORT,
            "last_error": last_error,
            "screenshot": _screenshot(page, task, turn, artifacts_dir)
            if backend.needs_screenshot else None,
        }

        action = backend.next_action(observation)
        if action is None:
            break

        fname, args = action
        record = execute_recorded(fname, args, page, width, height)
        run.add(record)
        run.turns_used = turn
        # The model is entitled to see its own errors and try again -- denying it
        # that feedback is precisely the silent-failure bug this repo documents.
        last_error = record.detail if record.is_fault else None
    else:
        run.turn_limit_hit = True

    run.passed = _evaluate_criterion(task, page, run)
    return run


def _evaluate_criterion(task, page, run):
    """Score the task from live page state.

    A criterion that itself blows up is a harness problem, so it is recorded as a
    driver error rather than silently scored as a model failure. Same principle
    as everywhere else here: never let the measuring instrument charge its own
    faults to the thing being measured.
    """
    try:
        return bool(page.evaluate(task.criterion))
    except Exception as e:
        run.add(ActionRecord(
            action="__criterion__",
            args={"expression": task.criterion},
            outcome=ActionOutcome.HARNESS_DRIVER_ERROR,
            detail=f"criterion evaluation failed -- {type(e).__name__}: {e}",
        ))
        return False


def _screenshot(page, task, turn, artifacts_dir):
    """Capture the frame sent to the model, optionally saving a copy.

    Saved copies go to `artifacts/`, which is gitignored. With fixtures they
    contain only synthetic content, so there is nothing sensitive in them -- that
    is a property of the fixture decision, not of care taken here.
    """
    shot = page.screenshot()
    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / f"{task.id}-turn{turn:02d}.png").write_bytes(shot)
    return shot
