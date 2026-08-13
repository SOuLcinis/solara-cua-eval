"""Failure taxonomy for computer-use agent runs.

The claim this package exists to support:

    A run containing a harness fault cannot be scored as a model failure.

Computer-use benchmarks overwhelmingly report a single success rate. When a task
fails, the failure is attributed to the model. But a harness that silently drops
an action produces exactly the same observable -- task not completed -- while the
model reasoned correctly the entire time.

Separating the two requires classifying every action, not just the final result.

Three tiers, because honesty about attribution matters more than a clean number:

  * DEFINITE harness faults -- the executor provably did not do what was asked.
    Nothing about the model can be concluded from a run containing one.
  * AMBIGUOUS faults -- the driver raised. This may be the harness's fault (a bad
    selector, a stale handle) or the model's (clicking outside the viewport).
    Reported separately rather than assigned to whichever side flatters the
    result.
  * Model-attributable -- the harness did everything asked and the task still
    did not complete.
"""
from enum import Enum


class ActionOutcome(str, Enum):
    """What happened when a single requested action was executed."""

    OK = "ok"
    """The action was performed."""

    NOOP_BY_DESIGN = "noop_by_design"
    """Genuinely nothing to do (e.g. open_browser when it is already open).

    Distinct from an unimplemented action, which is a fault. Collapsing these
    two into one code path is what made `triple_click` a silent no-op.
    """

    HARNESS_UNIMPLEMENTED = "harness_unimplemented"
    """The executor has no implementation for this action."""

    HARNESS_BAD_COORDINATE = "harness_bad_coordinate"
    """A coordinate could not be mapped into pixel space."""

    HARNESS_DRIVER_ERROR = "harness_driver_error"
    """The browser driver raised. Attribution is genuinely ambiguous."""

    HARNESS_TOKEN_BUDGET = "harness_token_budget"
    """The model's reply was cut off by max_tokens before it produced anything.

    Found while building the first backend, which is the only reason it is here.
    The local model reasons before answering, and with a max_tokens that looked
    generous the entire budget went to reasoning -- leaving `content: ""` and
    `finish_reason: "length"`. An empty reply is indistinguishable from a model
    that had nothing to say, so a benchmark would score it as a model failure.

    Filed AMBIGUOUS rather than as a definite harness fault, deliberately. Too
    small a budget is the harness's fault; a model that reasons forever and never
    answers is the model's. From one truncated response those are genuinely
    indistinguishable, and claiming otherwise would be the same overconfidence
    this taxonomy exists to prevent.
    """

    MODEL_UNPARSEABLE = "model_unparseable"
    """The model replied, but nothing in it could be read as an action.

    NOT a harness fault: the harness did its job and got something it could not
    use. Tracked separately from a wrong action because they fail differently --
    one is a formatting problem, the other a reasoning one, and a backend that
    scores badly needs to know which.

    Care is required here. The local model, asked the same question twice,
    returned `{"action":"click","x":600,"y":480}` under constrained decoding and
    `<click x="600" y="480"/>` without it. Identical grounding, different
    serialization. A strict JSON parser would have logged the second as a
    failure for a model that located the target perfectly -- so the parser is
    deliberately tolerant, and this outcome means "unreadable by any reading",
    not "not the format I wanted".
    """

    REFUSED_BY_USER = "refused_by_user"
    """A human declined the action at a safety gate. Not a fault on either side."""


DEFINITE_HARNESS_FAULTS = frozenset({
    ActionOutcome.HARNESS_UNIMPLEMENTED,
    ActionOutcome.HARNESS_BAD_COORDINATE,
})

AMBIGUOUS_FAULTS = frozenset({
    ActionOutcome.HARNESS_DRIVER_ERROR,
    ActionOutcome.HARNESS_TOKEN_BUDGET,
})

ALL_FAULTS = DEFINITE_HARNESS_FAULTS | AMBIGUOUS_FAULTS


class RunVerdict(str, Enum):
    """The scoreable outcome of a whole task run."""

    PASS = "pass"
    """The task's success criterion was met."""

    FAIL_MODEL = "fail_model"
    """Criterion not met, and every action the model requested was performed.

    This is the only verdict that says anything about the model.
    """

    CONTAMINATED = "contaminated"
    """A definite harness fault occurred. Not attributable to the model.

    A success-rate benchmark would silently count this as a model failure.
    """

    AMBIGUOUS = "ambiguous"
    """A driver error occurred and the task did not pass. Attribution unknown."""

    TURN_LIMIT = "turn_limit"
    """Ran out of turns with no fault and no success. Usually a budget problem."""

    REFUSED = "refused"
    """Stopped at a human safety gate."""


def classify_exception(exc):
    """Map an executor exception onto an ActionOutcome.

    Imported lazily so this module stays dependency-free for analysis of
    results captured elsewhere.
    """
    from solara_cua.executor import CoordinateError, UnhandledActionError

    if isinstance(exc, UnhandledActionError):
        return ActionOutcome.HARNESS_UNIMPLEMENTED
    if isinstance(exc, CoordinateError):
        return ActionOutcome.HARNESS_BAD_COORDINATE
    return ActionOutcome.HARNESS_DRIVER_ERROR


def verdict_for(outcomes, passed, turn_limit_hit=False):
    """Score a run from its action outcomes and whether the criterion was met.

    Ordering matters and encodes the whole argument:

      1. A refusal is a deliberate stop, not a result.
      2. A definite harness fault contaminates the run even if the task somehow
         still passed -- the pass cannot be trusted either.
      3. Only then may an outcome be attributed to the model.

    `passed` is the task's own success criterion; this function never guesses it.
    """
    outcomes = list(outcomes)

    if ActionOutcome.REFUSED_BY_USER in outcomes:
        return RunVerdict.REFUSED
    if any(o in DEFINITE_HARNESS_FAULTS for o in outcomes):
        return RunVerdict.CONTAMINATED
    if passed:
        return RunVerdict.PASS
    if any(o in AMBIGUOUS_FAULTS for o in outcomes):
        return RunVerdict.AMBIGUOUS
    if turn_limit_hit:
        return RunVerdict.TURN_LIMIT
    return RunVerdict.FAIL_MODEL
