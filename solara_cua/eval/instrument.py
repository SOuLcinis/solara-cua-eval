"""Instrumented execution: perform an action and record what actually happened.

This is the whole measurement apparatus. The executor already raises instead of
failing silently; this layer turns those raises into classified, countable data
so a run can be scored on more than whether the task finished.

Deliberately a thin wrapper. It must not change what the executor does -- if the
instrumented path and the production path differ, the measurement is fiction.
"""
from solara_cua.executor import NO_OP_ACTIONS, perform_action, settle
from solara_cua.eval.record import ActionRecord
from solara_cua.eval.taxonomy import ActionOutcome, classify_exception


def execute_recorded(
    fname,
    args,
    page,
    width,
    height,
    perform=perform_action,
    settle_fn=settle,
):
    """Execute one action, returning an ActionRecord instead of raising.

    Errors are captured rather than propagated because a single failed action
    should not end the run -- the model is entitled to see the error and try
    something else. That is exactly the feedback the old silent-failure paths
    denied it.
    """
    try:
        perform(fname, args, page, width, height)
    except Exception as e:
        return ActionRecord(
            action=fname,
            args=dict(args),
            outcome=classify_exception(e),
            detail=f"{type(e).__name__}: {e}",
        )

    outcome = (
        ActionOutcome.NOOP_BY_DESIGN if fname in NO_OP_ACTIONS else ActionOutcome.OK
    )

    # Only settle after an action that could have changed the page. Settling
    # after a declared no-op would charge it a second of latency for nothing.
    settled = True
    if outcome is ActionOutcome.OK:
        settled = settle_fn(page)

    return ActionRecord(action=fname, args=dict(args), outcome=outcome, settled=settled)


def record_refusal(fname, args, reason="declined at safety gate"):
    """Record a human declining an action, which is neither pass nor fault."""
    return ActionRecord(
        action=fname,
        args=dict(args),
        outcome=ActionOutcome.REFUSED_BY_USER,
        detail=reason,
    )
