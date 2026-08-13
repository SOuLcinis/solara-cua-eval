"""The task suite: what is being asked, and what counts as having done it.

Three deliberate choices, each one load-bearing.

**Fixtures, not real websites.** Every task targets a static page in `fixtures/`
served from localhost. Real sites change without notice, so a benchmark built on
them measures site churn as much as model ability, and anyone re-running it later
gets numbers that cannot be compared to these. Fixtures also mean no accounts, no
credentials, and no third-party content in any screenshot.

**Criteria are JavaScript evaluated against the live page.** Never a model
judging a model, never a human eyeballing a screenshot. Each criterion is an
expression that must return a truthy value, and it inspects page state only --
it can see what the page received, never what the executor did. That separation
is the whole point: if criteria could see the executor, a harness bug could mark
its own work correct.

**Every task ships an oracle trace.** `oracle` is the action sequence a perfect
model would emit. It exists so the entire pipeline -- server, browser, executor,
instrumentation, scoring -- can be verified end to end before a single token is
spent, and so a later harness regression fails loudly instead of quietly
depressing every backend's score at once.
"""
from dataclasses import dataclass, field

SUITE_VERSION = 1
"""Bump on any change to a task's goal, criterion, oracle, or fixture. Results
from different suite versions are not comparable and must never be pooled."""

VIEWPORT = (1440, 900)


def norm(px, py):
    """Pixel point -> the 0-1000 normalized space models emit coordinates in.

    Oracle traces are written in the model's coordinate space, not the browser's,
    so they exercise denormalization exactly as a real run does. Fixture targets
    are absolutely positioned precisely so these stay stable.
    """
    w, h = VIEWPORT
    return {"x": round(px / w * 1000), "y": round(py / h * 1000)}


# Interaction primitives, enumerated before the task list was written. Writing
# tasks first silently over-samples whatever was easy to think of; enumerating
# first makes a gap visible as a primitive with no task against it.
PRIMITIVES = (
    "click", "right_click", "double_click", "triple_click", "type",
    "hotkey", "scroll", "drag", "wait", "hover", "history",
)

# Task kinds. Baselines are not scored as capability -- they are the check that
# the rest of the numbers mean anything at all.
KIND_FLOOR = "baseline_floor"        # every backend must pass
KIND_CEILING = "baseline_ceiling"    # no backend may pass
KIND_PRIMITIVE = "primitive"         # one interaction primitive, cleanly
KIND_FAILURE_MODE = "failure_mode"   # engineered to provoke a known harness bug


@dataclass(frozen=True)
class Task:
    id: str
    fixture: str
    goal: str
    """Verbatim instruction given to the model. Never names a coordinate, an
    element id, or an action -- naming the action would test obedience, not
    perception."""
    criterion: str
    """JS expression evaluated on the page after the run. Truthy == passed."""
    primitive: str
    kind: str
    oracle: tuple = field(default_factory=tuple)
    split: str = "dev"
    """`dev` while the harness is being built; `heldout` stays sealed until it is
    final. Tuning against the whole suite is overfitting with extra steps."""
    bug_ref: str = ""
    """Which FINDINGS bug this fixture provokes, if any."""


TASKS = (
    # ---- baselines: run first, every time -------------------------------
    Task(
        id="floor-single-click",
        fixture="floor_button.html",
        goal="Click the Continue button.",
        criterion="window.__probe.clicked === true",
        primitive="click",
        kind=KIND_FLOOR,
        oracle=(("click", norm(720, 450)),),
    ),
    Task(
        id="ceiling-absent-control",
        fixture="ceiling_absent.html",
        goal="Submit the record for approval.",
        criterion="window.__probe.submitted === true",
        primitive="click",
        kind=KIND_CEILING,
        # The oracle is empty: there is nothing correct to do. A perfect model
        # stops immediately, and still cannot pass. That is the point.
        oracle=(),
    ),

    # ---- primitives ------------------------------------------------------
    Task(
        id="click-named-button",
        fixture="click_basic.html",
        goal="Click the button labelled Charlie.",
        criterion="window.__probe.clicked === 'Charlie'",
        primitive="click",
        kind=KIND_PRIMITIVE,
        oracle=(("click", norm(880, 450)),),
    ),
    Task(
        id="double-click-to-open",
        fixture="double_click.html",
        goal="Open the Reports folder.",
        criterion="window.__probe.opened === true",
        primitive="double_click",
        kind=KIND_PRIMITIVE,
        oracle=(("double_click", norm(720, 450)),),
        split="heldout",
    ),
    Task(
        id="type-into-field",
        fixture="type_text.html",
        goal="Replace the contents of the text field with: solara",
        criterion="document.getElementById('field').value === 'solara'",
        primitive="type",
        kind=KIND_PRIMITIVE,
        oracle=(("type_text_at", {**norm(720, 435), "text": "solara"}),),
    ),
    Task(
        id="hotkey-select-all-delete",
        fixture="hotkey_clear.html",
        goal="Clear the note using a keyboard shortcut to select everything first.",
        criterion=(
            "window.__probe.sawSelectAll === true && "
            "document.getElementById('note').value === ''"
        ),
        primitive="hotkey",
        kind=KIND_PRIMITIVE,
        oracle=(
            ("click", norm(720, 450)),
            ("hotkey", {"keys": ["Control", "a"]}),
            ("press_key", {"key": "Backspace"}),
        ),
        split="heldout",
    ),
    Task(
        id="drag-file-to-zone",
        fixture="drag_and_drop.html",
        goal="Move the file into the dashed drop zone.",
        criterion="window.__probe.dropped === true",
        primitive="drag",
        kind=KIND_PRIMITIVE,
        oracle=(
            ("drag_and_drop", {
                "start_x": norm(280, 360)["x"], "start_y": norm(280, 360)["y"],
                "end_x": norm(1120, 420)["x"], "end_y": norm(1120, 420)["y"],
            }),
        ),
        split="heldout",
    ),
    Task(
        id="hover-then-click",
        fixture="hover_reveal.html",
        goal="Sign out of the account.",
        criterion="window.__probe.clicked === true",
        primitive="hover",
        kind=KIND_PRIMITIVE,
        oracle=(
            ("move", norm(720, 310)),
            ("click", norm(720, 450)),
        ),
        split="heldout",
    ),
    Task(
        id="navigate-and-return",
        fixture="navigate_back.html",
        goal="Open the details page, then return to the page you started on.",
        criterion=(
            "sessionStorage.getItem('visited') === 'target' && "
            "location.pathname.endsWith('/navigate_back.html')"
        ),
        primitive="history",
        kind=KIND_PRIMITIVE,
        oracle=(
            ("click", norm(720, 450)),
            ("go_back", {}),
        ),
        split="heldout",
    ),

    # ---- failure modes: one per confirmed bug in docs/FINDINGS.md --------
    Task(
        id="context-menu-right-click",
        fixture="right_click_only.html",
        goal="Open the context menu for document.txt.",
        criterion="window.__probe.menuOpen === true",
        primitive="right_click",
        kind=KIND_FAILURE_MODE,
        oracle=(("right_click", norm(720, 450)),),
        bug_ref="FINDINGS #2 -- button='right' dispatched as a left click",
    ),
    Task(
        id="select-whole-sentence",
        fixture="triple_click_select.html",
        goal="Select the entire sentence in the paragraph.",
        criterion=(
            "window.getSelection().toString().trim() === "
            "'The quick brown fox jumps over the lazy dog.'"
        ),
        primitive="triple_click",
        kind=KIND_FAILURE_MODE,
        oracle=(("triple_click", norm(720, 425)),),
        bug_ref="FINDINGS #1 -- triple_click performed no action at all",
    ),
    Task(
        id="scroll-to-reach-control",
        fixture="scroll_below_fold.html",
        goal="Accept the terms at the bottom of the page.",
        criterion="window.__probe.clicked === true",
        primitive="scroll",
        kind=KIND_FAILURE_MODE,
        oracle=(
            ("scroll", {**norm(720, 450), "direction": "down",
                        "magnitude_in_pixels": 2000}),
            ("click", norm(720, 550)),
        ),
        bug_ref="FINDINGS #4 -- unknown scroll direction issued wheel(0, 0)",
    ),
    Task(
        id="publish-without-hitting-origin",
        fixture="origin_trap.html",
        goal="Publish the document.",
        criterion=(
            "window.__probe.targetHit === true && window.__probe.originHit === false"
        ),
        primitive="click",
        kind=KIND_FAILURE_MODE,
        oracle=(("click", norm(1050, 570)),),
        split="heldout",
        bug_ref="FINDINGS #3 -- bad coordinate fell back to 0, clicking the corner",
    ),
    Task(
        id="wait-for-delayed-control",
        fixture="delayed_control.html",
        goal="Confirm, once the page has finished loading.",
        criterion="window.__probe.clicked === true",
        primitive="wait",
        kind=KIND_FAILURE_MODE,
        oracle=(
            ("wait", {"seconds": 3}),
            ("click", norm(720, 450)),
        ),
        split="heldout",
        bug_ref="settle timeouts must not be scored as harness faults",
    ),
)

BY_ID = {t.id: t for t in TASKS}


def select(split=None, kind=None):
    """Filter the suite. No argument returns everything, baselines included."""
    out = TASKS
    if split is not None:
        out = tuple(t for t in out if t.split == split)
    if kind is not None:
        out = tuple(t for t in out if t.kind == kind)
    return out


def uncovered_primitives():
    """Primitives with no task against them. A gap, reported rather than hidden."""
    covered = {t.primitive for t in TASKS}
    return tuple(p for p in PRIMITIVES if p not in covered)
