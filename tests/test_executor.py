"""Regression tests for the computer-use action executor.

Every test here corresponds to a real defect found by auditing the original solara_agent.py.
The common shape of those defects: an action that did nothing, or did the wrong
thing, while reporting success to the model. The model then sees an unchanged
screenshot with no error to explain it, and burns turns retrying.
"""
import pytest

from solara_cua import executor
from solara_cua.fakes import FakePage
from solara_cua.executor import (
    CoordinateError,
    UnhandledActionError,
    denormalize_x,
    denormalize_y,
    perform_action,
)

WIDTH, HEIGHT = 1440, 900


@pytest.fixture
def page():
    return FakePage()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Keep the `wait` action from actually sleeping during tests."""
    monkeypatch.setattr(executor.time, "sleep", lambda _s: None)


def run(page, fname, **args):
    perform_action(fname, args, page, WIDTH, HEIGHT)


# --- coordinate mapping -----------------------------------------------------

def test_denormalize_maps_thousand_scale_to_pixels():
    assert denormalize_x(500, WIDTH) == 720
    assert denormalize_y(500, HEIGHT) == 450
    assert denormalize_x(0, WIDTH) == 0
    assert denormalize_y(1000, HEIGHT) == 900


@pytest.mark.parametrize("bad", ["not-a-number", None, {}])
def test_bad_coordinate_raises_instead_of_clicking_the_origin(bad):
    """Regression: denormalize returned 0 on any failure.

    A malformed coordinate silently clicked the top-left corner, which is a
    plausible-looking action the model cannot distinguish from a real one.
    """
    with pytest.raises(CoordinateError):
        denormalize_x(bad, WIDTH)
    with pytest.raises(CoordinateError):
        denormalize_y(bad, HEIGHT)


def test_bad_coordinate_propagates_out_of_perform_action(page):
    with pytest.raises(CoordinateError):
        run(page, "click", x="nope", y=100)
    assert page.log == [], "no interaction should be attempted on a bad coordinate"


# --- click dispatch ---------------------------------------------------------

def test_triple_click_actually_clicks(page):
    """Regression: triple_click was listed as a mouse action but had no branch.

    It fell through the entire if/elif chain and performed nothing at all -- no
    action, no warning, no error.
    """
    run(page, "triple_click", x=500, y=500)
    assert page.only("mouse") == [
        ("mouse", "click", (720, 450), {"button": "left", "click_count": 3})
    ]


def test_click_honors_right_button_in_args(page):
    """Regression: `click` with button="right" performed a LEFT click.

    The first branch caught ("click", "click_at") unconditionally, so the later
    branch checking args["button"] was unreachable.
    """
    run(page, "click", x=500, y=500, button="right")
    assert page.only("mouse") == [
        ("mouse", "click", (720, 450), {"button": "right"})
    ]


def test_right_click_action_uses_right_button(page):
    run(page, "right_click", x=100, y=200)
    assert page.only("mouse") == [
        ("mouse", "click", (144, 180), {"button": "right"})
    ]


def test_middle_click_action_uses_middle_button(page):
    run(page, "middle_click", x=100, y=200)
    assert page.only("mouse") == [
        ("mouse", "click", (144, 180), {"button": "middle"})
    ]


def test_action_name_overrides_conflicting_button_arg(page):
    """right_click stays a right click even if args disagree."""
    run(page, "right_click", x=0, y=0, button="left")
    assert page.only("mouse")[0][3]["button"] == "right"


def test_plain_click_defaults_to_left(page):
    run(page, "click_at", x=1000, y=1000)
    assert page.only("mouse") == [
        ("mouse", "click", (1440, 900), {"button": "left"})
    ]


def test_double_click_uses_dblclick(page):
    run(page, "double_click", x=500, y=500)
    assert page.only("mouse") == [
        ("mouse", "dblclick", (720, 450), {"button": "left"})
    ]


# --- unimplemented actions must be loud -------------------------------------

def test_unknown_action_raises(page):
    """Regression: unknown actions printed a warning and reported success."""
    with pytest.raises(UnhandledActionError):
        run(page, "summon_a_horse", x=1, y=1)


def test_unknown_scroll_direction_raises(page):
    """Regression: an unrecognized direction fell through to wheel(0, 0).

    A scroll that scrolls nothing, reported as a completed action.
    """
    with pytest.raises(UnhandledActionError):
        run(page, "scroll", x=500, y=500, direction="sideways")


@pytest.mark.parametrize("fname", ["open_web_browser", "open_app", "take_screenshot"])
def test_declared_noops_do_nothing_quietly(page, fname):
    """These are genuinely nothing-to-do, as opposed to not-implemented."""
    run(page, fname)
    assert page.log == []


# --- scroll -----------------------------------------------------------------

@pytest.mark.parametrize(
    "direction,expected",
    [
        ("down", (0, 300)),
        ("up", (0, -300)),
        ("right", (300, 0)),
        ("left", (-300, 0)),
    ],
)
def test_scroll_directions(page, direction, expected):
    run(page, "scroll", x=500, y=500, direction=direction)
    assert ("mouse", "wheel", expected, {}) in page.log


def test_scroll_respects_magnitude(page):
    run(page, "scroll", x=500, y=500, direction="down", magnitude_in_pixels=42)
    assert ("mouse", "wheel", (0, 42), {}) in page.log


# --- typing -----------------------------------------------------------------

def test_type_clears_field_before_typing(page):
    run(page, "type", text="hello")
    assert page.only("keyboard") == [
        ("keyboard", "press", ("Control+A",), {}),
        ("keyboard", "press", ("Backspace",), {}),
        ("keyboard", "type", ("hello",), {}),
    ]


def test_type_at_coordinates_clicks_first(page):
    run(page, "type_text_at", x=500, y=500, text="hi", press_enter=True)
    assert page.log[0] == ("mouse", "click", (720, 450), {})
    assert page.log[-1] == ("keyboard", "press", ("Enter",), {})


def test_type_without_coordinates_does_not_click(page):
    run(page, "type", text="hi")
    assert page.only("mouse") == []


# --- navigation and keys ----------------------------------------------------

def test_navigate(page):
    run(page, "navigate", url="https://example.com")
    assert ("page", "goto", ("https://example.com",), {}) in page.log


def test_hotkey_joins_keys(page):
    run(page, "hotkey", keys=["Control", "Shift", "T"])
    assert ("keyboard", "press", ("Control+Shift+T",), {}) in page.log


def test_drag_and_drop_sequence(page):
    run(page, "drag_and_drop", start_x=0, start_y=0, end_x=1000, end_y=1000)
    assert [c[1] for c in page.only("mouse")] == ["move", "down", "move", "up"]
    assert page.only("mouse")[2] == ("mouse", "move", (1440, 900), {"steps": 10})


# --- settle must not be blamed for action failures --------------------------

def test_settle_swallows_timeout_without_raising():
    """A slow page is not a failed action.

    settle() is called outside the action's except block, so a timeout here can
    never be recorded as an action error. Conflating the two would corrupt any
    success metric built on these results.
    """

    class SlowPage(FakePage):
        def wait_for_load_state(self, *a, **kw):
            raise TimeoutError("navigation timeout")

    executor.settle(SlowPage(), timeout_ms=1)  # must not raise
