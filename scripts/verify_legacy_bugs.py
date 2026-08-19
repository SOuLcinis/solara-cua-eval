"""Replays the ORIGINAL dispatch logic to prove the regression tests catch real bugs.

This is the evidence behind docs/FINDINGS.md: the bugs are confirmed by running
the original code, not by reading it.

Copied verbatim from the production agent before the fix. If these assertions
hold, the old code really did the wrong thing and the new tests are genuine
regressions.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from solara_cua.fakes import FakePage  # noqa: E402

WIDTH, HEIGHT = 1440, 900


def old_denormalize_x(x, width):
    try:
        return int(float(x) / 1000 * width)
    except Exception:
        return 0


def old_denormalize_y(y, height):
    try:
        return int(float(y) / 1000 * height)
    except Exception:
        return 0


def old_perform_action(fname, args, page, width, height):
    """The original if/elif chain, verbatim."""
    if fname in ("open_web_browser", "open_app"):
        pass
    elif fname in ("click", "click_at", "double_click", "triple_click",
                   "middle_click", "right_click", "move"):
        actual_x = old_denormalize_x(args["x"], width)
        actual_y = old_denormalize_y(args["y"], height)

        if fname in ("click", "click_at"):
            page.mouse.click(actual_x, actual_y)
        elif fname == "double_click":
            page.mouse.dblclick(actual_x, actual_y)
        elif fname in ("click", "click_at") and args.get("button") == "right" or fname == "right_click":
            page.mouse.click(actual_x, actual_y, button="right")
        elif fname == "middle_click":
            page.mouse.click(actual_x, actual_y, button="middle")
        elif fname == "move":
            page.mouse.move(actual_x, actual_y)
    elif fname == "scroll":
        actual_x = old_denormalize_x(args.get("x", 500), width)
        actual_y = old_denormalize_y(args.get("y", 500), height)
        direction = args["direction"]
        mag = args.get("magnitude_in_pixels", 300)
        delta_x, delta_y = 0, 0
        if direction == "down":
            delta_y = mag
        elif direction == "up":
            delta_y = -mag
        elif direction == "right":
            delta_x = mag
        elif direction == "left":
            delta_x = -mag
        page.mouse.move(actual_x, actual_y)
        page.mouse.wheel(delta_x, delta_y)
    else:
        print(f"Warning: Custom or unhandled function {fname}")


def check(label, fn):
    try:
        fn()
        print(f"  CONFIRMED BUG  {label}")
        return True
    except AssertionError as e:
        print(f"  not reproduced  {label}: {e}")
        return False


results = []

def triple_click_is_a_noop():
    p = FakePage()
    old_perform_action("triple_click", {"x": 500, "y": 500}, p, WIDTH, HEIGHT)
    assert p.log == [], f"expected NO interaction, got {p.log}"
results.append(check("triple_click performs no action at all", triple_click_is_a_noop))


def right_button_click_goes_left():
    p = FakePage()
    old_perform_action("click", {"x": 500, "y": 500, "button": "right"}, p, WIDTH, HEIGHT)
    assert p.log == [("mouse", "click", (720, 450), {})], f"got {p.log}"
    assert "button" not in p.log[0][3], "expected NO button kwarg -> defaults to LEFT"
results.append(check('click with button="right" performs a LEFT click', right_button_click_goes_left))


def bad_coord_clicks_origin():
    p = FakePage()
    old_perform_action("click", {"x": "not-a-number", "y": 500}, p, WIDTH, HEIGHT)
    assert p.log == [("mouse", "click", (0, 450), {})], f"got {p.log}"
results.append(check("malformed coordinate silently clicks x=0", bad_coord_clicks_origin))


def unknown_scroll_scrolls_nothing():
    p = FakePage()
    old_perform_action("scroll", {"x": 500, "y": 500, "direction": "sideways"}, p, WIDTH, HEIGHT)
    assert ("mouse", "wheel", (0, 0), {}) in p.log, f"got {p.log}"
results.append(check("unknown scroll direction issues wheel(0, 0)", unknown_scroll_scrolls_nothing))


def unknown_action_reports_success():
    p = FakePage()
    old_perform_action("summon_a_horse", {"x": 1, "y": 1}, p, WIDTH, HEIGHT)
    assert p.log == [], "no action taken, and no exception raised"
results.append(check("unknown action warns and returns as if successful", unknown_action_reports_success))

print(f"\n{sum(results)}/{len(results)} bugs confirmed present in the original code")
