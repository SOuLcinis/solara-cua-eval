"""Action executor for computer-use agents.

Takes the function calls a vision model emits and performs them against a page.

Deliberately imports no browser library: `page` is duck-typed, so the executor
runs against a real Playwright page or a recording fake with equal fidelity.
That is what makes the dispatch testable without a browser, and it is the reason
the bugs in `docs/FINDINGS.md` were findable at all.

Design rule, learned the hard way: an action that cannot be performed must
RAISE. The model's only feedback channel is the screenshot it gets next turn,
so an action that silently does nothing is indistinguishable from one that
worked. Every silent fallback here was once a real bug.
"""
import time


class UnhandledActionError(Exception):
    """The model asked for an action the executor does not implement.

    Deliberately loud. A silently-skipped action looks identical to a successful
    one in the screenshot the model receives next turn, so it burns turns while
    the model retries something that never ran.
    """


class CoordinateError(ValueError):
    """A coordinate could not be denormalized into pixel space."""


def denormalize_x(x, width):
    """Map a 0-1000 normalized x onto pixel space.

    Raises rather than falling back to 0: a bad coordinate used to click the
    top-left corner, which the model then had to diagnose from a screenshot with
    no error to explain it.
    """
    try:
        return int(float(x) / 1000 * width)
    except (TypeError, ValueError) as e:
        raise CoordinateError(f"bad x coordinate {x!r}: {e}") from e


def denormalize_y(y, height):
    """Map a 0-1000 normalized y onto pixel space. See denormalize_x."""
    try:
        return int(float(y) / 1000 * height)
    except (TypeError, ValueError) as e:
        raise CoordinateError(f"bad y coordinate {y!r}: {e}") from e


# The browser is already open and a fresh screenshot is captured every turn, so
# these need no interaction. Listed explicitly to distinguish "nothing to do"
# from "not implemented" -- the two used to be the same code path.
NO_OP_ACTIONS = frozenset({"open_web_browser", "open_app", "take_screenshot"})

# Mouse actions whose button is fixed by the action name rather than by args.
FIXED_MOUSE_BUTTON = {"right_click": "right", "middle_click": "middle"}

SCROLL_DELTAS = {"down": (0, 1), "up": (0, -1), "right": (1, 0), "left": (-1, 0)}


def perform_action(fname, args, page, width, height):
    """Execute one computer-use action against `page`.

    Raises UnhandledActionError for anything not implemented so that unknown
    actions surface as an error the model can see, rather than a silent no-op.
    """
    def xy(x_key="x", y_key="y"):
        return denormalize_x(args[x_key], width), denormalize_y(args[y_key], height)

    if fname in NO_OP_ACTIONS:
        return

    if fname in ("click", "click_at", "right_click", "middle_click"):
        x, y = xy()
        # The action name wins; a plain click may still ask for a button in args.
        button = FIXED_MOUSE_BUTTON.get(fname) or args.get("button", "left")
        page.mouse.click(x, y, button=button)
    elif fname == "double_click":
        x, y = xy()
        page.mouse.dblclick(x, y, button=args.get("button", "left"))
    elif fname == "triple_click":
        x, y = xy()
        page.mouse.click(x, y, button=args.get("button", "left"), click_count=3)
    elif fname == "move":
        x, y = xy()
        page.mouse.move(x, y)
    elif fname in ("type", "type_text_at"):
        if "x" in args and "y" in args:
            x, y = xy()
            page.mouse.click(x, y)
        page.keyboard.press("Control+A")
        page.keyboard.press("Backspace")
        page.keyboard.type(args["text"])
        if args.get("press_enter", False):
            page.keyboard.press("Enter")
    elif fname == "drag_and_drop":
        start_x, start_y = xy("start_x", "start_y")
        end_x, end_y = xy("end_x", "end_y")
        page.mouse.move(start_x, start_y)
        page.mouse.down()
        page.mouse.move(end_x, end_y, steps=10)
        page.mouse.up()
    elif fname == "navigate":
        page.goto(args["url"])
    elif fname == "go_back":
        page.go_back()
    elif fname == "go_forward":
        page.go_forward()
    elif fname == "wait":
        time.sleep(args.get("seconds", 1))
    elif fname == "press_key":
        page.keyboard.press(args["key"])
    elif fname == "key_down":
        page.keyboard.down(args["key"])
    elif fname == "key_up":
        page.keyboard.up(args["key"])
    elif fname == "hotkey":
        page.keyboard.press("+".join(args["keys"]))
    elif fname == "scroll":
        x = denormalize_x(args.get("x", 500), width)
        y = denormalize_y(args.get("y", 500), height)
        direction = args["direction"]
        if direction not in SCROLL_DELTAS:
            # Previously fell through to wheel(0, 0) -- a scroll that did nothing.
            raise UnhandledActionError(f"scroll direction {direction!r}")
        unit_x, unit_y = SCROLL_DELTAS[direction]
        mag = args.get("magnitude_in_pixels", 300)
        page.mouse.move(x, y)
        page.mouse.wheel(unit_x * mag, unit_y * mag)
    else:
        raise UnhandledActionError(fname)


def settle(page, timeout_ms=5000, settle_delay=1.0):
    """Let the page settle after an action.

    Returns True if the page reached domcontentloaded, False if it timed out.

    A timeout here is NOT an action failure and is never raised: a slow page is
    not a failed click. Conflating the two inflates the harness fault rate and
    corrupts any success metric built on these results.
    """
    reached = True
    try:
        page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
    except Exception as e:
        reached = False
        print(f"  [settle] no domcontentloaded within {timeout_ms}ms: {e}")
    time.sleep(settle_delay)
    return reached
