"""The instruction every backend receives. Shared on purpose.

Prompt text is the easiest place for a comparison to quietly stop being a
comparison. Tune the wording for one model and its score moves for reasons that
have nothing to do with capability -- and the tuning always happens to the model
that was underperforming, which is precisely the direction that manufactures a
result. So the system prompt and the action vocabulary live here, once, and no
adapter may edit them.

What an adapter *may* do is change the transport: how the image is attached, how
the reply is constrained, how the response is parsed back into an action. Those
are properties of the API, not of the task.
"""

NO_PARAMS = "(takes no parameters)"
"""Spelled out rather than shown as a symbol. See _vocabulary_block."""

# The executor's vocabulary, restated for the model. Kept deliberately small:
# every verb here is one the executor implements, so an unimplemented action is
# a model error rather than a gap in the harness.
ACTION_VOCABULARY = (
    ("click", "x, y", "left click at a point"),
    ("right_click", "x, y", "right click, opens a context menu"),
    ("double_click", "x, y", "double click, e.g. to open an item"),
    ("triple_click", "x, y", "triple click, selects a whole paragraph"),
    ("move", "x, y", "move the pointer without clicking, e.g. to hover"),
    ("type_text_at", "x, y, text", "click a field, clear it, and type text"),
    ("press_key", "key", "press one key, e.g. Enter or Backspace"),
    ("hotkey", "keys", "press a chord, e.g. keys: [Control, a]"),
    ("scroll", "x, y, direction, magnitude_in_pixels",
     "direction is up, down, left or right"),
    ("drag_and_drop", "start_x, start_y, end_x, end_y", "press, drag, release"),
    ("go_back", NO_PARAMS, "return to the previous page"),
    ("wait", "seconds", "let the page finish loading"),
    ("done", NO_PARAMS, "the task is complete, or cannot be done"),
)


def _vocabulary_block():
    """Render the table the model sees.

    The parameterless rows once showed "-" in the params column. The model
    copied it: 9 of 19 malformed replies in one experiment were
    `{"action": "go_back", "-"}` -- invalid JSON, recorded as MODEL_UNPARSEABLE,
    and charged to the model. The prompt caused it.

    Any placeholder that could be mistaken for content will eventually be echoed
    as content, so parameterless actions now say so in words.
    """
    return "\n".join(
        f"  {name:<14} {params:<38} {desc}" for name, params, desc in ACTION_VOCABULARY
    )


SYSTEM_PROMPT = f"""You operate a web browser by looking at a screenshot and \
emitting one action at a time.

The screenshot is 1440 by 900 pixels. All coordinates you emit are NORMALIZED to \
a 0-1000 scale on both axes, independent of pixel size: x=0 is the left edge, \
x=1000 the right edge, y=0 the top edge, y=1000 the bottom edge. Aim for the \
CENTRE of the element you intend to act on.

Available actions:
{_vocabulary_block()}

Reply with exactly one action as a JSON object, for example:
  {{"action": "click", "x": 500, "y": 500}}

Emit one action only. You will be shown a new screenshot after each one, so do \
not plan several moves ahead. When the goal is achieved, emit \
{{"action": "done"}}."""


def user_prompt(goal, turn, max_turns, last_error=None):
    """The per-turn instruction. Identical in shape for every backend.

    `last_error` is fed back deliberately. Denying a model sight of its own
    failed action is the exact defect documented in docs/FINDINGS.md, and a
    harness that repeats it while measuring models would be making the mistake
    it claims to detect.
    """
    lines = [f"Goal: {goal}", f"This is turn {turn} of {max_turns}."]
    if last_error:
        lines.append(
            f"Your previous action failed: {last_error}\n"
            "Do not repeat it unchanged."
        )
    lines.append("Look at the screenshot and emit the single next action.")
    return "\n\n".join(lines)
