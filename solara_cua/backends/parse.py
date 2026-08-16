"""Read a model's reply back into an action. Deliberately tolerant.

The local 4B, asked the same question twice, answered:

    {"action": "click", "x": 600, "y": 480}      (constrained decoding)
    <click x="600" y="480"/>                     (unconstrained)

Same grounding, same pixel, different serialization. A strict JSON parser scores
the second as a failure for a model that located the target perfectly -- which is
a harness fault wearing a model failure's clothes, the exact thing this repo
exists to catch. So the parser tries several readings before giving up, and
records WHICH reading worked, because "needed the fallback path" is itself a
finding about a backend.

Tolerance stops at the vocabulary. A reply naming an action the model was never
offered is a model error, not a harness gap -- see `_KNOWN` below.
"""
import json
import re
from dataclasses import dataclass

from solara_cua.backends.prompt import ACTION_VOCABULARY

_KNOWN = {name for name, _params, _desc in ACTION_VOCABULARY}

# A small, explicit alias table. Kept short on purpose: every entry is a place
# where the harness quietly forgives a model, and a long list becomes invisible
# per-model tuning. Anything not here is a model error, reported as one.
_ALIASES = {
    "left_click": "click",
    "click_at": "click",
    "type": "type_text_at",
    "type_text": "type_text_at",
    "back": "go_back",
    "finish": "done",
    "stop": "done",
}

_COORD_KEYS = ("x", "y", "start_x", "start_y", "end_x", "end_y")


@dataclass
class ParseResult:
    """What a reply turned out to be."""

    action: tuple = None
    """(name, args), or None if there is nothing to execute."""

    done: bool = False
    """The model declared the task finished. A claim, not a result."""

    form: str = ""
    """Which reading succeeded: json, fenced_json, embedded_json, xml_tag."""

    error: str = ""
    """Why the reply could not be read, when it could not be."""

    @property
    def ok(self):
        return self.action is not None or self.done


def parse_action(text):
    """Parse a model reply into a ParseResult. Never raises."""
    if not text or not text.strip():
        return ParseResult(error="empty reply")

    for form, candidate in _candidates(text):
        result = _to_action(candidate, form)
        if result is not None:
            return result

    return ParseResult(error=f"no action found in reply: {text.strip()[:200]!r}")


def _candidates(text):
    """Yield (form, dict) readings of a reply, best-supported first."""
    text = text.strip()

    try:
        yield "json", json.loads(text)
    except (ValueError, TypeError):
        pass

    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fenced:
        try:
            yield "fenced_json", json.loads(fenced.group(1).strip())
        except (ValueError, TypeError):
            pass

    # Innermost-to-outermost brace spans; a reasoning preamble often precedes
    # the object, and thinking models emit plenty of prose either side of it.
    for match in re.finditer(r"\{[^{}]*\}", text):
        try:
            yield "embedded_json", json.loads(match.group(0))
        except (ValueError, TypeError):
            continue

    # <click x="600" y="480"/> and friends -- the local model's unconstrained form.
    tag = re.search(r"<\s*([a-z_]+)((?:\s+[a-z_]+\s*=\s*\"[^\"]*\")*)\s*/?\s*>", text)
    if tag:
        attrs = dict(re.findall(r"([a-z_]+)\s*=\s*\"([^\"]*)\"", tag.group(2)))
        yield "xml_tag", {"action": tag.group(1), **attrs}


def _to_action(payload, form):
    """Validate one candidate reading. Returns None if it is not an action."""
    if not isinstance(payload, dict):
        return None

    name = payload.get("action") or payload.get("name") or payload.get("tool")
    if not isinstance(name, str):
        return None

    name = _ALIASES.get(name.strip().lower(), name.strip().lower())

    if name == "done":
        return ParseResult(done=True, form=form)

    if name not in _KNOWN:
        # The vocabulary was given in the system prompt, so inventing a verb is a
        # model error. Letting it reach the executor would raise
        # UnhandledActionError and be recorded as a HARNESS fault -- contaminating
        # a run over a mistake the harness did not make.
        return ParseResult(error=f"action {name!r} is not in the offered vocabulary")

    args = {k: v for k, v in payload.items()
            if k not in ("action", "name", "tool") and v is not None}

    form = _unpack_coord_pair(args) or form

    for key in _COORD_KEYS:
        if key in args:
            coerced = _to_int(args[key])
            if coerced is None:
                return ParseResult(error=f"{name}: {key}={args[key]!r} is not a number")
            args[key] = coerced

    if "keys" in args and isinstance(args["keys"], str):
        # "Control+a" and "Control, a" both appear in the wild.
        args["keys"] = [k.strip() for k in re.split(r"[+,]", args["keys"]) if k.strip()]

    return ParseResult(action=(name, args), form=form)


def _unpack_coord_pair(args):
    """Repair `{"x": [500, 828]}` -- both coordinates packed into one field.

    Observed 10 times in one unconstrained experiment. Same grounding as a
    well-formed reply, different packing, so rejecting it charges a
    serialization quirk to the model's capability.

    Deliberately narrow. It fires only when the pair is unambiguous: exactly two
    numbers, and no conflicting `y`. A model that sent `x: [a, b]` alongside a
    different `y` might mean a bounding box, and guessing there would be the
    harness inventing an intention.

    Returns a form suffix when it repairs something, so the tolerance is counted
    in every report rather than quietly improving a score. Any reader who
    thinks this is too generous can subtract it.
    """
    x = args.get("x")
    if not (isinstance(x, list) and len(x) == 2):
        return None
    if any(_to_int(v) is None for v in x):
        return None

    y = args.get("y")
    if y is not None and y != x:
        return None  # ambiguous -- leave it to fail loudly

    args["x"], args["y"] = x[0], x[1]
    return "coord_pair_repaired"


def _to_int(value):
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None
