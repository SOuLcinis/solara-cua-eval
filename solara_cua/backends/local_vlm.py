"""Backend for a local OpenAI-compatible vision server (llama.cpp / llama-server).

Zero marginal cost, no key, no data leaving the machine. That last part is not a
side benefit: it means screenshots never reach a third party, which is what makes
running this against anything richer than a fixture thinkable at all.

Measured against Huihui-Qwen3.5-4B (Q4_K_M, mmproj) on a local llama-server:

  * Vision is real. Asked to list the buttons on a fixture it returned
    "Alpha / Bravo / Charlie / Delta" and correctly declined to count the footer
    caption as a button.
  * Grounding is usable. Asked to click Charlie it emitted (600, 480) against a
    true centre of (611, 500) -- comfortably inside the target.
  * It is deterministic at temperature 0: byte-identical replies across repeats.
  * It ALWAYS reasons first. `enable_thinking: false` and a `/no_think` suffix
    both still produced `reasoning_content`, so the token budget must cover the
    reasoning or the answer never arrives -- see MIN_SANE_MAX_TOKENS.

Only transport lives here. The prompt is shared (see prompt.py) and the parser is
shared (see parse.py), because those are the two places a "comparison" quietly
turns into per-model tuning.
"""
import base64
import json
import urllib.error
import urllib.request

from solara_cua.backends.parse import parse_action
from solara_cua.backends.prompt import SYSTEM_PROMPT, user_prompt
from solara_cua.eval.runner import Backend, NonAction
from solara_cua.eval.taxonomy import ActionOutcome

DEFAULT_URL = "http://127.0.0.1:8080"

MIN_SANE_MAX_TOKENS = 512
"""Below this, a reasoning model spends the whole budget thinking and returns
`content: ""` with `finish_reason: "length"`. That empty string is
indistinguishable from a model with nothing to say, so a benchmark scores the
harness's own budget mistake as a model failure. Found the hard way, at 100."""

# Constrained decoding. llama-server honours a JSON schema, which removes
# formatting from the measurement entirely -- but it is a real difference from a
# backend that has no such feature, so it is recorded per run rather than
# silently enjoyed. Toggle with constrained=False to measure the model's
# unaided formatting.
ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string"},
        "x": {"type": "integer", "minimum": 0, "maximum": 1000},
        "y": {"type": "integer", "minimum": 0, "maximum": 1000},
        "start_x": {"type": "integer", "minimum": 0, "maximum": 1000},
        "start_y": {"type": "integer", "minimum": 0, "maximum": 1000},
        "end_x": {"type": "integer", "minimum": 0, "maximum": 1000},
        "end_y": {"type": "integer", "minimum": 0, "maximum": 1000},
        "text": {"type": "string"},
        "key": {"type": "string"},
        "keys": {"type": "array", "items": {"type": "string"}},
        "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
        "magnitude_in_pixels": {"type": "integer"},
        "seconds": {"type": "number"},
    },
    "required": ["action"],
    "additionalProperties": False,
}


class LocalVLMBackend(Backend):
    """Drives a local llama-server over its OpenAI-compatible chat endpoint."""

    needs_screenshot = True

    def __init__(self, url=DEFAULT_URL, model=None, max_tokens=1024,
                 temperature=0.0, constrained=True, timeout=300, name=None):
        if max_tokens < MIN_SANE_MAX_TOKENS:
            raise ValueError(
                f"max_tokens={max_tokens} is below MIN_SANE_MAX_TOKENS="
                f"{MIN_SANE_MAX_TOKENS}. This model reasons before answering; a "
                "smaller budget returns an empty reply that looks like a model "
                "failure and is not one."
            )
        self.url = url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.constrained = constrained
        self.timeout = timeout
        self.name = name or f"local-vlm{'' if constrained else '-free'}"
        self._goal = None
        self.last_form = None
        """Which parser reading the previous reply needed. Reported, not hidden."""

    def reset(self, task):
        self._goal = task.goal

    def next_action(self, observation):
        try:
            reply, finish_reason = self._complete(observation)
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
            # The server being down is not the model's fault, and must never be
            # scored as one.
            return NonAction(
                ActionOutcome.HARNESS_DRIVER_ERROR,
                f"local vision server at {self.url} unreachable or malformed: "
                f"{type(e).__name__}: {e}",
            )

        if not reply.strip() and finish_reason == "length":
            return NonAction(
                ActionOutcome.HARNESS_TOKEN_BUDGET,
                f"reply truncated at max_tokens={self.max_tokens} before any "
                "content; the reasoning consumed the whole budget",
            )

        parsed = parse_action(reply)
        self.last_form = parsed.form or None

        if parsed.done:
            return None
        if parsed.action is None:
            return NonAction(ActionOutcome.MODEL_UNPARSEABLE, parsed.error)
        return parsed.action

    def _complete(self, observation):
        content = [{
            "type": "text",
            "text": user_prompt(
                self._goal, observation["turn"], observation["max_turns"],
                observation.get("last_error"),
            ),
        }]
        shot = observation.get("screenshot")
        if shot:
            b64 = base64.b64encode(shot).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}})

        payload = {
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.model:
            payload["model"] = self.model
        if self.constrained:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "action", "strict": True,
                                "schema": ACTION_SCHEMA},
            }

        request = urllib.request.Request(
            f"{self.url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.load(response)

        choice = body["choices"][0]
        # `content` may be absent or null when the whole budget went to
        # reasoning_content; normalise to "" so the caller sees one shape.
        return (choice["message"].get("content") or "",
                choice.get("finish_reason", ""))


def server_reachable(url=DEFAULT_URL, timeout=5):
    """True if a model is being served. Checked so tests skip with a reason."""
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/v1/models", timeout=timeout) as r:
            return bool(json.load(r).get("data"))
    except Exception:
        return False
