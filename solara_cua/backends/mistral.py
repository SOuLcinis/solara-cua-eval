"""Backend for Mistral's vision models via their chat completions API.

Same prompt, same parser, same run loop as every other backend. The only
thing that lives here is how to get a base64 screenshot to api.mistral.ai
and read the answer back.

One format difference from OpenAI: Mistral's image_url field is a plain
string (the data URI itself), not a {"url": "..."} object. Getting that
wrong produces an opaque 422 that a backend should never charge to the
model.
"""
import base64
import json
import os
import time
import urllib.error
import urllib.request

from solara_cua.backends.parse import parse_action
from solara_cua.backends.prompt import SYSTEM_PROMPT, user_prompt
from solara_cua.eval.runner import Backend, NonAction
from solara_cua.eval.taxonomy import ActionOutcome

API_URL = "https://api.mistral.ai"


class MistralBackend(Backend):
    """Drives a Mistral vision model over the chat completions API."""

    needs_screenshot = True

    def __init__(self, model="mistral-small-latest", api_key=None,
                 max_tokens=1024, temperature=0.0, timeout=120, name=None):
        self.model = model
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No Mistral API key. Set MISTRAL_API_KEY in the environment "
                "or pass api_key= to MistralBackend."
            )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.name = name or model.removesuffix("-latest")
        self._goal = None
        self._meta = {}

    def turn_metadata(self):
        return dict(self._meta)

    def reset(self, task):
        self._goal = task.goal

    def next_action(self, observation):
        self._meta = {}
        try:
            reply, finish_reason = self._complete(observation)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()[:300]
            except Exception:
                pass
            return NonAction(
                ActionOutcome.HARNESS_DRIVER_ERROR,
                f"Mistral API error: HTTP {e.code}: {body or e.reason}",
            )
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
            return NonAction(
                ActionOutcome.HARNESS_DRIVER_ERROR,
                f"Mistral API error: {type(e).__name__}: {e}",
            )

        if not reply.strip() and finish_reason == "length":
            return NonAction(
                ActionOutcome.HARNESS_TOKEN_BUDGET,
                f"reply truncated at max_tokens={self.max_tokens} before any "
                "content arrived",
            )

        parsed = parse_action(reply)
        self._meta["parse_form"] = parsed.form or "unreadable"

        if not parsed.ok:
            self._meta["raw_reply"] = reply.strip()[:400]

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
            # Mistral: image_url is a plain string, not {"url": "..."}.
            content.append({
                "type": "image_url",
                "image_url": f"data:image/png;base64,{b64}",
            })

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        request = urllib.request.Request(
            f"{API_URL}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.load(response)
        elapsed = time.monotonic() - started

        choice = body["choices"][0]
        usage = body.get("usage") or {}

        self._meta.update({
            "latency_s": round(elapsed, 2),
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "finish_reason": choice.get("finish_reason", ""),
            "model": body.get("model", self.model),
        })

        return (choice["message"].get("content") or "",
                choice.get("finish_reason", ""))
