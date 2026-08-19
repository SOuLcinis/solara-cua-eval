"""Tests for the model-adapter layer. No server, no browser, no key.

Everything here is about attribution: which side of the harness/model line a
given kind of bad turn falls on. Getting that wrong is not a cosmetic bug -- it
moves numbers in the direction of whichever story the harness happens to tell.
"""
import pytest

from solara_cua.backends.gemini import GeminiBackend
from solara_cua.backends.local_vlm import (
    MIN_SANE_MAX_TOKENS,
    LocalVLMBackend,
)
from solara_cua.backends.mistral import MistralBackend
from solara_cua.backends.parse import parse_action
from solara_cua.backends.prompt import ACTION_VOCABULARY, SYSTEM_PROMPT, user_prompt
from solara_cua.eval.runner import Backend, NonAction, run_task
from solara_cua.eval.taxonomy import ActionOutcome, RunVerdict
from solara_cua.eval.tasks import BY_ID
from solara_cua.fakes import FakePage


# ------------------------------------------------------------------- parsing

def test_parses_strict_json():
    r = parse_action('{"action": "click", "x": 500, "y": 500}')
    assert r.action == ("click", {"x": 500, "y": 500})
    assert r.form == "json"


def test_parses_fenced_json():
    r = parse_action('Here you go:\n```json\n{"action":"click","x":1,"y":2}\n```')
    assert r.action == ("click", {"x": 1, "y": 2})
    assert r.form == "fenced_json"


def test_parses_json_embedded_in_prose():
    """Reasoning models put prose on both sides of the answer."""
    r = parse_action('I should press the button. {"action":"click","x":10,"y":20} Done.')
    assert r.action == ("click", {"x": 10, "y": 20})
    assert r.form == "embedded_json"


def test_parses_the_xml_form_the_local_model_actually_emitted():
    """Observed verbatim from the 4B without constrained decoding.

    It carried the same coordinates as its JSON reply to the same question, so
    rejecting it would log a model failure for a correct answer.
    """
    r = parse_action('<click x="600" y="480"/>')
    assert r.action == ("click", {"x": 600, "y": 480})
    assert r.form == "xml_tag"


def test_done_is_a_claim_not_an_action():
    r = parse_action('{"action": "done"}')
    assert r.done is True
    assert r.action is None


def test_aliases_are_accepted():
    assert parse_action('{"action":"left_click","x":1,"y":2}').action[0] == "click"
    assert parse_action('{"action":"type","x":1,"y":2,"text":"hi"}').action[0] == "type_text_at"


def test_unknown_action_is_a_model_error_not_a_harness_gap():
    """The vocabulary was given in the prompt, so inventing a verb is on the model.

    If this reached the executor it would raise UnhandledActionError and be
    recorded as a HARNESS fault -- contaminating a run over a mistake the harness
    did not make, and inflating the headline contamination number.
    """
    r = parse_action('{"action": "pinch_zoom", "x": 1, "y": 2}')
    assert r.action is None
    assert r.done is False
    assert "vocabulary" in r.error


def test_coordinates_are_coerced_from_strings_and_floats():
    assert parse_action('{"action":"click","x":"600","y":480.4}').action[1] == {
        "x": 600, "y": 480}


def test_coordinates_packed_into_one_field_are_repaired_and_the_repair_is_visible():
    """Observed 10x unconstrained: {"action":"click","x":[500,828]}.

    Same grounding, different packing. Rejecting it charges a serialization
    quirk to capability -- but the repair is reported as its own parse form so
    it is counted, not quietly enjoyed.
    """
    r = parse_action('{"action":"click","x":[500,828]}')
    assert r.action == ("click", {"x": 500, "y": 828})
    assert r.form == "coord_pair_repaired"


def test_a_packed_pair_with_a_conflicting_y_is_left_to_fail():
    """x:[a,b] beside a different y might be a bounding box. Guessing there
    would be the harness inventing an intention for the model."""
    r = parse_action('{"action":"click","x":[500,828],"y":300}')
    assert r.action is None
    assert "not a number" in r.error


def test_a_packed_pair_of_non_numbers_is_not_repaired():
    r = parse_action('{"action":"click","x":["left","top"]}')
    assert r.action is None


def test_non_numeric_coordinate_is_reported_not_guessed():
    r = parse_action('{"action":"click","x":"middle","y":480}')
    assert r.action is None
    assert "not a number" in r.error


def test_hotkey_string_is_split_into_keys():
    assert parse_action('{"action":"hotkey","keys":"Control+a"}').action[1] == {
        "keys": ["Control", "a"]}


@pytest.mark.parametrize("reply", ["", "   ", "I am not sure what to do here."])
def test_unreadable_replies_report_an_error(reply):
    r = parse_action(reply)
    assert not r.ok
    assert r.error


# -------------------------------------------------------------------- prompt

def test_system_prompt_offers_every_vocabulary_action():
    for name, _params, _desc in ACTION_VOCABULARY:
        assert name in SYSTEM_PROMPT


def test_no_vocabulary_row_uses_a_bare_placeholder_symbol():
    """The params column once read "-" for parameterless actions, and the model
    copied it: `{"action": "go_back", "-"}`. Invalid JSON, recorded as a model
    formatting failure, caused by the prompt.
    """
    for name, params, _desc in ACTION_VOCABULARY:
        assert params.strip() not in ("-", "--", "n/a", ""), (
            f"{name}: placeholder {params!r} can be echoed back as content"
        )


def test_unreadable_reply_is_kept_for_diagnosis():
    backend = _StubBackend("I shall click the thing, probably.")
    backend.next_action(_obs())
    assert "I shall click" in backend.turn_metadata()["raw_reply"]


def test_system_prompt_states_the_coordinate_convention():
    """The 0-1000 normalization is the single most misreadable part of the task."""
    assert "0-1000" in SYSTEM_PROMPT
    assert "1440" in SYSTEM_PROMPT and "900" in SYSTEM_PROMPT


def test_user_prompt_feeds_back_the_previous_error():
    text = user_prompt("do a thing", turn=2, max_turns=8, last_error="boom")
    assert "boom" in text
    assert "do a thing" in text


def test_user_prompt_omits_the_error_line_when_there_was_none():
    assert "previous action failed" not in user_prompt("g", turn=1, max_turns=8)


# ------------------------------------------------------------------- backend

class _StubBackend(LocalVLMBackend):
    """LocalVLMBackend with the network replaced, so the logic can be tested."""

    def __init__(self, reply, finish_reason="stop", **kw):
        super().__init__(**kw)
        self._reply = reply
        self._finish = finish_reason

    def _complete(self, observation):
        return self._reply, self._finish


def _obs(turn=1):
    return {"goal": "g", "turn": turn, "max_turns": 8, "last_error": None,
            "screenshot": None}


def test_rejects_a_token_budget_that_cannot_fit_reasoning():
    """Guards the bug that produced this class: max_tokens=100 on a model that
    always reasons returned an empty string that looked like model failure."""
    with pytest.raises(ValueError, match="MIN_SANE_MAX_TOKENS"):
        LocalVLMBackend(max_tokens=MIN_SANE_MAX_TOKENS - 1)


def test_truncated_empty_reply_is_a_budget_fault_not_a_model_failure():
    backend = _StubBackend("", finish_reason="length")
    result = backend.next_action(_obs())
    assert isinstance(result, NonAction)
    assert result.outcome is ActionOutcome.HARNESS_TOKEN_BUDGET


def test_unparseable_reply_is_attributed_to_the_model():
    backend = _StubBackend("I think I will click something.")
    result = backend.next_action(_obs())
    assert isinstance(result, NonAction)
    assert result.outcome is ActionOutcome.MODEL_UNPARSEABLE


def test_unreachable_server_is_a_driver_error():
    backend = LocalVLMBackend(url="http://127.0.0.1:1", timeout=2)
    result = backend.next_action(_obs())
    assert isinstance(result, NonAction)
    assert result.outcome is ActionOutcome.HARNESS_DRIVER_ERROR


def test_good_reply_becomes_an_action():
    backend = _StubBackend('{"action":"click","x":500,"y":500}')
    assert backend.next_action(_obs()) == ("click", {"x": 500, "y": 500})


def test_done_stops_the_run():
    assert _StubBackend('{"action":"done"}').next_action(_obs()) is None


def test_constrained_flag_shows_up_in_the_backend_name():
    """Constrained decoding is a real advantage over a backend without it, so it
    is recorded in the results rather than quietly enjoyed."""
    assert LocalVLMBackend(constrained=True).name == "local-vlm"
    assert LocalVLMBackend(constrained=False).name == "local-vlm-free"


# -------------------------------------------------------------- runner wiring

class _FixedBackend(Backend):
    """Emits a scripted sequence of whatever it is given, then stops.

    Subclasses Backend rather than duck-typing it, so a stub cannot drift out of
    the interface it stands in for and pass tests a real backend would fail.
    """

    name = "stub"

    def __init__(self, items):
        self._items = list(items)

    def next_action(self, observation):
        return self._items.pop(0) if self._items else None


class _CriterionPage(FakePage):
    """A fake page whose `evaluate` returns a fixed verdict.

    The runner also evaluates a storage-clearing snippet at the start of every
    run, so criterion checks are identified by expression rather than by call
    count -- a fake that counts every evaluate() is measuring the wrong thing.
    """

    SETUP = "sessionStorage.clear"

    def __init__(self, passed=False):
        super().__init__()
        self._passed = passed

    def evaluate(self, expression):
        if self.SETUP in expression:
            return None
        return self._criterion()

    def _criterion(self):
        return self._passed

    def wait_for_load_state(self, *_a, **_kw):
        return True


def test_runner_records_a_non_action_without_touching_the_page():
    page = _CriterionPage()
    backend = _FixedBackend([
        NonAction(ActionOutcome.MODEL_UNPARSEABLE, "gibberish"),
    ])
    run = run_task(BY_ID["click-named-button"], backend, page, "http://x", max_turns=4)

    assert [a.outcome for a in run.actions] == [ActionOutcome.MODEL_UNPARSEABLE]
    assert not page.only("mouse"), "a non-action must never reach the page"


def test_unparseable_output_does_not_contaminate_the_run():
    """A model that writes nonsense has failed. The harness has not."""
    page = _CriterionPage()
    backend = _FixedBackend([NonAction(ActionOutcome.MODEL_UNPARSEABLE, "x")])
    run = run_task(BY_ID["click-named-button"], backend, page, "http://x", max_turns=4)
    assert run.verdict is RunVerdict.FAIL_MODEL


def test_token_budget_fault_makes_the_run_unattributable():
    page = _CriterionPage()
    backend = _FixedBackend([NonAction(ActionOutcome.HARNESS_TOKEN_BUDGET, "x")])
    run = run_task(BY_ID["click-named-button"], backend, page, "http://x", max_turns=4)
    assert run.verdict is RunVerdict.AMBIGUOUS


class _SolveThenBreakPage(_CriterionPage):
    """The criterion goes true after one action, then false again.

    Models the real trace that exposed this bug: the local model completed a
    text-entry task, did not recognise it had succeeded, and typed over its own
    answer four turns later.
    """

    def __init__(self):
        super().__init__()
        self.checks = 0

    def _criterion(self):
        self.checks += 1
        return self.checks == 1


def test_a_solved_task_stays_a_pass_even_if_the_model_undoes_it():
    """The bug this locks down: end-state scoring charged a solved task to the
    model's capability because the model kept acting and destroyed its own work.
    """
    page = _SolveThenBreakPage()
    backend = _FixedBackend([("click", {"x": 500, "y": 500})] * 3)
    run = run_task(BY_ID["click-named-button"], backend, page, "http://x", max_turns=6)

    assert run.passed is True
    assert run.satisfied_at_turn == 1
    assert run.still_satisfied_at_end is False
    assert run.regressed is True


def test_the_episode_is_not_cut_short_on_success():
    """Stopping at success would score correctly and erase the signal: the model
    would never get the chance to say it was finished."""
    page = _SolveThenBreakPage()
    backend = _FixedBackend([("click", {"x": 500, "y": 500})] * 3)
    run = run_task(BY_ID["click-named-button"], backend, page, "http://x", max_turns=6)
    assert len(run.actions) == 3, "episode ended early and lost the stopping signal"


def test_a_clean_pass_is_not_marked_as_regressed():
    page = _CriterionPage(passed=True)
    backend = _FixedBackend([("click", {"x": 1, "y": 1})])
    run = run_task(BY_ID["click-named-button"], backend, page, "http://x", max_turns=4)
    assert run.passed is True
    assert run.still_satisfied_at_end is True
    assert run.regressed is False


def test_stopped_by_records_a_model_that_declared_itself_finished():
    """"Solved it" and "solved it and knew it" are different abilities."""
    page = _CriterionPage(passed=False)
    run = run_task(BY_ID["click-named-button"], _FixedBackend([]), page,
                   "http://x", max_turns=4)
    assert run.stopped_by == "model_done"
    assert run.self_terminated is True


def test_stopped_by_records_running_out_of_turns():
    page = _CriterionPage(passed=False)
    backend = _FixedBackend([("click", {"x": 1, "y": 1})] * 10)
    run = run_task(BY_ID["click-named-button"], backend, page, "http://x", max_turns=3)
    assert run.stopped_by == "turn_limit"
    assert run.turn_limit_hit is True


def test_turn_metadata_lands_on_the_action_record():
    """Latency and parse form have to survive to the results file, or the
    constrained/unconstrained comparison has nothing to compare."""

    class _MetaBackend(_FixedBackend):
        def turn_metadata(self):
            return {"parse_form": "xml_tag", "latency_s": 12.5,
                    "completion_tokens": 200}

    page = _CriterionPage(passed=False)
    backend = _MetaBackend([("click", {"x": 1, "y": 1})])
    run = run_task(BY_ID["click-named-button"], backend, page, "http://x", max_turns=2)

    assert run.actions[0].meta["parse_form"] == "xml_tag"
    assert run.actions[0].to_dict()["meta"]["latency_s"] == 12.5


def test_summary_reports_parse_forms_and_latency():
    from solara_cua.eval.report import format_summary, summarize

    rows = [{
        "backend": "local-vlm-free", "verdict": "pass", "actions": [
            {"outcome": "ok", "meta": {"parse_form": "xml_tag", "latency_s": 10.0,
                                       "completion_tokens": 100}},
            {"outcome": "ok", "meta": {"parse_form": "json", "latency_s": 20.0,
                                       "completion_tokens": 100}},
        ],
    }]
    summary = summarize(rows)
    assert summary["parse_forms"] == {"xml_tag": 1, "json": 1}
    assert summary["mean_turn_latency_s"] == 15.0
    assert summary["completion_tokens"] == 200
    assert "strict JSON would reject these" in format_summary(summary)


def test_summary_survives_results_without_metadata():
    """v1 and oracle results carry no meta. They must summarize, not crash."""
    from solara_cua.eval.report import summarize

    summary = summarize([{"backend": "oracle", "verdict": "pass",
                          "actions": [{"outcome": "ok"}]}])
    assert summary["parse_forms"] == {}
    assert summary["mean_turn_latency_s"] is None


def test_non_action_error_is_fed_back_to_the_model():
    """Denying a model sight of its own failure is the defect in FINDINGS.md."""
    seen = []

    class _Watcher(_FixedBackend):
        def next_action(self, observation):
            seen.append(observation["last_error"])
            return super().next_action(observation)

    page = _CriterionPage()
    backend = _Watcher([NonAction(ActionOutcome.MODEL_UNPARSEABLE, "bad output")])
    run_task(BY_ID["click-named-button"], backend, page, "http://x", max_turns=4)

    assert seen[0] is None
    assert seen[1] == "bad output"


# ------------------------------------------------------------ mistral backend

class _StubMistral(MistralBackend):
    """MistralBackend with the network replaced."""

    def __init__(self, reply, finish_reason="stop", **kw):
        kw.setdefault("api_key", "test-key")
        super().__init__(**kw)
        self._reply = reply
        self._finish = finish_reason

    def _complete(self, observation):
        return self._reply, self._finish


def test_mistral_requires_an_api_key():
    with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
        MistralBackend(api_key=None)


def test_mistral_good_reply():
    backend = _StubMistral('{"action":"click","x":500,"y":500}')
    assert backend.next_action(_obs()) == ("click", {"x": 500, "y": 500})


def test_mistral_truncated_reply_is_a_budget_fault():
    backend = _StubMistral("", finish_reason="length")
    result = backend.next_action(_obs())
    assert isinstance(result, NonAction)
    assert result.outcome is ActionOutcome.HARNESS_TOKEN_BUDGET


def test_mistral_unparseable_reply_is_a_model_failure():
    backend = _StubMistral("Let me think about clicking...")
    result = backend.next_action(_obs())
    assert isinstance(result, NonAction)
    assert result.outcome is ActionOutcome.MODEL_UNPARSEABLE


def test_mistral_done_stops_the_run():
    assert _StubMistral('{"action":"done"}').next_action(_obs()) is None


def test_mistral_metadata_includes_parse_form():
    backend = _StubMistral('{"action":"click","x":1,"y":2}')
    backend.next_action(_obs())
    assert backend.turn_metadata()["parse_form"] == "json"


def test_mistral_unreadable_reply_is_kept():
    backend = _StubMistral("I shall ponder this deeply.")
    backend.next_action(_obs())
    assert "ponder" in backend.turn_metadata()["raw_reply"]


def test_mistral_name_derives_from_model():
    assert _StubMistral("x", model="mistral-small-latest").name == "mistral-small"
    assert _StubMistral("x", model="mistral-large-latest").name == "mistral-large"
    assert _StubMistral("x", model="mistral-medium-2508").name == "mistral-medium-2508"


# ------------------------------------------------------------- gemini backend

class _StubGemini(GeminiBackend):
    """GeminiBackend with the network replaced."""

    def __init__(self, reply, finish_reason="stop", **kw):
        kw.setdefault("api_key", "test-key")
        super().__init__(**kw)
        self._reply = reply
        self._finish = finish_reason

    def _complete(self, observation):
        return self._reply, self._finish


def test_gemini_requires_an_api_key():
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiBackend(api_key=None)


def test_gemini_good_reply():
    backend = _StubGemini('{"action":"click","x":500,"y":500}')
    assert backend.next_action(_obs()) == ("click", {"x": 500, "y": 500})


def test_gemini_truncated_reply_is_a_budget_fault():
    backend = _StubGemini("", finish_reason="length")
    result = backend.next_action(_obs())
    assert isinstance(result, NonAction)
    assert result.outcome is ActionOutcome.HARNESS_TOKEN_BUDGET


def test_gemini_unparseable_reply_is_a_model_failure():
    backend = _StubGemini("I'm not sure what to do here.")
    result = backend.next_action(_obs())
    assert isinstance(result, NonAction)
    assert result.outcome is ActionOutcome.MODEL_UNPARSEABLE


def test_gemini_done_stops_the_run():
    assert _StubGemini('{"action":"done"}').next_action(_obs()) is None


def test_gemini_metadata_includes_parse_form():
    backend = _StubGemini('{"action":"click","x":1,"y":2}')
    backend.next_action(_obs())
    assert backend.turn_metadata()["parse_form"] == "json"


def test_gemini_unreadable_reply_is_kept():
    backend = _StubGemini("Perhaps I should click something.")
    backend.next_action(_obs())
    assert "Perhaps" in backend.turn_metadata()["raw_reply"]


def test_gemini_name_derives_from_model():
    assert _StubGemini("x", model="gemini-3.6-flash").name == "gemini-3.6-flash"
    assert _StubGemini("x", model="gemini-2.5-pro").name == "gemini-2.5-pro"
