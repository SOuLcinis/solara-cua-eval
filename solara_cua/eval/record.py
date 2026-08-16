"""Durable records for computer-use evaluation runs.

Everything here serializes to JSONL. That is deliberate: the reason three
previous builds of this system left nothing behind is that they logged prose to
a notes file instead of writing structured results. Prose does not survive a
rebuild, because nobody thinks to carry it forward. A results file with numbers
in it does.

Records are append-only and self-describing -- a run file should be readable
years later without the code that produced it.
"""
import json
from dataclasses import dataclass, field, asdict

from solara_cua.eval.taxonomy import (
    ALL_FAULTS,
    ActionOutcome,
    RunVerdict,
    verdict_for,
)

SCHEMA_VERSION = 2
"""v2 adds satisfied_at_turn and stopped_by. v1 results scored the criterion only
at the end of a run, which counted a solved task as a failure whenever the model
kept acting and undid its own work -- so v1 and v2 numbers are not comparable."""


@dataclass
class ActionRecord:
    """One action the model requested, and what actually happened to it."""

    action: str
    outcome: ActionOutcome
    args: dict = field(default_factory=dict)
    detail: str = ""
    settled: bool = True
    """False if the page never reached domcontentloaded. Not a fault -- see
    executor.settle -- but worth keeping, because a run full of unsettled pages
    is a run whose screenshots may have been captured mid-load."""

    meta: dict = field(default_factory=dict)
    """Per-turn facts from the backend: latency, tokens, which parser reading was
    needed. Kept beside the outcome rather than in it, because none of it changes
    whether the action was performed -- but all of it changes how the result
    should be read. A model that is 5% better and 40x slower is a different
    product, not a better one, and a bare accuracy column cannot say so."""

    @property
    def is_fault(self):
        return self.outcome in ALL_FAULTS

    def to_dict(self):
        d = asdict(self)
        d["outcome"] = self.outcome.value
        return d


@dataclass
class RunRecord:
    """A single task attempted by a single backend."""

    task_id: str
    backend: str
    actions: list = field(default_factory=list)
    passed: bool = False
    turn_limit_hit: bool = False
    turns_used: int = 0
    notes: str = ""

    satisfied_at_turn: int = None
    """The turn on which the success criterion FIRST became true, if ever.

    The criterion is checked after every action, not only at the end. Scoring
    end-state alone was wrong in a way that looked exactly like model
    incapability: the local model solved a text-entry task on turn 2, kept
    acting because it had not recognised success, and had typed over its own
    correct answer by turn 6. End-state scoring recorded that as a failure.
    """

    still_satisfied_at_end: bool = None
    """Whether the goal state survived to the end of the episode.

    Separate from `satisfied_at_turn` because "never achieved it" and "achieved
    it then undid it" are different failures with different fixes, and a single
    pass/fail bit cannot tell them apart.
    """

    stopped_by: str = ""
    """Why the episode ended: model_done, turn_limit, or criterion_error.

    The episode deliberately does NOT end when the task is solved. Stopping
    there scores correctly but destroys the signal -- the model never gets the
    chance to say it is finished, so every run ends by harness intervention and
    "solved it" becomes indistinguishable from "solved it and knew it". Those
    are different abilities, and letting the episode run is what makes the
    difference observable.
    """

    @property
    def regressed(self):
        """Reached the goal state and then left it. Reported, never hidden."""
        return bool(self.passed) and self.still_satisfied_at_end is False

    @property
    def self_terminated(self):
        """The model declared itself finished, rather than running out of turns."""
        return self.stopped_by == "model_done"

    def add(self, record):
        self.actions.append(record)
        return record

    @property
    def outcomes(self):
        return [a.outcome for a in self.actions]

    @property
    def faults(self):
        return [a for a in self.actions if a.is_fault]

    @property
    def verdict(self):
        """Never stored -- always derived, so a changed taxonomy rescores old runs."""
        return verdict_for(
            self.outcomes, passed=self.passed, turn_limit_hit=self.turn_limit_hit
        )

    def to_dict(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": self.task_id,
            "backend": self.backend,
            "passed": self.passed,
            "turns_used": self.turns_used,
            "turn_limit_hit": self.turn_limit_hit,
            "satisfied_at_turn": self.satisfied_at_turn,
            "still_satisfied_at_end": self.still_satisfied_at_end,
            "stopped_by": self.stopped_by,
            "regressed": self.regressed,
            "self_terminated": self.self_terminated,
            "verdict": self.verdict.value,
            "action_count": len(self.actions),
            "fault_count": len(self.faults),
            "actions": [a.to_dict() for a in self.actions],
            "notes": self.notes,
        }


def write_jsonl(runs, path):
    """Append runs to a JSONL results file. One run per line."""
    with open(path, "a", encoding="utf-8") as f:
        for run in runs:
            f.write(json.dumps(run.to_dict(), sort_keys=True) + "\n")


def read_jsonl(path):
    """Read raw run dicts back. Returns dicts, not RunRecords.

    Analysis should work on the durable on-disk shape, not on live objects --
    that is what lets a results file outlive the code that wrote it.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
