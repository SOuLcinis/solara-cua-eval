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

SCHEMA_VERSION = 1


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
