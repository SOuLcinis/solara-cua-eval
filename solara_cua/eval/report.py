"""Aggregate run records into the numbers that make the argument.

The headline comparison:

    naive success rate         passes / all runs
    attributable success rate  passes / runs whose outcome can be attributed

The gap between them is the size of the measurement error a single success-rate
number hides. Every contaminated run is a failure a naive benchmark charges to
the model's reasoning when the harness never performed the action at all.

Operates on plain dicts from a results file, not live objects, so old results
stay analysable after the code moves on.
"""
from collections import Counter

from solara_cua.eval.taxonomy import ALL_FAULTS, RunVerdict


def _verdict(row):
    return row["verdict"] if isinstance(row, dict) else row.verdict.value


def _field(row, name, default=None):
    """Read a field from a dict row or a live record.

    Defaults rather than raising, so a v1 results file -- written before these
    fields existed -- still summarizes instead of crashing. The missing fields
    read as zero, which is honest: those runs genuinely do not carry the signal.
    """
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def summarize(rows):
    """Compute the metrics for a set of run dicts.

    Returns counts alongside rates, never rates alone: a 100% success rate over
    two runs and over two hundred are not the same claim, and a bare percentage
    hides which one you are looking at.
    """
    rows = list(rows)
    total = len(rows)
    verdicts = Counter(_verdict(r) for r in rows)

    passes = verdicts[RunVerdict.PASS.value]
    contaminated = verdicts[RunVerdict.CONTAMINATED.value]
    ambiguous = verdicts[RunVerdict.AMBIGUOUS.value]

    # A run is attributable when nothing about the harness invalidates the result.
    attributable = total - contaminated - ambiguous

    fault_kinds = Counter()
    for r in rows:
        for a in r.get("actions", []) if isinstance(r, dict) else []:
            if a["outcome"] in {f.value for f in ALL_FAULTS}:
                fault_kinds[a["outcome"]] += 1

    # Two facts a success rate cannot carry. `regressed` counts runs that reached
    # the goal and then left it -- a real failure, but a different one from never
    # getting there. `self_terminated` counts runs the model ended itself rather
    # than running out of turns: knowing you are finished is a separate ability
    # from finishing, and an agent that never stops is a problem a pass rate
    # cannot see.
    regressed = sum(1 for r in rows if _field(r, "regressed"))
    self_terminated = sum(1 for r in rows if _field(r, "self_terminated"))

    parse_forms = Counter()
    latencies = []
    tokens = 0
    for r in rows:
        for a in (r.get("actions", []) if isinstance(r, dict) else []):
            meta = a.get("meta") or {}
            if meta.get("parse_form"):
                parse_forms[meta["parse_form"]] += 1
            if meta.get("latency_s") is not None:
                latencies.append(meta["latency_s"])
            tokens += meta.get("completion_tokens") or 0

    return {
        "runs": total,
        "verdicts": dict(verdicts),
        "passes": passes,
        "regressed": regressed,
        "self_terminated": self_terminated,
        "parse_forms": dict(parse_forms),
        "turns_timed": len(latencies),
        "mean_turn_latency_s": (sum(latencies) / len(latencies)) if latencies else None,
        "total_turn_seconds": round(sum(latencies), 1) if latencies else None,
        "completion_tokens": tokens or None,
        "contaminated": contaminated,
        "ambiguous": ambiguous,
        "attributable_runs": attributable,
        "naive_success_rate": (passes / total) if total else None,
        "attributable_success_rate": (passes / attributable) if attributable else None,
        "misattributed_failures": contaminated,
        "contamination_rate": (contaminated / total) if total else None,
        "fault_kinds": dict(fault_kinds),
    }


def _pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def format_summary(summary, backend=""):
    """Render a summary as plain text for a terminal or a writeup."""
    lines = []
    head = f"computer-use eval summary{f' -- {backend}' if backend else ''}"
    lines.append(head)
    lines.append("=" * len(head))
    lines.append(f"runs                       {summary['runs']}")
    lines.append(f"passes                     {summary['passes']}")
    lines.append("")
    lines.append(f"naive success rate         {_pct(summary['naive_success_rate'])}"
                 "   (passes / all runs)")
    lines.append(f"attributable success rate  {_pct(summary['attributable_success_rate'])}"
                 f"   (passes / {summary['attributable_runs']} attributable runs)")
    lines.append("")
    lines.append(f"contaminated runs          {summary['contaminated']}"
                 f"   ({_pct(summary['contamination_rate'])} of all runs)")
    lines.append(f"ambiguous runs             {summary['ambiguous']}")
    lines.append("")
    lines.append(f"reached goal then undid it {summary.get('regressed', 0)}"
                 "   (counted as a pass; the goal state WAS reached)")
    lines.append(f"model stopped on its own   {summary.get('self_terminated', 0)}"
                 f" of {summary['runs']}   (rest ran to the turn limit)")

    if summary["contaminated"]:
        lines.append("")
        lines.append(f"  {summary['misattributed_failures']} run(s) would be scored as MODEL")
        lines.append("  failures by a plain success-rate benchmark. The harness")
        lines.append("  never performed the requested action.")

    if summary.get("mean_turn_latency_s") is not None:
        lines.append("")
        lines.append(f"mean turn latency          {summary['mean_turn_latency_s']:.1f}s"
                     f"   ({summary['turns_timed']} turns, "
                     f"{summary['total_turn_seconds']}s total)")

    if summary.get("parse_forms"):
        lines.append("")
        lines.append("reply formats the parser had to accept")
        for form, n in sorted(summary["parse_forms"].items(), key=lambda kv: -kv[1]):
            note = "  <- strict JSON would reject these" if form in (
                "xml_tag", "embedded_json", "fenced_json") else ""
            lines.append(f"  {form:<28} {n}{note}")

    if summary["fault_kinds"]:
        lines.append("")
        lines.append("fault breakdown")
        for kind, n in sorted(summary["fault_kinds"].items(), key=lambda kv: -kv[1]):
            lines.append(f"  {kind:<28} {n}")

    return "\n".join(lines)


def compare_backends(rows):
    """Group runs by backend and summarize each.

    Cross-model comparison is the point: a failure mode that appears in one
    model's runs and not another's is a model property, while one that appears
    everywhere is almost certainly the harness.
    """
    by_backend = {}
    for r in rows:
        by_backend.setdefault(r["backend"], []).append(r)
    return {b: summarize(rs) for b, rs in sorted(by_backend.items())}
