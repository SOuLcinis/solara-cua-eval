#!/usr/bin/env python3
"""Offline demonstration of harness-fault attribution. No API key, no browser.

WHAT THIS IS
    A deterministic replay of scripted action traces through two executors --
    the original buggy one and the fixed one -- scored both ways.

WHAT THIS IS NOT
    An empirical result. There are no model outputs here. The action traces are
    hand-written to represent what a model plausibly emits; the point is to show
    what the SCORING does with them, not to measure any model.

    Real cross-model numbers require running scripts/run_suite.py against live
    backends. Those results go in results/ and are labelled with the backend.

Run:  python scripts/demo_offline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solara_cua.eval.instrument import execute_recorded  # noqa: E402
from solara_cua.eval.record import RunRecord, write_jsonl  # noqa: E402
from solara_cua.eval.report import compare_backends, format_summary  # noqa: E402
from solara_cua.fakes import FakePage  # noqa: E402

WIDTH, HEIGHT = 1440, 900


def legacy_executor(fname, args, page, width, height):
    """The original dispatch, reduced to the paths that mattered.

    Reproduces two real bugs: triple_click had no branch, and a click carrying
    button="right" was caught by the plain-click branch first.
    """
    if fname in ("click", "click_at", "triple_click", "right_click"):
        x = int(float(args["x"]) / 1000 * width)
        y = int(float(args["y"]) / 1000 * height)
        if fname in ("click", "click_at"):
            page.mouse.click(x, y)          # BUG: ignores args["button"]
        elif fname == "right_click":
            page.mouse.click(x, y, button="right")
        # BUG: triple_click falls through -- nothing happens, nothing raised
    elif fname == "navigate":
        page.goto(args["url"])
    elif fname == "hotkey":
        page.keyboard.press("+".join(args["keys"]))
    # BUG: anything else warns and returns as though it succeeded


# Each task: an action trace, plus an explicit criterion evaluated against what
# the page actually received. The criterion never inspects the executor -- it
# only asks whether the page ended up in the required state.
TASKS = [
    {
        "id": "select-and-copy-paragraph",
        "actions": [
            ("navigate", {"url": "https://example.com"}),
            ("triple_click", {"x": 500, "y": 400}),
            ("hotkey", {"keys": ["Control", "C"]}),
        ],
        # Passes only if a multi-click actually landed on the paragraph.
        "criterion": lambda page: any(
            c[1] == "click" and c[3].get("click_count") == 3 for c in page.only("mouse")
        ),
    },
    {
        "id": "open-context-menu",
        "actions": [
            ("navigate", {"url": "https://example.com"}),
            ("click", {"x": 300, "y": 300, "button": "right"}),
        ],
        "criterion": lambda page: any(
            c[3].get("button") == "right" for c in page.only("mouse")
        ),
    },
    {
        "id": "navigate-to-docs",
        # A genuine model error: it went to the wrong host. Both executors
        # perform the action correctly, so this must score as a model failure.
        "actions": [("navigate", {"url": "https://wrong-site.example"})],
        "criterion": lambda page: any(
            "docs" in str(c[2][0]) for c in page.only("page") if c[1] == "goto"
        ),
    },
    {
        "id": "pinch-zoom-image",
        # The model asks for a gesture neither executor implements.
        "actions": [("pinch_zoom", {"x": 500, "y": 500, "scale": 2})],
        "criterion": lambda page: False,
    },
]


def run_suite(backend, perform):
    runs = []
    for task in TASKS:
        page = FakePage()
        run = RunRecord(task_id=task["id"], backend=backend)
        for fname, args in task["actions"]:
            kw = {"perform": perform} if perform else {}
            run.add(execute_recorded(fname, args, page, WIDTH, HEIGHT,
                                     settle_fn=lambda _p, **_k: True, **kw))
        run.passed = task["criterion"](page)
        run.turns_used = len(task["actions"])
        runs.append(run)
    return runs


def main():
    all_runs = run_suite("legacy-executor", legacy_executor) + run_suite("fixed-executor", None)

    out = Path(__file__).resolve().parent.parent / "results" / "demo.jsonl"
    out.write_text("")  # demo output is regenerated, not appended
    write_jsonl(all_runs, out)

    rows = [r.to_dict() for r in all_runs]
    for backend, summary in compare_backends(rows).items():
        print(format_summary(summary, backend=backend))
        print()

    print("per-run verdicts")
    print("-" * 58)
    for r in all_runs:
        print(f"  {r.backend:<17} {r.task_id:<26} {r.verdict.value}")
    print()
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
