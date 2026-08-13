"""Ephemeral browser contexts for evaluation runs.

PRIVACY RULE, non-negotiable: this module never touches the production agent's
browser profile. That profile holds live logged-in cookies for a real account,
and the agent loop base64-encodes a full screenshot into every API request --
so a benchmark run against it would ship someone's signed-in session to three
model vendors, one screenshot at a time.

`browser.new_context()` allocates storage in memory and discards it on close:
no user_data_dir, no cookies carried in, nothing left on disk afterwards. It is
used here in deliberate contrast to `launch_persistent_context`, which is what
the production agent uses and what must never appear in this repo.

Playwright is imported lazily so the rest of the package -- executor, taxonomy,
scoring, the 50 tests -- keeps running with no browser installed.
"""
import contextlib

from solara_cua.eval.tasks import VIEWPORT


@contextlib.contextmanager
def ephemeral_page(viewport=VIEWPORT, headless=True):
    """Yield a Playwright page in a throwaway context. Nothing survives the exit."""
    from playwright.sync_api import sync_playwright

    width, height = viewport
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=headless)
        except Exception as e:
            # Distinguish "no browser installed" from a genuine driver failure.
            # Playwright's own message is long and buries the one-line fix.
            if "Executable doesn't exist" in str(e):
                raise RuntimeError(
                    "Chromium is not installed for Playwright. Run:\n"
                    "    playwright install chromium"
                ) from e
            raise
        try:
            context = browser.new_context(
                viewport={"width": width, "height": height},
                # Fixed so screenshots are byte-comparable across machines.
                device_scale_factor=1,
            )
            page = context.new_page()
            try:
                yield page
            finally:
                # Closed innermost-first. Letting the browser close out from
                # under a live page leaves Playwright's transport with pending
                # work and prints a TargetClosedError traceback at exit that
                # looks like a real failure in the run.
                page.close()
                context.close()
        finally:
            browser.close()


def playwright_available():
    """True if Playwright is importable, so the suite can skip with a reason
    instead of dying on an ImportError that reads like a code bug.

    Deliberately an import check only. Verifying the Chromium binary requires
    starting a driver, and entering `sync_playwright()` a second time in the
    same process orphans the first connection's task -- which prints a
    TargetClosedError traceback at exit that looks exactly like a failed run.
    A missing browser surfaces at launch instead, with an install hint.
    """
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        return False
    return True
