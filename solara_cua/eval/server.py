"""Serves the fixture pages on localhost for the duration of a run.

Bound to 127.0.0.1 on an ephemeral port -- never 0.0.0.0. The fixtures are
harmless, but a benchmark has no business opening a listener onto a LAN, and an
ephemeral port means concurrent runs never collide.

Pages are served over HTTP rather than opened as file:// URLs because file://
has a different origin model: sessionStorage and history behave differently
there, and two of the tasks depend on both.
"""
import contextlib
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures"


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        """Silence per-request stderr logging -- it drowns the run output."""

    def end_headers(self):
        # Fixtures are edited during development and must never be served from
        # cache; a stale fixture would silently invalidate a whole run.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


@contextlib.contextmanager
def serve_fixtures(directory=FIXTURES_DIR):
    """Yield the base URL of a local server for `directory`. Shuts down on exit."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"fixtures directory not found: {directory}")

    handler = partial(_QuietHandler, directory=str(directory))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
