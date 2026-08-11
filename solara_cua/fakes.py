"""A recording stand-in for a Playwright page.

Ships as library code rather than test-only scaffolding because it is the thing
that makes the executor testable at all: with no browser, no API key and no
network, every dispatch path can be asserted on directly.

The executor duck-types `page`, so this is a complete substitute for the real
object as far as the executor is concerned.
"""


class Recorder:
    """Records any method call into a shared log instead of performing it."""

    def __init__(self, log, namespace):
        self._log = log
        self._ns = namespace

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self._log.append((self._ns, name, args, kwargs))

        return record


class FakePage:
    """A page whose every interaction is appended to `self.log`.

    Entries are `(namespace, method, args, kwargs)` in call order, so tests can
    assert on both what was done and the sequence it was done in.
    """

    def __init__(self):
        self.log = []
        self.mouse = Recorder(self.log, "mouse")
        self.keyboard = Recorder(self.log, "keyboard")

    def __getattr__(self, name):
        log = self.__dict__["log"]

        def record(*args, **kwargs):
            log.append(("page", name, args, kwargs))

        return record

    def only(self, namespace):
        """Entries from one namespace: 'mouse', 'keyboard' or 'page'."""
        return [c for c in self.log if c[0] == namespace]


class BrokenPage(FakePage):
    """A page whose mouse raises, standing in for a driver-level failure."""

    def __init__(self, exc=None):
        super().__init__()
        self._exc = exc or RuntimeError("driver detached")

    @property
    def mouse(self):
        class _Boom:
            def __getattr__(_self, _name):
                def raiser(*_a, **_kw):
                    raise self._exc

                return raiser

        return _Boom()

    @mouse.setter
    def mouse(self, _value):
        pass  # base __init__ assigns a Recorder; the property supersedes it
