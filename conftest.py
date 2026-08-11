"""Present so pytest puts the repo root on sys.path.

pytest's default import mode prepends the first directory above a test file that
is not a package. `tests/` has no __init__.py, so without a conftest.py here that
would be `tests/` itself, and `import solara_cua` would fail. A rootdir conftest
makes the repo root importable instead.

Intentionally empty of fixtures -- shared test doubles live in solara_cua.fakes,
which ships as library code because it is useful outside the test suite.
"""
