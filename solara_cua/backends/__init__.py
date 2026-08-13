"""Model adapters.

One rule, and everything in this package follows from it: an adapter may change
only how a model is *reached*, never what it is asked or how its answer is
judged. The prompt (prompt.py) and the parser (parse.py) are shared, and the run
loop is identical for every backend. Per-model special-casing is the failure mode
that turns a cross-model comparison into a story about whichever model someone
spent the most time tuning.
"""
