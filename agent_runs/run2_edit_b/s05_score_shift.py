"""Counted: score query-shift edits. Each spec costs 4 passes. Usage: python s05_score_shift.py spec1 spec2 ..."""
import sys

import edits
from harness import Bench, show

bench = Bench()
for spec in sys.argv[1:]:
    model = bench.fresh()
    edits.apply_spec(model, spec)
    print(show(bench.score(model, f"shift {spec}")), flush=True)
