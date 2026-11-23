"""Counted: official score (4 passes each) of head-5.1 query-gain edits saved by s13_design2.py.
Usage: python s14_score_gain.py name1 name2 ..."""
import sys

import edits
from harness import Bench, show

bench = Bench()
for name in sys.argv[1:]:
    model = bench.fresh()
    edits.apply_gain_file(model, name)
    print(show(bench.score(model, f"gain {name}")), flush=True)
