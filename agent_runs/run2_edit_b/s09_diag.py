"""Counted diagnostics (1 pass per spec and cycle). Usage: python s09_diag.py <cycles e.g. 80,180> <shiftspec|none> spec1 spec2 ...
The baseline line is computed from the saved capture (free)."""
import sys

import numpy as np

import diag
import edits
from harness import OUT, Bench, passes_used

cycles = [int(c) for c in sys.argv[1].split(",")]
shift = sys.argv[2]
bench = Bench()
model = bench.fresh()
edits.apply_spec(model, shift)
for period in cycles:
    base = np.load(OUT / f"cap_base_{period}.npz")["median"].astype(np.float64)
    print(f"cycle {period} baseline (saved)      : {diag.summary(base, period)}")
    for spec in sys.argv[3:]:
        med, _ = diag.run(bench, model, period, spec, f"shift {shift}")
        print(f"cycle {period} {spec:28s}: {diag.summary(med, period)}", flush=True)
print("passes used", passes_used())
