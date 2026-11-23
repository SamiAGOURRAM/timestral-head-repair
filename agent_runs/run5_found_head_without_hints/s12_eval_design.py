"""Step 12: evaluate a design_lags edit of L5h1 (4 passes each).
Usage: s12_eval_design.py "D:a,11:b" ["D:a,11:b" ...]"""
import sys
import torch
from harness import Bench, show
from wsp import evaluate
from edits import design_lags

ref = torch.cat([torch.load(f"results/capture_base_{p}.pt", weights_only=False)["attn_in"][5] for p in (80, 180)])
b = Bench()
for spec in sys.argv[1:]:
    peaks = [(int(t.split(":")[0]), float(t.split(":")[1])) for t in spec.split(",")]
    m = b.fresh()
    design_lags(m, 5, 1, peaks, ref)
    r = evaluate(b, m, f"L5h1 design {spec}")
    print(show(r)); print("   profile:", r["profile"])
