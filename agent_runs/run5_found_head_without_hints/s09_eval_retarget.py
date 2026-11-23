"""Step 9: evaluate L5h1 retargeted to lag D (4 passes per D). Usage: s09_eval_retarget.py D [D ...]"""
import sys
import torch
from harness import Bench, show
from wsp import evaluate
from edits import retarget_lag

ref = torch.cat([torch.load(f"results/capture_base_{p}.pt", weights_only=False)["attn_in"][5] for p in (80, 180)])
b = Bench()
for D in map(int, sys.argv[1:]):
    m = b.fresh()
    retarget_lag(m, 5, 1, D, ref)
    r = evaluate(b, m, f"L5h1 retarget lag {D}")
    print(show(r)); print("   profile by 128-step lead bin:", r["profile"])
