"""Step 3: one hooked pass per diagnostic cycle (2 passes) on the unedited model; save
time-attention inputs, residual streams, RoPE tables and the raw median."""
import torch
from harness import Bench
from wsp import hooked_run

b = Bench()
m = b.fresh()
for p in (80, 180):
    x, meta = b.sets[p]
    st = hooked_run(m, x, f"capture baseline | cycle {p}")
    torch.save(st, f"results/capture_base_{p}.pt")
    print(p, "saved")
