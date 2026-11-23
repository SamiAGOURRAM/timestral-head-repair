"""Step 13: hooked pass per cycle (2 passes) with the dual-peak L5h1 edit (43:1.5, 11:0.5);
then the in-phase attention ratio per decoder token as in s05, before vs after."""
import numpy as np
import torch
from harness import Bench
from wsp import hooked_run, attention_probs
from edits import design_lags

ref = torch.cat([torch.load(f"results/capture_base_{p}.pt", weights_only=False)["attn_in"][5] for p in (80, 180)])
b = Bench()
m = b.fresh()
design_lags(m, 5, 1, [(43, 1.5), (11, 0.5)], ref)
W = {80: 6, 180: 8}
for p in (80, 180):
    x, meta = b.sets[p]
    st_new = hooked_run(m, x, f"capture dual43 | cycle {p}")
    st_old = torch.load(f"results/capture_base_{p}.pt", weights_only=False)
    q = 256 + np.arange(32); k = np.arange(288)
    off = ((q[:, None] - k[None, :]) * 32) % p; off = np.minimum(off, p - off)
    inphase = (off <= W[p]) & (k[None, :] < 256); chance = inphase[:, :256].mean(1)
    print(f"\n== cycle {p}: in-phase/chance per decoder j (every 2nd j), before -> after")
    for li, h in [(6, 1), (6, 2), (7, 0), (7, 2)]:
        rows = []
        for nm, st, mm in (("before", st_old, b.base), ("after", st_new, m)):
            P, _ = attention_probs(mm.transformer.layers[li].attns[0], st["attn_in"][li], st["cos"], st["sin"])
            a = (P.mean(0).numpy()[h] * inphase).sum(-1) / chance
            rows.append(f"{nm:6s} " + " ".join(f"{v:4.1f}" for v in a[::2]))
        print(f"L{li}h{h}\n  " + "\n  ".join(rows))
