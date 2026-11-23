"""Free: for position-like heads, the keys that get the most attention per decoder position
(averaged over series). Key index printed relative to the forecast origin: -1 = last context
patch, 0.. = decoder tokens. Usage: python s04_attn_top.py 80 [tag]"""
import sys

import numpy as np

from harness import OUT

period = int(sys.argv[1])
tag = sys.argv[2] if len(sys.argv) > 2 else "base"
heads = [(1, 2), (3, 1), (4, 0), (4, 1), (4, 2), (5, 0), (5, 1), (5, 2)]
if len(sys.argv) > 3:
    heads = [tuple(int(c) for c in s.split(".")) for s in sys.argv[3].split(",")]
attn = np.load(OUT / f"cap_{tag}_{period}.npz")["attn"].astype(np.float32)
for layer, h in heads:
    print(f"\n== head {layer}.{h}, cycle {period}: per decoder patch p, top keys (relative index: mass), then std over series of the top mass")
    w = attn[layer, :, h]  # (R, 32, 288)
    m = w.mean(0)
    for p in range(32):
        top = np.argsort(-m[p])[:5]
        print(f"  p={p:2d}: " + "  ".join(f"{k - 256:+4d}:{m[p, k]:.2f}" for k in top) + f"   sd {w[:, p, top[0]].std():.2f}")
