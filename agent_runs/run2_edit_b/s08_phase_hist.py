"""Free: for each head, the share of encoder attention by phase offset delta = 32 * (p - j) mod period
(0 = key patch at the same place in the cycle as the decoder patch; delta = 16 means the key
patch sits 16 steps EARLIER in the cycle). Usage: python s08_phase_hist.py 80 [tag] [heads]"""
import sys

import numpy as np

from harness import OUT

period = int(sys.argv[1])
tag = sys.argv[2] if len(sys.argv) > 2 else "base"
heads = [(6, 0), (6, 1), (6, 2), (7, 0), (7, 1), (7, 2)]
if len(sys.argv) > 3:
    heads = [tuple(int(c) for c in s.split(".")) for s in sys.argv[3].split(",")]
attn = np.load(OUT / f"cap_{tag}_{period}.npz")["attn"].astype(np.float32)
p = np.arange(32)[:, None] + 256
j = np.arange(256)[None, :]
delta = (32 * (p - j)) % period  # (32, 256)
nb = 10
bins = (delta * nb // period).astype(int)
groups = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)]
print(f"cycle {period} tag {tag}; bins of delta: " + " ".join(f"{b * period // nb:4d}" for b in range(nb)))
for layer, h in heads:
    w = attn[layer, :, h, :, :256].mean(0)  # (32, 256)
    print(f"== head {layer}.{h}")
    for a, b in groups:
        hist = np.zeros(nb)
        for pp in range(a, b):
            np.add.at(hist, bins[pp], w[pp])
        hist /= hist.sum()
        print(f"  p {a:2d}-{b - 1:2d}: " + " ".join(f"{v:4.2f}" for v in hist))
