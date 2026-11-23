"""Free: statistics of the saved decoder attention patterns. Usage: python s03_attn_stats.py 80 [tag]"""
import sys

import numpy as np

from harness import OUT

period = int(sys.argv[1])
tag = sys.argv[2] if len(sys.argv) > 2 else "base"
d = np.load(OUT / f"cap_{tag}_{period}.npz")
attn = d["attn"].astype(np.float32)  # (8, R, 3, 32, 288)
p = np.arange(32)[:, None] + 256
j = np.arange(288)[None, :]
delta = (32 * (p - j)) % period
align = np.cos(2 * np.pi * delta / period)  # (32, 288)
dist = (p - j).astype(np.float32)
groups = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)]
np.set_printoptions(precision=2, suppress=True, linewidth=220)
print(f"cycle {period}, tag {tag}. Columns: decoder patches 0-3, 4-7, ..., 28-31")
for name, f in [
    ("phase alignment of encoder reads (1 = same place in cycle, 0 = blind)", lambda w: (w[..., :256] * align[:, :256]).sum(-1) / w[..., :256].sum(-1).clip(1e-6)),
    ("attention mass on encoder keys", lambda w: w[..., :256].sum(-1)),
    ("mass on own token", lambda w: np.stack([w[..., i, 256 + i] for i in range(32)], -1)),
    ("mass on earlier decoder tokens", lambda w: w[..., 256:].sum(-1) - np.stack([w[..., i, 256 + i] for i in range(32)], -1)),
    ("mean distance (patches) of encoder reads", lambda w: (w[..., :256] * dist[:, :256]).sum(-1) / w[..., :256].sum(-1).clip(1e-6)),
    ("mass on the last 4 context patches", lambda w: w[..., 252:256].sum(-1)),
]:
    print("\n==", name)
    for layer in range(8):
        for h in range(3):
            v = f(attn[layer, :, h]).mean(0)  # (32,)
            print(f"  {layer}.{h}: " + " ".join(f"{v[a:b].mean():7.3f}" for a, b in groups))
