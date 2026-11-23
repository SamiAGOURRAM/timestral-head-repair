"""Free: where does the saved median forecast put each dip, relative to the true dip?
For every true dip, slide a window of the dip width over +-period/2 around the true start and
take the offset with the lowest window mean. Usage: python s07_dip_offsets.py 80 [tag]"""
import sys

import numpy as np

import check3_synthetic_lead_time as synth
from harness import OUT

period = int(sys.argv[1])
tag = sys.argv[2] if len(sys.argv) > 2 else "base"
x, meta = synth.make_series(period, synth.N_SERIES, seed=period)
med = np.load(OUT / f"cap_{tag}_{period}.npz")["median"].astype(np.float64)
rows = []
for i, m in enumerate(meta):
    w = m["width"]
    base = np.median(med[i])
    cs = np.r_[0, np.cumsum(med[i] - base)]
    for s in m["starts"]:
        best, bo = 0, None
        for o in range(-period // 2, period // 2):
            a = s + o
            if a < 0 or a + w > 1024:
                continue
            v = (cs[a + w] - cs[a]) / w / m["depth"]
            if v < best:
                best, bo = v, o
        rows.append((s, bo if bo is not None else np.nan, -best))
rows = np.array(rows, dtype=float)
print(f"cycle {period} tag {tag}: by lead of the true dip: median offset of the forecast dip (steps, + = late), quartiles, mean depth share at that best place, n")
for a in range(0, 1024, 64):
    sel = (rows[:, 0] >= a) & (rows[:, 0] < a + 64)
    if sel.sum() == 0:
        continue
    o = rows[sel, 1]
    print(f"  lead {a:4d}-{a + 64:4d}: offset median {np.nanmedian(o):6.1f}  q25 {np.nanpercentile(o, 25):6.1f} q75 {np.nanpercentile(o, 75):6.1f}   depth {rows[sel, 2].mean():.2f}   n {sel.sum()}")
