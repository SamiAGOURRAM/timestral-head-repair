"""Free: what does the saved (unmirrored, anchor-frame) median forecast look like far out?
Folds the forecast at the true cycle and at other candidate cycles. Usage: python s06_forecast_shape.py 80 [tag]"""
import sys

import numpy as np

import check3_synthetic_lead_time as synth
from harness import OUT

period = int(sys.argv[1])
tag = sys.argv[2] if len(sys.argv) > 2 else "base"
x, meta = synth.make_series(period, synth.N_SERIES, seed=period)
med = np.load(OUT / f"cap_{tag}_{period}.npz")["median"].astype(np.float64)  # (64, 1024)
truth = x[:, synth.CONTEXT:]
level = x[:, : synth.CONTEXT].mean(1, keepdims=True)
depth = np.array([m["depth"] for m in meta])[:, None]
z = (med - np.median(med, axis=1, keepdims=True)) / depth  # forecast in units of dip depth
zt = (truth - np.median(truth, axis=1, keepdims=True)) / depth
np.set_printoptions(precision=2, suppress=True, linewidth=250)
print("depth-normalised forecast, mean over series, by 64-step block of lead: min over a per-series view")
for a in range(0, 1024, 128):
    seg = z[:, a : a + 128]
    print(f"  lead {a:4d}-{a + 128:4d}: std within series {seg.std(1).mean():.3f}   min {seg.min(1).mean():.3f}   (truth std {zt[:, a:a+128].std(1).mean():.3f})")
# correlation between forecast and truth shifted by a lag, far region
print("\ncorrelation of forecast with truth shifted by lag (steps), lead 512-1024, mean over series")
far = slice(512, 1024)
lags = range(-period // 2, period // 2 + 1, max(1, period // 20))
out = []
for lag in lags:
    c = []
    for i in range(len(x)):
        t = x[i, synth.CONTEXT + 512 + lag : synth.CONTEXT + 1024 + lag] if lag <= 0 else np.r_[x[i, synth.CONTEXT + 512 + lag :], x[i, synth.CONTEXT + 1024 - period : synth.CONTEXT + 1024 - period + lag]]
        c.append(np.corrcoef(med[i, far], t)[0, 1])
    out.append(np.nanmean(c))
print("  lags:", list(lags))
print("  corr:", np.array(out))
print("same, lead 0-384")
out = []
for lag in lags:
    c = [np.corrcoef(med[i, 0:384], x[i, synth.CONTEXT + lag : synth.CONTEXT + 384 + lag])[0, 1] for i in range(len(x))]
    out.append(np.nanmean(c))
print("  corr:", np.array(out))
# autocorrelation of the far forecast
print("\nautocorrelation of the forecast over lead 384-1024 at candidate lags (mean over series)")
seg = z[:, 384:]
seg = seg - seg.mean(1, keepdims=True)
for lag in [32, 64, 80, 90, 96, 128, 160, 180, 192, 240, 256, 320, 360]:
    ac = (seg[:, lag:] * seg[:, :-lag]).mean(1) / seg.var(1)
    print(f"  lag {lag:4d}: {ac.mean():.3f}")
# one example series, forecast per 8-step bin
for i in (0, 1):
    print(f"\nseries {i}: dip starts {meta[i]['starts']}, width {meta[i]['width']}; forecast in depth units, 16-step means")
    print(z[i].reshape(64, 16).mean(1))
