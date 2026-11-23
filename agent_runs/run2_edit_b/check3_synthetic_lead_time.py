"""Check 3: on synthetic series, does Timestral-5M lose a repeating event at long lead times while 28M keeps it?

Hypothesis from checks 1 and 2 (bizitobs 10-second data, one-hour cycle with a dip): 5M
fails to forecast a dip that lies roughly 465 to 525 steps ahead; 28M does not.

Synthetic family, one variate, no covariates: constant level + white noise + a dip of
fixed depth and width once every PERIOD steps. Per series the level, noise size, dip depth
(3 to 6 noise standard deviations) and the position of the cycle are drawn at random; the
dip width is PERIOD / 6 (60 steps for a 360-step cycle, as in the real data). The model
sees 8192 steps and forecasts 1024 (its largest single-pass horizon).

Measure, for every true dip lying fully inside the horizon: the share of the dip the
median forecast shows at the right place,

    recovered = (mean forecast on the flanks - mean forecast inside the dip) / true depth

where the flanks are the dip-width stretches just before and just after the dip. 1 means
the dip is forecast at full depth in the right place, 0 means no dip there. Reported by
lead time (steps from the forecast origin to the start of the dip).

Run: .venv\\Scripts\\python.exe -u experiments\\gap_10s\\check3_synthetic_lead_time.py
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from timestral import TimestralForecaster

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "gap_10s"
SIZES = ["5M", "28M", "123M"]
PERIODS = [90, 180, 360, 720]
CONTEXT = 8192
HORIZON = 1024
N_SERIES = 64
LEAD_BIN = 128
DEVICE = os.environ.get("WSP_DEVICE", "cpu")


def make_series(period: int, n: int, seed: int) -> tuple[np.ndarray, list[dict]]:
    """`(n, CONTEXT + HORIZON)` series and, per series, its depth, width and dip starts in the horizon."""
    rng = np.random.default_rng(seed)
    total = CONTEXT + HORIZON
    width = period // 6
    x = np.empty((n, total), dtype=np.float64)
    meta = []
    for i in range(n):
        level = rng.uniform(20, 100)
        sd = rng.uniform(0.5, 2.0)
        depth = rng.uniform(3, 6) * sd
        phase = int(rng.integers(0, period))
        t = np.arange(total)
        in_dip = ((t - phase) % period) < width
        x[i] = level + rng.normal(0, sd, total) - depth * in_dip
        starts = [s - CONTEXT for s in range(phase, total, period) if s >= CONTEXT and s + width <= total]
        meta.append({"depth": depth, "width": width, "sd": sd, "starts": starts})
    return x, meta


def recovered_shares(median: np.ndarray, meta: list[dict]) -> list[tuple[int, int, float]]:
    """`(series, lead time, recovered share)` for every dip fully inside the horizon with at least one full flank."""
    out = []
    for i, m in enumerate(meta):
        w = m["width"]
        for s in m["starts"]:
            flanks = []
            if s - w >= 0:
                flanks.append(median[i, s - w : s].mean())
            if s + 2 * w <= HORIZON:
                flanks.append(median[i, s + w : s + 2 * w].mean())
            if not flanks:
                continue
            out.append((i, s, float((np.mean(flanks) - median[i, s : s + w].mean()) / m["depth"])))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    if DEVICE == "cpu":
        torch.set_num_threads(12)
    models = {s: TimestralForecaster.from_pretrained(f"mistralai/Timestral-{s}", device=DEVICE) for s in SIZES}
    rows = []
    for period in PERIODS:
        x, meta = make_series(period, N_SERIES, seed=period)
        truth_check = recovered_shares(x[:, CONTEXT:], meta)
        print(f"\n=== cycle of {period} steps, dip width {period // 6}: {N_SERIES} series, {len(truth_check)} dips scored; "
              f"measure applied to the true future = {np.mean([r for _, _, r in truth_check]):.3f} (should be near 1)", flush=True)
        for size in SIZES:
            t0 = time.perf_counter()
            with torch.inference_mode():
                median = models[size].forecast(x[:, None, :CONTEXT].astype(np.float32), HORIZON).median[:, 0].double().cpu().numpy()
            sd = np.array([m["sd"] for m in meta])[:, None]
            mae = float((np.abs(median - x[:, CONTEXT:]) / sd).mean())
            for i, lead, share in recovered_shares(median, meta):
                rows.append({"period": period, "model": size, "series": i, "lead": lead, "recovered": share})
            print(f"    {size:>4}: mean absolute error / noise sd over the horizon = {mae:.3f} ({time.perf_counter() - t0:.0f} s)", flush=True)
        df = pd.DataFrame([r for r in rows if r["period"] == period])
        df["lead time"] = (df["lead"] // LEAD_BIN) * LEAD_BIN
        mean = df.pivot_table(index="lead time", columns="model", values="recovered", aggfunc="mean")[SIZES]
        sem = df.pivot_table(index="lead time", columns="model", values="recovered", aggfunc="sem")[SIZES]
        count = df[df["model"] == SIZES[0]].groupby("lead time").size()
        table = pd.DataFrame({f"{s}": mean[s].round(3).astype(str) + " +/- " + sem[s].round(3).astype(str) for s in SIZES})
        table["dips"] = count
        table.index = [f"{a}-{a + LEAD_BIN}" for a in table.index]
        print("    share of the dip recovered, by lead time in steps (mean +/- standard error over dips):")
        print("    " + table.to_string().replace("\n", "\n    "), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "check3_synthetic.csv", index=False)

    df = pd.DataFrame(rows)
    df["half"] = np.where(df["lead"] < HORIZON // 2, "lead under 512", "lead 512 and over")
    summary = df.pivot_table(index=["period", "half"], columns="model", values="recovered", aggfunc="mean")[SIZES]
    print("\n=== Summary: mean share of the dip recovered")
    print(summary.round(3).to_string())
    print(f"\nWrote {(OUT / 'check3_synthetic.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
