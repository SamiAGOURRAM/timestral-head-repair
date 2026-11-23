"""Held-back evaluation of the editing experiment (PROTOCOL.md, "Data held back"), parts 1 to 4.

Run once, after the first agent run's final edit was frozen (`final_edit.py`). Part 5
(GIFT-Eval) is run separately with `evaluation/gift_eval_run.py --checkpoint`.

Models:
  unedited 5M in one pass; unedited 5M in chunks of 8 and of 12 patches;
  the first agent run's final edit; the three fine-tunes; unedited 28M and 123M.
Data, none of it seen while editing or while scoring the fine-tunes:
  1. dips, cycles 80 and 180, new seeds;
  2. dips, cycles 56, 112, 144, 250, 360 (not whole numbers of patches) and 96, 128;
  3. another shape: a sinusoid plus one short upward spike per cycle (cycles 80, 180, 250);
  4. real data: bizitobs_service, 21 series x 10 start points, 600-step horizon (MASE).

Measures on synthetic sets: share of each near (lead under 384 steps) and far (lead 512
and over) dip or spike that the median forecast shows at the right place, and the mean
absolute error in noise standard deviations.

Run: .venv\\Scripts\\python.exe -u experiments\\edit_5m\\heldback_eval.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "gap_10s"))
import check1_gap_source as real  # noqa: E402
import check3_synthetic_lead_time as synth  # noqa: E402
from timestral import TimestralForecaster, load_model  # noqa: E402

OUT = ROOT / "results" / "edit_5m"
DEVICE = os.environ.get("WSP_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
CK = ROOT / "checkpoints"
MODELS = [  # label, source, patches per pass
    ("5M unedited, one pass", "mistralai/Timestral-5M", 32),
    ("5M unedited, chunks of 8", "mistralai/Timestral-5M", 8),
    ("5M unedited, chunks of 12", "mistralai/Timestral-5M", 12),
    ("5M agent run 1, final edit", CK / "agent_run1_final", 32),
    ("5M fine-tune, 128 diagnostic series", CK / "finetune_fixed_300", 32),
    ("5M fine-tune, cycles 80 and 180", CK / "finetune_family_300", 32),
    ("5M fine-tune, many cycle lengths", CK / "finetune_anycycle_600", 32),
    ("28M unedited, one pass", "mistralai/Timestral-28M", 32),
    ("123M unedited, one pass", "mistralai/Timestral-123M", 32),
]
CANDIDATES = None  # labels checked against the criteria; default: the four edited 5M models above
MISALIGNED = (56, 112, 144, 250, 360)
ALIGNED = (96, 128)


def make_spike_series(period: int, n: int, seed: int):
    """Level + sinusoid + one short upward spike per cycle + noise; meta as in the dip family, plus the clean future."""
    rng = np.random.default_rng(seed)
    total = synth.CONTEXT + synth.HORIZON
    width = max(4, period // 12)
    x = np.empty((n, total))
    meta = []
    t = np.arange(total)
    for i in range(n):
        level, sd = rng.uniform(20, 100), rng.uniform(0.5, 2.0)
        amp, height = rng.uniform(2, 4) * sd, rng.uniform(4, 7) * sd
        phase, phi = int(rng.integers(0, period)), rng.uniform(0, 2 * np.pi)
        clean = level + amp * np.sin(2 * np.pi * t / period + phi) + height * (((t - phase) % period) < width)
        x[i] = clean + rng.normal(0, sd, total)
        starts = [s - synth.CONTEXT for s in range(phase, total, period) if s >= synth.CONTEXT and s + width <= total]
        meta.append({"width": width, "sd": sd, "starts": starts, "clean": clean[synth.CONTEXT :]})
    return x, meta


def bump(series: np.ndarray, s: int, w: int):
    """Mean inside [s, s + w) minus mean of the flanks, or None when no full flank fits in the horizon."""
    flanks = []
    if s - w >= 0:
        flanks.append(series[s - w : s].mean())
    if s + 2 * w <= len(series):
        flanks.append(series[s + w : s + 2 * w].mean())
    if not flanks:
        return None
    return series[s : s + w].mean() - np.mean(flanks)


def spike_shares(median: np.ndarray, meta: list[dict]) -> list[tuple[int, int, float]]:
    out = []
    for i, m in enumerate(meta):
        for s in m["starts"]:
            true = bump(m["clean"], s, m["width"])
            got = bump(median[i], s, m["width"])
            if true is not None and got is not None and abs(true) > 1e-9:
                out.append((i, s, float(got / true)))
    return out


def synthetic_sets() -> dict:
    sets = {}
    for c in (80, 180):
        sets[f"1. dips, cycle {c}, new seed"] = ("dip", "misaligned", *synth.make_series(c, synth.N_SERIES, seed=9000 + c))
    for c in MISALIGNED:
        sets[f"2. dips, cycle {c}"] = ("dip", "misaligned", *synth.make_series(c, synth.N_SERIES, seed=9000 + c))
    for c in ALIGNED:
        sets[f"2. dips, cycle {c} (whole patches)"] = ("dip", "aligned", *synth.make_series(c, synth.N_SERIES, seed=9000 + c))
    for c in (80, 180, 250):
        sets[f"3. spikes, cycle {c}"] = ("spike", "spike", *make_spike_series(c, synth.N_SERIES, seed=9500 + c))
    return sets


def main() -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    OUT.mkdir(parents=True, exist_ok=True)
    sets = synthetic_sets()
    series, official, horizon, season, _ = real.load("bizitobs_service")
    origins = [o for o in range(official[-1], real.FIRST_ORIGIN - 1, -horizon)][::-1]
    rows, real_rows = [], {}
    for label, source, chunk in MODELS:
        t0 = time.perf_counter()
        fc = TimestralForecaster(load_model(source, device=DEVICE), per_shot_horizon_patches=chunk)
        for name, (kind, group, x, meta) in sets.items():
            with torch.inference_mode():
                med = fc.forecast(x[:, None, : synth.CONTEXT].astype(np.float32), synth.HORIZON).median[:, 0].double().cpu().numpy()
            shares = synth.recovered_shares(med, meta) if kind == "dip" else spike_shares(med, meta)
            sd = np.array([m["sd"] for m in meta])[:, None]
            rows.append({"model": label, "set": name, "group": group,
                         "near": float(np.mean([s for _, lead, s in shares if lead < 384])),
                         "far": float(np.mean([s for _, lead, s in shares if lead >= 512])),
                         "error": float((np.abs(med - x[:, synth.CONTEXT :]) / sd).mean())})
        err = np.stack([real.scaled_errors(fc, series, o, horizon, season) for o in origins])
        real_rows[label] = np.nanmean(err, axis=(0, 2, 3))  # per series
        print(f"{label}: done in {time.perf_counter() - t0:.0f} s; bizitobs_service MASE {real_rows[label].mean():.4f}", flush=True)
        del fc
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "heldback_synthetic.csv", index=False)
    order = [m[0] for m in MODELS]
    for col, title in (("far", "FAR dips or spikes recovered (lead 512 steps and over)"),
                       ("near", "NEAR dips or spikes recovered (lead under 384 steps)"),
                       ("error", "mean absolute error / noise sd over the horizon")):
        print(f"\n=== {title}")
        print(df.pivot(index="set", columns="model", values=col)[order].round(3).T.to_string())

    print("\n=== Summary by group (mean over the sets of the group)")
    summary = df.groupby(["model", "group"])[["near", "far", "error"]].mean().unstack("group")
    print(summary.loc[order].round(3).to_string())

    per_series = pd.DataFrame(real_rows)
    per_series.to_csv(OUT / "heldback_bizitobs_per_series.csv", index=False)
    base = per_series[order[0]]
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(per_series), (10_000, len(per_series)))
    print("\n=== 4. bizitobs_service, 21 series x 10 start points, 600-step horizon")
    table = {}
    for label in order:
        a = per_series[label].to_numpy()
        r = a[idx].mean(axis=1) / base.to_numpy()[idx].mean(axis=1)
        table[label] = {"MASE": a.mean(), "ratio to unedited one pass": a.mean() / base.mean(),
                        "interval low": np.quantile(r, 0.025), "interval high": np.quantile(r, 0.975),
                        "series better than unedited": int((a < base.to_numpy()).sum())}
    print(pd.DataFrame(table).T.round(4).to_string())

    print("\n=== Criteria of the protocol, against unedited 5M in one pass")
    mis = summary.xs("misaligned", axis=1, level="group")
    for label in (CANDIDATES or order[3:7]):
        d_far = mis.loc[label, "far"] - mis.loc[order[0], "far"]
        d_near = mis.loc[label, "near"] - mis.loc[order[0], "near"]
        ratio = table[label]["ratio to unedited one pass"]
        print(f"{label}: far dips {d_far:+.3f} (needs at least +0.20: {'met' if d_far >= 0.20 else 'NOT met'}); "
              f"near dips {d_near:+.3f} (needs at least -0.05: {'met' if d_near >= -0.05 else 'NOT met'}); "
              f"bizitobs MASE ratio {ratio:.4f} (needs at most 1: {'met' if ratio <= 1 else 'NOT met'})")
    print("criterion 4 (GIFT-Eval) is evaluated separately")


if __name__ == "__main__":
    main()
