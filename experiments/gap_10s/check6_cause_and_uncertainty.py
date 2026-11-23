"""Check 6: (1) do the cycle-locked heads of Timestral-5M cause the dip to be drawn?
(2) when the far forecast goes flat, does the model still know a dip is coming?

Part 1, causal test. Check 5 found six heads in the last layers of 5M (layer.head 5.1, 6.1,
6.2, 7.0, 7.1, 7.2) that read history at the same place in the cycle. Here their output
is set to zero at the forecast positions only, and the share of each near dip recovered
(lead time under 384 steps, where 5M is good) is measured again. Controls: each head
alone, and three random sets of six heads taken from the other 18.

Part 2, uncertainty. The reported forecast is the median. If the model were unsure of
WHERE the dip falls but sure there is one, the median would go flat while the lower
quantiles drop. For each model, by lead time: the dip recovered in the 0.1, 0.5 and 0.9
quantile forecasts, the width of the 0.1-0.9 band in noise standard deviations (2.56 for
pure noise), and the share of true values inside that band (0.80 when the band is right),
inside the dips and outside them.

Synthetic series of check 3, 180-step cycle, one pass (default setting).

Run: .venv\\Scripts\\python.exe -u experiments\\gap_10s\\check6_cause_and_uncertainty.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check3_synthetic_lead_time as synth  # noqa: E402
import timestral.layers as layers  # noqa: E402
from timestral import TimestralForecaster  # noqa: E402

PERIOD = 180
LOCKED = [(5, 1), (6, 1), (6, 2), (7, 0), (7, 1), (7, 2)]
DEVICE = os.environ.get("WSP_DEVICE", "cpu")
OUT = synth.OUT
H_PATCHES = synth.HORIZON // 32


def forecast_with_heads_off(fc, x: np.ndarray, heads_off: list[tuple[int, int]]) -> np.ndarray:
    """Median forecast `(N, HORIZON)` with the given (layer, head) outputs zeroed at the forecast positions."""
    n_layers = len(fc.model.transformer.layers)
    n_heads = fc.config.num_heads
    n_ctx = synth.CONTEXT // 32
    original = layers._sdpa
    calls = {"n": 0}

    def spy(query, key, value, mask, is_causal=False):
        out = original(query, key, value, mask, is_causal)
        if query.shape[1] == H_PATCHES and key.shape[1] == n_ctx + H_PATCHES:  # forecast tokens, time attention
            layer = calls["n"] % n_layers
            calls["n"] += 1
            off = [h for (l, h) in heads_off if l == layer]
            if off:
                hd = out.shape[-1] // n_heads
                out = out.clone()
                for h in off:
                    out[..., h * hd : (h + 1) * hd] = 0
        return out

    layers._sdpa = spy
    try:
        with torch.inference_mode():
            return fc.forecast(x[:, None, : synth.CONTEXT].astype(np.float32), synth.HORIZON).median[:, 0].double().cpu().numpy()
    finally:
        layers._sdpa = original


def near_far(median: np.ndarray, meta: list[dict]) -> tuple[float, float]:
    shares = synth.recovered_shares(median, meta)
    near = [s for _, lead, s in shares if lead < 384]
    far = [s for _, lead, s in shares if lead >= 512]
    return float(np.mean(near)), float(np.mean(far))


def main() -> None:
    pd.set_option("display.width", 220)
    if DEVICE == "cpu":
        torch.set_num_threads(12)
    x, meta = synth.make_series(PERIOD, synth.N_SERIES, seed=PERIOD)
    sd = np.array([m["sd"] for m in meta])[:, None]
    truth = x[:, synth.CONTEXT :]

    print("=== Part 1. Timestral-5M, heads switched off at the forecast positions", flush=True)
    fc = TimestralForecaster.from_pretrained("mistralai/Timestral-5M", device=DEVICE)
    all_heads = [(l, h) for l in range(len(fc.model.transformer.layers)) for h in range(fc.config.num_heads)]
    others = [lh for lh in all_heads if lh not in LOCKED]
    rng = np.random.default_rng(0)
    cases = {"nothing off": [], "the 6 locked heads": LOCKED}
    for lh in LOCKED:
        cases[f"only {lh[0]}.{lh[1]}"] = [lh]
    for i in range(3):
        pick = [others[j] for j in sorted(rng.choice(len(others), 6, replace=False))]
        cases["6 other heads: " + " ".join(f"{l}.{h}" for l, h in pick)] = pick
    cases["all 18 other heads"] = others
    rows = {}
    for label, off in cases.items():
        median = forecast_with_heads_off(fc, x, off)
        near, far = near_far(median, meta)
        rows[label] = {"dip recovered, lead under 384": near, "dip recovered, lead 512+": far,
                       "error / noise sd": float((np.abs(median - truth) / sd).mean())}
    print(pd.DataFrame(rows).T.round(3).to_string(), flush=True)
    del fc

    print("\n=== Part 2. What the quantiles say where the median goes flat (one pass)", flush=True)
    in_dip = np.zeros_like(truth, dtype=bool)
    for i, m in enumerate(meta):
        for s in m["starts"]:
            in_dip[i, s : s + m["width"]] = True
    bins = [(0, 256), (256, 512), (512, 768), (768, 1024)]
    out_rows = []
    for size in ("5M", "28M", "123M"):
        fc = TimestralForecaster.from_pretrained(f"mistralai/Timestral-{size}", device=DEVICE)
        with torch.inference_mode():
            f = fc.forecast(x[:, None, : synth.CONTEXT].astype(np.float32), synth.HORIZON)
        q = {lv: f.quantile(lv)[:, 0].double().cpu().numpy() for lv in (0.1, 0.5, 0.9)}
        shares = {lv: synth.recovered_shares(q[lv], meta) for lv in q}
        inside = (truth >= q[0.1]) & (truth <= q[0.9])
        width = (q[0.9] - q[0.1]) / sd
        table = {}
        for a, b in bins:
            row = {f"dip in q{int(lv * 100)}": float(np.mean([s for _, lead, s in shares[lv] if a <= lead < b])) for lv in q}
            row["band width / sd, outside dips"] = float(width[:, a:b][~in_dip[:, a:b]].mean())
            row["band width / sd, inside dips"] = float(width[:, a:b][in_dip[:, a:b]].mean())
            row["truth in band, outside dips"] = float(inside[:, a:b][~in_dip[:, a:b]].mean())
            row["truth in band, inside dips"] = float(inside[:, a:b][in_dip[:, a:b]].mean())
            table[f"{a}-{b}"] = row
            out_rows.append({"model": size, "lead": f"{a}-{b}", **row})
        print(f"\n--- Timestral-{size} (columns: lead time in steps)")
        print(pd.DataFrame(table).round(3).to_string(), flush=True)
        del fc
    pd.DataFrame(out_rows).to_csv(OUT / "check6_quantiles.csv", index=False)
    depth = np.mean([m["depth"] / m["sd"] for m in meta])
    print(f"\nmean dip depth: {depth:.2f} noise sd; dips cover {in_dip.mean():.3f} of the horizon")


if __name__ == "__main__":
    main()
