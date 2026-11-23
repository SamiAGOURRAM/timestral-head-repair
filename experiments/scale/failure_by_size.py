"""Does the far-forecast failure fade with model size? Published Timestral 5M, 28M, 123M, 209M, no edit.

Measurement only.
For each size:
1. fixed-lag heads: time-attention heads with more than half of their attention on one lag
   (same measure as experiments/edit_5m/check_28m_fixed_lag.py: 16 series, cycle 180,
   8192 steps of history);
2. near / far dips and error on the held-back sets of experiments/edit_28m (same seeds, so
   the 5M and 28M rows must reproduce results/edit_28m_run.log), histories 1024 to 8192.

Run on CPU so the GPU stays free for other jobs:
    .venv\\Scripts\\python.exe -u experiments\\scale\\failure_by_size.py
Sizes can be restricted with WSP_SIZES=5M,28M.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "edit_5m"))
sys.path.insert(0, str(ROOT / "experiments" / "gap_10s"))
sys.path.insert(0, str(ROOT / "experiments" / "edit_28m"))
import check_28m_fixed_lag as c28  # noqa: E402
import heldback_eval as he  # noqa: E402
import select_and_heldback as sh  # noqa: E402
from timestral import load_model  # noqa: E402

SIZES = os.environ.get("WSP_SIZES", "5M,28M,123M,209M").split(",")
HISTORIES = (1024, 2048, 4096, 8192)
OUT = ROOT / "results" / "scale"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))
    sh.DEVICE = "cpu"
    sets = {}
    for c in sh.MISALIGNED:
        sets[f"dips {c}"] = ("dip", *he.synth.make_series(c, he.synth.N_SERIES, seed=5000 + c))
    for c in sh.ALIGNED:
        sets[f"dips {c} (whole patches)"] = ("dip", *he.synth.make_series(c, he.synth.N_SERIES, seed=5000 + c))
    for c in sh.SPIKES:
        sets[f"spikes {c}"] = ("spike", *he.make_spike_series(c, he.synth.N_SERIES, seed=5500 + c))
    groups = {"misaligned": [f"dips {c}" for c in sh.MISALIGNED],
              "aligned": [f"dips {c} (whole patches)" for c in sh.ALIGNED],
              "spikes": [f"spikes {c}" for c in sh.SPIKES]}
    diag = he.synth.make_series(180, he.synth.N_SERIES, seed=180)[0][:16]

    rows, heads = [], []
    for size in SIZES:
        model = load_model(f"mistralai/Timestral-{size}", device="cpu")
        n_params = sum(p.numel() for p in model.parameters())
        prof = c28.lag_profiles(model, diag)
        for li in range(prof.shape[0]):
            for hi in range(prof.shape[1]):
                lag = int(prof[li, hi].argmax())
                heads.append({"size": size, "head": f"{li}.{hi}", "lag": lag, "share": float(prof[li, hi, lag])})
        fixed = [h for h in heads if h["size"] == size and h["share"] > 0.5]
        print(f"Timestral-{size}: {n_params:,} weights, {prof.shape[0]} x {prof.shape[1]} time heads; "
              f"fixed-lag heads {len(fixed)}, of which lag 2-12: {sum(2 <= h['lag'] <= 12 for h in fixed)}", flush=True)
        for h in HISTORIES:
            s = sh.scores(model, sets, h)
            for g, names in groups.items():
                rows.append({"size": size, "history": h, "group": g,
                             **{k: float(np.mean([s[n][k] for n in names])) for k in ("near", "far", "error")}})
            print(f"  history {h} done", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "failure_by_size.csv", index=False)
        pd.DataFrame(heads).to_csv(OUT / "lag_heads_by_size.csv", index=False)
        del model

    df = pd.DataFrame(rows)
    for g in groups:
        for k in ("far", "near", "error"):
            t = df[df.group == g].pivot(index="history", columns="size", values=k)[SIZES]
            print(f"\n=== {g}, {k}\n{t.round(3).to_string()}")


if __name__ == "__main__":
    main()
