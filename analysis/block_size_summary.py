"""Summarise test B: head-5.1 lags 16, 24, 32 scored with forecast blocks of 16 and 24 tokens.

Each edit is compared with the published model at the same block size (per-configuration
MASE from `results/gift_eval/<name>/per_config/*.json`), on the lag sweep's seven configurations.
For reference, the default block (32): lag 32 against the published model from the lag sweep.
Writes `results/block_size_test.csv`.

Run: .venv\\Scripts\\python.exe analysis\\block_size_summary.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GE = ROOT / "results" / "gift_eval"
CONFIGS = ["solar/H/long", "solar/H/medium", "electricity/H/long", "m_dense/H/long", "ett1/H/long",
           "loop_seattle/H/long", "us_births/D/short"]


def mase(name: str) -> pd.Series:
    out = {}
    for c in CONFIGS:
        f = GE / name / "per_config" / (c.replace("/", "__") + ".json")
        d = json.loads(f.read_text())
        out[c] = d["metrics"]["MASE[0.5]"]
    return pd.Series(out, dtype=float)


def main() -> None:
    pd.set_option("display.width", 220)
    rows = []
    for chunk in (16, 24):
        ref = mase(f"Timestral-5M-chunk{chunk}")
        for lag in (16, 24, 32):
            r = mase(f"5M-lag{lag}-chunk{chunk}") / ref
            rows.append({"block": chunk, "lag": lag, "geo mean": float(np.exp(np.log(r).mean())), "worst": float(r.max()), **r.round(3).to_dict()})
    sweep = pd.read_csv(ROOT / "results" / "lag_sweep_gift.csv", index_col=0)
    for lag in (32, 33, 43):
        rows.append({"block": 32, "lag": lag, "geo mean": float(sweep.loc[f"lag {lag}", "geo mean"]), "worst": float(sweep.loc[f"lag {lag}", "worst"])})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "block_size_test.csv", index=False)
    print(df.round(3).to_string(index=False))
    for chunk in (16, 24):
        r = pd.read_csv(GE / f"Timestral-5M-chunk{chunk}" / "against_published.csv", index_col="config")["mase_ratio"].reindex(CONFIGS)
        print(f"published model, blocks of {chunk} vs its published scores (blocks of 32): geo mean {np.exp(np.log(r).mean()):.3f}")


if __name__ == "__main__":
    main()
