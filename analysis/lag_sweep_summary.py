"""Summarise the head-5.1 lag sweep on GIFT-Eval: MASE ratio per lag on the seven damaged configurations.

Reads `results/gift_eval/5M-lagsweep-<lag>/against_published.csv` (from `tools/lag_sweep.ps1`)
and, for reference, edit B on the same configurations. Writes `results/lag_sweep_gift.csv`.

Run: .venv\\Scripts\\python.exe analysis\\lag_sweep_summary.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LAGS = (11, 21, 28, 30, 31, 32, 33, 34, 36, 40, 42, 43)
CONFIGS = ["solar/H/long", "solar/H/medium", "electricity/H/long", "m_dense/H/long", "ett1/H/long",
           "loop_seattle/H/long", "us_births/D/short"]


def ratios(name: str) -> pd.Series:
    r = pd.read_csv(ROOT / "results" / "gift_eval" / name / "against_published.csv", index_col="config")["mase_ratio"]
    return r.reindex(CONFIGS)


def main() -> None:
    pd.set_option("display.width", 220)
    cols = {f"lag {lag}": ratios(f"5M-lagsweep-{lag}") for lag in LAGS}
    cols["edit B"] = ratios("5M-blind-run2-final")
    df = pd.DataFrame(cols).T
    assert df.notna().all().all(), df[df.isna().any(axis=1)]
    df["geo mean"] = np.exp(np.log(df[CONFIGS]).mean(axis=1))
    df["worst"] = df[CONFIGS].max(axis=1)
    print(df.round(3).to_string())
    df.to_csv(ROOT / "results" / "lag_sweep_gift.csv")
    syn = pd.read_csv(ROOT / "results" / "edit_5m" / "lag_sweep_synthetic.csv", index_col=0)
    print("\nlag: synthetic far dips | geometric mean ratio on the 7 configurations")
    for lag in LAGS:
        print(f"  {lag:3d}: {syn.loc[f'lag {lag}', 'far']:+.3f} | {df.loc[f'lag {lag}', 'geo mean']:.3f}")


if __name__ == "__main__":
    main()
