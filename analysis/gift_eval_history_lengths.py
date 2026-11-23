"""How long are the histories in the 97 GIFT-Eval configurations?

The two weight edits found by the blind agent runs hurt when the history is near their
reading distance (about 1024 steps for the 32-patch edit, 2000 to 3000 for the 60-patch
one). This lists, per configuration, the number of forecast windows and their history
lengths in steps, and counts the configurations in each range. Data only, no model.

Run: .venv\\Scripts\\python.exe analysis\\gift_eval_history_lengths.py
"""

import json
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("GIFT_EVAL", str(ROOT / "data" / "gifteval"))
warnings.filterwarnings("ignore")
sys.path.insert(0, str(ROOT / "evaluation"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import gift_eval_run as ge  # noqa: E402
from gift_eval.data import Dataset  # noqa: E402


def main() -> None:
    properties = json.load(open(ge.GIFT_REPO / "notebooks" / "dataset_properties.json"))
    rows = []
    for dataset_name, term, key in ge.all_configs(properties):
        ds = Dataset(name=dataset_name, term=term, to_univariate=False)
        lens = np.array([np.asarray(e["target"]).shape[-1] for e in ds.test_data.input])
        rows.append({"config": key, "windows": len(lens), "horizon": ds.prediction_length, "history min": int(lens.min()),
                     "history median": int(np.median(lens)), "history max": int(lens.max()),
                     "share 700-1500": float(((lens >= 700) & (lens < 1500)).mean()),
                     "share 1500-4000": float(((lens >= 1500) & (lens < 4000)).mean())})
        print(f"{key}: median history {rows[-1]['history median']}", flush=True)
    df = pd.DataFrame(rows).set_index("config")
    out = ROOT / "results" / "gift_eval_history_lengths.csv"
    df.to_csv(out)
    med = df["history median"]
    print("\nconfigurations by median history length (steps):")
    for lo, hi in ((0, 350), (350, 700), (700, 1500), (1500, 4000), (4000, 10**9)):
        sel = df[(med >= lo) & (med < hi)]
        print(f"  {lo:>5} to {hi if hi < 10**9 else 'more':>5}: {len(sel):2d} of {len(df)}")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
