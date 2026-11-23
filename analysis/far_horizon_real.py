"""Does edit B help the DISTANT part of real forecasts? Base 5M vs edit B, error by lead time.

Configurations: every GIFT-Eval configuration whose horizon is at least 600 steps (all 21 "long"
ones and the two 10-second "medium" ones), so that a distant block (512 steps ahead and over)
exists. Same data loading and batching as `evaluation/gift_eval_run.py`; median forecasts.

Per configuration: mean absolute error of the median, near block (steps 0-383) and distant
block (512 and over), for both models; ratio edit B / base per block. Also the dominant cycle
of the history (lag 24-2000 with the highest autocorrelation, median over windows) and its
strength (that autocorrelation), and whether the cycle is a whole number of 32-step patches.

Expectation (written before running): if the failure matters on real data and edit B repairs
it there, the distant-block ratio should be clearly below the near-block ratio in
configurations with a strong cycle (autocorrelation 0.5 or more) that is not a whole number
of patches, and close to the near-block ratio elsewhere.

Run: .venv\\Scripts\\python.exe -u analysis\\far_horizon_real.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import gift_eval_run as ge  # noqa: E402
from gift_eval.data import Dataset  # noqa: E402
from timestral import TimestralForecaster, load_model  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = {"base": "mistralai/Timestral-5M", "edit B": str(ROOT / "checkpoints" / "edit_b")}
NEAR, FAR = 384, 512


def dominant_cycle(x: np.ndarray) -> tuple[int, float]:
    x = x[~np.isnan(x)][-4096:]
    if len(x) < 600:
        return 0, float("nan")
    x = x - x.mean()
    den = float((x * x).sum())
    if den == 0:
        return 0, float("nan")
    best, lag = -1.0, 0
    for k in range(24, min(2000, len(x) // 2)):
        r = float((x[:-k] * x[k:]).sum() / den)
        if r > best:
            best, lag = r, k
    return lag, best


def medians(fc, ds) -> list[np.ndarray]:
    pred = ge.TimestralPredictor(fc, ds.prediction_length, use_past_covariates=bool(ds.past_feat_dynamic_real_dim))
    with torch.inference_mode():
        fcs = pred.predict(ds.test_data.input)
    # QuantileForecast gives (horizon,) or (horizon, variates); labels are (horizon,) or (variates, horizon)
    return [np.asarray(f.quantile(0.5)).T for f in fcs]


def main() -> None:
    pd.set_option("display.width", 220)
    props = json.load(open(ge.GIFT_REPO / "notebooks" / "dataset_properties.json"))
    configs = []
    for name, term, key in ge.all_configs(props):
        if term == "short":
            continue
        ds = Dataset(name=name, term=term, to_univariate=False)
        if ds.prediction_length >= 600:
            configs.append((name, term, key, ds))
    print(f"{len(configs)} configurations with a horizon of 600 steps or more", flush=True)
    fcs = {m: TimestralForecaster(load_model(src, device=DEVICE)) for m, src in MODELS.items()}
    rows = []
    for name, term, key, ds in configs:
        labels = [np.asarray(e["target"], dtype=float) for e in ds.test_data.label]
        inputs = [np.asarray(e["target"], dtype=float) for e in ds.test_data.input]
        cyc = [dominant_cycle(x if x.ndim == 1 else x[0]) for x in inputs[:: max(1, len(inputs) // 50)]]
        lag = int(np.median([c[0] for c in cyc])); strength = float(np.nanmedian([c[1] for c in cyc]))
        err = {}
        for m, fc in fcs.items():
            med = medians(fc, ds)
            a = np.concatenate([np.abs(p - y).reshape(-1, p.shape[-1]) for p, y in zip(med, labels)])
            err[m] = (np.nanmean(a[:, :NEAR]), np.nanmean(a[:, FAR:]))
        rows.append({"config": key, "horizon": ds.prediction_length, "cycle": lag, "cycle strength": strength,
                     "whole patches": lag % 32 == 0 if lag else None,
                     "near ratio": err["edit B"][0] / err["base"][0], "far ratio": err["edit B"][1] / err["base"][1]})
        r = rows[-1]
        print(f"{key}: cycle {lag} (acf {strength:.2f}), near {r['near ratio']:.3f}, far {r['far ratio']:.3f}", flush=True)
    df = pd.DataFrame(rows).set_index("config").sort_values("far ratio")
    df.to_csv(ROOT / "results" / "far_horizon_real.csv")
    print("\n" + df.round(3).to_string())
    strong = df[(df["cycle strength"] >= 0.5) & (df["whole patches"] == False)]  # noqa: E712
    rest = df.drop(strong.index)
    for label, d in (("strong cycle, not whole patches", strong), ("other", rest)):
        if len(d):
            print(f"\n{label} ({len(d)}): geometric mean near {np.exp(np.log(d['near ratio']).mean()):.3f}, "
                  f"far {np.exp(np.log(d['far ratio']).mean()):.3f}")


if __name__ == "__main__":
    main()
