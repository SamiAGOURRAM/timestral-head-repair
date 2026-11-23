"""Forecast examples for the README: published Timestral-5M against edit B.

1. Synthetic: a held-back-style series (level + noise + one dip every 150 steps, seed 7150),
   8192 steps of history, 1024-step forecast in one pass (top figure).
2. Real: error by distance ahead on the 10-second bizitobs tasks of GIFT-Eval
   (`bizitobs_service` and `bizitobs_application`), medium range (600 steps) and long range
   (900 steps). EVERY window and variate is used, none picked: per window, the absolute error
   of the median at each step divided by the series' recent spread (std of its last 2000
   history steps), so windows of different scale can be averaged.

Needs `checkpoints/edit_b` (`python edit/apply_edit_b.py`) and the GIFT-Eval data (GIFT_EVAL).
Writes `results/examples.npz`.

Run: python analysis/make_examples.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "gap_10s"))
sys.path.insert(0, str(ROOT / "evaluation"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import check3_synthetic_lead_time as synth  # noqa: E402
import gift_eval_run as ge  # noqa: E402
from gift_eval.data import Dataset  # noqa: E402
from timestral import TimestralForecaster, load_model  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def synthetic_example(fcs: dict, out: dict) -> None:
    x, _ = synth.make_series(150, 4, seed=7150)
    out["syn_history"] = x[0, synth.CONTEXT - 1024: synth.CONTEXT]
    out["syn_truth"] = x[0, synth.CONTEXT:]
    for name, fc in fcs.items():
        with torch.inference_mode():
            f = fc.forecast(x[:1, None, : synth.CONTEXT].astype(np.float32), synth.HORIZON)
        out[f"syn_{name}_median"] = f.median[0, 0].float().cpu().numpy()
        q = f.quantiles[0, 0].float().cpu().numpy()
        levels = list(fc.quantile_levels)
        out[f"syn_{name}_q10"] = q[:, levels.index(0.1)]
        out[f"syn_{name}_q90"] = q[:, levels.index(0.9)]


def real_forecasts(fcs: dict, term: str) -> list:
    """(task, window, variate, history, truth, {model: median}, scale) for every window of the bizitobs 10-second tasks."""
    rows = []
    for name in ("bizitobs_service", "bizitobs_application"):
        ds = Dataset(name=name, term=term, to_univariate=False)
        inputs = list(ds.test_data.input)
        labels = [np.atleast_2d(np.asarray(e["target"], dtype=float)) for e in ds.test_data.label]
        meds = {}
        for m, fc in fcs.items():
            pred = ge.TimestralPredictor(fc, ds.prediction_length, use_past_covariates=bool(ds.past_feat_dynamic_real_dim))
            with torch.inference_mode():
                meds[m] = [np.atleast_2d(np.asarray(f.quantile(0.5)).T) for f in pred.predict(inputs)]
        for i, y in enumerate(labels):
            hist = np.atleast_2d(np.asarray(inputs[i]["target"], dtype=float))
            for v in range(y.shape[0]):
                scale = np.nanstd(hist[v, -2000:])
                if scale > 0:
                    rows.append((f"{name}/10S/{term}", i, v, hist[v], y[v], {m: meds[m][i][v] for m in fcs}, scale))
    return rows


def real_error_by_distance(rows: list, out: dict, term: str) -> None:
    for m in ("published", "edit B"):
        out[f"dist_{term}_{m}_all"] = np.vstack([np.abs(r[5][m] - r[4]) / r[6] for r in rows])
    print(f"error by distance, {term}: {len(rows)} window-variates")


def main() -> None:
    fcs = {"published": TimestralForecaster(load_model("mistralai/Timestral-5M", device=DEVICE)),
           "edit B": TimestralForecaster(load_model(str(ROOT / "checkpoints" / "edit_b"), device=DEVICE))}
    out = {}
    synthetic_example(fcs, out)
    for term in ("medium", "long"):
        real_error_by_distance(real_forecasts(fcs, term), out, term)
    np.savez(ROOT / "results" / "examples.npz", **out)
    print("saved results/examples.npz")


if __name__ == "__main__":
    main()
