"""Gate for the 10-second regime: is the 5M-to-28M gap on the bizitobs datasets real, and what drives it?

Published GIFT-Eval MASE at the medium horizon: `bizitobs_service` 5M 1.3172, 28M 0.9607,
123M 0.9138; `bizitobs_application` 5M 2.8816, 28M 2.1835, 123M 2.1695. The first is 21
series, the second 1 series, each about one day of 10-second data with 34 or 35 past
covariates.

Per dataset this script:
  A. reproduces the published score on the official windows (a check of this script);
  B. repeats them without the past covariates;
  C. repeats them with the history cut to 1024 and 4096 steps;
  D. splits the error by position in the horizon;
  E. forecasts from every non-overlapping origin back to step 2400, then gives the 5M / 28M
     ratio per series and overall, with a bootstrap interval over series (service only:
     the application dataset has one series, so origins are resampled instead).

MASE as GIFT-Eval computes it: absolute error of the median divided by the seasonal-naive
error of that variate over the history before the origin.

Run: .venv\\Scripts\\python.exe -u experiments\\gap_10s\\check1_gap_source.py
"""

import os
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("GIFT_EVAL", str(ROOT / "data" / "gifteval"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from gluonts.time_feature import get_seasonality  # noqa: E402

from gift_eval.data import Dataset  # noqa: E402
from timestral import TimestralForecaster  # noqa: E402

SIZES = ["5M", "28M", "123M"]
PUBLISHED = {
    "bizitobs_service": {"5M": 1.3172, "28M": 0.9607, "123M": 0.9138},
    "bizitobs_application": {"5M": 2.8816, "28M": 2.1835, "123M": 2.1695},
}
FIRST_ORIGIN = 2400
OUT = ROOT / "results" / "gap_10s"
DEVICE = os.environ.get("WSP_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
SERIES_PER_BATCH = 3  # each series is up to 37 rows (2 targets + 35 covariates); sized for an 8 GB GPU


def load(name: str):
    """Per series: full target `(V, T)`, covariates `(C, T - H)`; official origins; horizon; season."""
    ds = Dataset(name=name, term="medium", to_univariate=False)
    horizon, season = ds.prediction_length, get_seasonality(ds.freq)
    series: dict[str, dict] = {}
    official: set[int] = set()
    for e, lab in zip(ds.test_data.input, ds.test_data.label):
        t = np.asarray(e["target"], dtype=np.float64)
        official.add(t.shape[-1])
        sid = str(e.get("item_id"))
        if sid not in series or t.shape[-1] > series[sid]["cov"].shape[-1]:
            full = np.concatenate([t, np.asarray(lab["target"], dtype=np.float64)], axis=-1)
            series[sid] = {"target": full, "cov": np.asarray(e["past_feat_dynamic_real"], dtype=np.float64)}
    return series, sorted(official), horizon, season, ds.freq


@torch.inference_mode()
def scaled_errors(fc, series: dict, origin: int, horizon: int, season: int, context=None, use_cov=True) -> np.ndarray:
    """`(S, V, H)` scaled absolute errors of the median, all series forecast in one batch."""
    ids = list(series)
    start = 0 if context is None else max(0, origin - context)
    medians = []
    for k in range(0, len(ids), SERIES_PER_BATCH):
        chunk = ids[k : k + SERIES_PER_BATCH]
        ctx = np.stack([series[i]["target"][:, start:origin] for i in chunk]).astype(np.float32)
        past = np.stack([series[i]["cov"][:, start:origin] for i in chunk]).astype(np.float32) if use_cov else None
        medians.append(fc.forecast(ctx, horizon, past_covariates=past).median.double().cpu().numpy())
    median = np.concatenate(medians)
    out = []
    for k, i in enumerate(ids):
        t = series[i]["target"]
        scale = np.nanmean(np.abs(t[:, season:origin] - t[:, : origin - season]), axis=-1)
        out.append(np.abs(median[k] - t[:, origin : origin + horizon]) / scale[:, None])
    return np.stack(out)


def run(name: str, models: dict) -> None:
    series, official, horizon, season, freq = load(name)
    ids = list(series)
    first = series[ids[0]]
    print(f"\n=================== {name}: {len(ids)} series, target {first['target'].shape}, covariates {first['cov'].shape}, "
          f"freq {freq}, season {season}, horizon {horizon}, official origins {official}", flush=True)

    def score(size, origins, **kw):
        return np.stack([scaled_errors(models[size], series, o, horizon, season, **kw) for o in origins])  # (O, S, V, H)

    t0 = time.perf_counter()
    err = {s: score(s, official) for s in SIZES}
    print(f"[A] official windows ({time.perf_counter() - t0:.0f} s on {DEVICE}). MASE, ours vs published:")
    ok = True
    for s in SIZES:
        ours = float(np.nanmean(err[s]))
        ok &= abs(ours / PUBLISHED[name][s] - 1) < 0.001
        print(f"    {s:>5}: {ours:.4f} vs {PUBLISHED[name][s]:.4f}")
    print("    reproduction:", "PASS (within 0.1 %)" if ok else "DIFFERS", flush=True)

    rows = {"official (covariates, full history)": {s: float(np.nanmean(err[s])) for s in SIZES}}
    rows["[B] without covariates"] = {s: float(np.nanmean(score(s, official, use_cov=False))) for s in SIZES}
    for context in (1024, 4096):
        rows[f"[C] history cut to {context}"] = {s: float(np.nanmean(score(s, official, context=context))) for s in SIZES}
        rows[f"[C] history cut to {context}, no covariates"] = {
            s: float(np.nanmean(score(s, official, context=context, use_cov=False))) for s in SIZES}
    table = pd.DataFrame(rows).T
    table["5M / 28M"] = table["5M"] / table["28M"]
    print("\n[B, C] MASE on the official windows")
    print(table.round(4).to_string(), flush=True)

    block = 96
    edges = list(range(0, horizon, block)) + [horizon]
    edges = sorted(set(edges))
    prof = pd.DataFrame(
        {s: [float(np.nanmean(err[s][..., a:b])) for a, b in zip(edges[:-1], edges[1:])] for s in SIZES},
        index=pd.Index([f"{a}-{b}" for a, b in zip(edges[:-1], edges[1:])], name="steps"),
    )
    prof["5M / 28M"] = prof["5M"] / prof["28M"]
    print("\n[D] MASE by position in the horizon (official windows)")
    print(prof.round(4).to_string(), flush=True)

    origins = [o for o in range(official[-1], FIRST_ORIGIN - 1, -horizon)][::-1]
    t0 = time.perf_counter()
    with_cov = {s: np.nanmean(score(s, origins), axis=(2, 3)) for s in ("5M", "28M")}  # (O, S)
    no_cov = {s: np.nanmean(score(s, origins, use_cov=False), axis=(2, 3)) for s in ("5M", "28M")}
    print(f"\n[E] {len(origins)} non-overlapping origins {origins} ({time.perf_counter() - t0:.0f} s)")
    per_origin = pd.DataFrame({
        "5M": with_cov["5M"].mean(axis=1), "28M": with_cov["28M"].mean(axis=1),
        "5M no cov": no_cov["5M"].mean(axis=1), "28M no cov": no_cov["28M"].mean(axis=1),
    }, index=pd.Index(origins, name="origin"))
    per_origin["5M / 28M"] = per_origin["5M"] / per_origin["28M"]
    per_origin["5M / 28M no cov"] = per_origin["5M no cov"] / per_origin["28M no cov"]
    per_origin["official"] = [o in official for o in origins]
    print(per_origin.round(4).to_string())
    per_series = pd.DataFrame({
        "5M": with_cov["5M"].mean(axis=0), "28M": with_cov["28M"].mean(axis=0),
        "5M no cov": no_cov["5M"].mean(axis=0), "28M no cov": no_cov["28M"].mean(axis=0),
    }, index=pd.Index(ids, name="series"))
    per_series["5M / 28M"] = per_series["5M"] / per_series["28M"]
    per_series["5M / 28M no cov"] = per_series["5M no cov"] / per_series["28M no cov"]
    print("\n[E1] Mean over origins, per series")
    print(per_series.round(4).to_string())
    OUT.mkdir(parents=True, exist_ok=True)
    per_origin.to_csv(OUT / f"check1_{name}_per_origin.csv")
    per_series.to_csv(OUT / f"check1_{name}_per_series.csv")

    rng = np.random.default_rng(0)
    axis = 1 if len(ids) > 1 else 0  # resample series when there are several, else origins
    unit = "series" if axis == 1 else "origins (one series: not independent)"
    n_units = with_cov["5M"].shape[axis]
    idx = rng.integers(0, n_units, (10_000, n_units))
    print(f"\n[E2] 5M / 28M over all origins and series, 95 % bootstrap interval over {unit}")
    for label, d in (("with covariates", with_cov), ("without covariates", no_cov)):
        a, b = np.moveaxis(d["5M"], axis, 0), np.moveaxis(d["28M"], axis, 0)
        r = a[idx].mean(axis=(1, 2)) / b[idx].mean(axis=(1, 2))
        worse = int((a.mean(axis=1) > b.mean(axis=1)).sum())
        print(f"    {label:>19}: {a.mean() / b.mean():.4f} [{np.quantile(r, 0.025):.4f}, {np.quantile(r, 0.975):.4f}]; "
              f"5M worse on {worse} of {n_units} {unit.split(' ')[0]}", flush=True)
    for s in ("5M", "28M"):
        print(f"    effect of covariates on {s}: MASE without / with = {no_cov[s].mean() / with_cov[s].mean():.4f}")


def main() -> None:
    pd.set_option("display.width", 220)
    models = {s: TimestralForecaster.from_pretrained(f"mistralai/Timestral-{s}", device=DEVICE) for s in SIZES}
    for name in PUBLISHED:
        run(name, models)


if __name__ == "__main__":
    main()
