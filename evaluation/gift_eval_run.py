"""Evaluate a Timestral checkpoint on GIFT-Eval and compare with the published results.

The protocol and the predictor are ported from `notebooks/timestral.ipynb` of the GIFT-Eval
repository (Apache-2.0), the notebook Mistral used for its leaderboard submissions:
multivariate datasets forecast jointly, `past_feat_dynamic_real` passed as past covariates,
full history as context (the model keeps the most recent 32768 steps), forecaster defaults,
point forecast = median, CRPS from the 9 quantiles 0.1 ... 0.9.

Added here: the share of true values falling below the 0.1 and 0.9 quantile forecasts
(`Coverage`), which the leaderboard does not report.

Each configuration's metrics are written to
`results/gift_eval/<name>/per_config/<config>.json` as soon as it is done; finished
configurations are skipped on restart.

Examples:
  .venv\\Scripts\\python.exe evaluation\\gift_eval_run.py --size 5M --only us_births,m4_hourly
  .venv\\Scripts\\python.exe evaluation\\gift_eval_run.py --size 28M
  .venv\\Scripts\\python.exe evaluation\\gift_eval_run.py --checkpoint checkpoints\\x --name x --compare-with Timestral-5M
"""

import argparse
import json
import logging
import math
import os
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("GIFT_EVAL", str(ROOT / "data" / "gifteval"))
warnings.filterwarnings("ignore", category=FutureWarning)  # pandas 'H'/'T' aliases used by gift-eval
warnings.filterwarnings("ignore", message="Using `json`-module")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from gluonts.ev.metrics import (  # noqa: E402
    MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE, SMAPE,
    Coverage, MeanWeightedSumQuantileLoss,
)
from gluonts.model import evaluate_model  # noqa: E402
from gluonts.model.forecast import QuantileForecast  # noqa: E402
from gluonts.time_feature import get_seasonality  # noqa: E402

from gift_eval.data import Dataset  # noqa: E402
from timestral import TimestralForecaster  # noqa: E402

GIFT_REPO = ROOT / "external" / "gift-eval"
# Batch budget: max rows (series x (variates + covariates)) and max rows x (context + horizon)
# steps. The notebook uses 1024 and 2**24; the step budget here is lower to fit an 8 GB GPU.
MAX_ROWS = 1024
MAX_TOKENS = 2**21

SHORT_DATASETS = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
MED_LONG_DATASETS = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"

PRETTY_DATASET_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
METRICS = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(quantile_levels=QUANTILE_LEVELS),
    Coverage(q=0.1),
    Coverage(q=0.9),
]
MASE_COL = "MASE[0.5]"
CRPS_COL = "mean_weighted_sum_quantile_loss"


def all_configs(properties: dict) -> list[tuple[str, str, str]]:
    """(dataset_name, term, config key) for the 97 configurations."""
    short_datasets = SHORT_DATASETS.split()
    med_long_datasets = MED_LONG_DATASETS.split()
    configs = []
    for dataset_name in dict.fromkeys(short_datasets + med_long_datasets):
        for term in ["short", "medium", "long"]:
            if term in ("medium", "long") and dataset_name not in med_long_datasets:
                continue
            if "/" in dataset_name:
                ds_key, ds_freq = dataset_name.split("/")
                ds_key = PRETTY_DATASET_NAMES.get(ds_key.lower(), ds_key.lower())
            else:
                ds_key = PRETTY_DATASET_NAMES.get(dataset_name.lower(), dataset_name.lower())
                ds_freq = properties[ds_key]["frequency"]
            configs.append((dataset_name, term, f"{ds_key}/{ds_freq}/{term}"))
    assert len(configs) == 97, len(configs)
    return configs


def left_pad(x: np.ndarray, length: int) -> np.ndarray:
    pad = length - x.shape[-1]
    if pad == 0:
        return x
    return np.concatenate([np.full((x.shape[0], pad), np.nan, dtype=np.float32), x], axis=-1)


class TimestralPredictor:
    """GluonTS-style predictor: predict(test_data.input) -> list[QuantileForecast].

    Series are sorted by length and batched within the MAX_ROWS / MAX_TOKENS budget; a batch
    that runs out of GPU memory is split in two.
    """

    def __init__(self, forecaster: TimestralForecaster, prediction_length: int, use_past_covariates: bool):
        self.fc = forecaster
        self.prediction_length = prediction_length
        self.use_past_covariates = use_past_covariates
        levels = list(forecaster.quantile_levels)
        self.q_index = [next(i for i, lv in enumerate(levels) if abs(lv - q) < 1e-9) for q in QUANTILE_LEVELS]
        self.forecast_keys = [str(q) for q in QUANTILE_LEVELS] + ["mean"]

    def predict(self, test_data_input) -> list[QuantileForecast]:
        entries = list(test_data_input)
        max_ctx = self.fc.max_context
        targets, covs = [], []
        for e in entries:
            t = np.asarray(e["target"], dtype=np.float32)
            targets.append((t[None] if t.ndim == 1 else t)[:, -max_ctx:])  # (V, T)
            c = e.get("past_feat_dynamic_real") if self.use_past_covariates else None
            if c is not None:
                c = np.asarray(c, dtype=np.float32)
                c = (c[None] if c.ndim == 1 else c)[:, -max_ctx:]
            covs.append(c)
        n_rows = targets[0].shape[0] + (covs[0].shape[0] if covs[0] is not None else 0)

        outputs = [None] * len(entries)
        order = sorted(range(len(entries)), key=lambda i: targets[i].shape[-1])
        start = 0
        while start < len(order):
            # Grow the batch from the shortest remaining series while the padded size fits the budget.
            end = start + 1
            while end < len(order):
                length = targets[order[end]].shape[-1] + self.prediction_length
                if (end + 1 - start) * n_rows > MAX_ROWS or (end + 1 - start) * n_rows * length > MAX_TOKENS:
                    break
                end += 1
            self._predict_batch(order[start:end], targets, covs, outputs)
            start = end

        forecasts = []
        for e, q in zip(entries, outputs):
            # q: (V, H, 9) -> (10, H, V), or (10, H) for a univariate target; "mean" is the median.
            arr = np.concatenate([q, q[..., QUANTILE_LEVELS.index(0.5), None]], axis=-1)
            arr = arr.transpose(2, 1, 0)
            if np.asarray(e["target"]).ndim == 1:
                arr = arr[..., 0]
            forecasts.append(
                QuantileForecast(
                    forecast_arrays=arr,
                    forecast_keys=self.forecast_keys,
                    start_date=e["start"] + np.asarray(e["target"]).shape[-1],
                    item_id=e.get("item_id"),
                )
            )
        return forecasts

    def _predict_batch(self, idx, targets, covs, outputs) -> None:
        try:
            q = self._run(idx, targets, covs)
        except RuntimeError as e:
            # Out of GPU memory arrives as torch.cuda.OutOfMemoryError (a RuntimeError) or as a
            # plain RuntimeError "CUDA error: out of memory"; anything else is a real error.
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            if len(idx) == 1:
                raise
            half = len(idx) // 2
            self._predict_batch(idx[:half], targets, covs, outputs)
            self._predict_batch(idx[half:], targets, covs, outputs)
            return
        for j, i in enumerate(idx):
            outputs[i] = q[j]

    def _run(self, idx, targets, covs) -> np.ndarray:
        # Left-pad with NaN to a common length (leading unobserved steps are ignored by the model).
        length = max(targets[i].shape[-1] for i in idx)
        ctx = np.stack([left_pad(targets[i], length) for i in idx])
        past = None
        if covs[idx[0]] is not None:
            past = np.stack([left_pad(covs[i], length) for i in idx])
        f = self.fc.forecast(ctx, self.prediction_length, past_covariates=past)
        return f.quantiles[..., self.q_index].float().cpu().numpy()  # (B, V, H, 9)


def evaluate_config(fc: TimestralForecaster, dataset_name: str, term: str) -> dict:
    ds = Dataset(name=dataset_name, term=term, to_univariate=False)
    predictor = TimestralPredictor(fc, ds.prediction_length, use_past_covariates=bool(ds.past_feat_dynamic_real_dim))
    res = evaluate_model(
        predictor,
        test_data=ds.test_data,
        metrics=METRICS,
        batch_size=512,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=get_seasonality(ds.freq),
    )
    return {k: float(res[k].iloc[0]) for k in res.columns}


def gmean(values: list[float]) -> float:
    return float(math.exp(sum(math.log(v) for v in values) / len(values)))


def compare_variant(out_dir: Path, reference: str, label: str) -> None:
    """A changed setting or checkpoint against the published `reference`: error ratios per configuration.

    Ratios are ours / published, so below 1 means the variant is better. `below_q10` and
    `below_q90` are the shares of true values under the 0.1 and 0.9 quantile forecasts
    (0.10 and 0.90 when the bands are right); the reference has no published values for them.
    """
    published = pd.read_csv(GIFT_REPO / "results" / reference / "all_results.csv").set_index("dataset")
    rows = []
    for p in sorted((out_dir / "per_config").glob("*.json")):
        r = json.loads(p.read_text())
        key, m = r["dataset"], r["metrics"]
        pub = published.loc[key]
        rows.append(
            {
                "config": key,
                "domain": pub["domain"],
                "term": key.split("/")[2],
                "mase": m[MASE_COL],
                "mase_ratio": m[MASE_COL] / pub[f"eval_metrics/{MASE_COL}"],
                "crps_ratio": m[CRPS_COL] / pub[f"eval_metrics/{CRPS_COL}"],
                "below_q10": m["coverage[0.1]"],
                "below_q90": m["coverage[0.9]"],
                "seconds": r["seconds"],
            }
        )
    df = pd.DataFrame(rows).set_index("config").sort_values("mase_ratio")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    print(f"\n{label} against published {reference} (ratio below 1 = better than published)")
    print(df.round(4).to_string())
    n = len(df)
    for col in ("mase_ratio", "crps_ratio"):
        r = df[col]
        print(f"{col}: geometric mean {gmean(list(r)):.4f}, median {r.median():.4f}, min {r.min():.4f}, max {r.max():.4f}; "
              f"better on {(r < 1).sum()} of {n}, by more than 5 % on {(r < 0.95).sum()}, worse by more than 5 % on {(r > 1.05).sum()}")
    for key in ("term", "domain"):
        g = df.groupby(key).agg(configurations=("mase_ratio", "size"), mase_ratio=("mase_ratio", lambda x: gmean(list(x))),
                                crps_ratio=("crps_ratio", lambda x: gmean(list(x))))
        print(f"\nby {key} (geometric mean of the ratio)")
        print(g.round(4).to_string())
    print(f"\ntotal evaluation time: {df['seconds'].sum():.0f} s")
    df.to_csv(out_dir / "against_published.csv")


def compare(out_dir: Path, reference: str) -> bool:
    """Print our metrics next to the published ones for `reference`; True when all agree within 0.1 %."""
    published = pd.read_csv(GIFT_REPO / "results" / reference / "all_results.csv").set_index("dataset")
    naive = pd.read_csv(GIFT_REPO / "results" / "Seasonal_Naive" / "all_results.csv").set_index("dataset")
    rows = []
    for p in sorted((out_dir / "per_config").glob("*.json")):
        r = json.loads(p.read_text())
        key, m = r["dataset"], r["metrics"]
        pub = published.loc[key]
        rows.append(
            {
                "config": key,
                "mase": m[MASE_COL],
                "mase_published": pub[f"eval_metrics/{MASE_COL}"],
                "crps": m[CRPS_COL],
                "crps_published": pub[f"eval_metrics/{CRPS_COL}"],
                "mase_vs_naive": m[MASE_COL] / naive.loc[key, f"eval_metrics/{MASE_COL}"],
                "crps_vs_naive": m[CRPS_COL] / naive.loc[key, f"eval_metrics/{CRPS_COL}"],
                "below_q10": m["coverage[0.1]"],
                "below_q90": m["coverage[0.9]"],
                "seconds": r["seconds"],
            }
        )
    df = pd.DataFrame(rows).set_index("config")
    df["mase_diff_%"] = (df["mase"] / df["mase_published"] - 1) * 100
    df["crps_diff_%"] = (df["crps"] / df["crps_published"] - 1) * 100
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    show = ["mase", "mase_published", "mase_diff_%", "crps", "crps_published", "crps_diff_%", "below_q10", "below_q90", "seconds"]
    print(df[show].round(4).to_string())
    worst = max(df["mase_diff_%"].abs().max(), df["crps_diff_%"].abs().max())
    print(f"\n{len(df)} configurations compared with published {reference}; largest absolute difference: {worst:.4f} %")
    print(f"total evaluation time: {df['seconds'].sum():.0f} s")
    if len(df) == 97:
        print(f"aggregate over 97 configurations: MASE {gmean(list(df['mase_vs_naive'])):.4f}   "
              f"CRPS {gmean(list(df['crps_vs_naive'])):.4f}")
    ok = bool(worst < 0.1)
    print("RESULT:", "PASS (all within 0.1 %)" if ok else "DIFFERS from published by more than 0.1 % somewhere")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", choices=["5M", "28M", "123M", "209M"], help="published checkpoint to evaluate")
    ap.add_argument("--checkpoint", help="local checkpoint directory (instead of --size)")
    ap.add_argument("--name", help="name of the results folder (default: Timestral-<size>)")
    ap.add_argument("--only", help="comma-separated substrings; keep the configurations whose key contains one")
    ap.add_argument("--compare-with", help="published model to compare with (default: Timestral-<size>)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, help="forecast in chunks of this many patches (default: the forecaster's "
                    "one pass of up to 32); results go to Timestral-<size>-chunk<N> and are reported as ratios")
    ap.add_argument("--terms", help="comma-separated horizon terms to keep (short, medium, long)")
    args = ap.parse_args()
    if bool(args.size) == bool(args.checkpoint):
        ap.error("give exactly one of --size and --checkpoint")
    if args.checkpoint and not args.name:
        ap.error("--checkpoint needs --name")

    name = args.name or (f"Timestral-{args.size}-chunk{args.chunk}" if args.chunk else f"Timestral-{args.size}")
    source = args.checkpoint or f"mistralai/Timestral-{args.size}"
    reference = args.compare_with or (f"Timestral-{args.size}" if args.size else None)
    out_dir = ROOT / "results" / "gift_eval" / name
    (out_dir / "per_config").mkdir(parents=True, exist_ok=True)

    logging.getLogger("gluonts.model.forecast").setLevel(logging.ERROR)
    properties = json.load(open(GIFT_REPO / "notebooks" / "dataset_properties.json"))
    configs = all_configs(properties)
    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
        configs = [c for c in configs if any(w in c[2] for w in wanted)]
        if not configs:
            ap.error(f"--only {args.only!r} matches no configuration")
    if args.terms:
        terms = {t.strip() for t in args.terms.split(",")}
        if not terms <= {"short", "medium", "long"}:
            ap.error(f"--terms {args.terms!r}: use short, medium, long")
        configs = [c for c in configs if c[1] in terms]

    fc = TimestralForecaster.from_pretrained(source, device=args.device, per_shot_horizon_patches=args.chunk or 32)
    print(f"evaluating {source} on {args.device}: {len(configs)} configurations, "
          f"{fc.per_shot_horizon_patches} patches per pass", flush=True)
    for dataset_name, term, key in configs:
        path = out_dir / "per_config" / (key.replace("/", "__") + ".json")
        if path.exists():
            continue
        t0 = time.perf_counter()
        metrics = evaluate_config(fc, dataset_name, term)
        seconds = time.perf_counter() - t0
        path.write_text(json.dumps({"dataset": key, "metrics": metrics, "seconds": seconds}, indent=1))
        print(f"{key}: MASE={metrics[MASE_COL]:.4f} CRPS={metrics[CRPS_COL]:.4f} ({seconds:.0f} s)", flush=True)

    if reference is None:
        return 0
    if args.chunk or args.checkpoint:
        compare_variant(out_dir, reference, name)
        return 0
    return 0 if compare(out_dir, reference) else 1


if __name__ == "__main__":
    sys.exit(main())
