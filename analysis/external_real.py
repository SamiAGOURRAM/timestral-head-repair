"""Real data outside GIFT-Eval: does it carry the failure, and does edit B fix it?

Public datasets, downloaded on first run into data/external_real/ (not redistributed here):
- tides: NOAA tide gauges, water level every 6 minutes, 2024-01-01 to 2024-03-05: Seattle
  (9447130), The Battery NY (8518750), Boston (8443970), Honolulu (1612340). Main tide period
  12.42 h = 124.2 steps (3.9 patches, not a whole number).
- solar: German Weather Service (DWD) open data, global radiation every 10 minutes, recent
  period, the first three stations of the listing with a long enough gap-free stretch. The
  daily cycle is 144 steps (4.5 patches, not a whole number), zero every night.
- nyc_taxi: NYC taxi passengers every 30 minutes, Jul 2014 - Jan 2015 (Numenta Anomaly
  Benchmark). Daily cycle 48 steps (1.5 patches), night dip every day.

Per series: the longest gap-free stretch; its last 8192 + 1024 + 99 x 64 steps give 100
forecasts of 1024 steps in one pass from 8192 steps of history, start points 64 steps apart
(fewer if the stretch is shorter; series too short for 10 forecasts are skipped). Published
5M vs edit B. Error = absolute error of the median divided by the series' std over its
history; reported for the near part (0-383 steps ahead) and the distant part (512+).

Expectation (written before running): a series carries the failure if the published model's
distant-part error is clearly above its near-part error; edit B then lowers it. Without the
failure, edit B changes little.

Larger models for reference (added in a second run, expectation written before it): the
published 28M and 123M, run on the same forecasts. Expected: on the tides, edit B (5M) has a
lower distant-part error than the published 28M; 28M is better on the near part; 123M is
better than both.

Needs `checkpoints/edit_b` (`python edit/apply_edit_b.py`). Writes `results/external_real.npz`
and `results/external_real_summary.csv`.

Run: python analysis/external_real.py
"""

import csv
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from timestral import TimestralForecaster, load_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "external_real"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
H, CTX, STRIDE, MAX_WINDOWS = 1024, 8192, 64, 100
UA = {"User-Agent": "research-script"}
TIDES = {"9447130": "Seattle", "8518750": "The Battery, NY", "8443970": "Boston", "1612340": "Honolulu"}
DWD = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/solar/recent/"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read()


def cached(name: str, url: str) -> bytes:
    f = DATA / name
    if not f.exists():
        f.write_bytes(fetch(url))
    return f.read_bytes()


def nyc_taxi() -> np.ndarray:
    t = cached("nyc_taxi.csv", "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv").decode()
    return np.array([float(r["value"]) for r in csv.DictReader(io.StringIO(t))])


def tide(station: str) -> np.ndarray:
    vals = []
    for b, e in (("20240101", "20240131"), ("20240201", "20240229"), ("20240301", "20240305")):
        url = ("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=water_level&application=research"
               f"&begin_date={b}&end_date={e}&datum=MLLW&station={station}&time_zone=gmt&units=metric&format=json")
        d = json.loads(cached(f"tide_{station}_{b}.json", url))
        vals += [float(x["v"]) if x.get("v") not in ("", None) else np.nan for x in d.get("data", [])]
    return np.array(vals, dtype=float)


def dwd_stations() -> list[str]:
    listing = cached("dwd_solar_listing.html", DWD).decode("latin-1")
    return sorted(set(re.findall(r"10minutenwerte_SOLAR_(\d{5})_akt\.zip", listing)))


def dwd_solar(station: str) -> np.ndarray:
    z = zipfile.ZipFile(io.BytesIO(cached(f"dwd_solar_{station}.zip", f"{DWD}10minutenwerte_SOLAR_{station}_akt.zip")))
    name = next(n for n in z.namelist() if n.startswith("produkt"))
    df = pd.read_csv(z.open(name), sep=";", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    t = pd.to_datetime(df["MESS_DATUM"].astype(str), format="%Y%m%d%H%M")
    s = pd.Series(df["GS_10"].astype(float).where(lambda v: v > -999).values, index=t)
    s = s[~s.index.duplicated()].asfreq("10min")  # missing timestamps become NaN
    return s.values


def clean_stretch(x: np.ndarray) -> np.ndarray:
    bad = np.isnan(x)
    edges = np.flatnonzero(np.diff(np.r_[1, bad.astype(int), 1]))
    runs = list(zip(edges[::2], edges[1::2]))
    if not runs:  # no observed value at all (e.g. a station that does not measure this quantity)
        return x[:0]
    s, e = max(runs, key=lambda t: t[1] - t[0])
    return x[s:e]


def evaluate(x: np.ndarray, fcs: dict) -> dict | None:
    x = clean_stretch(x)
    need = CTX + H + (MAX_WINDOWS - 1) * STRIDE
    x = x[-need:]
    starts = list(range(CTX, len(x) - H + 1, STRIDE))
    if len(starts) < 10:
        return None
    batch = np.stack([x[s - CTX: s] for s in starts]).astype(np.float32)[:, None]
    truth = np.stack([x[s: s + H] for s in starts])
    scale = np.stack([np.std(x[s - CTX: s]) for s in starts])[:, None]
    out = {"truth": truth, "history_tail": np.stack([x[s - 1024: s] for s in starts])}
    for m, fc in fcs.items():
        meds = []
        bs = 8 if "123M" in m else 32  # batches of forecasts sized for an 8 GB GPU
        for k in range(0, len(starts), bs):
            with torch.inference_mode():
                meds.append(fc.forecast(batch[k: k + bs], H).median[:, 0].float().cpu().numpy())
        med = np.concatenate(meds)
        out[f"{m}_median"], out[f"{m}_err"] = med, np.abs(med - truth) / scale
    return out


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    fcs = {"published": TimestralForecaster(load_model("mistralai/Timestral-5M", device=DEVICE)),
           "edit B": TimestralForecaster(load_model(str(ROOT / "checkpoints" / "edit_b"), device=DEVICE)),
           "published 28M": TimestralForecaster(load_model("mistralai/Timestral-28M", device=DEVICE)),
           "published 123M": TimestralForecaster(load_model("mistralai/Timestral-123M", device=DEVICE))}
    series = [(f"tide_{sid}", f"Tide, {place}", "tide", lambda sid=sid: tide(sid)) for sid, place in TIDES.items()]
    n_solar = 0
    for sid in dwd_stations():
        if n_solar == 3:
            break
        try:
            x = dwd_solar(sid)
        except Exception as e:  # malformed archive: skip this station, say so
            print(f"solar {sid}: skipped ({type(e).__name__}: {e})")
            continue
        if len(clean_stretch(x)) >= CTX + H + 9 * STRIDE:
            series.append((f"solar_{sid}", f"Solar radiation, DWD station {sid}", "solar", lambda x=x: x))
            n_solar += 1
    series.append(("taxi", "NYC taxi passengers", "taxi", nyc_taxi))

    save, rows = {}, []
    for key, label, kind, load in series:
        r = evaluate(load(), fcs)
        if r is None:
            print(f"{label}: gap-free stretch too short, skipped")
            continue
        p, b = r["published_err"], r["edit B_err"]
        pf, bf = p[:, 512:].mean(1), b[:, 512:].mean(1)
        row = {"key": key, "label": label, "kind": kind, "windows": len(pf),
               "published near": p[:, :384].mean(), "published far": p[:, 512:].mean(),
               "edit B near": b[:, :384].mean(), "edit B far": b[:, 512:].mean(), "edit B far better in": int((bf < pf).sum())}
        row["failure (published far / near)"] = row["published far"] / row["published near"]
        row["edit B / published, far"] = row["edit B far"] / row["published far"]
        row["edit B / published, near"] = row["edit B near"] / row["published near"]
        for big in ("28M", "123M"):
            e = r[f"published {big}_err"]
            row[f"published {big} near"], row[f"published {big} far"] = e[:, :384].mean(), e[:, 512:].mean()
            row[f"edit B far better than {big} in"] = int((bf < e[:, 512:].mean(1)).sum())
        rows.append(row)
        print(f"{label}: {len(pf)} windows | failure {row['failure (published far / near)']:.2f} | "
              f"edit B / published: far {row['edit B / published, far']:.3f}, near {row['edit B / published, near']:.3f} | "
              f"edit B better far ahead in {row['edit B far better in']} | far: 28M {row['published 28M far']:.3f}, "
              f"123M {row['published 123M far']:.3f}, edit B {row['edit B far']:.3f}; near: 28M {row['published 28M near']:.3f}, "
              f"edit B {row['edit B near']:.3f} | edit B far better than 28M in {row['edit B far better than 28M in']}", flush=True)
        for k, v in r.items():
            save[f"{key}__{k}"] = v
        save[f"{key}__typical"] = np.array(int(np.argsort(bf / pf)[len(pf) // 2]))  # median improvement, not the best
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "external_real_summary.csv", index=False)
    save["keys"] = np.array(df["key"].tolist())
    save["labels"] = np.array(df["label"].tolist())
    np.savez(ROOT / "results" / "external_real.npz", **save)
    print("saved results/external_real.npz and results/external_real_summary.csv")


if __name__ == "__main__":
    main()
