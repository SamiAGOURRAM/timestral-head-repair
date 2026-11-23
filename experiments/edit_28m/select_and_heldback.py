"""28M fixed-lag repair: selection on the diagnostic sets, then the held-back evaluation (PROTOCOL.md).

Phase 1 (selection): candidate heads x D in {32, 60} x g in {1.0, 1.5}, scored on the
diagnostic sets at histories 1024, 2048, 4096, 8192; admissibility and choice as in the
protocol. The selected edit is saved to checkpoints/28M_fixedlag_final.
Phase 2 (held-back): fresh sets, history sweep, real data, for unedited 28M, the selected
28M edit, unedited 5M and edit B (blind run 2).

Run: .venv\\Scripts\\python.exe -u experiments\\edit_28m\\select_and_heldback.py
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "edit_5m"))
sys.path.insert(0, str(ROOT / "experiments" / "gap_10s"))
import check_28m_fixed_lag as c28  # noqa: E402
import heldback_eval as he  # noqa: E402
from timestral import TimestralForecaster, load_model  # noqa: E402
from timestral.loading import resolve_checkpoint  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = ROOT / "results" / "edit_28m"
CKPT = ROOT / "checkpoints" / "28M_fixedlag_final"
SELECT_HISTORIES = (1024, 2048, 4096, 8192)
HELD_HISTORIES = (768, 1024, 1536, 2048, 3072, 4096, 8192)
MISALIGNED = (66, 90, 130, 175, 270)
ALIGNED = (192, 224)
SPIKES = (130, 270)
EDIT_B = ROOT / "checkpoints" / "edit_b"


def edited_copy(base, li: int, hi: int, delta: float, g: float):
    """Copy of `base` with the edit applied (the edit code works on CPU tensors), moved to DEVICE."""
    m = copy.deepcopy(base).cpu()
    c28.relag_and_gain(m, li, hi, delta, g)
    return m.to(DEVICE)


def scores(model, sets: dict, history: int) -> dict:
    """near / far / error per set, one pass, default forecaster."""
    fc = TimestralForecaster(model)
    out = {}
    for name, (kind, x, meta) in sets.items():
        with torch.inference_mode():
            med = fc.forecast(x[:, None, he.synth.CONTEXT - history : he.synth.CONTEXT].astype(np.float32),
                              he.synth.HORIZON).median[:, 0].double().cpu().numpy()
        shares = he.synth.recovered_shares(med, meta) if kind == "dip" else he.spike_shares(med, meta)
        sd = np.array([m["sd"] for m in meta])[:, None]
        out[name] = {"near": float(np.mean([s for _, lead, s in shares if lead < 384])),
                     "far": float(np.mean([s for _, lead, s in shares if lead >= 512])),
                     "error": float((np.abs(med - x[:, he.synth.CONTEXT :]) / sd).mean())}
    return out


def phase1(base) -> tuple:
    diag = {f"diag {c}": ("dip", *he.synth.make_series(c, he.synth.N_SERIES, seed=c)) for c in (80, 180)}
    prof = c28.lag_profiles(base.cpu(), diag["diag 180"][1][:16])
    base.to(DEVICE)
    heads = [(li, hi, int(prof[li, hi].argmax())) for li in range(prof.shape[0]) for hi in range(prof.shape[1])
             if prof[li, hi].max() > 0.5 and 2 <= prof[li, hi].argmax() <= 12]
    print("candidate heads (layer, head, lag):", heads, flush=True)
    ref = {h: scores(base, diag, h) for h in SELECT_HISTORIES}
    rows = []
    for li, hi, lag in heads:
        for d in (32, 60):
            for g in (1.0, 1.5):
                m = edited_copy(base, li, hi, float(d - lag), g)
                ok, far_gain = True, None
                row = {"head": f"{li}.{hi}", "lag": lag, "D": d, "g": g}
                for h in SELECT_HISTORIES:
                    s = scores(m, diag, h)
                    for k in diag:
                        dn = s[k]["near"] - ref[h][k]["near"]
                        de = s[k]["error"] - ref[h][k]["error"]
                        row[f"{h} {k} near"] = round(dn, 3)
                        row[f"{h} {k} err"] = round(de, 3)
                        ok &= dn >= -0.05 and de <= 0.05
                    if h == 8192:
                        far_gain = float(np.mean([s[k]["far"] - ref[h][k]["far"] for k in diag]))
                row.update({"admissible": ok, "far gain 8192": round(far_gain, 3)})
                rows.append(row)
                print(f"  {row['head']} lag {lag} -> {d}, g {g}: admissible {ok}, far gain {far_gain:.3f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "selection.csv", index=False)
    adm = table[table["admissible"]]
    if adm.empty:
        print("\nNO ADMISSIBLE CANDIDATE: negative result")
        return None, table
    best = adm.sort_values("far gain 8192", ascending=False).iloc[0]
    print(f"\nselected: head {best['head']}, lag {best['lag']} -> {best['D']}, g {best['g']}, far gain {best['far gain 8192']}")
    li, hi = (int(v) for v in best["head"].split("."))
    m = edited_copy(base, li, hi, float(best["D"] - best["lag"]), float(best["g"]))
    CKPT.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in m.state_dict().items()}, str(CKPT / "model.safetensors"))
    (CKPT / "config.json").write_text((Path(resolve_checkpoint("mistralai/Timestral-28M")) / "config.json").read_text())
    (OUT / "selected.json").write_text(json.dumps({k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()}, indent=1))
    return m, table


def phase2(models: dict) -> None:
    sets = {}
    for c in MISALIGNED:
        sets[f"dips {c}"] = ("dip", *he.synth.make_series(c, he.synth.N_SERIES, seed=5000 + c))
    for c in ALIGNED:
        sets[f"dips {c} (whole patches)"] = ("dip", *he.synth.make_series(c, he.synth.N_SERIES, seed=5000 + c))
    for c in SPIKES:
        sets[f"spikes {c}"] = ("spike", *he.make_spike_series(c, he.synth.N_SERIES, seed=5500 + c))
    groups = {"misaligned": [f"dips {c}" for c in MISALIGNED], "aligned": [f"dips {c} (whole patches)" for c in ALIGNED],
              "spikes": [f"spikes {c}" for c in SPIKES]}
    rows = []
    for label, model in models.items():
        for h in HELD_HISTORIES:
            s = scores(model, sets, h)
            for gname, names in groups.items():
                rows.append({"model": label, "history": h, "group": gname,
                             **{k: float(np.mean([s[n][k] for n in names])) for k in ("near", "far", "error")}})
        print(f"held-back synthetic done: {label}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "heldback_synthetic.csv", index=False)
    order = list(models)
    for g in ("misaligned", "aligned", "spikes"):
        for k in ("far", "near", "error"):
            t = df[df.group == g].pivot(index="history", columns="model", values=k)[order]
            print(f"\n=== {g}, {k}\n{t.round(3).to_string()}")

    series, official, horizon, season, _ = he.real.load("bizitobs_service")
    origins = [o for o in range(official[-1], he.real.FIRST_ORIGIN - 1, -horizon)][::-1]
    real = {}
    for label, model in models.items():
        fc = TimestralForecaster(model)
        err = np.stack([he.real.scaled_errors(fc, series, o, horizon, season) for o in origins])
        real[label] = np.nanmean(err, axis=(0, 2, 3))
    per = pd.DataFrame(real)
    per.to_csv(OUT / "heldback_bizitobs_per_series.csv", index=False)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(per), (10_000, len(per)))
    print("\n=== bizitobs_service, 21 series x 10 start points, 600-step horizon")
    for label, ref in (("28M edit", "28M unedited"), ("5M edit B", "5M unedited")):
        a, b = per[label].to_numpy(), per[ref].to_numpy()
        r = a[idx].mean(1) / b[idx].mean(1)
        print(f"  {label}: MASE {a.mean():.4f} vs {ref} {b.mean():.4f}; ratio {a.mean() / b.mean():.4f} "
              f"[{np.quantile(r, 0.025):.4f}, {np.quantile(r, 0.975):.4f}]; better on {int((a < b).sum())} of {len(a)}")

    m = df[(df.group == "misaligned") & (df.history == 8192)].set_index("model")
    print("\n=== Criteria 1-3, 28M edit against unedited 28M")
    d_far = m.loc["28M edit", "far"] - m.loc["28M unedited", "far"]
    d_near = m.loc["28M edit", "near"] - m.loc["28M unedited", "near"]
    ratio = per["28M edit"].mean() / per["28M unedited"].mean()
    print(f"far {d_far:+.3f} ({'met' if d_far >= 0.20 else 'NOT met'}); near {d_near:+.3f} ({'met' if d_near >= -0.05 else 'NOT met'}); "
          f"bizitobs ratio {ratio:.4f} ({'met' if ratio <= 1 else 'NOT met'})")
    print("=== Same, edit B against unedited 5M (third held-back check)")
    d_far = m.loc["5M edit B", "far"] - m.loc["5M unedited", "far"]
    d_near = m.loc["5M edit B", "near"] - m.loc["5M unedited", "near"]
    ratio = per["5M edit B"].mean() / per["5M unedited"].mean()
    print(f"far {d_far:+.3f}; near {d_near:+.3f}; bizitobs ratio {ratio:.4f}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_model("mistralai/Timestral-28M", device=DEVICE)
    edited, _ = phase1(base)
    if edited is None:
        return
    phase2({"28M unedited": base, "28M edit": edited,
            "5M unedited": load_model("mistralai/Timestral-5M", device=DEVICE), "5M edit B": load_model(EDIT_B, device=DEVICE)})


if __name__ == "__main__":
    main()
