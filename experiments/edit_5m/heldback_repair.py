"""Repair run 1: independent check of the delivered edit, then a FRESH held-back evaluation.

The first held-back sets have been looked at by this session and summarised to the repair
agent, so they no longer count as unseen for it. This script uses new seeds and cycle
lengths that appear nowhere else in the project.

1. What changed against the published weights.
2. Fresh sets, 8192 steps of history: dips on cycles 70, 100, 150, 220, 300 (not whole
   numbers of patches) and 64, 160 (whole patches); spikes on a sinusoid, cycles 100, 220;
   plus the real data `bizitobs_service`.
3. The fresh dip sets with the history cut to ten lengths, including lengths the repair
   agent was not asked about.

Models: unedited 5M, the blind run's edit, the repair run's edit.

Run: .venv\\Scripts\\python.exe -u experiments\\edit_5m\\heldback_repair.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

import heldback_eval as he
from timestral import TimestralForecaster, load_model

ARM = Path(__file__).resolve().parents[2] / "checkpoints"  # edit_b/ is built by edit/apply_edit_b.py
UNEDITED, BLIND, REPAIR = "5M unedited", "5M blind run 1 edit", "5M repair run 1 edit"
SOURCES = {UNEDITED: "mistralai/Timestral-5M", BLIND: ARM / "edit_b", REPAIR: ARM / "edit_b"}
OUT_NAME = "repair1"  # results folder under results/edit_5m; other runs reuse this script with other values
MISALIGNED = (70, 100, 150, 220, 300)
HISTORIES = (512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 6144, 8192)


def fresh_sets() -> dict:
    sets = {}
    for c in MISALIGNED:
        sets[f"dips, cycle {c}"] = ("dip", "misaligned", *he.synth.make_series(c, he.synth.N_SERIES, seed=7000 + c))
    for c in (64, 160):
        sets[f"dips, cycle {c} (whole patches)"] = ("dip", "aligned", *he.synth.make_series(c, he.synth.N_SERIES, seed=7000 + c))
    for c in (100, 220):
        sets[f"spikes, cycle {c}"] = ("spike", "spike", *he.make_spike_series(c, he.synth.N_SERIES, seed=7500 + c))
    return sets


def main() -> None:
    pd.set_option("display.width", 230)
    base = load_model(SOURCES[UNEDITED], device="cpu")
    edited = load_model(SOURCES[REPAIR], device="cpu")
    total = sum(p.numel() for p in base.parameters())
    print("=== 1. Repair edit: tensors that differ from the published weights")
    for (name, a), (_, b) in zip(base.named_parameters(), edited.named_parameters()):
        if not torch.equal(a, b):
            print(f"  {name}: {int((a != b).sum())} weights of {total} ({100 * int((a != b).sum()) / total:.3f} %), "
                  f"{int((a != b).reshape(a.shape[0], -1).any(dim=1).sum())} rows; relative size of the change {float((a - b).norm() / a.norm()):.3f}")

    print("\n=== 2. Fresh held-back sets, 8192 steps of history", flush=True)
    he.synthetic_sets = fresh_sets
    he.MODELS = [(label, src, 32) for label, src in SOURCES.items()]
    he.CANDIDATES = [BLIND, REPAIR]
    he.OUT = he.OUT / OUT_NAME
    he.main()

    print("\n=== 3. Fresh dip sets by length of history (mean over cycles 70, 100, 150, 220, 300)", flush=True)
    rows = []
    for label, src in SOURCES.items():
        fc = TimestralForecaster(load_model(src, device=he.DEVICE))
        for cycle in MISALIGNED:
            x, meta = he.synth.make_series(cycle, he.synth.N_SERIES, seed=7000 + cycle)
            sd = np.array([m["sd"] for m in meta])[:, None]
            for history in HISTORIES:
                ctx = x[:, None, he.synth.CONTEXT - history : he.synth.CONTEXT].astype(np.float32)
                with torch.inference_mode():
                    med = fc.forecast(ctx, he.synth.HORIZON).median[:, 0].double().cpu().numpy()
                shares = he.synth.recovered_shares(med, meta)
                err = np.abs(med - x[:, he.synth.CONTEXT :]) / sd
                rows.append({"model": label, "cycle": cycle, "history": history,
                             "near": float(np.mean([s for _, lead, s in shares if lead < 384])),
                             "far": float(np.mean([s for _, lead, s in shares if lead >= 512])),
                             "error": float(err.mean()), "error first 96": float(err[:, :96].mean())})
    df = pd.DataFrame(rows)
    df.to_csv(he.OUT / "history_length.csv", index=False)
    for col in ("near", "far", "error", "error first 96"):
        print(f"\n{col}")
        print(df.pivot_table(index="history", columns="model", values=col)[list(SOURCES)].round(3).to_string())
    worst = df.pivot_table(index=["history", "cycle"], columns="model", values="near")
    d = (worst[REPAIR] - worst[UNEDITED])
    print(f"\nrepair edit, largest drop of near dips on any cycle and history: {d.min():.3f} at {d.idxmin()}; "
          f"cells with a drop over 0.05: {int((d < -0.05).sum())} of {len(d)}")
    e = df.pivot_table(index=["history", "cycle"], columns="model", values="error")
    d = (e[REPAIR] - e[UNEDITED])
    print(f"repair edit, largest rise of the error: {d.max():.3f} at {d.idxmax()}; cells with a rise over 0.05: {int((d > 0.05).sum())} of {len(d)}")


if __name__ == "__main__":
    main()
