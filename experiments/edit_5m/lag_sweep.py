"""Lag sweep on head 5.1: is lag 32 a lone safe point, the edge of a safe range, or tied to the series' cycles?

One construction for every lag (blind run 5's, `results/edit_5m/blind_run5/agent_files/edits.py`,
single peak): each rotary frequency pair f of head 5.1's query rows is multiplied, as a
complex number, by m_f = |zq_f| e^{i arg zk_f} e^{-i w_f D} / zq_f, so the head's positional
score peaks at lag D patches. zq, zk = mean query / key components of the head per
frequency, taken from run 5's saved reference means (`l5h1_ref_means.json`). Only the 32
query rows of the 16 rotary pairs change (6 144 weights at most), as in edits B and 5.

Lags: 11 (the published head's own lag, re-phased: should change almost nothing), 21 and 42
(32 x lag is a whole number of weeks for hourly data), 28, 30, 31 (below 32: far forecast
tokens still read forecast tokens), 32, 33, 34, 36, 40, 43.

What would separate the explanations (written before running):
- only lag 32 safe on the hourly sets -> something specific to 32 (= the 32-patch forecast block);
- every lag up to 32 safe, every lag above harmful -> reading further back than the block is the harm;
- 21 and 42 safe while neighbours are not -> the series' weekly cycle;
- the same harm at every lag except 11 -> the construction itself, not the lag.

This script builds the checkpoints (`checkpoints/lagsweep_D`) and scores each on the
synthetic sets of `baseline_finetune.py` (far / near dips); `tools/lag_sweep.ps1` then runs
GIFT-Eval on seven hourly and daily configurations that runs 3-5 damage.

Run: .venv\\Scripts\\python.exe -u experiments\\edit_5m\\lag_sweep.py
"""

import json
import shutil

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

import baseline_finetune as bf
from timestral import load_model
from timestral.loading import resolve_checkpoint

LAGS = (11, 21, 28, 30, 31, 32, 33, 34, 36, 40, 42, 43)
LAYER, HEAD = 5, 1
REF = bf.ROOT / "edit" / "l5h1_ref_means.json"


def retarget(model, lag: int) -> int:
    ref = json.loads(REF.read_text())
    zq = np.array(ref["zq_re"]) + 1j * np.array(ref["zq_im"])
    zk = np.array(ref["zk_re"]) + 1j * np.array(ref["zk_im"])
    att = model.transformer.layers[LAYER].attns[0]
    om = model.transformer.rope.inv_freq[:16].double().cpu().numpy()
    mult = np.abs(zq) * np.exp(1j * np.angle(zk)) * np.exp(-1j * om * lag) / zq
    hd, nh = att.head_dim, att.num_heads
    rows = [d * nh + HEAD for d in range(hd)]  # query rows of this head (part 0), laid out (part, head_dim, n_heads)
    W = att.wQKV.weight.data
    for f in range(16):
        ra, rb = rows[f], rows[f + 32]
        a, b = W[ra].clone(), W[rb].clone()
        c, s = float(mult[f].real), float(mult[f].imag)
        W[ra] = c * a - s * b
        W[rb] = s * a + c * b
    return len(rows)


def main() -> None:
    pd.set_option("display.width", 200)
    base = load_model("mistralai/Timestral-5M", device=bf.DEVICE)
    before = bf.evaluate(base)
    rows = {"published": {"far": float(before["far dips"].mean()), "near": float(before["near dips"].mean()),
                          "error": float(before["error / noise sd"].mean()), "weights changed": 0}}
    for lag in LAGS:
        m = load_model("mistralai/Timestral-5M", device=bf.DEVICE)
        retarget(m, lag)
        changed = {n: int((p != q).sum()) for (n, p), (_, q) in zip(m.named_parameters(), base.named_parameters()) if not torch.equal(p, q)}
        assert set(changed) <= {"transformer.layers.5.attns.0.wQKV.weight"} and sum(changed.values()) <= 6144, changed
        after = bf.evaluate(m)
        rows[f"lag {lag}"] = {"far": float(after["far dips"].mean()), "near": float(after["near dips"].mean()),
                              "error": float(after["error / noise sd"].mean()), "weights changed": sum(changed.values())}
        ckpt = bf.ROOT / "checkpoints" / f"lagsweep_{lag}"
        ckpt.mkdir(parents=True, exist_ok=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in m.state_dict().items()}, str(ckpt / "model.safetensors"))
        shutil.copy(resolve_checkpoint("mistralai/Timestral-5M") / "config.json", ckpt / "config.json")
        print(f"lag {lag}: far {rows[f'lag {lag}']['far']:.3f}, near {rows[f'lag {lag}']['near']:.3f}", flush=True)
        del m
    df = pd.DataFrame(rows).T
    print("\nsynthetic sets of baseline_finetune.py (means over 7 sets)\n" + df.round(3).to_string())
    df.to_csv(bf.OUT / "lag_sweep_synthetic.csv")


if __name__ == "__main__":
    main()
