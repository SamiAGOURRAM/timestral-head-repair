"""Does the fixed-lag mechanism found in Timestral-5M by the blind run exist in Timestral-28M, and does the same edit work there?

Exploratory, not a protocol run (this session has seen the held-back results for 5M).

1. Find the fixed-lag heads of 28M: for each time-attention head, the attention of history
   positions to history positions, as a function of how many patches back, averaged over
   16 series and over the second half of the history. A head is "fixed-lag" when more than
   half of its attention sits on one lag.
2. For each fixed-lag head with a lag of 2 to 12 patches: apply the blind run's edit
   (rotate the rotary pairs of the head's query so it reads `60 - lag` patches further
   back, then multiply the query by 1.5; the `relag` and `qgain` of the blind run), and
   score near and far dips on cycles 80 and 180 with 8192 and 2048 steps of history.

Run: .venv\\Scripts\\python.exe -u experiments\\edit_5m\\check_28m_fixed_lag.py
"""

import copy
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "gap_10s"))
import check3_synthetic_lead_time as synth  # noqa: E402
import timestral.layers as layers  # noqa: E402
from timestral import TimestralForecaster, load_model  # noqa: E402

SIZE = os.environ.get("WSP_SIZE", "28M")
OUT = ROOT / "results" / "edit_5m"
N_CTX = synth.CONTEXT // 32
MAX_LAG = 80


def qkv_rows(attn, part: int, head: int) -> torch.Tensor:
    hd, nh = attn.head_dim, attn.num_heads
    return torch.tensor([part * hd * nh + d * nh + head for d in range(hd)])


def relag_and_gain(model, layer: int, head: int, delta: float, gain: float) -> None:
    """The blind run's edit (its `relag` then `qgain`), for any model size."""
    attn = model.transformer.layers[layer].attns[0]
    assert attn.is_time
    rows = qkv_rows(attn, 0, head)
    half = attn.head_dim // 2
    inv_freq = model.transformer.rope.inv_freq.double().cpu()
    w = attn.wQKV.weight.data
    q1, q2 = w[rows[:half]].double().clone(), w[rows[half:]].double().clone()
    ang = (inv_freq * delta)[:, None]
    c, s = torch.cos(ang), torch.sin(ang)
    w[rows[:half]] = ((q1 * c + q2 * s) * gain).to(w.dtype)
    w[rows[half:]] = ((-q1 * s + q2 * c) * gain).to(w.dtype)


def lag_profiles(model, x: np.ndarray) -> np.ndarray:
    """`(layers, heads, MAX_LAG + 1)`: share of attention of history positions by lag."""
    profiles = []
    original = layers._sdpa

    def spy(query, key, value, mask, is_causal=False):
        if query.shape[1] == N_CTX and key.shape[1] == N_CTX:  # history reading history, time attention
            q, k = query.transpose(1, 2).float(), key.transpose(1, 2).float()
            scores = (q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])).masked_fill(~mask, float("-inf"))
            a = scores.softmax(dim=-1).mean(dim=0)  # (heads, N, N)
            prof = torch.zeros(a.shape[0], MAX_LAG + 1)
            qs = range(N_CTX // 2, N_CTX)
            for lag in range(MAX_LAG + 1):
                prof[:, lag] = torch.stack([a[:, p, p - lag] for p in qs], dim=1).mean(dim=1)
            profiles.append(prof.numpy())
        return original(query, key, value, mask, is_causal)

    layers._sdpa = spy
    try:
        with torch.inference_mode():
            ctx = torch.as_tensor(x[:, : synth.CONTEXT], dtype=torch.float32)
            model.predict_block(ctx, torch.ones_like(ctx, dtype=torch.bool), torch.arange(len(x)), 32)
    finally:
        layers._sdpa = original
    return np.stack(profiles)


def score(model, sets: dict, history: int) -> dict:
    fc = TimestralForecaster(model)
    out = {}
    for cycle, (x, meta) in sets.items():
        with torch.inference_mode():
            med = fc.forecast(x[:, None, synth.CONTEXT - history : synth.CONTEXT].astype(np.float32), synth.HORIZON).median[:, 0].double().numpy()
        shares = synth.recovered_shares(med, meta)
        out[f"near{cycle}"] = float(np.mean([s for _, lead, s in shares if lead < 384]))
        out[f"far{cycle}"] = float(np.mean([s for _, lead, s in shares if lead >= 512]))
    return out


def main() -> None:
    pd.set_option("display.width", 200)
    torch.set_num_threads(8)
    base = load_model(f"mistralai/Timestral-{SIZE}", device="cpu")
    sets = {c: synth.make_series(c, 32, seed=c) for c in (80, 180)}
    prof = lag_profiles(base, sets[180][0][:16])
    n_layers, n_heads = prof.shape[:2]
    rows = []
    for li in range(n_layers):
        for hi in range(n_heads):
            lag = int(prof[li, hi].argmax())
            rows.append({"head": f"{li}.{hi}", "layer": li, "h": hi, "lag": lag, "share on that lag": float(prof[li, hi, lag]),
                         "share within 80 patches": float(prof[li, hi].sum())})
    heads = pd.DataFrame(rows)
    fixed = heads[(heads["share on that lag"] > 0.5)]
    print(f"=== Timestral-{SIZE}: {n_layers} layers x {n_heads} heads; heads with more than half of their attention on one lag")
    print(fixed[["head", "lag", "share on that lag"]].round(2).to_string(index=False), flush=True)

    results = {}
    for history in (8192, 2048):
        results[("unedited", history)] = score(base, sets, history)
    print("\nunedited:", {k: {m: round(v, 3) for m, v in r.items()} for k, r in results.items()}, flush=True)
    for _, r in fixed[(fixed["lag"] >= 2) & (fixed["lag"] <= 12)].iterrows():
        model = copy.deepcopy(base)
        relag_and_gain(model, int(r["layer"]), int(r["h"]), 60.0 - r["lag"], 1.5)
        for history in (8192, 2048):
            results[(f"head {r['head']} (lag {r['lag']} -> 60), x1.5", history)] = score(model, sets, history)
        print(f"done head {r['head']}", flush=True)
    table = pd.DataFrame(results).T
    table.index.names = ["edit", "history"]
    print("\n=== near and far dips recovered (32 series per cycle)")
    print(table.round(3).to_string())
    table.to_csv(OUT / f"check_{SIZE}_fixed_lag.csv")


if __name__ == "__main__":
    main()
