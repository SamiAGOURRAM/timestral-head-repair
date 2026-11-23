"""Check 5 (reading the model): do forecast positions keep attending to history at the right place in the cycle?

Context. Timestral forecasts a horizon with one identical learned query token per future
patch (32 steps). A query token knows where it is only through the rotary position
encoding of the time attention; it reads the whole history and the query tokens before it.
To draw a repeating dip at the right place, the token for a future patch has to read
history patches that sit at the same place in the cycle. Check 3 showed 5M stops placing
the dip beyond about 16 patches ahead, 28M beyond about 22, 123M not within 32.

Measure. On the synthetic series of check 3 (180-step cycle), for every time-attention
head and every future patch k: the share of that token's attention to the history that
lands on "in-phase" history patches, those whose distance to the future patch is within
8 steps of a whole number of cycles. Divided by the share expected if attention ignored
the cycle (the fraction of history patches that are in-phase), so 1 = no lock.

Reported per model: the heads with the strongest lock near the origin, and how their lock
changes with k; and the best lock over all heads at each k.

Run: .venv\\Scripts\\python.exe -u experiments\\gap_10s\\check5_attention_lock.py
"""

import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check3_synthetic_lead_time as synth  # noqa: E402
import timestral.layers as layers  # noqa: E402
from timestral import load_model  # noqa: E402

SIZES = ["5M", "28M", "123M"]
PERIOD = 180
PATCH = 32
H = 32  # future patches (1024 steps)
N_SERIES = 32
TOLERANCE = 8  # steps
KS = [0, 2, 4, 8, 12, 14, 16, 18, 20, 24, 28, 31]
DEVICE = os.environ.get("WSP_DEVICE", "cpu")
OUT = synth.OUT


def record_attention(model, x: np.ndarray) -> list[torch.Tensor]:
    """Per time-attention layer: `(H, heads, L + H)` attention of the future tokens, averaged over series."""
    captured: list[torch.Tensor] = []
    original = layers._sdpa
    n_ctx = x.shape[1] // PATCH

    def spy(query, key, value, mask, is_causal=False):
        if query.shape[1] == H and key.shape[1] == n_ctx + H:  # future tokens reading history + future
            q, k = query.transpose(1, 2).float(), key.transpose(1, 2).float()
            scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
            scores = scores.masked_fill(~mask, float("-inf"))
            captured.append(scores.softmax(dim=-1).mean(dim=0).permute(1, 0, 2).cpu())  # (H, heads, keys)
        return original(query, key, value, mask, is_causal)

    layers._sdpa = spy
    try:
        with torch.inference_mode():
            ctx = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
            model.predict_block(ctx, torch.ones_like(ctx, dtype=torch.bool), torch.arange(len(x), device=DEVICE), H)
    finally:
        layers._sdpa = original
    return captured


def in_phase(n_ctx: int, k: int) -> np.ndarray:
    """Boolean `(n_ctx,)`: history patches whose distance to future patch k is within TOLERANCE of whole cycles."""
    distance = PATCH * (n_ctx + k - np.arange(n_ctx))
    off = (distance + PERIOD / 2) % PERIOD - PERIOD / 2
    return np.abs(off) <= TOLERANCE


def main() -> None:
    pd.set_option("display.width", 220)
    if DEVICE == "cpu":
        torch.set_num_threads(12)
    x, _ = synth.make_series(PERIOD, N_SERIES, seed=PERIOD)
    x = x[:, : synth.CONTEXT]
    n_ctx = x.shape[1] // PATCH
    chance = float(np.mean([in_phase(n_ctx, k).mean() for k in KS]))
    print(f"{N_SERIES} series, cycle {PERIOD} steps, {n_ctx} history patches, {H} future patches; "
          f"in-phase history patches: {chance:.3f} of all (lock = 1 means attention ignores the cycle)", flush=True)
    rows = []
    for size in SIZES:
        model = load_model(f"mistralai/Timestral-{size}", device=DEVICE)
        attn = record_attention(model, x)
        n_layers, n_heads = len(attn), attn[0].shape[1]
        lock = np.zeros((n_layers, n_heads, len(KS)))
        hist_share = np.zeros((n_layers, n_heads, len(KS)))
        for li, a in enumerate(attn):
            a = a.numpy()
            for ki, k in enumerate(KS):
                to_hist = a[k, :, :n_ctx]  # (heads, n_ctx)
                mass = to_hist.sum(axis=-1)
                phase = in_phase(n_ctx, k)
                lock[li, :, ki] = (to_hist[:, phase].sum(axis=-1) / np.maximum(mass, 1e-12)) / phase.mean()
                hist_share[li, :, ki] = mass
        for li in range(n_layers):
            for hi in range(n_heads):
                for ki, k in enumerate(KS):
                    rows.append({"model": size, "layer": li, "head": hi, "k": k, "lock": lock[li, hi, ki],
                                 "attention_to_history": hist_share[li, hi, ki]})
        print(f"\n=== Timestral-{size}: {n_layers} time-attention layers x {n_heads} heads", flush=True)
        best = pd.DataFrame({"best lock over heads": lock.max(axis=(0, 1)),
                             "heads with lock above 3": (lock > 3).sum(axis=(0, 1)),
                             "mean attention to history": hist_share.mean(axis=(0, 1))},
                            index=pd.Index(KS, name="future patch k"))
        print(best.round(2).T.to_string())
        near = lock[:, :, [KS.index(2), KS.index(4), KS.index(8)]].mean(axis=-1)
        order = np.dstack(np.unravel_index(np.argsort(-near, axis=None), near.shape))[0][:8]
        table = pd.DataFrame({f"L{li}.H{hi}": lock[li, hi] for li, hi in order}, index=pd.Index(KS, name="future patch k")).T
        print("lock of the 8 heads most locked near the origin (rows: layer.head)")
        print(table.round(2).to_string(), flush=True)
        del model
    pd.DataFrame(rows).to_csv(OUT / "check5_attention_lock.csv", index=False)
    print(f"\nWrote {(OUT / 'check5_attention_lock.csv').relative_to(synth.ROOT)}")


if __name__ == "__main__":
    main()
