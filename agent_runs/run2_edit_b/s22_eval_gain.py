"""Free: robustness check of a head-5.1 gain design, with the exact offline attention of the
head (all 64 series of each diagnostic cycle). For every token with >= 34 patches behind it:
margin = logit of the key D back - best logit among all other keys except D-1, D+1.
Usage: python s22_eval_gain.py D name1 name2 ..."""
import sys

import numpy as np
import torch

import edits
from harness import OUT, Bench, qkv_rows, time_attention
from timestral import layers as L

LAYER, HEAD = 5, 1
D = int(sys.argv[1])
bench = Bench()
model = bench.base
attn = time_attention(model, LAYER)
Wq = attn.wQKV.weight[qkv_rows(attn, 0, HEAD)].detach().double()
Wk = attn.wQKV.weight[qkv_rows(attn, 1, HEAD)].detach().double()
causal = torch.arange(288)[None, :] <= torch.arange(288)[:, None]
data = {}
for period in (80, 180):
    resid = torch.tensor(np.load(OUT / f"cap_base_{period}.npz")["resid"][LAYER])
    with torch.no_grad():
        xn = model.transformer.layers[LAYER].attn_norms[0](resid).double()
    pos = torch.arange(288)[None].expand(len(xn), -1)
    cos, sin = model.transformer.rope(xn, pos)
    cos, sin = cos.double(), sin.double()
    q0, k0 = xn @ Wq.T, xn @ Wk.T
    _, kr = L._apply_rotary(k0[:, :, None, :], k0[:, :, None, :], cos, sin)
    data[period] = (q0, kr[:, :, 0], cos, sin)
for name in sys.argv[2:]:
    g = np.load(OUT / f"gain_{name}.npy")
    print(f"design {name}: |g| = {np.round(np.abs(g[:8]), 2)}")
    for period, (q0, kr, cos, sin) in data.items():
        q = q0 @ edits.gain_matrix(g).T
        qr, _ = L._apply_rotary(q[:, :, None, :], q[:, :, None, :], cos, sin)
        logits = (qr[:, :, 0] @ kr.transpose(-1, -2) / 8).masked_fill(~causal[None], float("-inf"))
        idx = torch.arange(34, 288)
        peak = logits[:, idx, idx - D]
        other = logits[:, idx].clone()
        for o in (-1, 0, 1):
            other[:, torch.arange(len(idx)), idx - D + o] = float("-inf")
        best, arg = other.max(-1)
        margin = (peak - best).numpy()
        w = logits.softmax(-1)
        mass = (w[:, idx, idx - D] + w[:, idx, idx - D - 1] + w[:, idx, idx - D + 1]).numpy()
        for label, sl in [("context 34-255", slice(0, 222)), ("decoder 0-15", slice(222, 238)), ("decoder 16-31", slice(238, 254))]:
            mg = margin[:, sl]
            rival = (idx[sl][None, :] - arg[:, sl]).numpy()  # patches back of the best rival key
            vals, counts = np.unique(rival[mg < np.percentile(mg, 5)], return_counts=True)
            top = sorted(zip(counts, vals), reverse=True)[:3]
            print(f"   cycle {period} {label:15s}: margin min {mg.min():6.1f}  1% {np.percentile(mg, 1):6.1f}  median {np.median(mg):6.1f}"
                  f" | mass on D-1..D+1: min {mass[:, sl].min():.3f} mean {mass[:, sl].mean():.3f} | rivals at the low end (patches back): {[int(v) for _, v in top]}")
