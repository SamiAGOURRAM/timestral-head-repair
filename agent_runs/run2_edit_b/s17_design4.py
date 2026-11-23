"""Free: design of head 5.1's query gains with the exact attention of that head as the objective.

s16_logit_check.py showed that head 5.1's attention weights, computed offline from the saved
layer-5 input residuals as q.k dot products with the rotary rotation, reproduce the recorded
pass to two decimals (the edit touches only this head's query rows, so its inputs do not move).
Here the same dot products are evaluated for candidate gains

    g_f = exp(i w_f s) + E_f * 8 * exp(-i w_f D) / c_f      (s = 11 - D; see s13_design2.py)

and E is chosen by coordinate grid search (no gradients) to maximise the attention mass that
tokens put on the token D patches back: worst mean over decoder patch groups 0-7, 8-15, 16-23,
24-31 and encoder patches 96-255, both diagnostic cycles, 16 series each.

Usage: python s17_design4.py name D [E for w=0.316 0.178 0.1 0.056 0.0316]
"""
import sys

import numpy as np
import torch

import edits
from harness import OUT, Bench, qkv_rows, time_attention
from timestral import layers as L

LAYER, HEAD = 5, 1
name, D = sys.argv[1], int(sys.argv[2])
s = 11 - D
FREQS = [2, 3, 4, 5, 6]
bench = Bench()
model = bench.base
inv = model.transformer.rope.inv_freq.double().numpy()[:16]
attn = time_attention(model, LAYER)
Wq = attn.wQKV.weight[qkv_rows(attn, 0, HEAD)].detach().double()
Wk = attn.wQKV.weight[qkv_rows(attn, 1, HEAD)].detach().double()
XN = []
for period in (80, 180):
    resid = torch.tensor(np.load(OUT / f"cap_base_{period}.npz")["resid"][LAYER][:16])
    with torch.no_grad():
        XN.append(model.transformer.layers[LAYER].attn_norms[0](resid).double())
XN = torch.cat(XN)  # (32, 288, 192)
pos = torch.arange(288)[None].expand(len(XN), -1)
cos, sin = model.transformer.rope(XN, pos)
cos, sin = cos.double(), sin.double()
Q0 = XN @ Wq.T
K0 = XN @ Wk.T
_, KR = L._apply_rotary(K0[:, :, None, :], K0[:, :, None, :], cos, sin)
KR = KR[:, :, 0]
causal = torch.arange(288)[None, :] <= torch.arange(288)[:, None]


def cplx(v):
    return v[..., :16] + 1j * v[..., 32:48]


zq = cplx(Q0.numpy()).mean((0, 1))
zk = cplx(K0.numpy()[:, 64:256]).mean((0, 1))
c = zq * np.conj(zk)


def gains(E):
    g = np.exp(1j * inv * s)
    for f, e in zip(FREQS, E):
        g[f] = g[f] + e * 8 * np.exp(-1j * inv[f] * D) / c[f]
    return g


def attention(g):
    q = Q0 @ edits.gain_matrix(g).T
    qr, _ = L._apply_rotary(q[:, :, None, :], q[:, :, None, :], cos, sin)
    logits = qr[:, :, 0] @ KR.transpose(-1, -2) / 8
    return logits.masked_fill(~causal[None], float("-inf")).softmax(-1)  # (32, 288, 288)


def masses(g):
    w = attention(g)
    idx = torch.arange(96, 288)
    on = w[:, idx, idx - D]  # (32, 192)
    near = on + w[:, idx, idx - D - 1] + w[:, idx, idx - D + 1]
    groups = {"enc96-255": slice(0, 160), "dec0-7": slice(160, 168), "dec8-15": slice(168, 176), "dec16-23": slice(176, 184), "dec24-31": slice(184, 192)}
    return {k: (float(on[:, v].mean()), float(near[:, v].mean())) for k, v in groups.items()}, w


def objective(E):
    g = gains(E)
    ms, _ = masses(g)
    return min(v[1] for v in ms.values()) - 0.002 * sum(abs(e) for e in E) - 0.2 * max(0.0, float(np.abs(g).max()) - 6.0)


if len(sys.argv) > 3:
    E = [float(v) for v in sys.argv[3:8]]
else:
    E = [0.0] * 5
    for rnd in range(3):
        for i in range(5):
            cands = [E[:i] + [v] + E[i + 1 :] for v in np.arange(-24, 25, 3.0)]
            E = max(cands, key=objective)
        print("round", rnd, E, round(objective(E), 3), flush=True)
g = gains(E)
np.save(OUT / f"gain_{name}.npy", g)
ms, w = masses(g)
print(f"D {D} s {s} E {dict(zip([round(float(inv[f]), 4) for f in FREQS], E))}")
print("|g_f|:", np.round(np.abs(g), 2))
print("mean attention on exactly D back / on D-1..D+1:", {k: (round(a, 3), round(b, 3)) for k, (a, b) in ms.items()})
m = w.mean(0).numpy()
for p in [0, 8, 16, 24, 31]:
    top = np.argsort(-m[256 + p])[:5]
    print(f"  decoder patch {p:2d}: top keys (patches back: mass) " + " ".join(f"{256 + p - t}:{m[256 + p, t]:.2f}" for t in top))
for j in [12, 20, 31, 40, 200]:
    top = np.argsort(-m[j])[:5]
    print(f"  context patch {j:3d}: top keys (patches back: mass) " + " ".join(f"{j - t}:{m[j, t]:.2f}" for t in top))
