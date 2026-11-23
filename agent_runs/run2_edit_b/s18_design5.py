"""Free: as s17_design4.py (exact offline attention of head 5.1 as the objective), with two additions:

- envelope terms allowed on the eight fastest rotary frequencies (w = 1 ... 0.0178);
- a second objective term: tokens that have fewer than 32 patches behind them (context patches
  0..31, which cannot read 32 back) should keep the ORIGINAL attention pattern of the head
  (overlap = sum over keys of min(new, original), 1 = identical). This is meant to leave short
  histories as they were.

objective = min over groups of mass on D-1..D+1  +  lam * mean overlap(patches 0..31)
            - 0.002 * sum|E| - 0.2 * max(0, max|g| - gcap)
Coordinate grid search, no gradients.

Usage: python s18_design5.py name D lam gcap [8 values of E]
"""
import sys

import numpy as np
import torch

import edits
from harness import OUT, Bench, qkv_rows, time_attention
from timestral import layers as L

LAYER, HEAD = 5, 1
name, D, lam, gcap = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
s = 11 - D
FREQS = list(range(8))
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
XN = torch.cat(XN)
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
    return logits.masked_fill(~causal[None], float("-inf")).softmax(-1)


W_ORIG = attention(np.ones(16, dtype=complex))
GROUPS = {"enc96-255": slice(0, 160), "dec0-7": slice(160, 168), "dec8-15": slice(168, 176), "dec16-23": slice(176, 184), "dec24-31": slice(184, 192)}


def stats(g):
    w = attention(g)
    idx = torch.arange(96, 288)
    on = w[:, idx, idx - D]
    near = on + w[:, idx, idx - D - 1] + w[:, idx, idx - D + 1]
    ms = {k: (float(on[:, v].mean()), float(near[:, v].mean())) for k, v in GROUPS.items()}
    overlap = torch.minimum(w[:, :32], W_ORIG[:, :32]).sum(-1).mean(0)  # (32,)
    return ms, overlap, w


def objective(E):
    g = gains(E)
    ms, overlap, _ = stats(g)
    return min(v[1] for v in ms.values()) + lam * float(overlap.mean()) - 0.002 * sum(abs(e) for e in E) - 0.2 * max(0.0, float(np.abs(g).max()) - gcap)


if len(sys.argv) > 5:
    E = [float(v) for v in sys.argv[5:13]]
else:
    E = [0.0, 0.0, 3.0, 0.0, 12.0, 24.0, 0.0, 0.0]  # start from design e
    for rnd in range(3):
        for i in [4, 5, 2, 3, 6, 7, 1, 0]:
            cands = [E[:i] + [v] + E[i + 1 :] for v in np.arange(-30, 31, 3.0)]
            E = max(cands, key=objective)
        print("round", rnd, E, round(objective(E), 3), flush=True)
g = gains(E)
np.save(OUT / f"gain_{name}.npy", g)
ms, overlap, w = stats(g)
print(f"D {D} s {s} E {dict(zip([round(float(inv[f]), 4) for f in FREQS], E))}")
print("|g_f|:", np.round(np.abs(g), 2))
print("mean attention on exactly D back / on D-1..D+1:", {k: (round(a, 3), round(b, 3)) for k, (a, b) in ms.items()})
print("overlap with the original attention, context patches 0..31:", np.round(overlap.numpy(), 2))
m = w.mean(0).numpy()
for p in [0, 16, 31]:
    top = np.argsort(-m[256 + p])[:5]
    print(f"  decoder patch {p:2d}: top keys (patches back: mass) " + " ".join(f"{256 + p - t}:{m[256 + p, t]:.2f}" for t in top))
for j in [5, 12, 20, 31, 32, 34, 40, 200]:
    top = np.argsort(-m[j])[:5]
    print(f"  context patch {j:3d}: top keys (patches back: mass) " + " ".join(f"{j - t}:{m[j, t]:.2f}" for t in top))
m0 = W_ORIG.mean(0).numpy()
for j in [3, 5, 8, 12, 20, 25, 31]:
    top = np.argsort(-m0[j])[:5]
    print(f"  ORIGINAL context patch {j:3d}: top keys (patches back: mass) " + " ".join(f"{j - t}:{m0[j, t]:.2f}" for t in top))
