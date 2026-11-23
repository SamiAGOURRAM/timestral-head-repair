"""Free: CLOSED-FORM design of head 5.1's query gains: one ridge least-squares solve.

    g_f = exp(i w_f s) + (A_f + i B_f) * 8 * exp(-i w_f D) / c_f ,   s = 11 - D,  f = 0..NF-1

The mean positional kernel is linear in (A, B):
    K(d) = Kshift(d) + sum_f Re[(A_f + i B_f) r_f exp(i w_f (d - D))],   d >= 1 (encoder keys)
    own key (d = 0), per token class: self = self_shift + sum_f Re[(A_f + i B_f) 8 exp(-i w_f D) s_f / c_f] / 8
with r_f = 1 (all-token mean query, mean encoder key) and s_f the mean of query*conj(key) of the
same token. Targets (logits), P = Kshift(D) + LIFT:
    d = D: P;  d = D-1, D+1: P - 6
    d = 11: P - SECOND;  d = 10, 12: P - SECOND - 6        (what a token with < 32 patches behind should read)
    d = 2, 3: P - THIRD                                    (what a token with < 11 patches behind should read)
    every other d in 1..287 and every own-key row: min(shifted value, P - FLOOR)
Ridge on (A, B). Then the exact offline attention of the head (as in s17) is printed as a check.
Averages of activations + least squares only; no search.

Usage: python s20_design_ls.py name D LIFT SECOND THIRD FLOOR ridge [NF]
"""
import sys

import numpy as np
import torch

import edits
from harness import OUT, Bench, qkv_rows, time_attention
from timestral import layers as L

LAYER, HEAD = 5, 1
name, D = sys.argv[1], int(sys.argv[2])
LIFT, SECOND, THIRD, FLOOR, RIDGE = (float(v) for v in sys.argv[3:8])
NF = int(sys.argv[8]) if len(sys.argv) > 8 else 8
s = 11 - D
bench = Bench()
model = bench.base
inv = model.transformer.rope.inv_freq.double().numpy()[:16]
attn = time_attention(model, LAYER)
Wq = attn.wQKV.weight[qkv_rows(attn, 0, HEAD)].detach().double()
Wk = attn.wQKV.weight[qkv_rows(attn, 1, HEAD)].detach().double()
XN = []
for period in (80, 180):
    resid = torch.tensor(np.load(OUT / f"cap_base_{period}.npz")["resid"][LAYER][:32])
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


def unrot(v):
    return np.concatenate([v[..., 16:32], v[..., 48:]], -1)


q_np, k_np = Q0.numpy(), K0.numpy()
zq = cplx(q_np).mean((0, 1))
zk = cplx(k_np[:, 64:256]).mean((0, 1))
c = zq * np.conj(zk)
c0 = unrot(q_np).mean((0, 1)) @ unrot(k_np[:, 64:256]).mean((0, 1)) / 8
gshift = np.exp(1j * inv * s)
ds = np.arange(1, 288)
Kshift = c0 + np.real((gshift * c)[None, :] * np.exp(1j * inv[None, :] * ds[:, None])).sum(1) / 8
P = Kshift[D - 1] + LIFT

rows, targets, weights = [], [], []
for i, d in enumerate(ds):
    ph = np.exp(1j * inv[:NF] * (d - D))
    rows.append(np.concatenate([ph.real, -ph.imag]))  # Re[(A + iB) e^{i w (d-D)}] = A cos - B sin
    if d == D:
        t, wt = P, 4.0
    elif d in (D - 1, D + 1):
        t, wt = P - 6, 1.0
    elif d == 11:
        t, wt = P - SECOND, 2.0
    elif d in (10, 12):
        t, wt = P - SECOND - 6, 1.0
    elif d in (2, 3):
        t, wt = P - THIRD, 2.0
    else:
        t, wt = min(Kshift[i], P - FLOOR), (1.0 if d < 64 else 0.5)
    targets.append(t - Kshift[i])
    weights.append(wt)
# own-key rows, one per token class
classes = {"ctx 1-10": slice(1, 11), "ctx 11-31": slice(11, 32), "ctx 64-255": slice(64, 256),
           "dec 0-7": slice(256, 264), "dec 8-15": slice(264, 272), "dec 16-23": slice(272, 280), "dec 24-31": slice(280, 288)}
for cname, sl in classes.items():
    sf = (cplx(q_np[:, sl]) * np.conj(cplx(k_np[:, sl]))).mean((0, 1))
    u = (unrot(q_np[:, sl]) * unrot(k_np[:, sl])).sum(-1).mean()
    self_shift = (u + np.real(gshift * sf).sum()) / 8
    coef = (np.exp(-1j * inv[:NF] * D) * sf[:NF] / c[:NF])  # Re[(A + iB) coef]
    rows.append(np.concatenate([coef.real, -coef.imag]))
    targets.append(min(self_shift, P - FLOOR) - self_shift)
    weights.append(3.0)
A = np.array(rows)
w = np.array(weights)
sol = np.linalg.solve(A.T @ (w[:, None] * A) + RIDGE * np.eye(2 * NF), A.T @ (w * np.array(targets)))
g = gshift.copy()
g[:NF] = g[:NF] + (sol[:NF] + 1j * sol[NF:]) * 8 * np.exp(-1j * inv[:NF] * D) / c[:NF]
np.save(OUT / f"gain_{name}.npy", g)
print(f"D {D} s {s} LIFT {LIFT} SECOND {SECOND} THIRD {THIRD} FLOOR {FLOOR} ridge {RIDGE} NF {NF}")
print(" A (cos):", np.round(sol[:NF], 1), "\n B (sin):", np.round(sol[NF:], 1))
print("|g_f|:", np.round(np.abs(g), 2))


def attention(gg):
    q = Q0 @ edits.gain_matrix(gg).T
    qr, _ = L._apply_rotary(q[:, :, None, :], q[:, :, None, :], cos, sin)
    logits = qr[:, :, 0] @ KR.transpose(-1, -2) / 8
    return logits.masked_fill(~causal[None], float("-inf")).softmax(-1)


W_ORIG = attention(np.ones(16, dtype=complex))
wn = attention(g)
idx = torch.arange(96, 288)
on = wn[:, idx, idx - D]
near = on + wn[:, idx, idx - D - 1] + wn[:, idx, idx - D + 1]
GROUPS = {"enc96-255": slice(0, 160), "dec0-7": slice(160, 168), "dec8-15": slice(168, 176), "dec16-23": slice(176, 184), "dec24-31": slice(184, 192)}
print("exact check, mean attention on exactly D back / on D-1..D+1:", {k: (round(float(on[:, v].mean()), 3), round(float(near[:, v].mean()), 3)) for k, v in GROUPS.items()})
print("   worst single decoder token, mass on D-1..D+1:", round(float(near[:, 160:].min()), 3))
overlap = torch.minimum(wn[:, :32], W_ORIG[:, :32]).sum(-1).mean(0)
print("overlap with the original attention, context patches 0..31:", np.round(overlap.numpy(), 2))
m = wn.mean(0).numpy()
for p in [0, 16, 28, 31]:
    top = np.argsort(-m[256 + p])[:5]
    print(f"  decoder patch {p:2d}: top keys (patches back: mass) " + " ".join(f"{256 + p - t}:{m[256 + p, t]:.2f}" for t in top))
for j in [3, 5, 8, 12, 20, 31, 32, 34, 40, 200]:
    top = np.argsort(-m[j])[:5]
    print(f"  context patch {j:3d}: top keys (patches back: mass) " + " ".join(f"{j - t}:{m[j, t]:.2f}" for t in top))
