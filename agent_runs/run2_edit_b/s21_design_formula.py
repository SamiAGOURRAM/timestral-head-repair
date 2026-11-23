"""Free: explicit-formula design of head 5.1's query gains (no search, no least squares).

For the four rotary frequencies where the head's mean query and key are strong
(F = {0, 1, 4, 5}: w = 1, 0.562, 0.1, 0.0562 rad/patch) the query's complex component is
REPLACED so that these frequencies form bumps that are all in phase at chosen distances:

    g_f = 8 * a_f * [ exp(-i w_f D) + beta * exp(-i w_f 11) + gamma * exp(-i w_f 2.5) ] / c_f     (f in F)
    g_f = exp(i w_f (11 - D))                                                                      (other f: plain shift)

c_f = (mean query component) * conj(mean key component), averages over the saved layer-5 inputs
of both diagnostic cycles. With the mean vectors the kernel part of F is
    sum_f a_f [ cos(w_f (d - D)) + beta cos(w_f (d - 11)) + gamma cos(w_f (d - 2.5)) ]
i.e. the main peak at d = D, a lower one at d = 11 (what the head read originally; used by tokens
with fewer than D patches behind them) and a still lower one at d = 2-3 (what the original
head read when fewer than 11 patches were behind).

The exact offline attention of the head (see s16/s17) is printed as a check.
Usage: python s21_design_formula.py name D a0 a1 a4 a5 beta gamma
"""
import sys

import numpy as np
import torch

import edits
from harness import OUT, Bench, qkv_rows, time_attention
from timestral import layers as L

LAYER, HEAD = 5, 1
name, D = sys.argv[1], int(sys.argv[2])
a = dict(zip([0, 1, 4, 5], (float(v) for v in sys.argv[3:7])))
beta, gamma = float(sys.argv[7]), float(sys.argv[8])
bench = Bench()
model = bench.base
inv = model.transformer.rope.inv_freq.double().numpy()[:16]
attn = time_attention(model, LAYER)
Wq = attn.wQKV.weight[qkv_rows(attn, 0, HEAD)].detach().double()
Wk = attn.wQKV.weight[qkv_rows(attn, 1, HEAD)].detach().double()
XN = []
for period in (80, 180):
    resid = torch.tensor(np.load(OUT / f"cap_base_{period}.npz")["resid"][LAYER])
    with torch.no_grad():
        XN.append(model.transformer.layers[LAYER].attn_norms[0](resid).double())
XN = torch.cat(XN)  # (128, 288, 192)
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
g = np.exp(1j * inv * (11 - D))
for f, af in a.items():
    # correction after try k1 (which failed: the slow bumps at 11 and 2.5 pulled the peak away from D):
    # the secondary bumps (beta at 11, gamma at 2.5) are built from the two FAST frequencies only;
    # the two slow frequencies (0.1, 0.0562) form a single hill centred on D.
    sec = (beta * np.exp(-1j * inv[f] * 11) + gamma * np.exp(-1j * inv[f] * 2.5)) if f in (0, 1) else 0.0
    g[f] = 8 * af * (np.exp(-1j * inv[f] * D) + sec) / c[f]
np.save(OUT / f"gain_{name}.npy", g)
np.save(OUT / f"gain_{name}_c.npy", c)
print(f"D {D} a {a} beta {beta} gamma {gamma}")
print("c_f (f = 0, 1, 4, 5):", np.round(c[[0, 1, 4, 5]], 2))
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
print("   worst single decoder token (128 series x 32 patches), mass on D-1..D+1:", round(float(near[:, 160:].min()), 3),
      "; 1st percentile:", round(float(np.percentile(near[:, 160:].numpy(), 1)), 3))
overlap = torch.minimum(wn[:, :32], W_ORIG[:, :32]).sum(-1).mean(0)
print("overlap with the original attention, context patches 0..31:", np.round(overlap.numpy(), 2))
m = wn.mean(0).numpy()
for p in [0, 16, 28, 31]:
    top = np.argsort(-m[256 + p])[:5]
    print(f"  decoder patch {p:2d}: top keys (patches back: mass) " + " ".join(f"{256 + p - t}:{m[256 + p, t]:.2f}" for t in top))
for j in [3, 5, 8, 12, 20, 31, 32, 34, 40, 200]:
    top = np.argsort(-m[j])[:5]
    print(f"  context patch {j:3d}: top keys (patches back: mass) " + " ".join(f"{j - t}:{m[j, t]:.2f}" for t in top))
