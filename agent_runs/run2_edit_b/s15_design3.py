"""Free: third design for head 5.1's query (same form as s13_design2.py):

    g_f = exp(i w_f s) + E_f * 8 * exp(-i w_f D) / c_f ,   s = 11 - D

but the margin is now checked separately for decoder patches 0-7, 8-15, 16-23, 24-31 (their
mean queries differ) and includes the token's own key (d = 0), which design b let through
(far decoder patches gave ~45 % of the head to themselves).

E_f is searched on a coarse grid over the slow frequencies w = 0.316, 0.1, 0.056, 0.0316
(coordinate search, a few rounds; objective = worst margin - 0.05 * sum |E|). No gradients.

Usage: python s15_design3.py name D [E0.316 E0.1 E0.056 E0.0316]
"""
import sys

import numpy as np
import torch

from harness import OUT, Bench, qkv_rows, time_attention

LAYER, HEAD = 5, 1
name, D = sys.argv[1], float(sys.argv[2])
Di = int(D)
s = 11 - D
bench = Bench()
model = bench.base
inv = model.transformer.rope.inv_freq.double().numpy()[:16]
FREQS = [2, 4, 5, 6]


def cplx(v):
    return v[..., :16] + 1j * v[..., 32:48]


def unrot(v):
    return np.concatenate([v[..., 16:32], v[..., 48:]], -1)


G = [(0, 8), (8, 16), (16, 24), (24, 32)]
acc = {}
for period in (80, 180):
    resid = torch.tensor(np.load(OUT / f"cap_base_{period}.npz")["resid"][LAYER])
    with torch.no_grad():
        xn = model.transformer.layers[LAYER].attn_norms[0](resid)
        attn = time_attention(model, LAYER)
        w = attn.wQKV.weight
        q = (xn @ w[qkv_rows(attn, 0, HEAD)].T).double().numpy()
        k = (xn @ w[qkv_rows(attn, 1, HEAD)].T).double().numpy()
    items = {"qe": q[:, 64:256], "ke": k[:, 64:256]}
    for i, (a, b) in enumerate(G):
        items[f"qd{i}"] = q[:, 256 + a : 256 + b]
        items[f"kd{i}"] = k[:, 256 + a : 256 + b]
    for key, v in items.items():
        acc.setdefault(key, []).append((cplx(v).mean((0, 1)), unrot(v).mean((0, 1))))
    # the token's own key (d = 0): mean of the PRODUCT query * conj(key) of the same token
    # (added after design b: the product of the means missed the query/key correlation)
    for key in ["e"] + [f"d{i}" for i in range(4)]:
        qq, kk = items["q" + key], items["k" + key]
        acc.setdefault("self" + key, []).append(((cplx(qq) * np.conj(cplx(kk))).mean((0, 1)), np.array((unrot(qq) * unrot(kk)).sum(-1).mean())))
m = {key: (sum(z for z, _ in v) / len(v), sum(u for _, u in v) / len(v)) for key, v in acc.items()}


def self_score(g, key):
    z, u = m["self" + key]
    return float(u + np.real(g * z).sum()) / 8
zq_all = (sum(m[f"qd{i}"][0] for i in range(4)) / 4 + m["qe"][0]) / 2
c = zq_all * np.conj(m["ke"][0])
ds = np.arange(0, 288)


def gains(E):
    g = np.exp(1j * inv * s)
    for f, e in zip(FREQS, E):
        g[f] = g[f] + e * 8 * np.exp(-1j * inv[f] * D) / c[f]
    return g


def kern(g, qk, kk):
    return (m[qk][1] @ m[kk][1]) / 8 + np.real((g * m[qk][0] * np.conj(m[kk][0]))[None, :] * np.exp(1j * inv[None, :] * ds[:, None])).sum(1) / 8


def margins(g):
    out = {}
    ke = kern(g, "qe", "ke")
    out["enc"] = ke[Di] - max(np.r_[ke[1 : Di - 1], ke[Di + 2 :]].max(), self_score(g, "e"))
    for i, (a, b) in enumerate(G):
        kq = kern(g, f"qd{i}", "ke")
        peak = kq[Di]
        # encoder keys: d from 1 (patch a, nearest) ... ; exclude D-1..D+1
        worst = np.r_[kq[1 : Di - 1], kq[Di + 2 :]].max()
        # own key and earlier decoder keys: d = 0 .. b-1, keys of each decoder group
        for j in range(i + 1):
            kd = kern(g, f"qd{i}", f"kd{j}")
            lo, hi = max(1, a - G[j][1] + 1), b - G[j][0]
            worst = max(worst, kd[lo:hi].max())
        worst = max(worst, self_score(g, f"d{i}"))
        out[f"self{a}"] = peak - self_score(g, f"d{i}")
        out[f"dec{a}-{b - 1}"] = peak - worst
    return out


def objective(E):
    g = gains(E)
    return min(margins(g).values()) - 0.05 * sum(abs(e) for e in E) - 3.0 * max(0.0, float(np.abs(g).max()) - 6.0)


if len(sys.argv) > 3:
    E = [float(v) for v in sys.argv[3:7]]
else:
    E = [0.0, 0.0, 0.0, 0.0]
    for rnd in range(4):
        for i in range(4):
            cands = [E[:i] + [v] + E[i + 1 :] for v in np.arange(-24, 25, 2.0)]
            E = max(cands, key=objective)
g = gains(E)
np.save(OUT / f"gain_{name}.npy", g)
print(f"D {D} s {s} E {dict(zip([round(float(inv[f]), 4) for f in FREQS], E))}")
print("|g_f|:", np.round(np.abs(g), 2))
print("margins of d = D over every other reachable key outside D-1..D+1 (logits):", {k2: round(float(v), 1) for k2, v in margins(g).items()})
for qk in ["qe", "qd0", "qd3"]:
    kq = kern(g, qk, "ke")
    print(f"  {qk} / enc keys: top " + " ".join(f"{o}:{kq[o]:.1f}" for o in np.argsort(-kq)[:8]) + "   | d<32 only: " + " ".join(f"{o}:{kq[o]:.1f}" for o in np.argsort(-kq[:32])[:5]))
print("  qd3 / own decoder group keys, d = 0..7:", np.round(kern(g, "qd3", "kd3")[:8], 1))
