"""Free: second design for head 5.1's query. Per rotary frequency f, the query's complex
component is multiplied by a complex gain

    g_f = exp(i w_f s) + E_f * 8 * exp(-i w_f D) / c_f

- exp(i w_f s) is the plain rotary shift: kernel K(d) -> K(d + s). With s = 11 - D the lobe at
  d = 11 moves to d = D.
- the second term adds E_f * cos(w_f (d - D)) to the kernel for a few slow frequencies (an
  envelope centred on D) so that the D lobe beats the other lobes that the shift brings into
  the causal range. c_f = mean query component * conj(mean key component) (averages of the
  saved activations of both diagnostic cycles).

Usage: python s13_design2.py name D [E0.1 E0.178 E0.056]   (no E given: small grid search, best saved)
"""
import itertools
import sys

import numpy as np
import torch

from harness import OUT, Bench, qkv_rows, time_attention

LAYER, HEAD = 5, 1
name, D = sys.argv[1], float(sys.argv[2])
s = 11 - D
bench = Bench()
model = bench.base
inv = model.transformer.rope.inv_freq.double().numpy()[:16]


def cplx(v):
    return v[..., :16] + 1j * v[..., 32:48]


parts = []
for period in (80, 180):
    resid = torch.tensor(np.load(OUT / f"cap_base_{period}.npz")["resid"][LAYER])
    with torch.no_grad():
        xn = model.transformer.layers[LAYER].attn_norms[0](resid)
        attn = time_attention(model, LAYER)
        w = attn.wQKV.weight
        q = (xn @ w[qkv_rows(attn, 0, HEAD)].T).double().numpy()
        k = (xn @ w[qkv_rows(attn, 1, HEAD)].T).double().numpy()
    parts.append(dict(
        zq_d=cplx(q[:, 256:]).mean((0, 1)), zq_e=cplx(q[:, 64:256]).mean((0, 1)),
        zk_d=cplx(k[:, 256:]).mean((0, 1)), zk_e=cplx(k[:, 64:256]).mean((0, 1)),
        c0=(q[:, 256:, 16:32].mean((0, 1)) @ k[:, 64:256, 16:32].mean((0, 1)) + q[:, 256:, 48:].mean((0, 1)) @ k[:, 64:256, 48:].mean((0, 1))) / 8,
    ))
m = {key: (parts[0][key] + parts[1][key]) / 2 for key in parts[0]}
zq = (m["zq_d"] + m["zq_e"]) / 2
c = zq * np.conj(m["zk_e"])
ds = np.arange(0, 320)
FREQ = {4: 0, 3: 1, 5: 2}  # frequency index -> position in E


def gains(E):
    g = np.exp(1j * inv * s)
    for f, i in FREQ.items():
        g[f] = g[f] + E[i] * 8 * np.exp(-1j * inv[f] * D) / c[f]
    return g


def kern(g, zqq, zkk):
    return m["c0"] + np.real((g * zqq * np.conj(zkk))[None, :] * np.exp(1j * inv[None, :] * ds[:, None])).sum(1) / 8


def gap(g):
    """Worst margin (logits) of d = D over every other d outside D-1..D+1, for the query/key pairs that occur."""
    out = []
    Di = int(D)
    for zqq, zkk, lo in [(m["zq_d"], m["zk_e"], 1), (m["zq_e"], m["zk_e"], 0)]:
        kk = kern(g, zqq, zkk)
        others = np.r_[kk[lo : Di - 1], kk[Di + 2 : 288]]
        out.append(kk[Di] - others.max())
    # decoder query reading decoder keys (d = 0..31)
    kk = kern(g, m["zq_d"], m["zk_d"])
    out.append(kern(g, m["zq_d"], m["zk_e"])[Di] - kk[:Di - 1].max())
    return min(out)


if len(sys.argv) > 3:
    best = tuple(float(v) for v in sys.argv[3:6])
else:
    grid = [0, 4, 8, 12, 16]
    best = max(itertools.product(grid, [-8, -4, 0, 4, 8], [-8, -4, 0, 4, 8]), key=lambda E: gap(gains(E)) - 0.02 * sum(abs(e) for e in E))
g = gains(best)
np.save(OUT / f"gain_{name}.npy", g)
np.set_printoptions(precision=1, suppress=True, linewidth=250)
print(f"D {D} s {s} E {best}: worst margin {gap(g):.1f} logits; |g| on the touched frequencies {[round(abs(g[f]), 2) for f in FREQ]}")
for label, zqq, zkk in [("dec query / enc key", m["zq_d"], m["zk_e"]), ("enc query / enc key", m["zq_e"], m["zk_e"]), ("dec query / dec key", m["zq_d"], m["zk_d"])]:
    new = kern(g, zqq, zkk)[:288]
    print(f"{label}: top " + " ".join(f"{o}:{new[o]:.1f}" for o in np.argsort(-new)[:10]))
    short = new[:32]
    print("   with fewer than 32 patches behind (d = 0..31 only): top " + " ".join(f"{o}:{short[o]:.1f}" for o in np.argsort(-short)[:6]))
