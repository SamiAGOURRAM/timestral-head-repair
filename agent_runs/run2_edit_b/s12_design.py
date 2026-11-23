"""Free: design per-frequency complex gains g_f for the query rows of a positional head so that
its positional kernel gets an extra peak at distance D (closed form: ridge least squares).

Kernel with mean query/key: K(d) = c0 + sum_f Re(c_f e^{i w_f d}) / 8,  c_f = zq_f conj(zk_f).
Multiplying the query's complex component of frequency f by g_f turns c_f into g_f c_f.
We look for dh_f = (g_f - 1) c_f with  sum_f Re(dh_f e^{i w_f d}) / 8  ~  H * exp(-(d - D)^2 / (2 sigma^2))
over d = 0..287 (ridge penalty on |g_f - 1|^2, i.e. on |dh_f|^2 / |c_f|^2).

Usage: python s12_design.py L.H D H sigma ridge out_name [nfreq]
"""
import sys

import numpy as np
import torch

from harness import OUT, Bench, qkv_rows, time_attention

layer, head = (int(c) for c in sys.argv[1].split("."))
D, H, sigma, ridge = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
name = sys.argv[6]
nfreq = int(sys.argv[7]) if len(sys.argv) > 7 else 16
bench = Bench()
model = bench.base
inv = model.transformer.rope.inv_freq.double().numpy()[:16]


def cplx(v):
    return v[..., :16] + 1j * v[..., 32:48]


parts = {}
for period in (80, 180):
    resid = torch.tensor(np.load(OUT / f"cap_base_{period}.npz")["resid"][layer])
    blk = model.transformer.layers[layer]
    with torch.no_grad():
        xn = blk.attn_norms[0](resid)
        attn = time_attention(model, layer)
        w = attn.wQKV.weight
        q = (xn @ w[qkv_rows(attn, 0, head)].T).double().numpy()
        k = (xn @ w[qkv_rows(attn, 1, head)].T).double().numpy()
    parts[period] = dict(
        zq_d=cplx(q[:, 256:]).mean((0, 1)), zq_e=cplx(q[:, 64:256]).mean((0, 1)),
        zk_d=cplx(k[:, 256:]).mean((0, 1)), zk_e=cplx(k[:, 64:256]).mean((0, 1)),
        c0=(q[:, 256:, 16:32].mean((0, 1)) @ k[:, 64:256, 16:32].mean((0, 1)) + q[:, 256:, 48:].mean((0, 1)) @ k[:, 64:256, 48:].mean((0, 1))) / 8,
    )
a, b = parts[80], parts[180]
print("agreement of the mean components between the two diagnostic cycles (|difference| / |mean|, per frequency):")
for key in ("zq_d", "zq_e", "zk_e"):
    print(f"  {key}: " + " ".join(f"{abs(a[key][f] - b[key][f]) / (abs(a[key][f] + b[key][f]) / 2 + 1e-9):.2f}" for f in range(16)))
m = {key: (a[key] + b[key]) / 2 for key in a}
zq = (m["zq_d"] + m["zq_e"]) / 2  # one query vector for the design
c = zq * np.conj(m["zk_e"])
ds = np.arange(0, 288)
target = H * np.exp(-((ds - D) ** 2) / (2 * sigma**2))
# unknowns: real and imaginary parts of dh_f, f < nfreq.   Re(dh e^{iwd}) = Re(dh) cos(wd) - Im(dh) sin(wd)
A = np.concatenate([np.cos(inv[None, :nfreq] * ds[:, None]), -np.sin(inv[None, :nfreq] * ds[:, None])], axis=1) / 8
pen = np.concatenate([1 / np.abs(c[:nfreq]), 1 / np.abs(c[:nfreq])])
lhs = A.T @ A + ridge * np.diag(pen**2)
sol = np.linalg.solve(lhs, A.T @ target)
dh = np.zeros(16, dtype=complex)
dh[:nfreq] = sol[:nfreq] + 1j * sol[nfreq:]
g = 1 + dh / c
np.save(OUT / f"gain_{name}.npy", g)
np.set_printoptions(precision=1, suppress=True, linewidth=250)
print("|g_f|:", np.round(np.abs(g), 2))
print("arg g_f:", np.round(np.angle(g), 2))
for label, zqq, zkk in [("dec query / enc key", m["zq_d"], m["zk_e"]), ("enc query / enc key", m["zq_e"], m["zk_e"]), ("dec query / dec key", m["zq_d"], m["zk_d"])]:
    old = m["c0"] + np.real((zqq * np.conj(zkk))[None, :] * np.exp(1j * inv[None, :] * ds[:, None])).sum(1) / 8
    new = m["c0"] + np.real((g * zqq * np.conj(zkk))[None, :] * np.exp(1j * inv[None, :] * ds[:, None])).sum(1) / 8
    order = np.argsort(-new)[:10]
    print(f"\n{label}: old top " + " ".join(f"{ds[o]}:{old[o]:.1f}" for o in np.argsort(-old)[:6]))
    print("   new top: " + " ".join(f"{ds[o]}:{new[o]:.1f}" for o in order))
    p = np.exp(new - new.max())
    print(f"   softmax over d = 0..287 (all keys present): mass at D-1, D, D+1 = {p[int(D) - 1] / p.sum():.3f} {p[int(D)] / p.sum():.3f} {p[int(D) + 1] / p.sum():.3f};"
          f"   over d = 0..31 only (short history): top " + " ".join(f"{o}:{v:.2f}" for o, v in sorted(enumerate(np.exp(new[:32] - new[:32].max()) / np.exp(new[:32] - new[:32].max()).sum()), key=lambda t: -t[1])[:4]))
    print("   new K(0..47):", np.round(new[:48], 1))
