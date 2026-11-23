"""Free: per rotary frequency, the complex mean query / key components of a head and their
spread, and the true positional kernel K(d) = c0 + sum_f Re(c_f e^{i w_f d}), d = query pos - key pos.
Usage: python s11_freq_parts.py 5.1 [period]"""
import sys

import numpy as np
import torch

from harness import OUT, Bench, qkv_rows, time_attention

layer, head = (int(c) for c in sys.argv[1].split("."))
period = int(sys.argv[2]) if len(sys.argv) > 2 else 80
bench = Bench()
model = bench.base
resid = torch.tensor(np.load(OUT / f"cap_base_{period}.npz")["resid"][layer])
blk = model.transformer.layers[layer]
with torch.no_grad():
    xn = blk.attn_norms[0](resid)
    attn = time_attention(model, layer)
    w = attn.wQKV.weight
    q = (xn @ w[qkv_rows(attn, 0, head)].T).double().numpy()
    k = (xn @ w[qkv_rows(attn, 1, head)].T).double().numpy()
inv = model.transformer.rope.inv_freq.double().numpy()[:16]


def cplx(v):
    return v[..., :16] + 1j * v[..., 32:48]


zq_d, zq_e = cplx(q[:, 256:]), cplx(q[:, 128:256])
zk_d, zk_e = cplx(k[:, 256:]), cplx(k[:, 128:256])
print(f"head {layer}.{head} cycle {period}")
print("freq(rad/patch) |mean zq dec| sd  |mean zq enc| sd  angle(dec)-angle(enc) | |mean zk dec| sd |mean zk enc| sd  angle diff")
for f in range(16):
    a, b, c, d = zq_d[..., f], zq_e[..., f], zk_d[..., f], zk_e[..., f]
    print(f"  {inv[f]:8.5f}   {abs(a.mean()):6.2f} {abs(a - a.mean()).mean():5.2f}   {abs(b.mean()):6.2f} {abs(b - b.mean()).mean():5.2f}   {np.angle(a.mean() / b.mean()):6.2f}"
          f"  |  {abs(c.mean()):6.2f} {abs(c - c.mean()).mean():5.2f}   {abs(d.mean()):6.2f} {abs(d - d.mean()).mean():5.2f}   {np.angle(c.mean() / d.mean()):6.2f}")
# unrotated part
c0_d = (q[:, 256:, 16:32].mean((0, 1)) @ k[:, 128:256, 16:32].mean((0, 1)) + q[:, 256:, 48:].mean((0, 1)) @ k[:, 128:256, 48:].mean((0, 1))) / 8
print(f"unrotated part of the score (dec query, enc key): {c0_d:.2f}")
ds = np.arange(0, 289)
cf = zq_d.mean((0, 1)) * np.conj(zk_e.mean((0, 1)))
kern = c0_d + (np.real(cf[None, :] * np.exp(1j * inv[None, :] * ds[:, None]))).sum(1) / 8
order = np.argsort(-kern)[:12]
print("true kernel, dec query / enc key: top d:", " ".join(f"{ds[o]}:{kern[o]:.1f}" for o in order))
print("K(0..47):", np.round(kern[:48], 1))
