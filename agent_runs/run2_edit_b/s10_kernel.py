"""Free: positional kernel of a time head, from the weights and the MEAN query / key vectors
(averages of saved activations): K(d) = mean_q^T T_d mean_k / sqrt(head_dim), d = query position - key position.
Usage: python s10_kernel.py 5.1 [period]"""
import sys

import numpy as np
import torch

import edits
from harness import OUT, Bench, qkv_rows, time_attention

layer, head = (int(c) for c in sys.argv[1].split("."))
period = int(sys.argv[2]) if len(sys.argv) > 2 else 80
bench = Bench()
model = bench.base
resid = torch.tensor(np.load(OUT / f"cap_base_{period}.npz")["resid"][layer])  # (R, 288, 192) entering the layer
blk = model.transformer.layers[layer]
with torch.no_grad():
    xn = blk.attn_norms[0](resid)
    attn = time_attention(model, layer)
    w = attn.wQKV.weight
    q = xn @ w[qkv_rows(attn, 0, head)].T  # (R, 288, 64)
    k = xn @ w[qkv_rows(attn, 1, head)].T
qd, qe = q[:, 256:].mean((0, 1)).double(), q[:, 128:256].mean((0, 1)).double()
kd, ke = k[:, 256:].mean((0, 1)).double(), k[:, 128:256].mean((0, 1)).double()
print(f"head {layer}.{head}: |mean q dec| {qd.norm():.2f} |mean q enc| {qe.norm():.2f} cos {torch.cosine_similarity(qd, qe, 0):.3f};"
      f" rms spread of q around mean: dec {(q[:, 256:] - qd.float()).norm(dim=-1).mean():.2f} enc {(q[:, 128:256] - qe.float()).norm(dim=-1).mean():.2f}")
print(f"           |mean k dec| {kd.norm():.2f} |mean k enc| {ke.norm():.2f} cos {torch.cosine_similarity(kd, ke, 0):.3f};"
      f" rms spread of k: dec {(k[:, 256:] - kd.float()).norm(dim=-1).mean():.2f} enc {(k[:, 128:256] - ke.float()).norm(dim=-1).mean():.2f}")
ds = list(range(-48, 97))
np.set_printoptions(precision=1, suppress=True, linewidth=250)
for name, qq, kk in [("dec query, enc key", qd, ke), ("dec query, dec key", qd, kd), ("enc query, enc key", qe, ke)]:
    kern = np.array([float(qq @ edits.rotary_matrix(model, d) @ kk) / 8.0 for d in ds])
    # T_d on the key side with +d: rotary(q at m) . rotary(k at n) depends on m - n
    print(f"\n{name}: K(d) for d = -48..96 (d>0: key d patches earlier)")
    for a in range(0, len(ds), 16):
        print(f"  d {ds[a]:4d}..: " + " ".join(f"{v:6.1f}" for v in kern[a : a + 16]))
