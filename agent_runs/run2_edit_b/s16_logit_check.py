"""Free check of the kernel algebra: from the saved layer-5 input residuals (unchanged by an
edit of layer 5's own query rows), compute head 5.1's attention logits for the DECODER queries
(q.k dot products with the rotary rotation, nothing else of the model is run) with a given gain
file, and compare the top keys with a recorded pass.
Usage: python s16_logit_check.py 180 gainname|none [recorded tag]"""
import sys

import numpy as np
import torch

import edits
from harness import OUT, Bench, qkv_rows, time_attention
from timestral import layers as L

period, gname = int(sys.argv[1]), sys.argv[2]
tag = sys.argv[3] if len(sys.argv) > 3 else None
LAYER, HEAD = 5, 1
bench = Bench()
model = bench.fresh()
if gname != "none":
    edits.apply_gain_file(model, gname)
resid = torch.tensor(np.load(OUT / f"cap_base_{period}.npz")["resid"][LAYER])
with torch.no_grad():
    xn = model.transformer.layers[LAYER].attn_norms[0](resid)
    attn = time_attention(model, LAYER)
    w = attn.wQKV.weight
    q = (xn @ w[qkv_rows(attn, 0, HEAD)].T)[:, :, None, :]  # (R, 288, 1, 64)
    k = (xn @ w[qkv_rows(attn, 1, HEAD)].T)[:, :, None, :]
    pos = torch.arange(288)[None].expand(len(q), -1)
    cos, sin = model.transformer.rope(q, pos)
    q, k = L._apply_rotary(q, k, cos, sin)
    logits = q[:, 256:, 0] @ k[:, :, 0].transpose(-1, -2) / 8  # (R, 32, 288)
    causal = torch.arange(288)[None, :] <= (torch.arange(32)[:, None] + 256)
    logits = logits.masked_fill(~causal[None], float("-inf"))
    wgt = logits.softmax(-1).numpy()
m = wgt.mean(0)
rec = np.load(OUT / f"cap_{tag}_{period}.npz")["attn"][LAYER, :, HEAD].astype(np.float32).mean(0) if tag else None
lg = logits.numpy()
for p in range(0, 32, 3):
    top = np.argsort(-m[p])[:4]
    line = f"p={p:2d}: computed " + " ".join(f"{t - 256:+4d}:{m[p, t]:.2f}" for t in top)
    if rec is not None:
        top = np.argsort(-rec[p])[:4]
        line += "   | recorded " + " ".join(f"{t - 256:+4d}:{rec[p, t]:.2f}" for t in top)
    d32, own = lg[:, p, 256 + p - 32], lg[:, p, 256 + p]
    line += f"   | logit at 32 back {d32.mean():6.1f} sd {d32.std():4.1f}; own key {own.mean():6.1f} sd {own.std():4.1f}"
    print(line)
