"""Step 11 (no model run): grid over dual-peak weights (main lag D weight a, fallback lag 11
weight b) for L5h1; offline attention masses from captured layer-5 inputs (both cycles)."""
import numpy as np
import torch
from timestral import load_model
from timestral.layers import _apply_rotary
from edits import design_lags

caps = {p: torch.load(f"results/capture_base_{p}.pt", weights_only=False) for p in (80, 180)}
ref = torch.cat([caps[p]["attn_in"][5] for p in (80, 180)])
base = load_model("mistralai/Timestral-5M", local_files_only=True)

def probs(m, st):
    att = m.transformer.layers[5].attns[0]
    with torch.no_grad():
        q, k, _ = att._qkv(st["attn_in"][5])
        q, k = _apply_rotary(q, k, st["cos"], st["sin"])
        s = (q[:, :, 1] @ k[:, :, 1].transpose(-1, -2)) / 8.0
    n = s.shape[-1]
    allow = torch.tril(torch.ones(n, n, dtype=torch.bool)); allow[:256, 256:] = False
    return s.masked_fill(~allow, float("-inf")).softmax(-1).mean(0).numpy()

def lagmass(P, rows, d):
    return np.mean([P[r, r - d] for r in rows if r - d >= 0])

import copy
for D in (32, 43):
    for a in (1.0, 1.5, 2.0):
        for b in (0.0, 0.5, 0.75, 1.0):
            res = []
            for p in (80, 180):
                m = copy.deepcopy(base)
                design_lags(m, 5, 1, [(D, a), (11, b)], ref)
                P = probs(m, caps[p])
                res.append((lagmass(P, range(100, 256), D), lagmass(P, range(12, 30), 11), lagmass(P, range(256, 288), D)))
            r = np.mean(res, 0)
            print(f"D={D} a={a} b={b}: late ctx lagD {r[0]:.2f} | early ctx lag11 {r[1]:.2f} | dec lagD {r[2]:.2f}")
