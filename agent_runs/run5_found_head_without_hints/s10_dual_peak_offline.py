"""Step 10 (no model run): L5h1 attention, from captured layer-5 inputs, for designs
single D and dual (D main, 11 fallback). Context tokens at position p only see keys <= p
(causal), so queries at p < D show what a short context would do."""
import numpy as np
import torch
from timestral import load_model
from timestral.layers import _apply_rotary
from edits import design_lags

st = torch.load("results/capture_base_180.pt", weights_only=False)
ref = torch.cat([torch.load(f"results/capture_base_{p}.pt", weights_only=False)["attn_in"][5] for p in (80, 180)])

def probs(m):
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

designs = {"original": None}
for D in (32, 43):
    designs[f"single {D}"] = [(D, 1.0)]
    for beta in (0.5, 0.7):
        designs[f"dual {D} + 11*{beta}"] = [(D, 1.0), (11, beta)]
for name, peaks in designs.items():
    m = load_model("mistralai/Timestral-5M", local_files_only=True)
    if peaks:
        design_lags(m, 5, 1, peaks, ref)
    P = probs(m)
    D = peaks[0][0] if peaks else 43
    late = range(100, 256); early = range(12, 30); dec = range(256, 288)
    print(f"{name:20s} late ctx: lagD {lagmass(P, late, D):.2f} lag11 {lagmass(P, late, 11):.2f} | early ctx(p 12-29): lag11 {lagmass(P, early, 11):.2f} | dec: lagD {lagmass(P, dec, D):.2f} lag11 {lagmass(P, dec, 11):.2f}")
