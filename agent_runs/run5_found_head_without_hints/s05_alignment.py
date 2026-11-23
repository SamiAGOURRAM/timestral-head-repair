"""Step 5 (no model run): per decoder token j, how much of each head's attention falls on
context keys whose content is in phase with the query patch (step offset (q-k)*32 mod
cycle within +-W of 0), divided by the share of in-phase keys (1 = chance)."""
import numpy as np
import torch
from timestral import load_model
from wsp import attention_probs

m = load_model("mistralai/Timestral-5M", local_files_only=True)
N_ENC = 256
W = {80: 6, 180: 8}
for p in (80, 180):
    st = torch.load(f"results/capture_base_{p}.pt", weights_only=False)
    q = N_ENC + np.arange(32)
    k = np.arange(N_ENC + 32)
    off = ((q[:, None] - k[None, :]) * 32) % p
    off = np.minimum(off, p - off)
    inphase = (off <= W[p]) & (k[None, :] < N_ENC)
    chance = inphase[:, :N_ENC].mean(1)
    print(f"\n===== cycle {p}; columns j=0..31 (aligned mass / chance)")
    for li in range(4, 8):
        att = m.transformer.layers[li].attns[0]
        P, _ = attention_probs(att, st["attn_in"][li], st["cos"], st["sin"])
        P = P.mean(0).numpy()
        for h in range(3):
            a = (P[h] * inphase).sum(-1) / chance
            ctxm = P[h, :, :N_ENC].sum(-1)
            print(f"L{li}h{h} ctx-mass {ctxm[:16].mean():.2f}/{ctxm[16:].mean():.2f} :", " ".join(f"{v:4.1f}" for v in a))
