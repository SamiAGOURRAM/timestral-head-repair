"""Step 8 (no model run): apply retarget_lag to L5h1 for several D and recompute its
attention from the captured (unchanged) layer-5 inputs: where do context and decoder
queries now look?"""
import numpy as np
import torch
from timestral import load_model
from wsp import attention_probs
from edits import retarget_lag
from timestral.layers import _apply_rotary

for p in (80, 180):
    st = torch.load(f"results/capture_base_{p}.pt", weights_only=False)
    for D in (11, 43, 40, 48):
        m = load_model("mistralai/Timestral-5M", local_files_only=True)
        retarget_lag(m, 5, 1, D, st["attn_in"][5])
        att = m.transformer.layers[5].attns[0]
        with torch.no_grad():
            q, k, v = att._qkv(st["attn_in"][5])
            q, k = _apply_rotary(q, k, st["cos"], st["sin"])
            s = (q[:, :, 1] @ k[:, :, 1].transpose(-1, -2)) / 8.0
            n = s.shape[-1]
            allow = torch.tril(torch.ones(n, n, dtype=torch.bool))
            allow[:256, 256:] = False
            allow[256:, 256:] = torch.tril(torch.ones(32, 32, dtype=torch.bool))
            Pm = s.masked_fill(~allow, float("-inf")).softmax(-1).mean(0).numpy()
        def onlag(rows):
            hit = np.mean([Pm[r, r - D] for r in rows if r - D >= 0])
            return hit
        print(f"cycle {p} D={D}: mass on lag D: ctx q 100-255 {onlag(range(100,256)):.2f}, dec q {onlag(range(256,288)):.2f}; dec mass on decoders {Pm[256:,256:].sum(-1).mean():.2f}")
