"""Step 6 (no model run): unmasked score profile vs relative distance (query pos - key pos)
for the fixed-lag heads, context queries and decoder queries, from captured layer inputs.
Shows whether a head's preferred lag holds for context tokens too, and what it would score
at negative distances (keys after the query) if its query were shifted."""
import numpy as np
import torch
from timestral import load_model
from timestral.layers import _apply_rotary

m = load_model("mistralai/Timestral-5M", local_files_only=True)
st = torch.load("results/capture_base_180.pt", weights_only=False)
N = 288
for li, h in [(3, 1), (4, 0), (4, 1), (5, 0), (5, 1), (2, 2)]:
    att = m.transformer.layers[li].attns[0]
    with torch.no_grad():
        q, k, v = att._qkv(st["attn_in"][li])
        q, k = _apply_rotary(q, k, st["cos"], st["sin"])
        s = (q[:, :, h] @ k[:, :, h].transpose(-1, -2)).mean(0).numpy() / 8.0  # N x N
    for name, qs in (("ctx q 150-250", range(150, 250)), ("dec q 0-31", range(256, 288))):
        prof = {}
        for qi in qs:
            for d in range(-40, 140):
                kk = qi - d
                if 0 <= kk < N:
                    prof.setdefault(d, []).append(s[qi, kk])
        prof = {d: np.mean(vv) for d, vv in prof.items()}
        top = sorted(prof, key=lambda d: -prof[d])[:8]
        neg = max(prof[d] for d in prof if d < 0)
        med = np.median(list(prof.values()))
        print(f"L{li}h{h} {name}: top lags " + " ".join(f"{d}:{prof[d]:.1f}" for d in top) + f" | max over neg lags {neg:.1f} | median {med:.1f}")
