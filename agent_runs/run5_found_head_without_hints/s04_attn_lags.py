"""Step 4 (no model run): decoder-query attention per layer and head, from the captured
inputs. Prints mass on context vs decoder, and the top lags (query position - key position)
for near (j<12) and far (j>=16) decoder tokens."""
import numpy as np
import torch
from harness import Bench
from timestral import load_model
from wsp import attention_probs

m = load_model("mistralai/Timestral-5M", local_files_only=True)
N_ENC = 256
for p in (80, 180):
    st = torch.load(f"results/capture_base_{p}.pt", weights_only=False)
    print(f"\n===== cycle {p}")
    for li in range(8):
        att = m.transformer.layers[li].attns[0]
        P, _ = attention_probs(att, st["attn_in"][li], st["cos"], st["sin"])  # R,H,32,288
        P = P.mean(0).numpy()  # H,32,288
        for h in range(3):
            dec_mass = P[h, :, N_ENC:].sum(-1)
            # lag profile
            lag = {}
            for grp, js in (("near", range(0, 12)), ("far", range(16, 32))):
                acc = np.zeros(N_ENC + 32)
                for j in js:
                    q = N_ENC + j
                    for kpos in range(q + 1):
                        acc[q - kpos] += P[h, j, kpos]
                acc /= len(js)
                top = np.argsort(-acc)[:6]
                lag[grp] = " ".join(f"{d}:{acc[d]:.2f}" for d in top)
            print(f"L{li}h{h} dec-mass near {dec_mass[:12].mean():.2f} far {dec_mass[16:].mean():.2f} | near lags {lag['near']} | far lags {lag['far']}")
