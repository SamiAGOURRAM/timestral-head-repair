"""Step 7 (no model run): per RoPE frequency pair, the query/key 2-D components of L5h1
(before rotation): mean norm, coherence |mean|/mean|.| over tokens, and the positional
score each frequency contributes as a function of lag."""
import numpy as np
import torch
from timestral import load_model

m = load_model("mistralai/Timestral-5M", local_files_only=True)
st = torch.load("results/capture_base_180.pt", weights_only=False)
om = m.transformer.rope.inv_freq[:16].numpy()
for li, h in [(5, 1), (3, 1), (2, 2)]:
    att = m.transformer.layers[li].attns[0]
    with torch.no_grad():
        q, k, _ = att._qkv(st["attn_in"][li])  # R,N,H,64
    q = q[:, :, h].numpy(); k = k[:, :, h].numpy()
    print(f"\n=== L{li}h{h}")
    qd = q[:, 256:].reshape(-1, 64); qc = q[:, 100:256].reshape(-1, 64); kc = k[:, 100:256].reshape(-1, 64)
    def z(a, f):
        return a[:, f] + 1j * a[:, f + 32]
    for f in range(16):
        out = []
        for nm, a in (("qdec", qd), ("qctx", qc), ("kctx", kc)):
            c = z(a, f)
            out.append(f"{nm} |{np.abs(c).mean():5.2f}| coh {np.abs(c.mean()) / np.abs(c).mean():.2f} ang {np.angle(c.mean()):+.2f}")
        print(f"f{f:2d} w={om[f]:.4f} " + " ; ".join(out))
    un = list(range(16, 32)) + list(range(48, 64))
    print("unrotated norms qdec %.2f qctx %.2f kctx %.2f" % (np.linalg.norm(qd[:, un], axis=1).mean(), np.linalg.norm(qc[:, un], axis=1).mean(), np.linalg.norm(kc[:, un], axis=1).mean()))
    print("unrotated mean-vector score qdec.kctx %.2f" % (qd[:, un].mean(0) @ kc[:, un].mean(0)))
