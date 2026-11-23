"""Step 15 (no model run): save the reference statistics used by the final edit (mean
complex query / key component of L5h1 per RoPE frequency, over the captured layer-5
inputs of both cycles) so the edit can be rebuilt without the 236 MB capture files."""
import json
import numpy as np
import torch
from timestral import load_model
from edits import head_mean_angles

ref = torch.cat([torch.load(f"results/capture_base_{p}.pt", weights_only=False)["attn_in"][5] for p in (80, 180)])
m = load_model("mistralai/Timestral-5M", local_files_only=True)
zq, zk = head_mean_angles(m.transformer.layers[5].attns[0], ref, 1)
json.dump({"zq_re": zq.real.tolist(), "zq_im": zq.imag.tolist(), "zk_re": zk.real.tolist(), "zk_im": zk.imag.tolist()},
          open("results/l5h1_ref_means.json", "w"), indent=1)
for f in range(16):
    print(f"f{f:2d} |zq| {abs(zq[f]):6.2f} ang {np.angle(zq[f]):+.3f} | |zk| {abs(zk[f]):6.2f} ang {np.angle(zk[f]):+.3f}")
