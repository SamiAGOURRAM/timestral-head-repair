"""Free (no model run): build the final edited model (design h on head 5.1's query rows) and save
final_checkpoint/model.safetensors, final_checkpoint/config.json and results/patch_final.safetensors.
config.json is written from the loaded model's config (TimestralConfig.save), not copied from the
Hugging Face cache, which is outside the folders I may read."""
import numpy as np
import torch
from safetensors.torch import load_file, save_file

import edits
from harness import OUT, ROOT, Bench, passes_used, qkv_rows, time_attention

bench = Bench()
model = bench.fresh()
g = np.load(OUT / "gain_h.npy")
edits.gain_head(model, 5, 1, g)
out = ROOT / "final_checkpoint"
out.mkdir(exist_ok=True)
save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(out / "model.safetensors"))
model.config.save(out / "config.json")
bench.save_patch(model, "final")

# checks
back = load_file(str(out / "model.safetensors"))
assert set(back) == set(bench.base.state_dict())
changed = [k for k, v in bench.base.state_dict().items() if not torch.equal(v, back[k])]
print("tensors that differ from the published model:", changed)
attn = time_attention(bench.base, 5)
rows = qkv_rows(attn, 0, 1)
diff = back["transformer.layers.5.attns.0.wQKV.weight"] - attn.wQKV.weight.detach()
mask = torch.zeros(diff.shape[0], dtype=torch.bool)
mask[rows] = True
print("rows changed:", int((diff.abs().sum(1) > 0).sum()), "of which query rows of head 5.1:", int((diff[mask].abs().sum(1) > 0).sum()),
      "; max |change| outside those rows:", float(diff[~mask].abs().max()))
np.set_printoptions(precision=4, suppress=True, linewidth=200)
print("gains g_f, f = 0..15 (real):", g.real)
print("gains g_f, f = 0..15 (imag):", g.imag)
print("passes used", passes_used())
