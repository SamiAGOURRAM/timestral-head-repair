"""Step 14: build the final edit, save final_checkpoint/ (model.safetensors + config.json)
and the patch, reload the checkpoint from disk and score it (4 passes)."""
import shutil
from pathlib import Path
import torch
from safetensors.torch import save_file
from harness import Bench, show
from timestral import load_model
from timestral.loading import resolve_checkpoint
from edits import design_lags

ref = torch.cat([torch.load(f"results/capture_base_{p}.pt", weights_only=False)["attn_in"][5] for p in (80, 180)])
b = Bench()
m = b.fresh()
mult = design_lags(m, 5, 1, [(43, 1.5), (11, 0.5)], ref)
print("per-frequency complex multipliers (|m|, angle):")
for f, z in enumerate(mult):
    print(f"  f{f:2d} |m|={abs(z):.4f} angle={__import__('numpy').angle(z):+.4f}")
out = Path("final_checkpoint"); out.mkdir(exist_ok=True)
save_file({k: v.detach().contiguous().cpu() for k, v in m.state_dict().items()}, str(out / "model.safetensors"))
src = resolve_checkpoint("mistralai/Timestral-5M", local_files_only=True)
shutil.copy(src / "config.json", out / "config.json")
print(b.save_patch(m, "final"))
shutil.copy("results/patch_final.safetensors", out / "patch_final.safetensors")
m2 = load_model(str(out))
same = all(torch.equal(a, c) for a, c in zip(m.state_dict().values(), m2.state_dict().values()))
print("reloaded == edited:", same)
print(show(b.score(m2, "final checkpoint reloaded (L5h1 dual 43:1.5, 11:0.5)")))
