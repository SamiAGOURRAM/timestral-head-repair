"""Build the edited Timestral-5M ("edit B") from the published weights and the 6,144-weight patch.

The patch file holds one tensor, `transformer.layers.5.attns.0.wQKV.weight` (576 x 192), as
edited: the fused query/key/value projection of layer 5's time attention. It differs from the
published tensor in 32 rows only, the rotary query rows of head 1 (6,144 weights).

Writes `checkpoints/edit_b/` (model.safetensors + config.json), which the experiment scripts
load as "edit B".

Run: python edit/apply_edit_b.py
"""

import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from timestral import load_model
from timestral.loading import resolve_checkpoint

ROOT = Path(__file__).resolve().parents[1]
TENSOR = "transformer.layers.5.attns.0.wQKV.weight"


def main() -> None:
    model = load_model("mistralai/Timestral-5M", device="cpu")
    patch = load_file(str(ROOT / "edit" / "edit_b_patch.safetensors"))
    assert set(patch) == {TENSOR}, set(patch)
    w = dict(model.named_parameters())[TENSOR]
    new = patch[TENSOR]
    diff = w.detach() != new
    print(f"{TENSOR}: {int(diff.any(dim=1).sum())} rows, {int(diff.sum())} weights changed")
    with torch.no_grad():
        w.copy_(new)
    out = ROOT / "checkpoints" / "edit_b"
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(out / "model.safetensors"))
    shutil.copy(resolve_checkpoint("mistralai/Timestral-5M") / "config.json", out / "config.json")
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
