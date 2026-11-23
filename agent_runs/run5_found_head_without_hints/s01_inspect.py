"""Step 1: read the config and list the parameter tensors of Timestral-5M (no model run)."""
import json
from timestral.loading import resolve_checkpoint
from timestral import load_model

p = resolve_checkpoint("mistralai/Timestral-5M", local_files_only=True)
print(p)
print(json.dumps(json.loads((p / "config.json").read_text()), indent=1))
m = load_model(p)
for n, t in m.named_parameters():
    print(f"{n:60s} {tuple(t.shape)}")
print(m.config.rope_dim, m.transformer.rope.inv_freq)
