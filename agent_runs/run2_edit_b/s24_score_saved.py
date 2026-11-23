"""Counted (4 passes): official score of the model reloaded from final_checkpoint/ (checks the deliverable itself)."""
from harness import DEVICE, ROOT, Bench, show
from timestral import load_model

bench = Bench()
model = load_model(ROOT / "final_checkpoint", device=DEVICE)
print(show(bench.score(model, "final checkpoint reloaded from disk")))
