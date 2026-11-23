"""Edit B against gradient training of the same 6 144 weights, on the same fresh held-back sets.

The restricted fine-tune (`finetune_same_weights.py`, learning rate 1e-3 kept by its rule) is
scored on the sets that scored edit B, runs 3 and 4 and the full fine-tune (seeds 7000+, dip
cycles 70, 100, 150, 220, 300, 64, 160; spikes on cycles 100, 220; `bizitobs_service`; dip
sets by length of history). Models: unedited 5M, edit B (blind run 2), the restricted fine-tune.

Run: .venv\\Scripts\\python.exe -u experiments\\edit_5m\\heldback_sameweights.py
"""

from pathlib import Path

import heldback_repair as hr

hr.BLIND = "5M edit B (blind run 2)"
hr.REPAIR = "5M fine-tune of the same 6 144 weights"
hr.SOURCES = {hr.UNEDITED: "mistralai/Timestral-5M", hr.BLIND: hr.ARM / "edit_b",
              hr.REPAIR: Path(__file__).resolve().parents[2] / "checkpoints" / "finetune_sameweights_lr0.001"}
hr.OUT_NAME = "sameweights_vs_b"

if __name__ == "__main__":
    hr.main()
