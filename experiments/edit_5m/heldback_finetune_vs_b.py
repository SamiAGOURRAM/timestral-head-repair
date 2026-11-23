"""Edit B against our best fine-tune, on the same fresh held-back sets (those of `heldback_run3.py`).

The fine-tune (`checkpoints/finetune_anycycle_600`: all weights, 600 gradient steps on synthetic
series with many cycle lengths) was scored so far on other held-back sets than edit B. This
puts both on the same series: seeds 7000+, dip cycles 70, 100, 150, 220, 300, 64, 160; spikes
on cycles 100, 220; `bizitobs_service`; and the dip sets by length of history.
Models: unedited 5M, edit B (blind run 2), the fine-tune.

Run: .venv\\Scripts\\python.exe -u experiments\\edit_5m\\heldback_finetune_vs_b.py
"""

from pathlib import Path

import heldback_repair as hr

hr.BLIND = "5M edit B (blind run 2)"
hr.REPAIR = "5M fine-tune, many cycle lengths"
hr.SOURCES = {hr.UNEDITED: "mistralai/Timestral-5M", hr.BLIND: hr.ARM / "edit_b",
              hr.REPAIR: Path(__file__).resolve().parents[2] / "checkpoints" / "finetune_anycycle_600"}
hr.OUT_NAME = "finetune_vs_b"

if __name__ == "__main__":
    hr.main()
