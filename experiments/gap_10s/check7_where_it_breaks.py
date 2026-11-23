"""Check 7: at which depth of Timestral-5M does a far forecast position lose the cycle?

Method (activation patching). Use a 96-step cycle, exactly 3 patches, so future patches
k and k - 18 sit at the same place in the cycle. For one layer L at a time, the internal
state of each far forecast token (patches 18 to 31) entering layer L is replaced by the
state of the near token 18 patches earlier (patches 0 to 13), which the model handles
well. Layers below L have then run "as if near"; layers from L on run at the true far
position. The share of far dips recovered tells where the damage is done:

  - patching at layer 0 changes nothing (all forecast tokens start identical): baseline;
  - patching after the last layer copies the near forecast: the ceiling;
  - the layer where the rescue appears is where far tokens stop carrying what is needed.

Run: .venv\\Scripts\\python.exe -u experiments\\gap_10s\\check7_where_it_breaks.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check3_synthetic_lead_time as synth  # noqa: E402
from timestral import TimestralForecaster  # noqa: E402

# Cycle length in steps and patch shift, from the environment. The shift must bring a future
# patch back to the same place in the cycle: 96 steps = 3 patches (shift 18); 80 steps is not
# a whole number of patches but repeats its position every 5 patches (shift 20).
PERIOD = int(os.environ.get("WSP_PERIOD", "96"))
SHIFT = int(os.environ.get("WSP_SHIFT", "18"))
assert (SHIFT * 32) % PERIOD == 0, "the shift must be a whole number of cycles"
FAR = list(range(SHIFT, 32))
DEVICE = os.environ.get("WSP_DEVICE", "cpu")
OUT = synth.OUT
N_CTX = synth.CONTEXT // 32


def far_near_shares(median: np.ndarray, meta: list[dict]) -> tuple[float, float]:
    shares = synth.recovered_shares(median, meta)
    near = [s for _, lead, s in shares if lead < 384]
    far = [s for _, lead, s in shares if lead >= SHIFT * 32]
    return float(np.mean(near)), float(np.mean(far))


def run(fc, x: np.ndarray, hooks: list) -> np.ndarray:
    try:
        with torch.inference_mode():
            return fc.forecast(x[:, None, : synth.CONTEXT].astype(np.float32), synth.HORIZON).median[:, 0].double().cpu().numpy()
    finally:
        for h in hooks:
            h.remove()


def copy_near_into_far(x: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    far = torch.tensor([N_CTX + k for k in FAR], device=x.device)
    x[:, far] = x[:, far - SHIFT]
    return x


def main() -> None:
    pd.set_option("display.width", 200)
    if DEVICE == "cpu":
        torch.set_num_threads(12)
    x, meta = synth.make_series(PERIOD, synth.N_SERIES, seed=PERIOD)
    sd = np.array([m["sd"] for m in meta])[:, None]
    truth = x[:, synth.CONTEXT :]
    fc = TimestralForecaster.from_pretrained("mistralai/Timestral-5M", device=DEVICE)
    blocks = fc.model.transformer.layers
    n = len(blocks)

    base = run(fc, x, [])
    shares = synth.recovered_shares(base, meta)
    by_patch = pd.Series({k: np.mean([s for _, lead, s in shares if lead // 32 == k]) for k in range(0, 32, 2)
                          if any(lead // 32 == k for _, lead, _ in shares)})
    print(f"cycle of {PERIOD} steps ({PERIOD / 32:g} patches), {synth.N_SERIES} series. Unpatched 5M, dip recovered by future patch:")
    print(by_patch.round(2).to_frame("recovered").T.to_string(), flush=True)

    print(f"\n=== A. State of far tokens (patches {FAR[0]}-{FAR[-1]}) entering layer L replaced by the near token's "
          f"(patches {FAR[0] - SHIFT}-{FAR[-1] - SHIFT})")
    rows = {}
    for layer in range(n + 1):
        if layer < n:
            hook = blocks[layer].register_forward_pre_hook(lambda m, args: (copy_near_into_far(args[0]), *args[1:]))
            label = f"entering layer {layer}"
        else:
            hook = fc.model.transformer.out_norm.register_forward_pre_hook(lambda m, args: (copy_near_into_far(args[0]),))
            label = "after the last layer (ceiling)"
        med = run(fc, x, [hook])
        near, far = far_near_shares(med, meta)
        rows[label] = {"far dips recovered": far, "near dips recovered": near,
                       "error / noise sd, far steps": float((np.abs(med - truth) / sd)[:, SHIFT * 32 :].mean())}
    print(pd.DataFrame(rows).T.round(3).to_string(), flush=True)
    pd.DataFrame(rows).T.to_csv(OUT / f"check7_patching_cycle{PERIOD}.csv")


if __name__ == "__main__":
    main()
