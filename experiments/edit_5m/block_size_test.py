"""Test B: is lag 32 safe because the model forecasts 32 future tokens in one block?

Build head-5.1 edits for lags 16 and 24 (same construction as `lag_sweep.py`; lag 32 is
`checkpoints/lagsweep_32`), then `tools/block_size_test.ps1` scores lags 16, 24 and 32, and
the published model, with the forecaster set to blocks of 16 and of 24 future tokens
(`--chunk`), on the seven configurations of the lag sweep. Each edit is compared with the
published model at the same block size.

Expectation (written before running): if the safe lag is tied to the block length, then with
blocks of 16 lag 16 is safe and lag 32 harmful, and with blocks of 24 lag 24 is safe; if lag 32
stays the only safe lag at every block size, the block-length explanation is rejected.

Run: .venv\\Scripts\\python.exe -u experiments\\edit_5m\\block_size_test.py
"""

import shutil

import torch
from safetensors.torch import save_file

import baseline_finetune as bf
from lag_sweep import retarget
from timestral import load_model
from timestral.loading import resolve_checkpoint


def main() -> None:
    base = load_model("mistralai/Timestral-5M", device="cpu")
    for lag in (16, 24):
        m = load_model("mistralai/Timestral-5M", device="cpu")
        retarget(m, lag)
        changed = {n: int((p != q).sum()) for (n, p), (_, q) in zip(m.named_parameters(), base.named_parameters()) if not torch.equal(p, q)}
        assert set(changed) <= {"transformer.layers.5.attns.0.wQKV.weight"} and sum(changed.values()) <= 6144, changed
        ckpt = bf.ROOT / "checkpoints" / f"lagsweep_{lag}"
        ckpt.mkdir(parents=True, exist_ok=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in m.state_dict().items()}, str(ckpt / "model.safetensors"))
        shutil.copy(resolve_checkpoint("mistralai/Timestral-5M") / "config.json", ckpt / "config.json")
        print(f"built lag {lag}: {sum(changed.values())} weights changed", flush=True)


if __name__ == "__main__":
    main()
