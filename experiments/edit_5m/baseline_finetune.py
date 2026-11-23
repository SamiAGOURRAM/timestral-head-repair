"""Baseline for the editing experiment: can ordinary training fix one-pass forecasting in Timestral-5M?

This is gradient training, so it is NOT weight-space programming and is outside the
agent's pass budget (PROTOCOL.md, "Baselines"). It answers one question: is there any
setting of the weights that makes 5M place a repeating dip far ahead in one pass?

Two versions:
  fixed   trains on the 128 diagnostic series only (cycles 80 and 180, the same series the
          agent saw);
  family  trains on fresh series of the same family at every step (cycles 80 and 180).
  anycycle  trains on fresh series with cycle lengths drawn from 40 to 400 steps, never
          the ones used for scoring or held back: the test of whether training can fix
          the behaviour in general, not for two cycle lengths.

Loss: quantile (pinball) loss over the 21 quantiles and the 1024-step horizon, forecast in
one pass through the forecaster's own decoding code (so the per-patch rescaling it applies
at inference is applied in training too). No mirrored series, no clipping.

Scored before and after, with the forecaster's default settings, on sets that are neither
the agent's diagnostic set nor the held-back data of the protocol:
  - the diagnostic series themselves (for comparison with the agent's edits);
  - new series, cycles 80 and 180;
  - cycles not trained on: 70 and 200 (not whole numbers of patches), 64 (2 patches).
The held-back data is not touched here.

Run: .venv\\Scripts\\python.exe -u experiments\\edit_5m\\baseline_finetune.py fixed|family [steps]
"""

import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "gap_10s"))
import check3_synthetic_lead_time as synth  # noqa: E402
from timestral import TimestralForecaster, load_model  # noqa: E402
from timestral.loading import resolve_checkpoint  # noqa: E402

OUT = ROOT / "results" / "edit_5m"
DEVICE = os.environ.get("WSP_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
TRAIN_CYCLES = (80, 180)
BATCH = int(os.environ.get("WSP_BATCH", "32"))
LR = 1e-4
# "anycycle" mode: cycle lengths drawn at each step from 40 to 400 steps, leaving out every
# cycle used for scoring here (64, 70, 80, 180, 200) and every cycle of the protocol's
# held-back data (56, 96, 112, 128, 144, 250, 360).
EXCLUDED = {64, 70, 80, 180, 200, 56, 96, 112, 128, 144, 250, 360}
ANY_CYCLES = np.array([c for c in range(40, 401) if c not in EXCLUDED])
EVAL_SETS = {  # name: (cycle, seed)
    "diagnostic 80": (80, 80), "diagnostic 180": (180, 180),
    "new series 80": (80, 1080), "new series 180": (180, 1180),
    "unseen cycle 70": (70, 1070), "unseen cycle 200": (200, 1200), "unseen cycle 64 (2 patches)": (64, 1064),
}


def pinball(q: torch.Tensor, y: torch.Tensor, levels: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    err = y[..., None] - q
    return (torch.maximum(levels * err, (levels - 1) * err).mean(dim=(1, 2)) / scale).mean()


def train_loss(fc: TimestralForecaster, x: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    ctx, y = x[:, : synth.CONTEXT], x[:, synth.CONTEXT :]
    mask = torch.ones_like(ctx, dtype=torch.bool)
    q = fc._block_ar_decode(ctx[:, None], mask[:, None], synth.HORIZON, None, None, None)[:, 0]  # (B, H, Q), one pass
    scale = (ctx - ctx.mean(dim=1, keepdim=True)).abs().mean(dim=1)
    return pinball(q.float(), y, levels, scale)


@torch.inference_mode()
def evaluate(model) -> pd.DataFrame:
    fc = TimestralForecaster(model)
    rows = {}
    for name, (cycle, seed) in EVAL_SETS.items():
        x, meta = synth.make_series(cycle, synth.N_SERIES, seed=seed)
        med = fc.forecast(x[:, None, : synth.CONTEXT].astype(np.float32), synth.HORIZON).median[:, 0].double().cpu().numpy()
        shares = synth.recovered_shares(med, meta)
        sd = np.array([m["sd"] for m in meta])[:, None]
        rows[name] = {"near dips": float(np.mean([s for _, lead, s in shares if lead < 384])),
                      "far dips": float(np.mean([s for _, lead, s in shares if lead >= 512])),
                      "error / noise sd": float((np.abs(med - x[:, synth.CONTEXT :]) / sd).mean())}
    return pd.DataFrame(rows).T


def main() -> None:
    mode = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    assert mode in ("fixed", "family", "anycycle")
    pd.set_option("display.width", 200)
    torch.manual_seed(0)
    if DEVICE == "cpu":
        torch.set_num_threads(12)
    base = load_model("mistralai/Timestral-5M", device=DEVICE)
    model = copy.deepcopy(base)
    fc = TimestralForecaster(model)
    levels = torch.tensor(model.config.quantile_knots, device=DEVICE)
    print(f"mode {mode}, {steps} steps, batch {BATCH}, lr {LR}, device {DEVICE}", flush=True)

    before = evaluate(base)
    print("\nbefore training\n" + before.round(3).to_string(), flush=True)

    fixed = np.concatenate([synth.make_series(c, synth.N_SERIES, seed=c)[0] for c in TRAIN_CYCLES])  # the 128 diagnostic series
    rng = np.random.default_rng(0)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    for p in model.parameters():
        p.requires_grad_(True)
    t0 = time.perf_counter()
    history = []
    for step in range(steps):
        if mode == "fixed":
            x = fixed[rng.choice(len(fixed), BATCH, replace=False)]
        elif mode == "anycycle":
            cycles = rng.choice(ANY_CYCLES, 8, replace=False)
            x = np.concatenate([synth.make_series(int(c), BATCH // 8, seed=200_000 + 10 * step + i)[0] for i, c in enumerate(cycles)])
        else:
            half = BATCH // 2
            x = np.concatenate([synth.make_series(c, half, seed=100_000 + 10 * step + i)[0] for i, c in enumerate(TRAIN_CYCLES)])
        loss = train_loss(fc, torch.as_tensor(x, dtype=torch.float32, device=DEVICE), levels)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        history.append(float(loss))
        if step % 25 == 0 or step == steps - 1:
            print(f"  step {step:4d}  loss {float(loss):.5f}  ({time.perf_counter() - t0:.0f} s)", flush=True)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    after = evaluate(model)
    print("\nafter training\n" + after.round(3).to_string(), flush=True)
    print("\nchange (after - before)\n" + (after - before).round(3).to_string(), flush=True)

    ckpt = ROOT / "checkpoints" / f"finetune_{mode}_{steps}"
    ckpt.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(ckpt / "model.safetensors"))
    src = resolve_checkpoint("mistralai/Timestral-5M")
    (ckpt / "config.json").write_text((src / "config.json").read_text())
    moved = {n: float((p.detach() - q.detach()).norm() / q.detach().norm()) for (n, p), (_, q) in
             zip(model.named_parameters(), base.named_parameters())}
    top = sorted(moved.items(), key=lambda kv: -kv[1])[:10]
    print("\nlargest relative weight changes (tensor: |change| / |original|)")
    for n, v in top:
        print(f"  {v:.4f}  {n}")
    after.to_csv(OUT / f"baseline_finetune_{mode}_{steps}.csv")
    (OUT / f"baseline_finetune_{mode}_{steps}_loss.json").write_text(json.dumps(history))
    print(f"\ncheckpoint saved to {ckpt.relative_to(ROOT)} (not tracked by git)")


if __name__ == "__main__":
    main()
