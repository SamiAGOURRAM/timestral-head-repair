"""Control for edit B: gradient training restricted to the 6 144 weights edit B changed.

Question: does edit B's advantage over the full fine-tune come from the agent's reasoning,
or only from touching so few weights? Same data, loss and settings as the full fine-tune
(`baseline_finetune.py`, mode anycycle: fresh series, cycle lengths 40-400 steps excluding
every scored or held-back cycle, one-pass 1024-step forecast, pinball loss, batch 32, 600
steps, AdamW, gradient clipping 1.0), but only the rows of
`transformer.layers.5.attns.0.wQKV.weight` that differ between edit B and the published
weights are trained (read from edit B's checkpoint: 32 rows x 192 = 6 144 weights). Every
other weight is frozen; the script checks at the end that nothing else moved.

With so few weights, the full fine-tune's learning rate (1e-4) may barely move them, so
three learning rates are trained: 1e-4, 1e-3, 1e-2.

Selection rule, fixed before running: on the script's own scoring sets (diagnostic cycles
80 and 180, new series 80 and 180, unseen cycles 70, 200, 64; none of them held back),
keep the learning rate with the highest mean far-dip share among those whose mean
near-dip share drops by at most 0.05 against the published model. The kept checkpoint is
then scored, once, on the held-back sets and on GIFT-Eval, like edit B and the full fine-tune.

Run: .venv\\Scripts\\python.exe -u experiments\\edit_5m\\finetune_same_weights.py
"""

import copy
import json
import time

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

import baseline_finetune as bf
from timestral import TimestralForecaster, load_model
from timestral.loading import resolve_checkpoint

TENSOR = "transformer.layers.5.attns.0.wQKV.weight"
EDIT_B = bf.ROOT / "checkpoints" / "edit_b"
STEPS = 600
RATES = (1e-4, 1e-3, 1e-2)


def target_rows(base) -> torch.Tensor:
    b = dict(load_model(EDIT_B, device="cpu").named_parameters())[TENSOR]
    a = dict(base.named_parameters())[TENSOR].detach().cpu()
    rows = (a != b).any(dim=1)
    assert int((a != b).sum()) == 6144 and int(rows.sum()) == 32, "edit B is not the expected 32-row edit"
    return rows


def train(base, rows: torch.Tensor, lr: float):
    model = copy.deepcopy(base)
    fc = TimestralForecaster(model)
    levels = torch.tensor(model.config.quantile_knots, device=bf.DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    w = dict(model.named_parameters())[TENSOR]
    w.requires_grad_(True)
    frozen = (~rows).to(bf.DEVICE)[:, None]
    original = w.detach().clone()
    opt = torch.optim.AdamW([w], lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    history, t0 = [], time.perf_counter()
    for step in range(STEPS):
        cycles = rng.choice(bf.ANY_CYCLES, 8, replace=False)
        x = np.concatenate([bf.synth.make_series(int(c), bf.BATCH // 8, seed=200_000 + 10 * step + i)[0] for i, c in enumerate(cycles)])
        loss = bf.train_loss(fc, torch.as_tensor(x, dtype=torch.float32, device=bf.DEVICE), levels)
        opt.zero_grad()
        loss.backward()
        w.grad.masked_fill_(frozen, 0.0)
        torch.nn.utils.clip_grad_norm_([w], 1.0)
        opt.step()
        with torch.no_grad():
            w.copy_(torch.where(frozen, original, w))
        history.append(float(loss))
        if step % 100 == 0 or step == STEPS - 1:
            print(f"  lr {lr:g} step {step:4d}  loss {float(loss):.5f}  ({time.perf_counter() - t0:.0f} s)", flush=True)
    w.requires_grad_(False)
    model.eval()
    changed = {n: int((p != q).sum()) for (n, p), (_, q) in zip(model.named_parameters(), base.named_parameters()) if not torch.equal(p, q)}
    assert set(changed) <= {TENSOR} and changed.get(TENSOR, 0) <= 6144, f"weights outside the 6 144 moved: {changed}"
    rel = float((w - original).norm() / original[~frozen[:, 0]].norm())
    print(f"  lr {lr:g}: {changed.get(TENSOR, 0)} weights changed; relative size of the change on the trained rows {rel:.3f}", flush=True)
    return model, history


def main() -> None:
    pd.set_option("display.width", 200)
    base = load_model("mistralai/Timestral-5M", device=bf.DEVICE)
    rows = target_rows(base)
    print(f"trained rows of {TENSOR}: {rows.nonzero().flatten().tolist()}", flush=True)
    before = bf.evaluate(base)
    print("\npublished model\n" + before.round(3).to_string(), flush=True)
    summary = {}
    for lr in RATES:
        model, history = train(base, rows, lr)
        after = bf.evaluate(model)
        print(f"\nlr {lr:g}\n" + after.round(3).to_string() + "\nchange\n" + (after - before).round(3).to_string(), flush=True)
        summary[lr] = {"far": float(after["far dips"].mean()), "near change": float((after["near dips"] - before["near dips"]).mean()),
                       "error": float(after["error / noise sd"].mean())}
        ckpt = bf.ROOT / "checkpoints" / f"finetune_sameweights_lr{lr:g}"
        ckpt.mkdir(parents=True, exist_ok=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(ckpt / "model.safetensors"))
        (ckpt / "config.json").write_text((resolve_checkpoint("mistralai/Timestral-5M") / "config.json").read_text())
        (bf.OUT / f"finetune_sameweights_lr{lr:g}_loss.json").write_text(json.dumps(history))
        del model
        torch.cuda.empty_cache()
    s = pd.DataFrame(summary).T
    s.index.name = "lr"
    print("\nsummary (means over the scoring sets)\n" + s.round(3).to_string())
    s.to_csv(bf.OUT / "finetune_sameweights_summary.csv")
    ok = s[s["near change"] >= -0.05]
    kept = ok["far"].idxmax() if len(ok) else None
    print(f"\nkept by the rule: {f'lr {kept:g} -> checkpoints/finetune_sameweights_lr{kept:g}' if kept is not None else 'none (every rate loses near dips)'}")


if __name__ == "__main__":
    main()
