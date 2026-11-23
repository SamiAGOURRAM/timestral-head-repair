"""Harness for the editing task on Timestral-5M: pass counter, diagnostic score, weight patches.

See TASK.md. Every model run must go through `Bench.score` or `Bench.forecast`, which count
passes and log them to `results/passes.jsonl`. Weight edits are plain functions applied to a
fresh copy of the model (`Bench.fresh()`), and are saved with `Bench.save_patch`.
"""

import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import check3_synthetic_lead_time as synth  # noqa: E402
from timestral import TimestralForecaster, load_model  # noqa: E402

OUT = ROOT / "results"
PASS_LOG = OUT / "passes.jsonl"
BUDGET = 200
DIAGNOSTIC_CYCLES = (80, 180)
DEVICE = os.environ.get("WSP_DEVICE", "cpu")


def passes_used() -> int:
    if not PASS_LOG.exists():
        return 0
    return sum(json.loads(line)["passes"] for line in PASS_LOG.read_text().splitlines() if line.strip())


class Bench:
    """Diagnostic set and pass accounting."""

    def __init__(self) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        if DEVICE == "cpu":
            torch.set_num_threads(5)
        self.base = load_model("mistralai/Timestral-5M", device=DEVICE)
        self.sets = {p: synth.make_series(p, synth.N_SERIES, seed=p) for p in DIAGNOSTIC_CYCLES}

    def fresh(self):
        return copy.deepcopy(self.base)

    def forecast(self, model, x: np.ndarray, label: str, chunk: int = 32):
        """One counted `forecast` call on at most 64 series (2 passes: the series and its mirror)."""
        assert len(x) <= 64
        n = 2 * -(-synth.HORIZON // (chunk * 32))
        used = passes_used()
        if used + n > BUDGET:
            raise RuntimeError(f"pass budget exhausted: {used} used, {n} more asked, budget {BUDGET}")
        fc = TimestralForecaster(model, per_shot_horizon_patches=chunk)
        with torch.inference_mode():
            f = fc.forecast(x[:, None, : synth.CONTEXT].astype(np.float32), synth.HORIZON)
        with PASS_LOG.open("a") as fh:
            fh.write(json.dumps({"time": time.strftime("%H:%M:%S"), "label": label, "passes": n}) + "\n")
        return f

    def score(self, model, label: str) -> dict:
        """Diagnostic score of a model: dips recovered near and far, and error, per diagnostic cycle."""
        out = {"label": label}
        for period, (x, meta) in self.sets.items():
            median = self.forecast(model, x, f"{label} | cycle {period}").median[:, 0].double().cpu().numpy()
            shares = synth.recovered_shares(median, meta)
            sd = np.array([m["sd"] for m in meta])[:, None]
            out[f"near{period}"] = float(np.mean([s for _, lead, s in shares if lead < 384]))
            out[f"far{period}"] = float(np.mean([s for _, lead, s in shares if lead >= 512]))
            out[f"err{period}"] = float((np.abs(median - x[:, synth.CONTEXT :]) / sd).mean())
        out["passes_used"] = passes_used()
        with (OUT / "scores.jsonl").open("a") as fh:
            fh.write(json.dumps(out) + "\n")
        return out

    def save_patch(self, model, name: str) -> Path:
        """Save the edit as differences from the published weights; returns the file path."""
        base = dict(self.base.named_parameters())
        diff = {n: (p.detach() - base[n].detach()).cpu() for n, p in model.named_parameters()
                if not torch.equal(p.detach(), base[n].detach())}
        path = OUT / f"patch_{name}.safetensors"
        save_file(diff, str(path))
        changed = sum(int((d != 0).sum()) for d in diff.values())
        total = sum(p.numel() for p in self.base.parameters())
        print(f"patch {name}: {len(diff)} tensors, {changed} weights changed of {total} ({100 * changed / total:.4f} %)")
        return path


def show(row: dict) -> str:
    return (f"{row['label'][:58]:58s} | cycle 80: near {row['near80']:6.3f} far {row['far80']:6.3f} err {row['err80']:5.3f}"
            f" | cycle 180: near {row['near180']:6.3f} far {row['far180']:6.3f} err {row['err180']:5.3f} | passes {row['passes_used']}")


def qkv_rows(attn, part: int, head: int) -> torch.Tensor:
    """Row indices of `attn.wQKV.weight` for one of q (0), k (1), v (2) of one head.

    The fused projection's output is laid out as (part, head_dim, n_heads).
    """
    hd, nh = attn.head_dim, attn.num_heads
    return torch.tensor([part * hd * nh + d * nh + head for d in range(hd)])


def time_attention(model, layer: int):
    """The time attention module of a layer (the first attention of the block)."""
    return model.transformer.layers[layer].attns[0]
