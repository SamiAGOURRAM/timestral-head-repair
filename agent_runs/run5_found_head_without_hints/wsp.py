"""Shared helpers for run 5: counted, hooked model runs and attention reconstruction.

`hooked_run` runs ONE pass (one batch of <= 64 series, no mirror) of `model.predict_block`
on the diagnostic context and logs it to results/passes.jsonl (1 pass), capturing the
input of every time attention (after its norm), the residual stream after every layer,
and the RoPE tables.
"""
import json
import time

import numpy as np
import torch

import harness
from harness import Bench, PASS_LOG, BUDGET, passes_used, synth
from timestral.layers import _apply_rotary


def log_pass(label: str, n: int = 1) -> None:
    used = passes_used()
    if used + n > BUDGET:
        raise RuntimeError(f"pass budget exhausted: {used} used")
    with PASS_LOG.open("a") as fh:
        fh.write(json.dumps({"time": time.strftime("%H:%M:%S"), "label": label, "passes": n}) + "\n")


def hooked_run(model, x: np.ndarray, label: str, sign: float = 1.0):
    """One counted pass. Returns dict with 'attn_in'[layer] (R,N,D), 'resid'[layer] (R,N,D), 'cos','sin','preds' (median, raw space)."""
    assert len(x) <= 64
    log_pass(label, 1)
    ctx = torch.tensor(sign * x[:, : synth.CONTEXT], dtype=torch.float32)
    mask = torch.ones_like(ctx, dtype=torch.bool)
    store = {"attn_in": {}, "resid": {}, "cos": None, "sin": None}
    hs = []
    for li, layer in enumerate(model.transformer.layers):
        att = layer.attns[0]
        hs.append(att.register_forward_hook(lambda m, a, o, li=li: store["attn_in"].__setitem__(li, a[0].detach().clone())))
        hs.append(layer.register_forward_hook(lambda m, a, o, li=li: store["resid"].__setitem__(li, o.detach().clone())))
    hs.append(model.transformer.rope.register_forward_hook(lambda m, a, o: store.update(cos=o[0].clone(), sin=o[1].clone())))
    try:
        with torch.inference_mode():
            q = model.predict_block(ctx, mask, torch.arange(len(x)), 32)
    finally:
        for h in hs:
            h.remove()
    store["preds"] = q[..., model.config.median_index].numpy() * sign
    return store


def attention_probs(att, xin, cos, sin, n_enc=256):
    """Attention probabilities of decoder queries over all keys, per head: (R, H, Ndec, N)."""
    with torch.no_grad():
        q, k, v = att._qkv(xin)
        q, k = _apply_rotary(q, k, cos, sin)
        q = q.transpose(1, 2)  # R,H,N,hd
        k = k.transpose(1, 2)
        s = q[:, :, n_enc:] @ k.transpose(-1, -2) / np.sqrt(att.head_dim)
        n = s.shape[-1]
        ndec = n - n_enc
        allow = torch.ones(ndec, n, dtype=torch.bool)
        allow[:, n_enc:] = torch.tril(torch.ones(ndec, ndec, dtype=torch.bool))
        s = s.masked_fill(~allow, float("-inf"))
        return s.softmax(-1), v.transpose(1, 2)


def evaluate(bench, model, label: str) -> dict:
    """Same as Bench.score (4 passes, through Bench.forecast) plus a per-lead profile (bins of 128 steps)."""
    from harness import OUT
    out = {"label": label}
    prof = {}
    for period, (x, meta) in bench.sets.items():
        median = bench.forecast(model, x, f"{label} | cycle {period}").median[:, 0].double().cpu().numpy()
        shares = synth.recovered_shares(median, meta)
        sd = np.array([m["sd"] for m in meta])[:, None]
        out[f"near{period}"] = float(np.mean([s for _, lead, s in shares if lead < 384]))
        out[f"far{period}"] = float(np.mean([s for _, lead, s in shares if lead >= 512]))
        out[f"err{period}"] = float((np.abs(median - x[:, synth.CONTEXT:]) / sd).mean())
        bins = {}
        for _, lead, s in shares:
            bins.setdefault(lead // 128, []).append(s)
        prof[period] = [round(float(np.mean(bins[k])), 3) for k in sorted(bins)]
    out["passes_used"] = passes_used()
    with (OUT / "scores.jsonl").open("a") as fh:
        fh.write(json.dumps(out) + "\n")
    out["profile"] = prof
    return out
