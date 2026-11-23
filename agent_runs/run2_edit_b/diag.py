"""Diagnostic runs with interventions inside the attention (hooks). Every run is counted:
1 pass per call of `run` (one batch of 64 unmirrored diagnostic series), logged to results/passes.jsonl.

Interventions are for understanding only (they are not weight edits):
  block:L.H        decoder queries of time head L.H cannot read decoder keys (incl. themselves)
  blockenc:L.H     decoder queries of time head L.H cannot read encoder keys
  zero:L.H:K       output of time head L.H set to 0 on decoder patches >= K
  zeroenc:L.H      output of time head L.H set to 0 on encoder tokens
  force:L.H:D      every token of time head L.H reads exactly the token D patches back (one-hot)
"""
import json
import time

import numpy as np
import torch

import check3_synthetic_lead_time as synth
from harness import BUDGET, OUT, PASS_LOG, passes_used
from timestral import layers as L

_orig_forward = L.Attention.forward
STATE = {"iv": [], "rec": None}


def _forward(self, x, mask, cos, sin):
    tag = getattr(self, "_tag", None)
    if not self.is_time or tag is None:
        return _orig_forward(self, x, mask, cos, sin)
    layer = tag
    rows, num_patches, _ = x.shape
    q, k, v = self._qkv(x)
    q, k = L._apply_rotary(q, k, cos, sin)
    n = mask.enc_len
    out_enc = L._sdpa(q[:, :n], k[:, :n], v[:, :n], mask.mask[:, :, :n, :n])
    qd = q[:, n:].transpose(1, 2)
    logits = qd @ k.transpose(1, 2).transpose(-1, -2) / (self.head_dim ** 0.5)
    logits = logits.masked_fill(~mask.mask[:, :, n:, :], float("-inf"))
    for kind, l, h, arg in STATE["iv"]:
        if l == layer and kind == "block":
            logits[:, h, :, n:] = float("-inf")
        if l == layer and kind == "blockenc":
            logits[:, h, :, :n] = float("-inf")
    w = logits.softmax(-1)
    od = w @ v.transpose(1, 2)  # (R, H, 32, hd)
    for kind, l, h, arg in STATE["iv"]:
        if l == layer and kind == "zero":
            od[:, h, arg:, :] = 0
    out_dec = od.transpose(1, 2).contiguous().view(rows, num_patches - n, -1)
    for kind, l, h, arg in STATE["iv"]:
        if l == layer and kind == "zeroenc":
            out_enc = out_enc.clone().view(rows, n, self.num_heads, self.head_dim)
            out_enc[:, :, h, :] = 0
            out_enc = out_enc.view(rows, n, -1)
    out = torch.cat([out_enc, out_dec], dim=1)
    for kind, l, h, arg in STATE["iv"]:
        if l == layer and kind == "force":
            # every token (encoder and decoder) of this head reads exactly the token `arg` patches back
            idx = (torch.arange(num_patches) - arg).clamp(min=0)
            o4 = out.clone().view(rows, num_patches, self.num_heads, self.head_dim)
            o4[:, :, h, :] = v[:, idx, h, :]
            out = o4.view(rows, num_patches, -1)
    if STATE["rec"] is not None:
        STATE["rec"].setdefault("attn", []).append(w.to(torch.float16).numpy())
    out = out * self._gate(x).sigmoid()
    return self.wO(out.reshape(rows, num_patches, self.embed_dim))


def parse(spec: str):
    iv = []
    for item in spec.split(","):
        if item in ("", "none"):
            continue
        parts = item.split(":")
        l, h = parts[1].split(".")
        iv.append((parts[0], int(l), int(h), int(parts[2]) if len(parts) > 2 else 0))
    return iv


def run(bench, model, period: int, spec: str, label: str, record: bool = False):
    """One counted pass. Returns (median (64, 1024) in raw space / anchor frame, recorded dict or None)."""
    for i, layer in enumerate(model.transformer.layers):
        layer.attns[0]._tag = i
    L.Attention.forward = _forward
    STATE["iv"] = parse(spec)
    STATE["rec"] = {} if record else None
    x, meta = bench.sets[period]
    ctx = torch.tensor(x[:, : synth.CONTEXT].astype(np.float32))
    assert passes_used() + 1 <= BUDGET
    try:
        with torch.inference_mode():
            q = model.predict_block(ctx, torch.ones_like(ctx, dtype=torch.bool), torch.arange(len(ctx)), 32)
    finally:
        L.Attention.forward = _orig_forward
    with PASS_LOG.open("a") as fh:
        fh.write(json.dumps({"time": time.strftime("%H:%M:%S"), "label": f"diag {label} [{spec}] | cycle {period}", "passes": 1}) + "\n")
    rec = STATE["rec"]
    STATE["rec"] = None
    return q[..., 10].double().numpy(), rec


def summary(median: np.ndarray, period: int) -> str:
    """near / far shares and dip offsets by lead on a raw one-sided median (not the official score)."""
    x, meta = synth.make_series(period, synth.N_SERIES, seed=period)
    sh = synth.recovered_shares(median, meta)
    near = np.mean([s for _, lead, s in sh if lead < 384])
    far = np.mean([s for _, lead, s in sh if lead >= 512])
    rows = []
    for i, m in enumerate(meta):
        w = m["width"]
        cs = np.r_[0, np.cumsum(median[i] - np.median(median[i]))]
        for s in m["starts"]:
            best, bo = 0, np.nan
            for o in range(-period // 2, period // 2):
                a = s + o
                if a < 0 or a + w > 1024:
                    continue
                val = (cs[a + w] - cs[a]) / w / m["depth"]
                if val < best:
                    best, bo = val, o
            rows.append((s, bo, -best))
    rows = np.array(rows)
    txt = f"one-sided near {near:.3f} far {far:.3f} | offset median / depth by lead(128): "
    for a in range(0, 1024, 128):
        sel = (rows[:, 0] >= a) & (rows[:, 0] < a + 128)
        txt += f"[{np.nanmedian(rows[sel, 1]):.0f} {rows[sel, 2].mean():.2f}] "
    return txt
