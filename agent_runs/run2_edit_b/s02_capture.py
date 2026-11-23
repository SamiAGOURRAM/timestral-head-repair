"""Counted diagnostic pass: one direct predict_block call per diagnostic cycle on the 64
unmirrored series (1 pass each, logged to results/passes.jsonl), with a recording version of
Attention.forward. Saves residual streams, decoder attention patterns and per-head outputs.

Usage: python s02_capture.py 80 [tag]     (tag = name of an edit function in edits.py, default none)
"""
import json
import sys
import time

import numpy as np
import torch

import check3_synthetic_lead_time as synth
from harness import BUDGET, OUT, PASS_LOG, Bench, passes_used
from timestral import layers as L

period = int(sys.argv[1])
tag = sys.argv[2] if len(sys.argv) > 2 else "base"

bench = Bench()
model = bench.fresh()
if tag != "base":
    # tag is "name=shiftspec", e.g. "s51m16=5.1:-16" (added after the first two captures)
    import edits
    tag, spec = tag.split("=")
    if spec.startswith("gain:"):  # "name=gain:b" = gain file results/gain_b.npy on head 5.1
        edits.apply_gain_file(model, spec[5:])
    else:
        edits.apply_spec(model, spec)

rec = {"attn": [], "headout": [], "names": []}


def recording_forward(self, x, mask, cos, sin):
    rows, num_patches, _ = x.shape
    inputs = x if self.is_time else x.permute(1, 0, 2)
    q, k, v = self._qkv(inputs)
    if self.is_time:
        q, k = L._apply_rotary(q, k, cos, sin)
        n = mask.enc_len
        out_enc = L._sdpa(q[:, :n], k[:, :n], v[:, :n], mask.mask[:, :, :n, :n])
        # decoder part by hand, to keep the attention weights
        qd = q[:, n:].transpose(1, 2)  # (R, H, 32, hd)
        kk = k.transpose(1, 2)
        vv = v.transpose(1, 2)
        logits = qd @ kk.transpose(-1, -2) / (self.head_dim ** 0.5)
        logits = logits.masked_fill(~mask.mask[:, :, n:, :], float("-inf"))
        w = logits.softmax(-1)  # (R, H, 32, S)
        od = (w @ vv)  # (R, H, 32, hd)
        out_dec = od.transpose(1, 2).contiguous().view(rows, num_patches - n, -1)
        out = torch.cat([out_enc, out_dec], dim=1)
        rec["attn"].append(w.to(torch.float16).numpy())
    else:
        out = L._sdpa(q, k, v, mask, is_causal=False)
    gate = self._gate(inputs).sigmoid()
    out = out * gate
    if self.is_time:
        out = out.reshape(rows, num_patches, self.embed_dim)
    else:
        out = out.reshape(num_patches, rows, self.embed_dim).permute(1, 0, 2)
    # per-head contribution to the residual stream on decoder tokens. Layout of `out`: (heads, head_dim)
    dec = out[:, -32:, :].reshape(rows, 32, self.num_heads, self.head_dim)
    wo = self.wO.weight.reshape(self.embed_dim, self.num_heads, self.head_dim)
    ho = torch.einsum("rphd,ehd->rhpe", dec, wo)
    rec["headout"].append(ho.to(torch.float16).numpy())
    rec["names"].append("time" if self.is_time else "variate")
    return self.wO(out)


L.Attention.forward = recording_forward

resid = []
for layer in model.transformer.layers:
    layer.register_forward_pre_hook(lambda mod, args: resid.append(args[0].detach().clone()))
final = []
model.transformer.out_norm.register_forward_pre_hook(lambda mod, args: final.append(args[0].detach().clone()))

x, meta = bench.sets[period]
ctx = torch.tensor(x[:, : synth.CONTEXT].astype(np.float32))
used = passes_used()
assert used + 1 <= BUDGET
with torch.inference_mode():
    q = model.predict_block(ctx, torch.ones_like(ctx, dtype=torch.bool), torch.arange(len(ctx)), 32)
with PASS_LOG.open("a") as fh:
    fh.write(json.dumps({"time": time.strftime("%H:%M:%S"), "label": f"capture {tag} | cycle {period}", "passes": 1}) + "\n")

resid = torch.stack(resid + final).numpy()  # (9, R, 288, 192)
np.savez(
    OUT / f"cap_{tag}_{period}.npz",
    resid=resid.astype(np.float32),
    attn=np.stack(rec["attn"]),  # (8, R, 3, 32, 288)
    headout=np.stack(rec["headout"]),  # (10, R, 3, 32, 192)
    names=np.array(rec["names"]),
    median=q[..., 10].numpy(),
)
print("saved", resid.shape, np.stack(rec["attn"]).shape, rec["names"], "passes used", passes_used())
