"""Weight edits (closed form, no gradients).

shift_head: change the relative distance at which a time-attention head reads, by rotating
the head's query rows with the model's own rotary rotation.

With RoPE the score of query (position m) and key (position n) depends on m - n through
rotations of the rotary dimension pairs. Replacing the query projection W_q of one head by
T_s W_q, where T_s is the rotary rotation for a position offset s, makes every query of that
head behave exactly as if its token sat s patches later (s < 0: earlier). A head that read
the token d patches back now reads the token d - s patches back. Nothing else changes:
values, keys, gate and output projection are untouched, and the unrotated dimensions too.
"""
import torch

from harness import qkv_rows, time_attention


def rotary_matrix(model, s: float) -> torch.Tensor:
    """(head_dim, head_dim) matrix T_s with T_s q = q cos(w s) + rotate_half(q) sin(w s)."""
    inv = model.transformer.rope.inv_freq.double()
    ang = torch.cat([inv, inv]) * s
    hd = ang.numel()
    half = hd // 2
    t = torch.diag(ang.cos())
    for d in range(half):
        t[d, d + half] += -ang[d].sin()  # rotate_half(q)[d] = -q[d + half]
        t[d + half, d] += ang[d + half].sin()  # rotate_half(q)[d + half] = q[d]
    return t


def shift_head(model, layer: int, head: int, s: float, part: int = 0) -> None:
    attn = time_attention(model, layer)
    rows = qkv_rows(attn, part, head)
    w = attn.wQKV.weight.data
    t = rotary_matrix(model, s)
    w[rows] = (t @ w[rows].double()).to(w.dtype)


def gain_matrix(g) -> torch.Tensor:
    """(64, 64) matrix multiplying the complex component (q[f] + i q[f + 32]) of rotary frequency f by g[f];
    identity on the unrotated dimensions (16..31, 48..63)."""
    t = torch.eye(64, dtype=torch.float64)
    for f in range(16):
        re, im = float(g[f].real), float(g[f].imag)
        t[f, f], t[f, f + 32] = re, -im
        t[f + 32, f], t[f + 32, f + 32] = im, re
    return t


def gain_head(model, layer: int, head: int, g) -> None:
    """Query rows of one head <- G @ rows, G = gain_matrix(g). g: 16 complex gains (numpy)."""
    attn = time_attention(model, layer)
    rows = qkv_rows(attn, 0, head)
    w = attn.wQKV.weight.data
    w[rows] = (gain_matrix(g) @ w[rows].double()).to(w.dtype)


def apply_gain_file(model, name: str, layer: int = 5, head: int = 1) -> None:
    import numpy as np

    from harness import OUT

    gain_head(model, layer, head, np.load(OUT / f"gain_{name}.npy"))


def apply_spec(model, spec: str) -> None:
    """spec like '5.1:-16,4.1:-4' = head 5.1 queries moved 16 patches earlier, head 4.1 by 4."""
    if spec in ("", "none"):
        return
    for item in spec.split(","):
        hd, s = item.split(":")
        layer, head = hd.split(".")
        shift_head(model, int(layer), int(head), float(s))
