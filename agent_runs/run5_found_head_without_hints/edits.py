"""Weight edits (closed form, no gradients).

`retarget_lag(model, layer, head, D, ref)`: for a positional time-attention head whose
per-frequency query and key components are nearly constant over tokens, rotate the query
weight rows of each RoPE frequency pair (dims f and f+32 of the head) so that every
frequency's positional score cos(w_f * lag + ang_q - ang_k) peaks at lag D. ang_q and
ang_k are the angles of the mean query / key component, measured on reference layer
inputs `ref` (activations averaged over the diagnostic series - a closed-form statistic).
Amplitudes and the unrotated (content) dims are untouched.
"""
import numpy as np
import torch

from harness import qkv_rows, time_attention


def head_mean_angles(att, xin, head, n_ctx=256):
    with torch.no_grad():
        q, k, _ = att._qkv(xin)
    q = q[:, :, head].reshape(-1, 64).double().numpy()
    k = k[:, :n_ctx, head].reshape(-1, 64).double().numpy()
    zq = (q[:, :16] + 1j * q[:, 32:48]).mean(0)
    zk = (k[:, :16] + 1j * k[:, 32:48]).mean(0)
    return zq, zk


def retarget_lag(model, layer, head, D, ref_xin):
    att = time_attention(model, layer)
    om = model.transformer.rope.inv_freq[:16].double().numpy()
    zq, zk = head_mean_angles(att, ref_xin, head)
    target = np.angle(zk) - om * D  # ang_q' = ang_k - w D  -> cos(w(lag - D))
    psi = target - np.angle(zq)
    rows = qkv_rows(att, 0, head)  # 64 rows of the query of this head
    W = att.wQKV.weight.data
    for f in range(16):
        ra, rb = rows[f], rows[f + 32]
        a, b = W[ra].clone(), W[rb].clone()
        c, s = float(np.cos(psi[f])), float(np.sin(psi[f]))
        W[ra] = c * a - s * b  # z -> z * e^{i psi}, z = a + i b
        W[rb] = s * a + c * b
    return psi


def design_lags(model, layer, head, peaks, ref_xin):
    """Generalisation of retarget_lag: positional score sum_p w_p * S(lag - D_p), with
    S(x) = sum_f |q_f||k_f| cos(w_f x). `peaks` = [(D, w), ...]. Each query frequency pair
    is multiplied (as a complex number) by m_f = zq'_f / zq_f with
    zq'_f = |zq_f| e^{i ang zk_f} sum_p w_p e^{-i w_f D_p}."""
    att = time_attention(model, layer)
    om = model.transformer.rope.inv_freq[:16].double().numpy()
    zq, zk = head_mean_angles(att, ref_xin, head)
    tgt = np.abs(zq) * np.exp(1j * np.angle(zk)) * sum(w * np.exp(-1j * om * D) for D, w in peaks)
    mult = tgt / zq
    rows = qkv_rows(att, 0, head)
    W = att.wQKV.weight.data
    for f in range(16):
        ra, rb = rows[f], rows[f + 32]
        a, b = W[ra].clone(), W[rb].clone()
        c, s = float(mult[f].real), float(mult[f].imag)
        W[ra] = c * a - s * b
        W[rb] = s * a + c * b
    return mult
