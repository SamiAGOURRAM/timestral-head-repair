# REPORT — run 5 (blind arm): Timestral-5M, far dips edited without training

## Result

| model | near80 | far80 | err80 | near180 | far180 | err180 |
|---|---|---|---|---|---|---|
| published | 0.696 | -0.254 | 1.496 | 0.955 | -0.027 | 1.227 |
| **final edit** (reloaded from `final_checkpoint/`) | **0.879** | **0.747** | **0.873** | **0.956** | **0.888** | **0.843** |

Far dips: +1.00 (cycle 80) and +0.92 (cycle 180). Near dips: +0.18 and +0.00. Error: down
by 42 % and 31 %. One tensor changed, 6,144 weights out of 5,208,960 (0.118 %): the 64
query rows of head 1 in the time attention of layer 5. Passes used: 32 of 200.

## Mechanism

1. The 1024-step horizon is one block of 32 decoder tokens. They all start as the same
   learned vector and differ only by their RoPE position. They read all context patches
   and the earlier decoder tokens.
2. Layers 6 and 7 hold content-matching ("induction") heads: L6h1, L6h2, L7h0, L7h1 and
   L7h2. For decoder token j they read the context patches that are in phase with patch j.
   In cycle 180 they read lags 17+45k and 28+45k; in cycle 80, multiples of 5 patches.
3. Their query is built by **L5h1**, a purely positional head. Its per-frequency query
   and key components are constant over tokens (coherence 0.96–1.00), and it reads the
   patch **11 patches back**. Every token, context or decoder, therefore carries "the
   content 11 patches earlier". Layers 6–7 match the decoder's "content 11 back" against
   each context key's "content 11 back", then copy the matched patch.
4. For decoder tokens j >= 11, "11 back" is itself a decoder token, which holds no
   observed content. The matching fails. Measured in-phase attention of the layer-6/7
   heads starts falling at exactly j = 12 and reaches chance by j ≈ 16–19, which is
   500–600 steps ahead. It does this in both cycles.
5. Cycle 96 (3 patches) has no limit, probably through L2h2. That positional head reads
   lags 63 and 126, both multiples of 3 patches. This is inferred, not tested: 96 is not a
   diagnostic cycle.
6. Proof by edit: if L5h1 is re-phased to read lag 11 again, nothing changes (control).
   If it reads a lag >= 32, so that every decoder's source is observed context, far dips
   come back. After the edit, the layer-6/7 heads stay 2–8x above chance in-phase for all
   32 decoder tokens (s13).

## Edits tried (Bench-equivalent scores, 4 passes each)

| edit of L5h1 | near80 | far80 | err80 | near180 | far180 | err180 |
|---|---|---|---|---|---|---|
| lag 43 (single peak) | 0.878 | 0.748 | 0.874 | 0.955 | 0.896 | 0.842 |
| lag 11 re-phased (control) | 0.759 | -0.259 | 1.482 | 0.956 | -0.025 | 1.225 |
| lag 32 | 0.894 | 0.706 | 0.898 | 0.950 | 0.828 | 0.864 |
| lag 64 | 0.734 | 0.445 | 1.092 | 0.932 | 0.884 | 0.854 |
| **lag 43 x1.5 + lag 11 x0.5 (final)** | 0.879 | 0.747 | 0.873 | 0.956 | 0.888 | 0.843 |

## Final edit, precisely

Tensor: `transformer.layers.5.attns.0.wQKV.weight`, rows of the query of head 1:
`qkv_rows(attn, 0, 1)`, which are rows `3*d + 1` for d = 0..63. Only the 16 rotated
frequency pairs change, (d = f, d = f+32) for f = 0..15. Each pair of rows (a, b) is
multiplied as a complex number `a + i b` by

    m_f = zq'_f / zq_f,   zq'_f = |zq_f| * exp(i arg zk_f) * (1.5 exp(-i w_f 43) + 0.5 exp(-i w_f 11))
    a' = Re(m_f) a - Im(m_f) b,   b' = Im(m_f) a + Re(m_f) b

- `w_f = 10000^(-f/16)` is the RoPE frequency.
- `zq_f` and `zk_f` are the mean complex query and key components of L5h1 at frequency f.
  They are averaged over the captured layer-5 inputs of the 64 + 64 diagnostic series:
  all tokens for zq, context tokens for zk. The values are in
  `results/l5h1_ref_means.json`.

The positional score of the head becomes

    1.5 S(lag - 43) + 0.5 S(lag - 11),   with S(x) = sum_f |q_f||k_f| cos(w_f x)

- When 43 patches of history exist, it reads lag 43 (mass 0.99–1.00).
- When they do not (a context under 1376 steps, or early context tokens), it falls back to
  lag 11 like the published model (mass 0.93 against 0.85 for the published model,
  measured offline on causal early positions).

Lag 43 = 11 + 32 is the original lag moved past the 32-patch decode block. The unrotated
(content) query dims, the keys, the values and everything else are untouched.

Code: `edits.py` (`design_lags`, `retarget_lag`) and `s14_final.py`.

Closed-form steps used: activation means (complex mean of query/key components per
frequency), per-frequency phase alignment, and a complex scaling of weight-row pairs.
No gradient, optimizer or training of any kind.

## Expectation on unseen data (honest)

- **Other cycle lengths and pattern shapes.** The fix is not cycle-specific: induction
  with lag 43 works for any repeat, the same way lag 11 did. I expect far dips to improve
  where the published model's limit came from the lag-11 source falling inside the
  decoder block. Not tested: only cycles 80 and 180 were run.
- **Short contexts (< 43 patches).** The fallback keeps the published lag-11 behaviour
  inside layer 5. This was checked offline on attention only, not on forecasts.
  - Contexts between 11 and about 43 patches are a mixed case. Decoders whose lag-43 key
    exists read lag 43, while context keys carry lag-11 content. Those tokens may be no
    better than in the published model.
- **General forecasting.** For any context of 43 patches or more, every token's L5h1
  read moves from 11 to 43 patches back. On the diagnostic series this lowered the error
  everywhere, near horizons included. On real, non-periodic data I cannot tell whether
  anything downstream relied on lag 11 specifically, so a regression on the general
  benchmark is possible. I have no data to bound it.
- **Tuning.** The lag was chosen among 32, 43 and 64 partly from the diagnostic scores
  (43 > 32 on far dips). The dual-peak weights (1.5, 0.5) were chosen from attention
  masses only.

## Blindness

- **Context shown to me.** My session context contained the git status and recent commit
  titles of another folder (<project>). I did not open, list or
  search it, but I saw the text:
  - File names under `experiments/edit_5m/`, `results/edit_5m/` and
    `results/gift_eval/5M-finetune-sameweights/`.
  - Commit titles mentioning an "edit B", blind runs finding "a gradient-free fix", a fine-tune
    comparison and a 28M transfer.

  None of it names a layer, head or tensor, and none of it guided where I looked. The
  location came from the attention analysis in s04–s07. It does hint that the general
  benchmark is GIFT-Eval; that played no role in the edit.
- **Files I accessed.** Outside run5 I only read the `timestral` package source, ran the
  given interpreter, and loaded the published weights from the Hugging Face cache through
  `harness.py` and `load_model`.
- **Model runs.** Only on `bench.sets[80]` and `bench.sets[180]`, all logged in
  `results/passes.jsonl` (32 passes).

## Files

- `journal.md`
- Scripts: `s01`–`s15`, `wsp.py`, `edits.py`.
- `final_checkpoint/` (model.safetensors, config.json, patch_final.safetensors).
- `results/` (passes.jsonl, scores.jsonl, patch_final.safetensors, l5h1_ref_means.json).
- `results/capture_base_*.pt`: two 236 MB activation captures used by the analysis scripts.
