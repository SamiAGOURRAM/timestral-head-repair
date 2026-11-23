# Journal — run2

Rules I hold myself to: every model run goes through `Bench.score` / `Bench.forecast` or is a
logged direct call (1 pass per batch of 64). Offline, I only do linear algebra on saved
activations and weights (averages, least squares, single-matrix products); I do not re-run
model layers offline on saved activations as a way around the budget.

## Step 0 — reading (0 passes)
- Read TASK.md, harness.py, check3 (generator + measure), and the `timestral` package
  (model.py, layers.py, masks.py, forecaster.py, scaler.py, config.py, loading.py).
- Architecture facts that matter: 256 context patches (encoder tokens, causal among
  themselves) then 32 decoder tokens that all start as the same `forecast_query` vector;
  decoder tokens read the whole context and earlier decoder tokens. Time attention uses RoPE on
  16 of the 32 dimension pairs of each head (frequencies 1 … 1.8e-4 rad/patch), a sigmoid
  gate on the attention output, no biases. Layers 3 and 7 also have a variate attention (with
  one variate it only reads its own token). Normalised output is in the frame of the last
  context patch; the forecaster then re-frames patch by patch.
- `s01_inspect.py`: config + weight shapes. block_size 32, trained context 32768.

## Step 1 — one recorded pass per cycle (2 passes, total 2)
- `s02_capture.py 80` and `180`: direct `predict_block` on the 64 unmirrored series with a
  recording `Attention.forward`; saves residual streams, decoder attention patterns, per-head
  outputs, raw median (`results/cap_base_*.npz`). Logged 1 pass each.
- `s03_attn_stats.py`, `s04_attn_top.py` (free). Findings:
  - Fixed-offset "relay" heads, the same pattern for every series: 3.1 reads the token 3
    patches back, 4.0 one back, 4.1 two back, 5.0 one back, **5.1 eleven back** (mass ~0.9).
    They read whatever sits there: a context patch, or an earlier decoder token.
  - 1.2, 4.2, 5.2 point at the last context patch(es); 1.2 stops doing so from patch ~23 on.
  - Heads 6.1, 6.2, 7.0, 7.1, 7.2 (and 6.0) read the context at the same place in the cycle
    for the first decoder patches (alignment 0.9+) and lose it further out.
- `s06_forecast_shape.py`, `s07_dip_offsets.py`, `s08_phase_hist.py` (free): the far forecast is
  NOT flat. Cycle 80: from lead 384 on, dips of depth 0.6-0.9 are placed exactly 16 steps
  late. Cycle 180: about 40 steps late, weaker. The dip heads read context patches at the
  wrong place in the cycle (16 steps earlier for cycle 80, then 32).

## Step 2 — first edit idea: rotate the query of 5.1 (12 passes, total 14)
- `edits.py: shift_head`: multiplying a head's query rows by the rotary rotation T_s makes its
  positional kernel K(d) become K(d + s). `s05_score_shift.py`:
  - none: 80 near 0.696 far -0.254 err 1.496 | 180 near 0.955 far -0.027 err 1.227 (as in TASK.md)
  - 5.1:-16: 80 near 0.272 far -0.243 err 1.615 | 180 near 0.857 far -0.117 err 1.321
  - 5.1:-21: 80 near 0.242 far -0.221 err 1.605 | 180 near 0.480 far -0.125 err 1.436
  Failed. Reason found in step 4: the kernel has other lobes that the shift brings into reach.

## Step 3 — interventions to find the cause (6 passes, total 20)
- `diag.py` / `s09_diag.py`: hook interventions, 1 logged pass each, one-sided raw median.
  - block:5.1 (decoder queries cannot read decoder tokens): cycle 80 dips 16 steps EARLY at lead
    384-640; cycle 180 far dips strong (0.9) but 16-23 late.
  - zero:5.1:11: cycle 180 far dips vanish (depth 0.04); cycle 80 still 16 late.
  - block all five relay heads: dips strong everywhere, misplaced even near.
  Reading: the dip heads match "what head 5.1 brought to me" with "what head 5.1 brought to a
  context patch". Both must come from the same distance back. A decoder patch >= 11 gets an
  earlier decoder token instead of real history, so the match is shifted.

## Step 4 — check of the shift (1 pass, total 21)
- `s02_capture.py 80 s51m16=5.1:-16`: shifted 5.1 reads 6-7 back (0.9) and 27 back (0.1), not
  27 back. `s10_kernel.py`, `s11_freq_parts.py` (free; note s10 prints K(-d)): kernel from mean
  query/key has lobes at d = 11, 23, 35 and, on the masked side, -9.5, -21, -33. Head 5.1 is
  almost purely positional (query spread ~10 % of its mean).
- With the shift, patches 0-3 (which still read real history on both sides) stayed right:
  the match is symmetric, as hoped.

## Step 5 — proof of mechanism by intervention (4 passes, total 25)
- force:5.1:11 (every token of 5.1 reads exactly 11 back) reproduces the baseline.
- **force:5.1:32** (every token reads exactly 32 back, always real history for a 32-patch
  horizon): one-sided cycle 80 near 0.895 far 0.682; cycle 180 near 0.958 far 0.914; all
  dips at offset 0. So the far limit is head 5.1's reading distance.

## Step 6 — turning it into weights (in progress)
- `s12_design.py`: ridge least squares for per-frequency complex gains on 5.1's query rows,
  adding a kernel peak at d = 32. First try (H 50, sigma 1, ridge 0.5) far too weak: peak stays
  at 11. Session was interrupted here; 25 passes used (checked against results/passes.jsonl).
- `s13_design2.py`: other form: g_f = exp(i w_f s) (plain shift, s = 11 - 32 = -21, moves the
  d = 11 lobe to d = 32) + an envelope E_f cos(w_f (d - 32)) on slow frequencies so that the
  d = 32 lobe wins over the lobes the shift brings in. Design "b" (E: 0.1 -> 12, 0.178 -> 4,
  0.056 -> 8), edit = `edits.gain_head` (query rows of 5.1 multiplied by a block matrix).
  - **score b (4 passes, total 29): 80 near 0.836 far 0.566 err 0.995 | 180 near 0.929 far 0.432 err 0.986.**
- Capture with b on cycle 180 (1 pass, total 30): far decoder patches give ~45 % of head 5.1 to
  themselves (own key), which my mean-vector kernel had not predicted.
- `s16_logit_check.py` (free): head 5.1's attention computed offline from the saved layer-5
  input (q.k dot products + rotary only; layer-5 input does not depend on this edit) matches
  the recorded passes to 2 decimals, for the base model and for b. I use this as the design
  objective from here on. It is not a model run: no layer is executed, only this head's scores.
- `s17_design4.py`: coordinate grid search of E on that exact objective. Design "e"
  (0.316 -> 3, 0.1 -> 12, 0.056 -> 24): 0.997 of the head on 31-33 back for every decoder group.
  - **score e (4 passes, total 34): 80 near 0.832 far 0.594 err 0.979 | 180 near 0.933 far 0.846 err 0.860.**
- Worry: histories shorter than 32 patches. Tokens that cannot read 32 back should behave as
  before (read 11 back), otherwise short series change. `s18_design5.py` / `s19_design6.py` add
  that to the objective (overlap with the original attention for context patches < 32) and
  allow cosine + sine envelope terms on 8 frequencies. Design "h": 0.97 on exactly 32 back
  for all decoder groups; context patches 11..31 read 11 back as before (overlap 0.55-0.8);
  context patches 1..10 read themselves instead of 2-3 back (not fixed).
  - **score h (4 passes, total 38): 80 near 0.875 far 0.673 err 0.907 | 180 near 0.947 far 0.897 err 0.845.**

## Step 7 — alternatives and robustness (8 passes, total 46)
Session interrupted again after s22; pass log re-read: 38 used, no half-written run.
- `s19_design6.py i` (extra weight on context patches 1..10): did not restore them (overlap 0.02).
- `s20_design_ls.py` (one ridge least-squares solve on the mean kernel): FAILED offline, 0.5-0.6
  of the head on 32 back, rivals at 100 and 11-12 back. Not scored.
- `s21_design_formula.py` k1 (explicit bumps at 32, 11, 2.5 on four frequencies): FAILED offline
  (head reads 12 back). k2 (fast frequencies: bumps at 32 and 11; slow ones: one hill at 32):
  1.00 of the head on 31-33 back, best margins (`s22_eval_gain.py`), but tokens with fewer than
  32 patches behind read the oldest token instead of 11 back.
- **score k2: 80 near 0.880 far 0.672 err 0.925 | 180 near 0.952 far 0.905 err 0.843** (total 42)
- **score i : 80 near 0.868 far 0.649 err 0.920 | 180 near 0.946 far 0.892 err 0.845** (total 46)
- h, i, k2 give the same scores within 0.03, and the same as the hook intervention
  force:5.1:32 (one-sided far 0.68 / 0.91). The weight edit reaches what this mechanism gives.
- `s19_design6.py j` (weight 30 on patches 1..10): the search gave up the main goal (0 on 32 back).
  Not scored. Restoring the first 10 patches and reading 32 back is not reachable in this family.

## Step 8 — choice and delivery (5 passes, total 51)
- Final = design h: same score as k2, and it keeps the original reading (11 back) for tokens
  with 11 to 31 patches behind them, which is what short series see. k2 is slightly more
  robust in margin on the diagnostic data but changes short histories more. Judgement call,
  not measured: I may not run the model on short series.
- Capture with h on cycle 80 (1 pass, total 47): dips at offset 0 at every lead, depth 0.98
  falling to 0.70 at lead 960+.
- `s23_final.py` (free): final_checkpoint/model.safetensors, config.json (written from the
  loaded config, not copied from the HF cache), results/patch_final.safetensors. One tensor
  differs (layer 5 time attention wQKV), 32 rows, 6144 weights (0.118 %).
- `s24_score_saved.py` (4 passes, total 51): checkpoint reloaded from disk scores
  80 near 0.875 far 0.673 err 0.907 | 180 near 0.947 far 0.897 err 0.845.
- Stopped at 51 of 200 passes. Not explored: why cycle 80 far stays at 0.67 (depth, not place).

