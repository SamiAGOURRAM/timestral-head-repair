# Journal — run 5 (blind arm)

## Step 0 — setup (0 passes)
Read TASK.md, harness.py, check3_synthetic_lead_time.py, and the `timestral` package source
(model.py, layers.py, masks.py, forecaster.py, scaler.py, config.py, loading.py).

Facts from the source:
- 8 layers `time,time,time,time_variate,time,time,time,time_variate`; width 192, 3 heads of 64.
- RoPE on time attention, theta 1e4, only 32 of 64 head dims rotate (16 frequencies,
  1 down to 1.78e-4 rad/patch); dims i and i+32 pair (i < 16); dims 16-31 and 48-63 unrotated.
- 8192-step context = 256 patches; the 1024-step horizon = 32 decoder tokens in ONE block
  (block_size 32), so no autoregressive feedback. All 32 decoder tokens start as the same
  learned vector `forecast_query`; they differ only by their RoPE position and by what they
  read. Decoder tokens read all context patches and earlier decoder tokens.
- Each series is run alone (variate attention = self only).
- Cycle 80 = 2.5 patches (pattern of patches repeats every 5 patches = 160 steps);
  cycle 180 = 5.625 patches (patch-aligned repeat every 45 patches = 1440 steps);
  cycle 96 = 3 patches (repeat every 3 patches).

## Step 1 — config and parameter list (s01_inspect.py, 0 passes)

## Step 2 — baseline (s02_baseline.py, 4 passes; total 4)
cycle 80: near 0.696 far -0.254 err 1.496 | cycle 180: near 0.955 far -0.027 err 1.227. Matches TASK.md.

## Step 3 — hooked capture (s03_capture.py, 2 passes; total 6)
One un-mirrored pass per cycle through `predict_block` with hooks; saved inputs of every
time attention, residuals, RoPE tables (results/capture_base_*.pt; not delivered if large).

## Step 4 — decoder attention by lag (s04_attn_lags.py, 0 passes)
Fixed-lag positional heads read by decoders: L3h1 lag 3, L4h0 lag 1, L4h1 lag 2 (+65,128),
L5h0 lag 1, **L5h1 lag 11**, L2h2 lags 63/126. Layer 6 heads (and L7h0/L7h2) read the
context at lags that are multiples of the patch-aligned period (cycle 80: 10,20,45,...;
cycle 180: 17+45k, 28+45k): content-matching ("induction") heads.

## Step 5 — in-phase attention per decoder token (s05_alignment.py, 0 passes)
Share of attention on context keys in phase with the query patch / chance. L6h1, L6h2,
L7h0, L7h1, L7h2 are 3-8x above chance for decoder tokens j=0..11, and decay from
**j=12** to chance by j~16-19 (≈ 500-600 steps), in both cycles. j=11 is exactly where
L5h1 (lag 11) stops reading the context and starts reading decoder token j-11, which holds
no observed content. Hypothesis: induction with a lag of 11 patches — L5h1 writes into each
token the content of the patch 11 earlier; the layer-6/7 heads match the decoder's
"content 11 back" against the context keys' "content 11 back" and copy the matched patch.
For decoders j>=11 the "11 back" source is itself a decoder, so matching fails.
(Cycle 96 = 3 patches probably survives through the lag-63/126 positional head L2h2,
63 and 126 being multiples of 3 patches — not tested, it is not a diagnostic cycle.)

## Step 6-7 — L5h1 score structure (s06_lag_heads.py, s07_l5h1_freq.py, 0 passes)
L5h1 is purely positional: per RoPE frequency, query and key components are constant over
tokens (coherence 0.96-1.00 on all high-amplitude frequencies). Its unmasked score also
peaks at negative lags (-9, -21), so a plain RoPE shift of the query would land back on
lag 11; instead re-phase each frequency.

## Step 8 — edit design (edits.py `retarget_lag`, s08_check_retarget.py, 0 passes)
Closed form: for each frequency pair f of L5h1's query rows, rotate the 2 weight rows
(dims f, f+32) by psi_f = (ang(mean k_f) - w_f D) - ang(mean q_f), so every frequency's
positional score is |q_f||k_f| cos(w_f (lag - D)): a single sharp peak at lag D.
Means are activation averages over the captured layer-5 inputs (both cycles).
Offline (exact, layer-5 inputs do not change): mass on lag D = 0.99 for context and
decoder queries, for D in {11, 40, 43, 48}.

## Step 9 — evaluations (s09_eval_retarget.py)
| edit | near80 | far80 | err80 | near180 | far180 | err180 | passes total |
|---|---|---|---|---|---|---|---|
| baseline | 0.696 | -0.254 | 1.496 | 0.955 | -0.027 | 1.227 | 4 |
| L5h1 lag 43 | 0.878 | 0.748 | 0.874 | 0.955 | 0.896 | 0.842 | 10 |
| L5h1 lag 11 (re-phased, control) | 0.759 | -0.259 | 1.482 | 0.956 | -0.025 | 1.225 | 14 |
| L5h1 lag 32 | 0.894 | 0.706 | 0.898 | 0.950 | 0.828 | 0.864 | 18 |
| L5h1 lag 64 | 0.734 | 0.445 | 1.092 | 0.932 | 0.884 | 0.854 | 22 |

Re-phasing alone (D=11) changes nothing: the lag is the cause. Moving the lag past the
32-patch block (D >= 32) restores far dips in both cycles; D=64 is weaker on cycle 80.

## Step 10-11 — keeping the original behaviour for short contexts (s10, s11, 0 passes)
Risk of a lag-43 head on unseen data: with a context shorter than 43 patches (1376 steps)
the lag-43 key does not exist, and the edited head would no longer read lag 11 as the
published model does. `design_lags` (edits.py) generalises the edit: the positional score
becomes a*S(lag-43) + b*S(lag-11), S(x)=sum_f |q_f||k_f| cos(w_f x), by multiplying each
query frequency pair by a complex number. Because the encoder is causal, context queries
at positions 12-29 of the captured run see exactly what a short context would see, so the
fallback can be checked offline. Grid over a in {1,1.5,2}, b in {0,.5,.75,1}, D in {32,43}
(attention masses only, no scores). Chosen a=1.5, b=0.5, D=43: late context on lag 43
1.00, decoders on lag 43 0.99, early context (p<43) on lag 11 0.93 (published model 0.85).

## Step 12 — evaluate the dual design (s12_eval_design.py, 4 passes; total 26)
L5h1 43:1.5 + 11:0.5 | cycle 80 near 0.879 far 0.747 err 0.873 | cycle 180 near 0.956 far 0.888 err 0.843.
Same as the single-peak lag 43.

## Step 13 — mechanism check after the edit (s13_capture_edit.py, 2 passes; total 28)
In-phase attention / chance per decoder token, before -> after: L6h1, L6h2, L7h0, L7h2 now
stay 2-8x above chance for all 32 decoder tokens (before: chance from j~16). Confirms the
lag-11 induction hypothesis.

## Step 14 — final checkpoint (s14_final.py, 4 passes; total 32)
Saved final_checkpoint/model.safetensors (full state dict), config.json (copy of published),
patch (results/patch_final.safetensors, copied into final_checkpoint/). Reloaded from disk:
identical weights; Bench.score: cycle 80 near 0.879 far 0.747 err 0.873 | cycle 180 near
0.956 far 0.888 err 0.843. 1 tensor, 6144 weights changed (0.118 %).

## Step 15 — reference statistics saved (s15_save_ref_means.py, 0 passes)
results/l5h1_ref_means.json: the per-frequency mean query/key components used by the edit.

Passes used: 32 of 200. Stopped: the mechanism is identified and fixed; further tuning on
two cycles would mostly fit the diagnostic set.
