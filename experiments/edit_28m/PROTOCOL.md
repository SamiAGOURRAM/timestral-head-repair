# Does the fixed-lag repair carry over to Timestral-28M? Protocol

Written on 2026-11-17, before any 28M edit was scored on the data below.

## Question

Blind run 2 fixed Timestral-5M's far forecasts by making one fixed-lag head (5.1, 11
patches back) read 32 patches back. An exploratory check (`experiments/edit_5m/check_28m_fixed_lag.py`)
found 21 fixed-lag heads in 28M, and that the blind run 1 recipe (rotate a head's query to
read 60 back, query x 1.5) applied to head 6.0 or 4.1 restores far dips with 8192 steps of
history and hurts with 2048. This protocol asks whether a rule fixed in advance, applied
to 28M, yields an edit that passes the same four criteria as edit B did for 5M.

This is not a blind agent run: this session chose the rule and has seen the exploratory
28M scores. It tests whether the recipe transfers, not whether an agent can find it.

## Edit family and selection rule

- Candidate heads: 28M time-attention heads with more than half of their attention on one
  lag between 2 and 12 patches, measured as in the exploratory check (16 diagnostic
  series, cycle 180, 8192 steps of history).
- Edit: rotate the rotary pairs of the head's query so it reads D patches back, D in
  {32, 60}, then multiply the query by g in {1.0, 1.5}. Rotation and scaling only; no
  gradient.
- Selection on the diagnostic sets only (cycles 80 and 180, 64 series each, seeds 80 and
  180), histories 1024, 2048, 4096 and 8192. A candidate is admissible when, against
  unedited 28M at each of these histories, near dips are not lower by more than 0.05 on
  either cycle and the error is not higher by more than 0.05. Among admissible
  candidates, the one with the largest mean gain in far dips at 8192 is selected. If none
  is admissible, the result is negative and is reported as such.

## Held-back evaluation, run once on the selected edit

Fresh data used nowhere else in the project:
- dips, cycles 66, 90, 130, 175, 270 (not whole numbers of patches) and 192, 224 (whole
  patches), seeds 5000 + cycle;
- spikes on a sinusoid, cycles 130, 270, seeds 5500 + cycle;
- histories 768, 1024, 1536, 2048, 3072, 4096 and 8192;
- real data: `bizitobs_service`, 21 series x 10 start points, 600-step horizon;
- GIFT-Eval, all 97 configurations, against the published Timestral-28M results.

The same fresh synthetic and real sets are run on unedited 5M and edit B, as a third
held-back check of edit B.

## Success criteria (as for 5M)

Against unedited 28M, one pass:
1. far dips, misaligned cycles, 8192 steps of history: up by at least 0.20;
2. near dips, same sets: down by no more than 0.05;
3. `bizitobs_service` MASE: not higher;
4. GIFT-Eval: geometric mean of the MASE ratio and of the CRPS ratio at most 1.01.

## Changes after the first held-back run

None.
