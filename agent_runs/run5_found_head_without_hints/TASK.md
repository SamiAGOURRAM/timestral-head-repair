# Task: edit the weights of Timestral-5M, without training, so that it keeps a repeating pattern further ahead

You are the agent in a weight-space programming experiment. You read and change the
weights of a neural network directly. You do not train it.

## The model

Timestral-5M is a time-series forecasting model by Mistral (5.2 million weights, 8 layers,
width 192, 3 attention heads per layer). It cuts a series into patches of 32 steps, and
forecasts up to 32 future patches (1024 steps) in one pass, as 21 quantiles per step. Its
inference code is the Python package `timestral`, installed in the environment given
below; you may read that package's source. The published weights are downloaded by the
harness from the Hugging Face repository `mistralai/Timestral-5M` (already in the local
cache).

## The problem

On series made of a constant level, noise, and one dip per cycle, Timestral-5M forecasting
1024 steps in one pass stops placing the dip beyond about 500 steps ahead, when the cycle
length is not a whole number of 32-step patches. With a cycle of 96 steps (3 patches) it
has no such limit.

Nothing is known about where in the network this happens: finding that is part of the task.

## Your goal

Change the weights of Timestral-5M so that, forecasting in one pass with the forecaster's
default settings, the share of FAR dips it places correctly goes up, without losing the
NEAR dips. "Far" = the dip starts 512 steps or more after the forecast origin; "near" =
under 384 steps.

The diagnostic set you work on: cycles of 80 and 180 steps, 64 series each (fixed seeds),
8192 steps of history, 1024-step horizon. `Bench.score(model, label)` in `harness.py`
returns, per cycle: `near` and `far` (share of each dip that the median forecast shows at
the right place; 1 = full depth at the right place, 0 = no dip) and `err` (mean absolute
error in noise standard deviations).

Unedited model: cycle 80, near 0.696, far -0.254, err 1.496; cycle 180, near 0.955,
far -0.027, err 1.227.

Your edit will afterwards be scored by someone else on data you do not have: other series,
other cycle lengths, another pattern shape, a real dataset and a general forecasting
benchmark. It counts as a success only if, there, far dips go up by at least 0.20, near
dips go down by no more than 0.05, and forecasting in general does not get worse. An edit
that only fits the two diagnostic cycles is worth nothing.

## Rules

1. No gradient, no optimizer, no training step, no backward pass, no automatic
   differentiation of any kind. Closed-form computations (least squares, averages of
   activations, scaling, low-rank updates) are allowed; say which ones you used.
2. Reading and analysing the weights with Python is free and unlimited.
3. Running the model is limited: 200 passes in total. One pass = one batch of at most 64
   series through the model. `Bench.score` costs 4 passes (2 cycles, each run on the
   series and on its mirror image). Every run of the model MUST go through `Bench.score`
   or `Bench.forecast`, or be logged by you to `results/passes.jsonl` in the same format
   if you need hooks around a direct call (1 pass per batch of at most 64 series). Do not
   run the model in any uncounted way. When the budget is used up, stop.
4. Run the model only on the diagnostic series (`bench.sets[80]`, `bench.sets[180]`). Do
   not generate other series to run the model on.
5. Work only inside this folder:
   `<agent folder>`
   You may also read the `timestral` package source at
   `<project>\.venv\Lib\site-packages\timestral`.
   Do NOT read, list or search anything else under `<project>`
   (it contains another run's notes and results, and reading them would void this run),
   nor any other project folder, nor the memory or notes of other sessions.
6. No web access. Do not use other language models or other agents.
7. Only the 5M model. Do not load other Timestral sizes.

## Environment

- Python: `<project>\.venv\Scripts\python.exe` (PyTorch, numpy,
  pandas, safetensors, timestral installed). Run scripts from this folder.
- Set the environment variable `WSP_DEVICE=cpu`: the GPU is in use by another job. The
  model is small; a `Bench.score` call takes a few seconds on the CPU.
- Shell: Windows. Write every script you run as a file in this folder (do not pipe code
  through the shell), so that what you ran can be read afterwards.
- `harness.py`: `Bench()` loads the model and the diagnostic sets; `bench.fresh()` gives a
  copy of the published model to edit; `bench.score(model, label)`; `bench.forecast(...)`;
  `bench.save_patch(model, name)`; helpers `time_attention(model, layer)` and
  `qkv_rows(attn, part, head)`. `check3_synthetic_lead_time.py` holds the series generator
  and the dip measure.

## What to deliver, in this folder

1. `journal.md`: kept as you go. For each step: what you looked at or tried, why, the
   result, passes used so far.
2. Every script you ran.
3. `final_checkpoint/` with `model.safetensors` (full state dict of your final edited
   model) and `config.json` (copy of the published one), and the patch saved by
   `bench.save_patch(model, "final")`. If no edit helped, deliver the unedited model and
   say so.
4. `REPORT.md`: what you understood about the mechanism, each edit tried with its
   diagnostic scores, your final edit described precisely (which tensors, what formula),
   the closed-form steps used, passes used, and an honest statement of whether you expect
   it to hold on data you have not seen.

A negative result, clearly reported, is an acceptable outcome. Do not claim more than the
scores show.
