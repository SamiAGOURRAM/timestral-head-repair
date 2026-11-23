"""Test A: does an edited head 5.1 do its damage through the history tokens or the future tokens?

Inference runs the sequence [history tokens ; future tokens] (`TwoStreamMask`, split at
`enc_len`). A wrapper around layer 5's time attention computes the published and the edited
attention and keeps, per token, one or the other:
  - "full": edited everywhere (the plain weight edit);
  - "future only": edited on future tokens, published on history tokens;
  - "history only": edited on history tokens, published on future tokens.
Only head 5.1's query rows differ between the two, so this is head 5.1's edit applied per
token. A diagnostic, not a weight edit.

Edits: lags 32, 33 and 43 (`checkpoints/lagsweep_<lag>`). Scored on the synthetic sets of
`baseline_finetune.py` (far / near dips) and on the lag sweep's seven GIFT-Eval
configurations (MASE ratio to the published model run the same way).

Expectation (written before running): far dips need the edit on the future tokens, so
"future only" should keep most of the synthetic gain. If the GIFT-Eval damage of lags 33 and
43 comes from re-reading the history, "history only" carries it and "future only" is safe;
if it comes from the future tokens, the reverse.

Run: .venv\\Scripts\\python.exe -u experiments\\edit_5m\\split_history_future.py
"""

import copy
import json
import sys

import numpy as np
import pandas as pd
import torch

import baseline_finetune as bf

sys.path.insert(0, str(bf.ROOT / "evaluation"))
import gift_eval_run as ge  # noqa: E402
from timestral import TimestralForecaster, load_model  # noqa: E402
from timestral.layers import TwoStreamMask  # noqa: E402

LAGS = (32, 33, 43)
MODES = ("full", "future only", "history only")
CONFIGS = ["solar/H/long", "solar/H/medium", "electricity/H/long", "m_dense/H/long", "ett1/H/long",
           "loop_seattle/H/long", "us_births/D/short"]


class Split(torch.nn.Module):
    def __init__(self, orig, edited, mode: str) -> None:
        super().__init__()
        self.orig, self.edited, self.mode = orig, edited, mode
        self.is_time = True

    def forward(self, x, mask, cos, sin):
        a = self.orig(x, mask, cos, sin)
        if self.mode == "published":
            return a
        b = self.edited(x, mask, cos, sin)
        if self.mode == "full":
            return b
        assert isinstance(mask, TwoStreamMask), "expected the [history ; future] layout"
        n = mask.enc_len
        if self.mode == "future only":
            return torch.cat([a[:, :n], b[:, n:]], dim=1)
        return torch.cat([b[:, :n], a[:, n:]], dim=1)


def build(lag: int | None, mode: str):
    m = load_model("mistralai/Timestral-5M", device=bf.DEVICE)
    orig = m.transformer.layers[5].attns[0]
    if lag is None:
        m.transformer.layers[5].attns[0] = Split(orig, None, "published")
        return m
    e = load_model(str(bf.ROOT / "checkpoints" / f"lagsweep_{lag}"), device=bf.DEVICE)
    edited = copy.deepcopy(e.transformer.layers[5].attns[0])
    m.transformer.layers[5].attns[0] = Split(orig, edited, mode)
    return m


def gift(model) -> pd.Series:
    fc = TimestralForecaster(model)
    props = json.load(open(ge.GIFT_REPO / "notebooks" / "dataset_properties.json"))
    out = {}
    for name, term, key in ge.all_configs(props):
        if key in CONFIGS:
            with torch.inference_mode():
                out[key] = ge.evaluate_config(fc, name, term)[ge.MASE_COL]
    return pd.Series(out)[CONFIGS]


def main() -> None:
    pd.set_option("display.width", 220)
    base = build(None, "published")
    syn0 = bf.evaluate(base)
    ref = gift(base)
    print("published:", ref.round(4).to_dict(), flush=True)
    rows = []
    for lag in LAGS:
        for mode in MODES:
            m = build(lag, mode)
            syn = bf.evaluate(m)
            r = gift(m) / ref
            rows.append({"lag": lag, "mode": mode, "far dips": float(syn["far dips"].mean()),
                         "near change": float((syn["near dips"] - syn0["near dips"]).mean()),
                         "gift geo mean": float(np.exp(np.log(r).mean())), "gift worst": float(r.max()), **r.round(4).to_dict()})
            print(f"lag {lag}, {mode}: far {rows[-1]['far dips']:.3f}, gift {rows[-1]['gift geo mean']:.3f}", flush=True)
            del m
            torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(bf.ROOT / "results" / "edit_5m" / "split_history_future.csv", index=False)
    print(f"\npublished synthetic far dips {syn0['far dips'].mean():.3f}\n")
    print(df[["lag", "mode", "far dips", "near change", "gift geo mean", "gift worst"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
