"""Draw every figure of the README from the stored results (no model runs).

Inputs: the CSV / JSON / NPZ files under `results/`. A few numbers that exist only in the
logs of the original runs are written here with their source.
Outputs: `figures/*.png`.

Run: python make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent
R, F = ROOT / "results", ROOT / "figures"
F.mkdir(exist_ok=True)

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
PUB, EDB, SAME, FULL, CHUNK = "#898781", "#eb6834", "#1baf7a", "#2a78d6", "#6250d6"
SIZE_COL = {"5M": "#eb6834", "28M": "#2a78d6", "123M": "#1baf7a", "209M": "#6250d6"}
GOOD, BAD = "#0ca30c", "#d03b3b"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 2,
})


def save(fig, name: str) -> None:
    fig.savefig(F / f"{name}.png")
    plt.close(fig)
    print("wrote figures/" + name + ".png")


def gmean(s) -> float:
    return float(np.exp(np.log(np.asarray(s, dtype=float)).mean()))


# ---------------------------------------------------------------- 1. architecture
def architecture() -> None:
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 64); ax.axis("off"); ax.grid(False)
    # input series
    t = np.linspace(0, 1, 300)
    y = 54 + np.where((t * 6) % 1 < 0.12, -4, 0) + 0.4 * np.sin(t * 90)
    ax.plot(3 + t * 16, y, color=FULL, lw=1.5)
    ax.text(3, 61, "Input series", fontsize=10, weight="bold", color=INK)
    ax.text(3, 47.5, "a dip every N steps", fontsize=8, color=MUTED)
    ax.text(3, 41, "Patches of 32 steps", fontsize=10, weight="bold", color=INK)
    for i in range(8):
        ax.add_patch(Rectangle((3 + i * 2, 36), 1.6, 3, fc="#cde2fb", ec=FULL, lw=0.8))
    ax.text(3, 33.5, "each → a token of 192 numbers", fontsize=8, color=MUTED)
    ax.text(3, 27, "+ 32 future tokens", fontsize=10, weight="bold", color=INK)
    for i in range(8):
        ax.add_patch(Rectangle((3 + i * 2, 22), 1.6, 3, fc="white", ec=MUTED, lw=0.8, ls="--"))
    ax.text(3, 19.5, "1,024 steps to forecast", fontsize=8, color=MUTED)
    ax.add_patch(FancyArrowPatch((21, 31), (26, 31), arrowstyle="-|>", mutation_scale=14, color="#c3c2b7", lw=1.5))
    # blocks
    bx, bw, bh, gap, top = 27, 50, 6.0, 1.0, 62
    for b in range(8):
        y0 = top - (b + 1) * bh - b * gap
        hot = b == 5
        ax.add_patch(FancyBboxPatch((bx, y0), bw, bh, boxstyle="round,pad=0,rounding_size=0.8",
                                    fc="#fde7da" if hot else "white", ec=EDB if hot else GRID, lw=1.4 if hot else 1))
        ax.text(bx + 1.5, y0 + bh / 2, f"Layer {b}", va="center", fontsize=9, weight="bold", color=INK)
        ax.text(bx + 9, y0 + bh / 2, "time attention", va="center", fontsize=8, color=MUTED)
        for h in range(3):
            cx = bx + 25 + h * 5
            ax.add_patch(Circle((cx, y0 + bh / 2), 1.8, fc="white", ec="#c3c2b7", lw=1))
            ax.text(cx, y0 + bh / 2, f"{b}.{h}", ha="center", va="center", fontsize=6.5, color=INK2)
            if f"{b}.{h}" in ("6.1", "6.2", "7.2"):
                ax.add_patch(Circle((cx, y0 + bh / 2), 2.4, fc="none", ec=CHUNK, lw=1.2, ls="--"))
            if b == 5 and h == 1:
                ax.add_patch(Circle((cx, y0 + bh / 2), 2.7, fc="none", ec=EDB, lw=2.6))
        ax.add_patch(Rectangle((bx + 41, y0 + 1.3), 7, bh - 2.6, fc="#f0efec", ec="none"))
        ax.text(bx + 44.5, y0 + bh / 2, "MLP", ha="center", va="center", fontsize=7.5, color=INK2)
    y5 = top - 6 * bh - 5 * gap + bh / 2
    ax.annotate("head 5.1\nalways looks 11 patches back\n→ the cause of the failure", xy=(bx + 30 + 2.7, y5), xytext=(bx + bw + 3, y5 + 4),
                fontsize=8.5, color=EDB, weight="bold", arrowprops=dict(arrowstyle="-", color=EDB, lw=1.2))
    y6 = top - 7 * bh - 6 * gap + bh / 2
    ax.annotate("heads 6.1, 6.2, 7.2\nfind the matching place in\nthe history and copy the pattern", xy=(bx + 35 + 2.4, y6), xytext=(bx + bw + 3, y6 - 9),
                fontsize=8.5, color=CHUNK, arrowprops=dict(arrowstyle="-", color=CHUNK, lw=1))
    ax.text(bx + bw / 2, 1.5, "Output: 21 quantiles for each of the 32 future patches", ha="center", fontsize=9, color=INK2, weight="bold")
    ax.set_title("Timestral-5M: 8 layers × 3 attention heads (layers 3 and 7 also attend across variables)", loc="left")
    save(fig, "01_architecture")


# ---------------------------------------------------------------- 2. failure, example
def example_synthetic() -> None:
    d = np.load(R / "examples.npz")
    h, truth = d["syn_history"], d["syn_truth"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, sharey=True)
    xh, xf = np.arange(-len(h), 0), np.arange(len(truth))
    for ax, name, col in ((axes[0], "published", PUB), (axes[1], "edit B", EDB)):
        ax.axvspan(512, len(truth), color="#fde7da" if name == "edit B" else "#f0efec", lw=0, zorder=0)
        ax.plot(xh, h, color="#b7b6ae", lw=0.8)
        ax.plot(xf, truth, color="#b7b6ae", lw=0.8)
        ax.fill_between(xf, d[f"syn_{name}_q10"], d[f"syn_{name}_q90"], color=col, alpha=0.18, lw=0)
        ax.plot(xf, d[f"syn_{name}_median"], color=col, lw=1.8)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_title(("Published Timestral-5M" if name == "published" else "Head edit (6,144 weights changed)"), loc="left")
        ax.text(530, ax.get_ylim()[0] if False else np.nanmin(truth) - 0.2, "512+ steps ahead", fontsize=8, color=INK2)
    axes[1].set_xlabel("steps (0 = forecast start)")
    for ax in axes:
        ax.set_ylabel("value")
    fig.suptitle("A dip every 150 steps: the published model stops drawing it about 500 steps ahead", x=0.01, ha="left", weight="bold")
    fig.text(0.01, -0.01, "Grey: the series (history and true future). Line: median forecast. Band: 10-90 % range. Synthetic held-back series, 8,192 steps of history.", fontsize=8, color=MUTED)
    save(fig, "02_failure_example")


def lead_time() -> None:
    df = pd.read_csv(R / "gap_10s" / "check3_synthetic.csv")
    df = df[df.period == 180]
    df["bin"] = (df["lead"] // 128) * 128
    t = df.groupby(["model", "bin"])["recovered"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.axvspan(512, 1024, color="#f0efec", lw=0, zorder=0)
    for m in ("5M", "28M", "123M"):
        ax.plot(t.index + 64, t[m], marker="o", ms=4, color=SIZE_COL[m], label=f"Timestral-{m}", lw=2.6 if m == "5M" else 1.8)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_xlabel("how far ahead the dip is (steps)")
    ax.set_ylabel("share of the dip drawn")
    ax.set_ylim(-0.2, 1.08)
    ax.set_title("Share of each dip the forecast draws, by distance ahead (cycle of 180 steps)", loc="left")
    ax.legend(loc="lower left")
    save(fig, "03_lead_time")


# ---------------------------------------------------------------- 3. locating the head
def head_lags() -> None:
    df = pd.read_csv(R / "scale" / "lag_heads_by_size.csv")
    df = df[df["size"] == "5M"].copy()
    df["head"] = df["head"].map(lambda v: f"{float(v):.1f}")
    fig, ax = plt.subplots(figsize=(10, 3.6))
    cols = [EDB if h == "5.1" else ("#c3c2b7" if s < 0.5 else FULL) for h, s in zip(df["head"], df["share"])]
    ax.bar(df["head"], df["share"], color=cols, width=0.7)
    for i, (h, lag, s) in enumerate(zip(df["head"], df["lag"], df["share"])):
        ax.text(i, s + 0.02, f"{lag}", ha="center", fontsize=7.5, color=INK if h == "5.1" else INK2, weight="bold" if h == "5.1" else None)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("share of attention on its\nfavourite distance")
    ax.set_xlabel("attention head (layer.head)   —   number above each bar: the distance back, in patches")
    ax.set_title("Where each head looks: head 5.1 puts 91 % of its attention exactly 11 patches back", loc="left")
    plt.setp(ax.get_xticklabels(), fontsize=8)
    save(fig, "04_head_distances")


def lock_heatmap() -> None:
    df = pd.read_csv(R / "gap_10s" / "check5_attention_lock.csv")
    df = df[df.model == "5M"].copy()
    df["head"] = df["layer"].astype(str) + "." + df["head"].astype(str)
    t = df.pivot_table(index="head", columns="k", values="lock")
    t = t.loc[[h for h in t.index if int(h.split(".")[0]) >= 3]]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.grid(False)
    im = ax.imshow(t.values, aspect="auto", cmap="Blues", vmin=0, vmax=9)
    ax.set_yticks(range(len(t.index)), t.index, fontsize=8)
    ks = list(t.columns)
    ax.set_xticks(range(len(ks)), ks, fontsize=8)
    ax.set_xlabel("future patch being forecast (1 patch = 32 steps)")
    ax.set_ylabel("head (layers 3-7)")
    for lab in ax.get_yticklabels():
        if lab.get_text() in ("5.1", "6.1", "6.2", "7.0", "7.1", "7.2"):
            lab.set_color(EDB); lab.set_fontweight("bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("cycle lock (1 = chance, 9 = max)", fontsize=8)
    ax.set_title("Which heads read the history at the right point of the cycle, and where they lose it", loc="left")
    save(fig, "05_cycle_lock")


def knockout_and_patching() -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.1), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(wspace=0.28)
    # knockout: results/gap_10s/check6.log, part 1 (near dips recovered, cycle 180)
    labels = ["nothing", "one other\nlocked head", "6 random\nheads", "head 5.1\nalone", "all 6 locked\nheads"]
    vals = [0.955, 0.89, 0.650, 0.426, 0.123]
    lo = [None, 0.847, 0.559, None, None]; hi = [None, 0.932, 0.738, None, None]
    cols = [PUB, "#c3c2b7", "#c3c2b7", EDB, INK2]
    a1.bar(range(5), vals, color=cols, width=0.65)
    for i, (v, l, h) in enumerate(zip(vals, lo, hi)):
        if l is not None:
            a1.plot([i, i], [l, h], color=INK2, lw=1.5)
        a1.text(i, (h or v) + 0.03, f"{v:.2f}" if l is None else f"{l:.2f}-{h:.2f}", ha="center", fontsize=8, color=INK2)
    a1.set_xticks(range(5), labels, fontsize=8)
    a1.set_ylim(0, 1.1)
    a1.set_ylabel("near dips still drawn")
    a1.set_title("Switching heads off: 5.1 matters most", loc="left")
    # patching: results/gap_10s/check7_patching_cycle80.csv
    p = pd.read_csv(R / "gap_10s" / "check7_patching_cycle80.csv", index_col=0)
    v = p["far dips recovered"].values
    labs = [str(i) for i in range(8)] + ["end"]
    cols2 = [EDB if i == 6 else (FULL if i < 8 else PUB) for i in range(9)]
    a2.bar(range(9), v, color=cols2, width=0.65)
    a2.axhline(0, color="#c3c2b7", lw=1)
    for i, x in enumerate(v):
        a2.text(i, x + (0.03 if x >= 0 else -0.08), f"{x:.2f}", ha="center", fontsize=7.5, color=INK2)
    a2.set_xticks(range(9), labs)
    a2.set_xlabel("layer where far patches receive a near patch's state")
    a2.set_ylabel("far dips drawn")
    a2.set_ylim(-0.35, 0.85)
    a2.set_title("Swapping states: the loss happens in layer 5", loc="left")
    a2.annotate("good state from layer 6 on\n→ far dips come back", xy=(6, 0.70), xytext=(1.2, 0.6), fontsize=8, color=EDB,
                arrowprops=dict(arrowstyle="-", color=EDB))
    save(fig, "06_knockout_and_swap")


def reach_diagram() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 3.6))
    NH, NF, j = 44, 32, 17
    for ax, lag, title in ((axes[0], 11, "Published head 5.1: looks 11 patches back"),
                           (axes[1], 32, "Head edit: looks 32 patches back")):
        ax.set_xlim(-1, NH + NF + 1); ax.set_ylim(-1.6, 3.4); ax.axis("off"); ax.grid(False)
        for i in range(NH + NF):
            fut = i >= NH
            ax.add_patch(Rectangle((i, 0), 0.85, 1, fc="white" if fut else "#cde2fb", ec=MUTED if fut else FULL, lw=0.6, ls="--" if fut else "-"))
        src, dst = NH + j, NH + j - lag
        ok = dst < NH
        ax.add_patch(Rectangle((src - 0.05, -0.05), 0.95, 1.1, fc="none", ec=INK, lw=1.8))
        ax.add_patch(Rectangle((dst - 0.05, -0.05), 0.95, 1.1, fc="#d8f3dc" if ok else "#fbdcdc", ec=GOOD if ok else BAD, lw=1.8))
        ax.add_patch(FancyArrowPatch((src + 0.42, 1.1), (dst + 0.42, 1.1), connectionstyle="arc3,rad=0.35", arrowstyle="-|>",
                                     mutation_scale=12, color=EDB, lw=1.6))
        ax.plot([NH - 0.08, NH - 0.08], [-0.3, 1.3], color=INK, lw=1.4)
        ax.text(0, -1.2, "history (real data)", fontsize=8, color=INK2)
        ax.text(NH, -1.2, "future patches 0-31 (no data yet)", fontsize=8, color=INK2)
        ax.text(-1, 3.0, title, fontsize=10, weight="bold", color=INK)
        ax.text(NH + NF + 1, 3.0, ("lands on a future patch: nothing to match" if not ok else "lands on real history"),
                fontsize=9, color=BAD if not ok else GOOD, ha="right", weight="bold")
    fig.suptitle(f"Where future patch {j} (≈ {j * 32} steps ahead) reads from", x=0.01, ha="left", fontsize=10, color=INK2, y=1.0)
    save(fig, "07_reach")


# ---------------------------------------------------------------- 4. comparison
def comparison() -> None:
    fb = pd.read_csv(R / "edit_5m" / "finetune_vs_b" / "heldback_synthetic.csv")
    sw = pd.read_csv(R / "edit_5m" / "sameweights_vs_b" / "heldback_synthetic.csv")
    allm = pd.concat([fb, sw[sw.model.str.contains("same")]])
    # The protocol's criteria use the dip sets whose cycle is not a whole number of patches
    # (cycles 70, 100, 150, 220, 300): that is where the failure lives.
    dips = allm[allm.group == "misaligned"]
    m = dips.groupby("model")[["near", "far"]].mean()
    base = m.loc["5M unedited"]
    names = {"5M edit B (blind run 2)": "Head edit\n(no gradients)", "5M fine-tune of the same 6 144 weights": "Same weights,\ngradients",
             "5M fine-tune, many cycle lengths": "Full\nfine-tune"}
    cols = [EDB, SAME, FULL]
    far = [m.loc[k, "far"] - base["far"] for k in names]
    near = [m.loc[k, "near"] - base["near"] for k in names]
    # bizitobs_service MASE ratio and 95 % interval: results/edit_5m/heldback_finetune_vs_b.log, heldback_sameweights.log
    biz = [0.8752, 0.8884, 1.2366]; biz_lo = [0.8373, 0.8487, 1.1782]; biz_hi = [0.9144, 0.9277, 1.3033]
    gift = [gmean(pd.read_csv(R / "gift_eval" / g / "against_published.csv")["mase_ratio"])
            for g in ("5M-blind-run2-final", "5M-finetune-sameweights", "5M-finetune-anycycle")]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.9))
    specs = [(far, "Distant dips drawn\n(gain; higher = better)", (0, 1), None, None),
             (near, "Near dips drawn\n(change; ≥ −0.05 passes)", (-0.1, 0.06), None, None),
             (biz, "Real server metrics\n(error ratio; < 1 = better)", (0.7, 1.35), 1.0, None),
             (gift, "97-task benchmark\n(error ratio; ≤ 1.01 passes)", (0.9, 1.18), 1.01, None)]
    x = np.arange(3)
    for ax, (vals, title, ylim, ref, note) in zip(axes, specs):
        ax.bar(x, vals, color=cols, width=0.62)
        if ref is not None:
            ax.axhline(ref, color=INK2, lw=1, ls="--")
        ax.axhline(0, color="#c3c2b7", lw=1)
        for i, v in enumerate(vals):
            ax.text(i, v + (ylim[1] - ylim[0]) * 0.02 if v >= 0 else v - (ylim[1] - ylim[0]) * 0.07, f"{v:+.2f}" if ylim[0] < 0 or title.startswith("Distant") else f"{v:.3f}",
                    ha="center", fontsize=8.5, color=INK)
        if vals is biz:
            ax.errorbar(x, biz, yerr=[np.array(biz) - biz_lo, np.array(biz_hi) - biz], fmt="none", ecolor=INK2, lw=1, capsize=3)
        ax.set_xticks(x, list(names.values()), fontsize=8)
        ax.set_ylim(*ylim)
        ax.set_title(title, loc="left", fontsize=9.5)
    fig.suptitle("Three ways to fix the failure, scored on the same unseen data", x=0.01, ha="left", weight="bold", y=1.06)
    fig.subplots_adjust(wspace=0.35)
    save(fig, "08_comparison")


def gift_domains() -> None:
    rows = {}
    for label, g in (("Head edit", "5M-blind-run2-final"), ("Same weights, gradients", "5M-finetune-sameweights"),
                     ("Full fine-tune", "5M-finetune-anycycle")):
        df = pd.read_csv(R / "gift_eval" / g / "against_published.csv")
        rows[label] = df.groupby("domain")["mase_ratio"].apply(gmean)
        rows[label]["All 97 tasks"] = gmean(df["mase_ratio"])
    t = pd.DataFrame(rows)
    order = ["All 97 tasks"] + [d for d in t.index if d != "All 97 tasks"]
    t = t.loc[order]
    counts = pd.read_csv(R / "gift_eval" / "5M-blind-run2-final" / "against_published.csv").domain.value_counts()
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    x = np.arange(len(t)); w = 0.26
    for i, (label, col) in enumerate((("Head edit", EDB), ("Same weights, gradients", SAME), ("Full fine-tune", FULL))):
        ax.bar(x + (i - 1) * w, t[label], width=w, color=col, label=label)
    ax.axhline(1.0, color=INK2, lw=1, ls="--")
    ax.set_xticks(x, [f"{d}\n({counts.get(d, 97)} tasks)" for d in t.index], fontsize=8)
    top = 1.6
    ax.set_ylim(0.85, top)
    for i, v in enumerate(t["Full fine-tune"]):
        ax.text(i + w, min(v, top - 0.03) + 0.01, f"{v:.2f}", ha="center", fontsize=7.5, color=FULL)
    ax.set_ylabel("error ratio vs published\n(below 1 = better)")
    ax.legend(loc="upper left", ncol=3)
    ax.set_title("GIFT-Eval by domain: the head edit is at most 1 % worse anywhere; the full fine-tune is worse everywhere", loc="left")
    save(fig, "09_benchmark_domains")


def real_far_near() -> None:
    df = pd.read_csv(R / "far_horizon_real.csv").sort_values("far ratio")
    df = df[df["cycle strength"] >= 0.5]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    y = np.arange(len(df))
    for i, (_, r) in enumerate(df.iterrows()):
        ax.plot([r["near ratio"], r["far ratio"]], [i, i], color="#d6d5cf", lw=2, zorder=1)
    ax.scatter(df["near ratio"], y, color=FULL, s=30, label="near part (0-383 steps ahead)", zorder=2)
    ax.scatter(df["far ratio"], y, color=EDB, s=38, label="distant part (512+ steps ahead)", zorder=3)
    ax.axvline(1.0, color=INK2, lw=1, ls="--")
    ax.set_yticks(y, df["config"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("error of the head edit ÷ error of the published model   (left of 1 = head edit better)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.set_title("Real series with a strong cycle: the head edit helps where the failure lives, far ahead", loc="left")
    save(fig, "10_real_far_vs_near")


def tide() -> None:
    d = np.load(R / "external_real.npz")
    k = "tide_9447130"
    i = int(d[f"{k}__typical"])
    h, t = d[f"{k}__history_tail"][i], d[f"{k}__truth"][i]
    mp, mb = d[f"{k}__published_median"][i], d[f"{k}__edit B_median"][i]
    pe, be = d[f"{k}__published_err"], d[f"{k}__edit B_err"]
    n = len(pe)
    better = int((be[:, 512:].mean(1) < pe[:, 512:].mean(1)).sum())
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.6), gridspec_kw={"height_ratios": [1.35, 1]})
    a1.axvspan(512, len(t), color="#fde7da", lw=0, zorder=0)
    a1.plot(np.arange(-len(h), 0), h, color="#b7b6ae", lw=1.1, label="real water level (history and true future)")
    a1.plot(np.arange(len(t)), t, color="#b7b6ae", lw=1.1)
    a1.plot(mp, color=PUB, lw=2, label="published")
    a1.plot(mb, color=EDB, lw=2.2, label="head edit")
    a1.axvline(0, color=INK2, lw=1)
    a1.set_ylabel("water level (m)")
    a1.set_xlabel("steps of 6 minutes (0 = forecast start; 240 steps = 1 day)")
    a1.legend(loc="lower left", ncol=3, fontsize=8.5)
    a1.set_title(f"A typical forecast (the median of {n} windows, not the best): from ≈ 2 days ahead the published model loses the tide",
                 loc="left", fontsize=9.5)
    bins = np.arange(0, pe.shape[1] + 1, 32)
    c = (bins[:-1] + bins[1:]) / 2
    a2.axvspan(512, pe.shape[1], color="#fde7da", lw=0, zorder=0)
    a2.plot(c, [pe[:, s:e].mean() for s, e in zip(bins[:-1], bins[1:])], color=PUB, lw=2, label="published")
    a2.plot(c, [be[:, s:e].mean() for s, e in zip(bins[:-1], bins[1:])], color=EDB, lw=2.4, label="head edit")
    ge28 = d[f"{k}__published 28M_err"]
    a2.plot(c, [ge28[:, s:e].mean() for s, e in zip(bins[:-1], bins[1:])], color=FULL, lw=1.6, ls="--", label="published 28M (5.4× bigger)")
    a2.set_ylabel("error (× series spread)")
    a2.set_xlabel("steps ahead (6 minutes each; shaded: distant part, 512+ steps)")
    a2.legend(loc="upper left", fontsize=8.5)
    a2.set_title(f"Error by distance ahead, all {n} windows: the head edit is better far ahead in {better} of them", loc="left", fontsize=10)
    fig.suptitle("Real ocean tides, NOAA gauge 9447130 (Seattle), outside the benchmark", x=0.01, ha="left", weight="bold")
    fig.subplots_adjust(hspace=0.45)
    save(fig, "15_real_tide")


def external_overview() -> None:
    s = pd.read_csv(R / "external_real_summary.csv")
    s = s.iloc[::-1].reset_index(drop=True)
    col = {"tide": EDB, "solar": SAME, "taxi": PUB}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True, gridspec_kw={"width_ratios": [1, 1.25]})
    y = np.arange(len(s))
    a1.barh(y, s["failure (published far / near)"], color=[col[k] for k in s["kind"]], height=0.6)
    a1.axvline(1, color=INK2, lw=1, ls="--")
    for i, v in enumerate(s["failure (published far / near)"]):
        a1.text(v + 0.06, i, f"{v:.1f}×", va="center", fontsize=8.5, color=INK2)
    a1.set_yticks(y, s["label"], fontsize=8.5)
    a1.set_xlim(0, 6)
    a1.set_xlabel("published model: distant-part error ÷ near-part error")
    a1.set_title("How strong is the failure?\n(1 = none; higher = loses the pattern far ahead)", loc="left", fontsize=10)
    far, near = s["edit B / published, far"], s["edit B / published, near"]
    a2.barh(y, far, color=[col[k] for k in s["kind"]], height=0.6)
    a2.scatter(near, y, color=INK, s=22, zorder=3)
    a2.axvline(1, color=INK2, lw=1, ls="--")
    for i, (v, nb, n) in enumerate(zip(far, s["edit B far better in"], s["windows"])):
        txt = "no change" if abs(v - 1) < 0.01 else (f"{(1 - v) * 100:.0f} % less" if v < 1 else f"{(v - 1) * 100:.0f} % more")
        a2.text(v + 0.02, i + 0.22, f"{txt} · better in {nb}/{n}", va="center", fontsize=7.8, color=INK)
    a2.set_xlim(0, 1.5)
    a2.set_xlabel("head edit's error ÷ published model's error  (below 1 = head edit better)")
    a2.set_title("Head edit against the published model\n(bars: distant part; dots: near part)", loc="left", fontsize=10)
    fig.suptitle("Real data outside the benchmark: wherever the failure is present, the head edit removes most of it", x=0.01, ha="left", weight="bold", y=1.07)
    save(fig, "16_external_overview")


def tides_vs_bigger() -> None:
    s = pd.read_csv(R / "external_real_summary.csv")
    s = s[s.kind == "tide"].reset_index(drop=True)
    models = [("published far", "published 5M", PUB), ("edit B far", "head edit (5M)", EDB),
              ("published 28M far", "published 28M", FULL), ("published 123M far", "published 123M", SAME)]
    near = [("published near", PUB), ("edit B near", EDB), ("published 28M near", FULL), ("published 123M near", SAME)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    x = np.arange(len(s)); w = 0.2
    for j, (col, label, c) in enumerate(models):
        a1.bar(x + (j - 1.5) * w, s[col], width=w, color=c, label=label)
    for i, (nb, n) in enumerate(zip(s["edit B far better than 28M in"], s["windows"])):
        a1.text(i, max(s.loc[i, [m[0] for m in models]]) + 0.03, f"head edit beats 28M\nin {nb}/{n}", ha="center", fontsize=8, color=INK2)
    a1.set_xticks(x, [l.replace("Tide, ", "") for l in s["label"]])
    a1.set_ylim(0, 1.08)
    a1.set_ylabel("error (× series spread)")
    a1.set_title("Distant part (512+ steps, ≈ 2-4 days ahead)", loc="left", fontsize=10)
    fig.legend(*a1.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=4, fontsize=9)
    for j, (col, c) in enumerate(near):
        a2.bar(x + (j - 1.5) * w, s[col], width=w, color=c)
    a2.set_xticks(x, [l.replace("Tide, ", "") for l in s["label"]])
    a2.set_title("Near part (0-383 steps, ≈ first 1.5 days)", loc="left", fontsize=10)
    fig.suptitle("Real tides: far ahead, edited 5M matches the 5.4× larger published 28M; near ahead, 28M stays better",
                 x=0.01, ha="left", weight="bold")
    save(fig, "18_tides_vs_bigger_models")


def external_examples() -> None:
    d = np.load(R / "external_real.npz")
    s = pd.read_csv(R / "external_real_summary.csv").set_index("key")
    picks = [("tide_8443970", "6 minutes"), ("tide_1612340", "6 minutes"),
             (next(k for k in s.index if k.startswith("solar_")), "10 minutes"), ("taxi", "30 minutes")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.6))
    for ax, (k, step) in zip(axes.ravel(), picks):
        i = int(d[f"{k}__typical"])
        h, t = d[f"{k}__history_tail"][i][-512:], d[f"{k}__truth"][i]
        ax.axvspan(512, len(t), color="#fde7da", lw=0, zorder=0)
        ax.plot(np.arange(-len(h), 0), h, color="#b7b6ae", lw=0.9)
        ax.plot(np.arange(len(t)), t, color="#b7b6ae", lw=0.9, label="real series")
        ax.plot(d[f"{k}__published_median"][i], color=PUB, lw=1.7, label="published")
        ax.plot(d[f"{k}__edit B_median"][i], color=EDB, lw=1.9, label="head edit")
        ax.axvline(0, color=INK2, lw=1)
        r = s.loc[k]
        ax.set_title(f"{r['label']} (failure {r['failure (published far / near)']:.1f}×)", loc="left", fontsize=9.5)
        ax.set_xlabel(f"steps of {step} (0 = forecast start; shaded: 512+ steps ahead)", fontsize=8.5)
        if k == "taxi":  # 1,024 half-hours are too dense to read: zoom on the start of the distant part
            ax.set_xlim(-60, 760)
            ax.set_xlabel("steps of 30 minutes (0 = forecast start; shaded: 512+ steps ahead; zoomed)", fontsize=8.5)
    axes[0, 0].legend(loc="lower left", fontsize=8, ncol=3)
    fig.suptitle("Typical forecasts (median improvement, not the best): tides lose the pattern without the edit; solar and taxi don't, and the two lines overlap",
                 x=0.01, ha="left", weight="bold", fontsize=10.5)
    fig.subplots_adjust(hspace=0.45, wspace=0.15)
    save(fig, "17_external_examples")


def real_error_by_distance() -> None:
    d = np.load(R / "examples.npz")
    fig, axes = plt.subplots(2, 2, figsize=(12, 5.8), sharex="col", gridspec_kw={"height_ratios": [1.2, 1]})
    for col, (term, label) in enumerate((("medium", "600-step forecasts"), ("long", "900-step forecasts"))):
        a, b = d[f"dist_{term}_published_all"], d[f"dist_{term}_edit B_all"]
        n, H = a.shape
        edges = np.arange(0, H + 1, 32)
        edges[-1] = H
        centers = (edges[:-1] + edges[1:]) / 2
        pa = np.array([np.nanmean(a[:, s:e], axis=1) for s, e in zip(edges[:-1], edges[1:])]).T
        pb = np.array([np.nanmean(b[:, s:e], axis=1) for s, e in zip(edges[:-1], edges[1:])]).T
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(2000):
            idx = rng.integers(0, n, n)
            boot.append(pb[idx].mean(0) / pa[idx].mean(0))
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        a1, a2 = axes[0, col], axes[1, col]
        for ax in (a1, a2):
            ax.axvspan(512, H, color="#fde7da", lw=0, zorder=0)
        a1.plot(centers, pa.mean(0), color=PUB, lw=2, label="published")
        a1.plot(centers, pb.mean(0), color=EDB, lw=2.4, label="head edit")
        a1.set_title(f"{label}, all {n} windows", loc="left", fontsize=10)
        a2.fill_between(centers, lo, hi, color=EDB, alpha=0.2, lw=0)
        a2.plot(centers, pb.mean(0) / pa.mean(0), color=EDB, lw=2)
        a2.axhline(1, color=INK2, lw=1, ls="--")
        a2.set_ylim(0.3, 1.6)
        r = pb.mean(0) / pa.mean(0)
        print(f"  {term}: largest ratio {r.max():.2f} at step {centers[r.argmax()]:.0f}; smallest {r.min():.2f} at {centers[r.argmin()]:.0f}")
        a2.set_xlabel("steps ahead (10 s each; 360 = 1 hour)")
        far = edges[:-1] >= 512
        print(f"  {term}: distant-part ratio {pb[:, far].mean() / pa[:, far].mean():.3f}, near (0-383) {pb[:, :12].mean() / pa[:, :12].mean():.3f}")
    axes[0, 0].set_ylabel("forecast error\n(× series spread)")
    axes[1, 0].set_ylabel("head edit ÷ published\n(band: 95 % interval)")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Real server metrics (bizitobs, 10-second data): error by distance ahead, every window averaged", x=0.01, ha="left", weight="bold")
    save(fig, "11_real_error_by_distance")


# ---------------------------------------------------------------- 5. why 32
def lag_sweep() -> None:
    syn = pd.read_csv(R / "edit_5m" / "lag_sweep_synthetic.csv", index_col=0)
    gift = pd.read_csv(R / "lag_sweep_gift.csv", index_col=0)
    lags = [11, 21, 28, 30, 31, 32, 33, 34, 36, 40, 42, 43]
    s = [syn.loc[f"lag {l}", "far"] for l in lags]
    g = [gift.loc[f"lag {l}", "geo mean"] for l in lags]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True)
    x = np.arange(len(lags))
    a1.bar(x, s, color=[PUB if l == 11 else (EDB if l == 32 else FULL) for l in lags], width=0.62)
    a1.axhline(0, color="#c3c2b7", lw=1)
    a1.set_ylabel("distant dips drawn")
    a1.set_title("Synthetic series: almost every reach from 28 to 43 repairs the dips", loc="left")
    a2.bar(x, g, color=[PUB if l == 11 else (GOOD if l == 32 else BAD) for l in lags], width=0.62)
    a2.axhline(1.0, color=INK2, lw=1, ls="--")
    for i, v in enumerate(g):
        a2.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color=INK2)
    a2.set_ylim(0.8, 1.8)
    a2.set_ylabel("error ratio on 7 real\nhourly / daily tasks")
    a2.set_title("Real data: only 32 is safe", loc="left")
    a2.set_xticks(x, [str(l) for l in lags])
    a2.set_xlabel("how many patches back head 5.1 is made to look (published: 11)")
    save(fig, "12_reach_sweep")


def history_cost() -> None:
    a = pd.read_csv(R / "edit_5m" / "finetune_vs_b" / "history_length.csv")
    b = pd.read_csv(R / "edit_5m" / "sameweights_vs_b" / "history_length.csv")
    df = pd.concat([a, b[b.model.str.contains("same")]])
    t = df.groupby(["history", "model"])["near"].mean().unstack()
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for col, label, c, ls in (("5M unedited", "published", PUB, "--"), ("5M edit B (blind run 2)", "head edit", EDB, "-"),
                              ("5M fine-tune of the same 6 144 weights", "same weights, gradients", SAME, "-"),
                              ("5M fine-tune, many cycle lengths", "full fine-tune", FULL, "-")):
        ax.plot(t.index, t[col], marker="o", ms=3.5, color=c, ls=ls, label=label, lw=2.4 if label == "head edit" else 1.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks([512, 1024, 2048, 4096, 8192], ["512", "1,024", "2,048", "4,096", "8,192"])
    ax.set_xlabel("steps of history given to the model")
    ax.set_ylabel("near dips drawn")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.set_title("The cost every fix shares: short histories (≈ 1,000-1,500 steps)", loc="left")
    save(fig, "13_short_history_cost")


def scale() -> None:
    fs = pd.read_csv(R / "scale" / "failure_by_size.csv")
    f = fs[(fs.history == 8192) & (fs.group == "misaligned")].set_index("size")["far"]
    sizes = ["5M", "28M", "123M", "209M"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(4)
    px = np.array([-0.18, 1, 2, 3])  # published bars; 5M shares its slot with head edit
    ax.bar(px, [f[s] for s in sizes], width=0.34, color=[SIZE_COL[s] for s in sizes], alpha=0.45, label="published")
    edit_b = 0.822  # results/edit_5m/heldback_finetune_vs_b.log (head edit, misaligned)
    ax.bar([0.18], [edit_b], width=0.34, color=[SIZE_COL["5M"]], label="head edit")
    for i, s in enumerate(sizes):
        ax.text(px[i], f[s] + 0.02, f"{f[s]:.2f}", ha="center", fontsize=8, color=INK2)
    ax.text(0.18, edit_b + 0.02, f"{edit_b:.2f}", ha="center", fontsize=8, color=INK, weight="bold")
    ax.set_xticks(x, [f"Timestral-{s}" for s in sizes])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("distant dips drawn")
    ax.legend(loc="upper left")
    ax.set_title("The failure shrinks with model size; head edit repairs 5M", loc="left")
    save(fig, "14_model_sizes")


if __name__ == "__main__":
    architecture(); example_synthetic(); lead_time(); head_lags(); lock_heatmap(); knockout_and_patching()
    reach_diagram(); comparison(); gift_domains(); real_far_near(); tide(); external_overview(); tides_vs_bigger(); external_examples(); real_error_by_distance(); lag_sweep(); history_cost(); scale()
