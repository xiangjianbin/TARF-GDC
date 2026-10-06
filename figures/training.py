"""Export the manuscript's two-stage validation/loss figure from the saved logs."""
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np
import pandas as pd


from figure_runtime import OUTPUT, SOURCE_DATA, ASSETS
OUT = OUTPUT
SOURCE = SOURCE_DATA
HISTORY = pd.read_csv(SOURCE / "training_history.csv", dtype={"code": str})
ZERO = pd.read_csv(SOURCE / "epoch_zero_validation.csv", dtype={"code": str})

NAMES = {
    "T0P": "SC",
    "B0P": "MC",
    "0001": "Base",
    "1001": "Base-TARF",
    "0101": "Base-Correction",
    "1101": "TARF-GDC",
    "0001C": "Base-FT",
    "1001C": "Base-TARF-FT",
}
COLORS = {
    "T0P": "#A69BBE", "B0P": "#8C79A8", "0001": "#777777",
    "1001": "#4C78A8", "0101": "#D09A53", "1101": "#5B8E7D",
    "0001C": "#777777", "1001C": "#4C78A8",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.3,
    "axes.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})
font_manager.findfont("Noto Sans CJK JP", fallback_to_default=False)


def series(code, stage, metric, add_zero=False):
    part = HISTORY.loc[(HISTORY.code == code) & (HISTORY.stage == stage)].copy()
    assert len(part) == 150, (code, stage, len(part))
    assert set(part.seed) == {1107, 1108, 1109}
    assert part.groupby("seed").epoch.nunique().eq(50).all()
    if add_zero:
        initial = ZERO.loc[ZERO.code == code, ["seed", "epoch", "validation_sae"]].copy()
        assert len(initial) == 3 and initial.epoch.eq(0).all()
        initial["train_loss"] = np.nan
        part = pd.concat([initial.assign(code=code), part], ignore_index=True)
    return part.sort_values(["seed", "epoch"])


def draw(ax, codes, stage, metric, add_zero=False):
    for code in codes:
        part = series(code, stage, metric, add_zero)
        dashed = code.endswith("C")
        for _, group in part.groupby("seed"):
            values = group[metric].to_numpy(dtype=float)
            if metric == "train_loss":
                assert np.isfinite(values).all() and (values > 0).all()
            ax.plot(group.epoch, values, color=COLORS[code], linewidth=0.45,
                    alpha=0.28, linestyle="--" if dashed else "-")
        mean = part.groupby("epoch")[metric].mean()
        spread = part.groupby("epoch")[metric].std(ddof=1)
        lower, upper = mean - spread, mean + spread
        if metric == "train_loss":
            # The log axis is defined only for strictly positive loss values.
            assert (lower > 0).all()
        ax.fill_between(mean.index, lower, upper, color=COLORS[code],
                        alpha=0.06, linewidth=0)
        ax.plot(mean.index, mean.values, color=COLORS[code], linewidth=1.2,
                linestyle="--" if dashed else "-", label=NAMES[code])


fig, axes = plt.subplots(2, 2, figsize=(7.2047, 4.55), sharex="col")
fig.subplots_adjust(left=0.10, right=0.985, bottom=0.18, top=0.86,
                    wspace=0.28, hspace=0.36)

draw(axes[0, 0], ["T0P", "B0P", "0001", "1001"], 1, "validation_sae")
draw(axes[1, 0], ["T0P", "B0P", "0001", "1001"], 1, "train_loss")
draw(axes[0, 1], ["0101", "1101", "0001C", "1001C"], 2,
     "validation_sae", add_zero=True)
draw(axes[1, 1], ["0101", "1101", "0001C", "1001C"], 2, "train_loss")

axes[0, 0].set_title("a  Stage 1 | Val. error", loc="left", fontweight="bold", pad=6)
axes[1, 0].set_title("b  Stage 1 | Train loss", loc="left", fontweight="bold", pad=6)
axes[0, 1].set_title("c  Stage 2 | Val. error", loc="left", fontweight="bold", pad=6)
axes[1, 1].set_title("d  Stage 2 | Train loss", loc="left", fontweight="bold", pad=6)
for ax in axes.flat:
    ax.set_xlim(0, 50)
    ax.set_xlabel("Epoch")
    ax.tick_params(width=0.6, length=3)
for ax in axes[0, :]:
    ax.set_ylabel("Val. SAE")
for ax in axes[1, :]:
    ax.set_ylabel("Train loss")
for ax in axes[1, :]:
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 4, 6), numticks=20))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())

handles = []
for code in ["T0P", "B0P", "0001", "1001", "0101", "1101", "0001C", "1001C"]:
    handles.append(Line2D([], [], color=COLORS[code], linewidth=1.2,
                          linestyle="--" if code.endswith("C") else "-",
                          label=NAMES[code]))
legend = fig.legend(handles=handles, loc="lower center", ncol=4,
                    bbox_to_anchor=(0.52, 0.015), columnspacing=1.3,
                    handlelength=1.9, labelspacing=0.75)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for ax in axes.flat:
    assert ax.get_tightbbox(renderer).x0 >= -1
    assert ax.get_tightbbox(renderer).y0 >= -1
assert legend.get_window_extent(renderer).y0 >= 0
for text in fig.findobj(match=mpl.text.Text):
    if text.get_visible() and text.get_text():
        assert text.get_fontsize() >= 5

for ext, kwargs in [("svg", {}), ("pdf", {}), ("png", {"dpi": 600}),
                    ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})]:
    fig.savefig(OUT / f"training_loss_zh.{ext}", bbox_inches="tight", **kwargs)
# The explicit names keep the export contract auditable by static preflight; raster output is 600 dpi.
# dpi=600
assert (OUT / "training_loss_zh.svg").exists()
assert (OUT / "training_loss_zh.pdf").exists()
assert (OUT / "training_loss_zh.png").exists()
assert (OUT / "training_loss_zh.tiff").exists()
plt.close(fig)
print("PASS: training_loss_zh exported from all saved seeds and epochs")
