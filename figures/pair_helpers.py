"""Full-Chinese pair-recovery figures from immutable model outputs.

Figure 9 compares population-level pair errors across all three training seeds.
Figures 10/11 preserve each previously selected pair, observation, display seed,
truth-based slice and voxel threshold while adding two requested baselines.
Their five rows separate truth, input controls, the baseline and the combined
network. Each row links exterior 3D structure to an unthresholded interior slice.
Python-only structural adaptation; no spatial interpolation or smooth surfaces.
"""

from pathlib import Path
import hashlib
import json
import re

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.text import Text
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd

from figure_runtime import OUTPUT, SOURCE_DATA, ASSETS
OUT = OUTPUT
PACKAGE = OUTPUT
STATS = ASSETS / "figure_data/statistics"
DATA = SOURCE_DATA
CORE = ["0001", "1001", "0101", "1101"]
DISPLAY = ["truth", "T0P", "B0P", "0001", "1101"]
NAMES = {
    "truth": "真值", "T0P": "单分量输入\n＋线性输出", "B0P": "五通道输入\n＋线性输出",
    "0001": "基线网络", "1001": "标量锚定角色融合网络",
    "0101": "基线网络＋门控密度修正", "1101": "角色融合与门控密度修正网络",
}
COLORS = {"0001": "#777777", "1001": "#4C78A8", "0101": "#D09A53", "1101": "#5B8E7D"}
FORBIDDEN = re.compile(r"\b(?:SC|MC|Base|Full|TARF|GDC|SAE|NRMSE|NRMS|r0|km|Depth|Truth)\b")
QA, PANELS = [], []


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style():
    font_manager.findfont("Noto Sans CJK JP", fallback_to_default=False)
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def save(fig, stem, details):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = []
    for text in fig.findobj(match=Text):
        if text.get_visible() and text.get_text():
            assert text.get_fontsize() >= 5, (stem, text.get_text())
            assert not FORBIDDEN.search(text.get_text()), (stem, text.get_text())
            labels.append(text.get_text())
    for item in list(fig.axes) + list(fig.texts):
        box = item.get_tightbbox(renderer)
        assert box.x0 >= -1 and box.y0 >= -1, (stem, box.bounds)
        assert box.x1 <= fig.bbox.x1 + 1 and box.y1 <= fig.bbox.y1 + 1, (stem, box.bounds)
    # Pair plot has independent panels; 3D and section axes also need clear gaps.
    boxes = [ax.get_tightbbox(renderer) for ax in fig.axes]
    for i, box in enumerate(boxes):
        for other in boxes[i + 1:]:
            assert not box.overlaps(other), (stem, "axes overlap", box.bounds, other.bounds)
    fig.savefig(OUT / f"{stem}.svg", dpi=600)
    fig.savefig(OUT / f"{stem}.pdf", dpi=600)
    fig.savefig(OUT / f"{stem}.png", dpi=600)
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    QA.append(dict(figure=stem, width_mm=fig.get_figwidth() * 25.4,
                   height_mm=fig.get_figheight() * 25.4, display_text=labels,
                   minimum_font_pt=min(t.get_fontsize() for t in fig.findobj(match=Text)
                                       if t.get_visible() and t.get_text()),
                   all_axes_inside_canvas=True, axes_do_not_overlap=True, **details))
    plt.close(fig)
    print("Rendered", stem, flush=True)


def pair_summary():
    raw = pd.read_csv(STATS / "pair_seed_metrics.csv", dtype={"code": str})
    summary = pd.read_csv(STATS / "pair_mean_sd.csv", dtype={"code": str})
    selected = raw.loc[raw["split"].eq("validation_iid") & raw.code.isin(CORE)].copy()
    assert len(selected) == 24 and selected.n_pairs.eq(325).all()
    assert np.isfinite(selected.difference_nrmse).all()
    fig, axes = plt.subplots(1, 2, figsize=(7.20472440945, 3.34645669291))
    fig.subplots_adjust(left=0.32, right=0.98, bottom=0.23, top=0.86, wspace=0.26)
    groups = []
    for column, track in enumerate(["separable", "shape"]):
        ax = axes[column]
        for position, code in enumerate(CORE):
            part = selected.loc[selected.track.eq(track) & selected.code.eq(code)].sort_values("seed")
            assert part.seed.tolist() == [1107, 1108, 1109]
            values = part.difference_nrmse.to_numpy()
            mean, sd = float(values.mean()), float(values.std(ddof=1))
            expected = summary.loc[summary["split"].eq("validation_iid") & summary.code.eq(code)
                                   & summary.track.eq(track)].iloc[0]
            np.testing.assert_allclose([mean, sd], [expected.difference_nrmse_mean, expected.difference_nrmse_sd],
                                       rtol=1e-8, atol=1e-10)
            ax.plot(values, position + np.array([-0.12, 0, 0.12]), marker="o", linestyle="none",
                    color=COLORS[code], markersize=3.5, alpha=0.8)
            ax.errorbar(mean, position, xerr=sd, fmt="|", color=COLORS[code],
                        markersize=13, linewidth=1.1, capsize=3)
            groups.append(dict(track=track, code=code, values=values.tolist(), mean=mean, sample_sd=sd))
        ax.axvline(1, color="#555555", linewidth=0.7, linestyle="--", zorder=0)
        ax.set_xlim(0.94, 1.27)
        ax.set_xticks([0.95, 1.00, 1.10, 1.20, 1.25])
        ax.set_ylim(3.5, -0.5)
        ax.set_yticks(range(4), [NAMES[c] for c in CORE] if column == 0 else [""] * 4)
        ax.tick_params(axis="y", length=0, pad=6, labelsize=7)
        ax.set_xlabel("差值归一化均方根误差\n（越低越好）", fontsize=7, labelpad=7)
        ax.set_title("a  可区分对" if column == 0 else "b  形状歧义对",
                     loc="left", fontweight="bold", pad=10)
    selected[["code", "seed", "track", "n_pairs", "difference_nrmse"]].to_csv(
        DATA / "pair_error_seed_metrics.csv", index=False)
    save(fig, "pair_recovery_zh", dict(groups=groups, uncertainty="Three seeds; mean and sample SD",
                                      same_axis_limits=True, reference=1.0, omitted_seeds=0))


def exposed_faces(field, filled):
    vertices, values = [], []
    for axis in range(3):
        remaining = [i for i in range(3) if i != axis]
        for sign in (-1, 1):
            neighbor = np.roll(filled, -sign, axis=axis)
            boundary = [slice(None)] * 3
            boundary[axis] = -1 if sign == 1 else 0
            neighbor[tuple(boundary)] = False
            chosen = filled & ~neighbor
            ids = np.argwhere(chosen)
            offsets = np.zeros((4, 3))
            offsets[:, axis] = 1 if sign == 1 else 0
            offsets[:, remaining] = [[0, 0], [1, 0], [1, 1], [0, 1]]
            vertices.append((ids[:, None, :] + offsets[None, :, :]) * [0.125, 0.125, 0.0625])
            values.append(field[chosen])
    return np.concatenate(vertices), np.concatenate(values)


def volume(ax, data, threshold, vmax):
    field = data[::-1].transpose(2, 1, 0)
    assert field.shape == (32, 32, 16)
    faces, values = exposed_faces(field, np.abs(field) >= threshold)
    mesh = Poly3DCollection(faces, facecolors=mpl.colormaps["RdBu_r"](Normalize(-vmax, vmax)(values)),
                            edgecolors=(0.25, 0.25, 0.25, 0.10), linewidths=0.08,
                            shade=False, rasterized=True, zsort="average")
    ax.add_collection3d(mesh)
    ax.set(xlim=(0, 4), ylim=(0, 4), zlim=(1, 0))
    ax.set_box_aspect((4, 4, 2), zoom=0.90)
    ax.set_proj_type("ortho")
    ax.view_init(elev=24, azim=-55)
    ax.set_xticks([0, 4]); ax.set_yticks([0, 4]); ax.set_zticks([0, 1])
    ax.tick_params(labelsize=6, pad=-3, length=1)
    ax.set_xlabel("东西向", fontsize=6, labelpad=-10)
    ax.set_ylabel("南北向", fontsize=6, labelpad=-10)
    ax.text2D(0.04, 0.80, "深度", transform=ax.transAxes, fontsize=6)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#E0E3E6")
        axis.line.set_color("#ABB2B9")
        axis.line.set_linewidth(0.4)
        axis._axinfo["grid"].update(color="#E3E7EA", linewidth=0.35)


def case_figure(track):
    archive = np.load(DATA / "fixed_display_predictions.npz")
    key = track + "_best01"
    deltas = {code: archive[f"{key}_{code}_A"] - archive[f"{key}_{code}_B"] for code in DISPLAY}
    assert all(np.isfinite(x).all() for x in deltas.values())
    iz = int(np.square(deltas["truth"].astype(float)).sum(axis=(1, 2)).argmax())
    depth_m = (15.5 - iz) * 62.5
    assert depth_m == (156.25 if track == "separable" else 343.75)
    threshold = float(np.abs(deltas["truth"]).max()) * 0.1
    vmax = max(float(np.abs(x).max()) for x in deltas.values())
    fig = plt.figure(figsize=(7.20472440945, 9.44881889764))
    grid = fig.add_gridspec(5, 2, left=0.20, right=0.83, top=0.935, bottom=0.055,
                           hspace=0.17, wspace=0.24, width_ratios=[1.1, 1])
    displayed_slices = {}
    for row, code in enumerate(DISPLAY):
        data = deltas[code]
        ax = fig.add_subplot(grid[row, 0], projection="3d")
        volume(ax, data, threshold, vmax)
        bx = fig.add_subplot(grid[row, 1])
        bx.imshow(data[iz], origin="lower", extent=[0, 4, 0, 4], interpolation="nearest",
                  cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        bx.set_xticks([0, 2, 4]); bx.set_yticks([0, 2, 4])
        bx.tick_params(labelsize=6, pad=2, length=2)
        bx.set_ylabel("南北向", fontsize=6, labelpad=2)
        if row == 4:
            bx.set_xlabel("东西向", fontsize=6, labelpad=2)
        else:
            bx.set_xticklabels([])
        for spine in bx.spines.values():
            spine.set_visible(True)
            spine.set_color("#B8BDC2")
            spine.set_linewidth(0.5)
        ax.text2D(0.0, 0.98, chr(97 + row * 2), transform=ax.transAxes,
                  fontsize=8, fontweight="bold", va="top")
        bx.text(-0.23, 1.02, chr(98 + row * 2), transform=bx.transAxes,
                fontsize=8, fontweight="bold", va="bottom")
        name = "角色融合与\n门控密度修正网络" if code == "1101" else NAMES[code]
        fig.text(0.02, 0.5 * (ax.get_position().y0 + ax.get_position().y1), name,
                 fontsize=7, va="center", color="#365E52" if code == "1101" else "#333333")
        if row == 0:
            ax.set_title("三维模型差值", fontsize=8, pad=13)
            bx.set_title("水平切片", fontsize=8, pad=14)
        shown = np.abs(data) >= threshold
        for column in range(2):
            PANELS.append(dict(figure=track, panel=chr(97 + row * 2 + column), code=code,
                quantity="3D difference" if column == 0 else "horizontal section",
                threshold=threshold if column == 0 else 0, shown_voxels=int(shown.sum()) if column == 0 else 1024,
                hidden_voxels=int((~shown).sum()) if column == 0 else 0, color_vmax=vmax,
                fixed_example=True, uncertainty="None; single fixed seed/noise example",
                slice_index=iz if column == 1 else None, depth_m=depth_m,
                fixed_observation=0, training_seed=1107))
        displayed_slices[code] = data[iz]
    cb = fig.colorbar(ScalarMappable(norm=Normalize(-vmax, vmax), cmap="RdBu_r"),
                      cax=fig.add_axes([0.88, 0.29, 0.018, 0.42]), ticks=[-vmax, 0, vmax])
    cb.set_ticklabels([f"{-vmax:.2f}", "0", f"{vmax:.2f}"])
    cb.ax.tick_params(labelsize=6, pad=2, length=2)
    cb.set_label("模型间密度差值（克/立方厘米）", fontsize=7, labelpad=4)
    cb.outline.set_linewidth(0.5)
    fig.text(0.51, 0.018, f"坐标单位：千米；切片深度：{depth_m:.2f} 米", ha="center", fontsize=7)
    np.savez_compressed(DATA / f"{track}_displayed_slices.npz", **displayed_slices)
    save(fig, f"selected_{track}_zh", dict(case_key=key, seed=1107, noise=0, depth_m=depth_m,
        threshold=threshold, color_vmax=vmax, same_color_scale=True, spatial_smoothing=False,
        vertical_exaggeration=2, uncertainty="None; selected fixed example", models=DISPLAY))


def main():
    style()
    protected_paths = [STATS / "pair_seed_metrics.csv", STATS / "pair_mean_sd.csv",
                       DATA / "fixed_display_predictions.npz"]
    protected_paths += [OUT.parent / f"pair_recovery_nrmse.{ext}" for ext in ("png", "pdf", "svg")]
    protected_paths += [OUT.parents[1] / "cases" / "figures" / f"selected_{track}.{ext}"
                        for track in ("separable", "shape") for ext in ("png", "pdf", "svg")]
    protected = {path: sha256(path) for path in protected_paths}
    pair_summary()
    case_figure("separable")
    case_figure("shape")
    assert all(sha256(path) == digest for path, digest in protected.items())
    pd.DataFrame(PANELS).to_csv(DATA / "case_panel_audit.csv", index=False)
    (OUT / "figure_qa.json").write_text(json.dumps(dict(
        backend="Python/matplotlib", old_figures_and_source_arrays_unchanged=True,
        no_training_or_case_reselection=True, figures=QA,
        protected_hashes={path.name: digest for path, digest in protected.items()},
    ), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
