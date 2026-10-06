"""Two-row case comparisons with unchanged predictions and display transforms.

Claim: in the two previously selected cases, the combined network recovers
the main difference structure better than the displayed input controls.
These selected examples do not establish population-level performance.
Image-plate archetype: truth and four matched predictions in five columns;
top-row exterior geometry and bottom-row unthresholded amplitude sections.
Structural adaptation of the existing Python renderer; no inference, new
selection, smoothing, spatial crop, resampling or uncertainty aggregation.
"""

from pathlib import Path
import hashlib
import json
import re
import sys

sys.dont_write_bytecode = True
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.text import Text
import numpy as np
import pandas as pd

from figure_runtime import OUTPUT, SOURCE_DATA, ASSETS
OUT = OUTPUT
SOURCE = SOURCE_DATA
DATA = SOURCE_DATA
sys.path.insert(0, str(SOURCE))
import pair_helpers as shared

DISPLAY = ["truth", "T0P", "B0P", "0001", "1101"]
NAMES = ["Truth", "SC", "MC", "Base", "TARF-GDC"]
FORBIDDEN = re.compile(r"(?:真值|单分量|五通道|基线网络|角色融合|门控密度修正|东西向|南北向|模型间密度差值|坐标单位|深度)")
QA, PANELS = [], []


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, stem, details):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(match=Text) if t.get_visible() and t.get_text()]
    assert min(t.get_fontsize() for t in texts) >= 5
    assert all(not FORBIDDEN.search(t.get_text()) for t in texts)
    boxes = [item.get_tightbbox(renderer) for item in list(fig.axes) + list(fig.texts)]
    fig.savefig(OUT / f"{stem}.png", dpi=600)
    for box in boxes:
        assert box.x0 >= -1 and box.y0 >= -1, (stem, "canvas lower", box.bounds)
        assert box.x1 <= fig.bbox.x1 + 1 and box.y1 <= fig.bbox.y1 + 1, (stem, "canvas upper", box.bounds)
    for i, box in enumerate(boxes):
        for j, other in enumerate(boxes[i + 1:], i + 1):
            assert not box.overlaps(other), (stem, "collision", i, j, box.bounds, other.bounds)
    fig.savefig(OUT / f"{stem}.svg", dpi=600)
    fig.savefig(OUT / f"{stem}.pdf", dpi=600)
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    QA.append(dict(figure=stem, width_mm=fig.get_figwidth() * 25.4,
        height_mm=fig.get_figheight() * 25.4, layout="2 rows x 5 columns",
        top_row="3D difference", bottom_row="horizontal section",
        model_order=DISPLAY, display_text=[t.get_text() for t in texts],
        minimum_font_pt=min(t.get_fontsize() for t in texts),
        all_axes_and_text_inside_canvas=True, no_axes_or_title_collisions=True, **details))
    plt.close(fig)
    print("Rendered", stem, flush=True)


def draw_case(track, archive, errors):
    key = track + "_best01"
    deltas = {code: archive[f"{key}_{code}_A"] - archive[f"{key}_{code}_B"] for code in DISPLAY}
    assert all(x.shape == (16, 32, 32) and np.isfinite(x).all() for x in deltas.values())
    iz = int(np.square(deltas["truth"].astype(float)).sum(axis=(1, 2)).argmax())
    depth_m = (15.5 - iz) * 62.5
    assert depth_m == (156.25 if track == "separable" else 343.75)
    threshold = float(np.abs(deltas["truth"]).max()) * 0.1
    vmax = max(float(np.abs(x).max()) for x in deltas.values())
    old_qa = json.loads((SOURCE / "figure_qa.json").read_text())
    old = next(x for x in old_qa["figures"] if x["figure"] == f"selected_{track}_zh")
    assert threshold == old["threshold"] and vmax == old["color_vmax"]
    old_slices = np.load(DATA / f"{track}_displayed_slices.npz")
    fig = plt.figure(figsize=(7.20472440945, 4.13385826772))
    grid = fig.add_gridspec(2, 5, left=0.06, right=0.97, top=0.81, bottom=0.20,
                           hspace=0.34, wspace=0.34)
    for column, code in enumerate(DISPLAY):
        field = deltas[code]
        np.testing.assert_array_equal(field[iz], old_slices[code])
        ax = fig.add_subplot(grid[0, column], projection="3d")
        shared.volume(ax, field, threshold, vmax)
        # The shared renderer is also used by the Chinese source figures;
        # override only this English-label plate so the requested abbreviations
        # and coordinate labels are visible without changing those originals.
        ax.set_xlabel("x (km)", fontsize=6, labelpad=-10)
        ax.set_ylabel("y (km)", fontsize=6, labelpad=-10)
        for annotation in ax.texts:
            if annotation.get_text() == "深度":
                annotation.set_text("z (km)")
        bx = fig.add_subplot(grid[1, column])
        bx.imshow(field[iz], origin="lower", extent=[0, 4, 0, 4],
                  interpolation="nearest", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  aspect="equal")
        bx.set_xticks([0, 2, 4]); bx.set_yticks([0, 2, 4])
        bx.tick_params(labelsize=6, pad=2, length=2)
        bx.set_xlabel("x (km)", fontsize=6, labelpad=2)
        if column == 0:
            bx.set_ylabel("y (km)", fontsize=6, labelpad=2)
        else:
            bx.set_yticklabels([])
        for spine in bx.spines.values():
            spine.set_visible(True)
            spine.set_color("#B8BDC2")
            spine.set_linewidth(0.5)
        ax.text2D(0.0, 1.01, chr(97 + column), transform=ax.transAxes,
                  fontsize=8, fontweight="bold", va="bottom")
        bx.text(0.0, 1.04, chr(102 + column), transform=bx.transAxes,
                fontsize=8, fontweight="bold", va="bottom")
        position = grid[0, column].get_position(fig)
        fig.text(0.5 * (position.x0 + position.x1), 0.955, NAMES[column],
                 fontsize=6.5, ha="center", va="top",
                 color="#365E52" if code == "1101" else "#333333")
        for row in range(2):
            visible = np.abs(field) >= threshold
            PANELS.append(dict(figure=track, panel=chr(97 + row * 5 + column),
                row=row + 1, column=column + 1, code=code,
                quantity="3D difference" if row == 0 else "horizontal section",
                summary="raw difference; no aggregation",
                uncertainty="None; one fixed case, training seed and observation",
                seed=1107, noise=0, threshold=threshold if row == 0 else 0,
                shown_voxels=int(visible.sum()) if row == 0 else 1024,
                hidden_voxels=int((~visible).sum()) if row == 0 else 0,
                slice_index=iz if row == 1 else None, depth_m=depth_m, color_vmax=vmax,
                displayed_values_match_previous=True, geometry_qa="pass"))
        if code != "truth":
            expected = errors.loc[errors.case_key.eq(key) & errors.code.eq(code), "difference_nrmse"].item()
            # Reproduce the previously recorded whole-volume double-precision
            # norm; differences must be formed after promoting each endpoint.
            prediction = archive[f"{key}_{code}_A"].astype(float) - archive[f"{key}_{code}_B"].astype(float)
            truth = archive[f"{key}_truth_A"].astype(float) - archive[f"{key}_truth_B"].astype(float)
            actual = np.linalg.norm(prediction - truth) / np.linalg.norm(truth)
            assert abs(actual - expected) < 1e-10, (code, actual, expected)
    cb = fig.colorbar(ScalarMappable(norm=Normalize(-vmax, vmax), cmap="RdBu_r"),
                      cax=fig.add_axes([0.30, 0.089, 0.40, 0.021]),
                      orientation="horizontal", ticks=[-vmax, 0, vmax])
    cb.set_ticklabels([f"{-vmax:.2f}", "0", f"{vmax:.2f}"])
    cb.ax.tick_params(labelsize=6, pad=2, length=2)
    cb.set_label("Model-pair density difference (g/cm³)", fontsize=6.5, labelpad=2)
    cb.ax.xaxis.set_label_position("top")
    cb.outline.set_linewidth(0.5)
    fig.text(0.5, 0.010, f"Coordinates: km; horizontal-slice depth: {depth_m:.2f} m",
             fontsize=6.5, ha="center", va="bottom")
    save(fig, f"selected_{track}_horizontal_zh", dict(case_key=key, seed=1107, noise=0,
        depth_m=depth_m, threshold=threshold, color_vmax=vmax,
        projection="orthographic", vertical_exaggeration=2, spatial_smoothing=False,
        all_predictions_slices_and_difference_errors_unchanged=True))


def main():
    shared.style()
    mpl.rcParams.update({"font.family": "sans-serif", "font.size": 7,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    protected_paths = [SOURCE / "plot_section45.py", SOURCE / "figure_qa.json",
                       SOURCE / "inference_qa.json"]
    protected_paths += list(DATA.glob("*"))
    protected = {path: sha256(path) for path in protected_paths if path.is_file()}
    with np.load(DATA / "fixed_display_predictions.npz") as archive:
        errors = pd.read_csv(DATA / "case_difference_errors.csv", dtype={"code": str})
        draw_case("separable", archive, errors)
        draw_case("shape", archive, errors)
    assert all(sha256(path) == digest for path, digest in protected.items())
    pd.DataFrame(PANELS).sort_values(["figure", "row", "column"]).to_csv(
        OUT / "panel_audit.csv", index=False)
    (OUT / "figure_qa.json").write_text(json.dumps(dict(backend="Python/matplotlib",
        source_data_and_inference_unchanged=True, labels_regenerated=True,
        no_inference_or_reselection=True,
        figures=QA, protected_hashes={path.name: digest for path, digest in protected.items()}),
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
