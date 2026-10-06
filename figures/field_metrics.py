"""Audit frozen field predictions and plot the requested single-method subset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


COMPONENTS = ("Txx", "Txy", "Txz", "Tyy", "Tyz", "Tzz")
SEEDS = (1107, 1108, 1109)
METRICS = ("r2", "relative_residual", "rms_ratio")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recompute(source, seed):
    with np.load(source / f"vinton_Full_s{seed}.npz") as arrays:
        observed = np.asarray(arrays["observed_d6_33"], dtype=np.float64)
        predicted = np.asarray(arrays["predicted_d6_33"], dtype=np.float64)
    assert observed.shape == predicted.shape == (6, 33, 33)
    assert np.isfinite(observed).all() and np.isfinite(predicted).all()
    records = []
    for k, name in enumerate(COMPONENTS):
        actual, estimate = observed[k].ravel(), predicted[k].ravel()
        error = actual - estimate
        variance_sum = np.square(actual - actual.mean()).sum()
        actual_rms = np.sqrt(np.square(actual).mean())
        assert variance_sum > 0 and actual_rms > 0
        records.append({
            "seed": seed, "component": name,
            "r2": 1 - np.square(error).sum() / variance_sum,
            "relative_residual": np.sqrt(np.square(error).mean()) / actual_rms,
            "rms_ratio": np.sqrt(np.square(estimate).mean()) / actual_rms,
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    source, out = args.source_dir, args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    source_files = [source / "vinton_component_metrics.csv", source / "vinton_geometry.csv"]
    source_files += [source / f"vinton_Full_s{seed}.npz" for seed in SEEDS]
    original_hashes = {path.name: digest(path) for path in source_files}
    frame = pd.read_csv(source_files[0])
    selected = frame.loc[frame["method"].eq("Full")].copy()
    assert len(selected) == 18
    assert set(selected["seed"]) == set(SEEDS)
    assert set(selected["component"]) == set(COMPONENTS)
    assert not selected.duplicated(["seed", "component"]).any()
    assert np.isfinite(selected[list(METRICS)].to_numpy()).all()
    recomputed = pd.DataFrame([row for seed in SEEDS for row in recompute(source, seed)])
    joined = selected.merge(recomputed, on=["seed", "component"], suffixes=("_stored", "_check"), validate="one_to_one")
    errors = {}
    for key in METRICS:
        errors[key] = float(np.max(np.abs(joined[f"{key}_stored"] - joined[f"{key}_check"])))
        assert errors[key] < 1e-6, (key, errors[key])
    selected["method"] = "TARF-GDC"
    selected.to_csv(out / "component_metrics.csv", index=False)
    seed_summary = selected.groupby("seed")[list(METRICS)].mean().reindex(SEEDS)
    seed_summary.to_csv(out / "seed_metrics.csv")
    component_summary = selected.groupby("component")[list(METRICS)].agg(["mean", "std"]).reindex(COMPONENTS)
    component_summary.to_csv(out / "component_summary.csv")
    geometry = pd.read_csv(source_files[1])
    geometry = geometry.loc[geometry["method"].eq("Full")].copy()
    assert set(geometry["seed"]) == set(SEEDS) and len(geometry) == 3
    geometry["method"] = "TARF-GDC"
    geometry.to_csv(out / "geometry.csv", index=False)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.204724409448819, 3.149606299212599))
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.25, top=0.86, wspace=0.27)
    x = np.arange(len(COMPONENTS))
    mean_color, point_color = "#4C7B6C", "#667581"
    markers = ("o", "s", "^")
    specs = (("relative_residual", "a  Response NRMS", (0, 0.75)),
             ("rms_ratio", "b  RMS ratio", (0, 1.4)))
    panels = []
    for ax, (metric, title, limits) in zip(axes, specs):
        values = selected.pivot(index="component", columns="seed", values=metric).reindex(index=COMPONENTS, columns=SEEDS).to_numpy()
        means, spread = values.mean(axis=1), values.std(axis=1, ddof=1)
        assert np.all(means - spread >= limits[0]) and np.all(means + spread <= limits[1])
        assert np.all(values >= limits[0]) and np.all(values <= limits[1])
        ax.errorbar(x - 0.14, means, yerr=spread, fmt="D", color=mean_color,
                    markersize=4, capsize=3, linewidth=1.15, zorder=4)
        for i, marker in enumerate(markers):
            ax.scatter(x + 0.12 + (i - 1) * 0.085, values[:, i], s=17,
                       marker=marker, facecolors="white", edgecolors=point_color,
                       linewidths=0.8, zorder=5)
        ax.set_title(title, loc="left", fontweight="bold", pad=10)
        ax.set_xticks(x, COMPONENTS)
        ax.set_xlim(-0.5, 5.55)
        ax.set_ylim(*limits)
        ax.grid(axis="y", color="#E1E5E7", linewidth=0.55)
        ax.set_axisbelow(True)
        if metric == "rms_ratio":
            ax.axhline(1, color="#7A7A7A", linestyle="--", linewidth=0.8, zorder=2)
        panels.append({"metric": metric, "n_seeds": 3, "n_components": 6,
                       "center": "arithmetic mean", "uncertainty": "sample SD, ddof=1",
                       "all_seed_values_shown": True, "y_limits": list(limits),
                       "all_points_and_errorbars_within_limits": True})
    handles = [Line2D([], [], color=mean_color, marker="D", linestyle="none", markersize=4,
                      label="Mean ± SD")]
    handles += [Line2D([], [], color=point_color, marker=marker, markerfacecolor="white",
                       linestyle="none", markersize=4, label=f"Seed {seed}")
                for seed, marker in zip(SEEDS, markers)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.52, 0.035),
               ncol=4, handletextpad=0.5, columnspacing=1.6)
    fig.canvas.draw()
    fig.savefig(out / "Figure13_TARF_GDC.svg")
    fig.savefig(out / "Figure13_TARF_GDC.pdf")
    fig.savefig(out / "Figure13_TARF_GDC.png", dpi=600)
    fig.savefig(out / "Figure13_TARF_GDC.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(out / "Figure13_preview.png", dpi=300)
    for index, ax in enumerate(axes):
        bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(out / f"panel_{index + 1}_preview.png", dpi=300, bbox_inches=bbox.expanded(1.035, 1.05))
    plt.close(fig)
    assert original_hashes == {path.name: digest(path) for path in source_files}
    report = {
        "status": "PASS", "backend": "python", "size_mm": [183, 80],
        "source_rows": len(frame), "displayed_rows": 18, "displayed_method": "TARF-GDC",
        "selection_rule": "Author-requested single-method chapter; all six components and all three seeds retained",
        "excluded_source_rows": len(frame) - 18, "source_data_unchanged": True,
        "source_hashes": original_hashes, "recomputed_metric_max_absolute_errors": errors,
        "seed_metrics": seed_summary.reset_index().to_dict(orient="records"),
        "overall_mean": seed_summary.mean().to_dict(),
        "overall_sample_sd": seed_summary.std(ddof=1).to_dict(),
        "panels": panels, "statistical_tests": "none",
        "claim_boundary": "Field-response consistency only; no density-truth or method-ranking claim",
    }
    (out / "figure_data_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"status": "PASS", "metric_audit": errors,
                      "seed_metrics": report["seed_metrics"], "mean": report["overall_mean"],
                      "sd": report["overall_sample_sd"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
