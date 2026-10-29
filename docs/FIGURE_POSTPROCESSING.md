# Figure post-processing provenance

The final manuscript artwork was assembled in several presentation-only
steps. The scientific arrays, model predictions, thresholds and reported
metrics were kept unchanged.

## Located post-processing steps

| Date | Evidence | Scope | Current release location |
|---|---|---|---|
| 22 October 2026, 16:15 | `unify_figure1_labels.py` in the local audit archive | Replaced nine Figure 1 title terms inside fixed title bands and wrote a 600 dpi PNG. | `figures/dataset_figures.py` now emits the final terminology directly; `scripts/postprocess_submission_figures.py figure1` preserves the historical raster operation for an archived pre-edit PNG. |
| 23 October 2026 | `plot_source/figures/*.py` and the GitHub synchronization copy | Replayed Figures 1--13 from frozen source arrays, with SVG/PDF/PNG export and English submission labels. | `figures/` and `scripts/render_figures.py` |
| 24 October 2026, 16:33 | `audit28_patch.py` and `audit28_update_figures.py` | Changed the Figure 6 TARF-GDC title label to black and removed the Figure 7 coordinate/depth footer. The selected slices, thresholds, color limits and prediction differences were protected by hash checks. | `figures/pairs.py` |
| 26 October 2026 onward | `TARF-GDC.pptx`, `TARF-GDC.svg` and `TARF-GDC.pdf` in the author drawing folder | Replaced the earlier simplified Figure 4 drawing with the author-provided TARF-GDC artwork and its revised palette. | `figures/assets/Figure04.*`, `figures/assets/Figure04_provenance.json`, `figures/architecture.py` |
| 28 October 2026 | `render_final_*_20261028` and `render_verify_*_20261028b` outputs | Replayed the current release routes and checked final PNG/SVG/PDF exports. | `scripts/render_figures.py`, `tests/test_figure_routes.py`, `tests/test_release.py` |

`figures/pair_helpers.py` is retained as a repository-only helper for
historical Chinese-labelled diagnostic plots. It is not a standalone
manuscript-figure route; `figures/pairs.py` overrides its displayed labels with
English for the current submission.

## Figure 4 palette

The checked-in Figure 4 provenance records the audited colors: framework
`#F2F2F2`, GDC main panel `#DAE3F3`, TARF `#EFEDFA`, GLIN `#FBF1DC`, GDC
transformer `#ECFFF0`, blue nodes `#4472C4`, and orange skip connections
`#F4B183`. `figures/architecture.py` copies the three formats from this one
audited asset set, so a replay cannot silently regenerate a different
simplified schematic.

## What could not be recovered byte-for-byte

The original `Figures_independent_images/fig01.png` input used by the historical title
band script is no longer present in the working archive. The current source
plot regenerates Figure 1 with the same final labels and geometry, but it does
not claim byte-identical pixels to that old raster edit. The same limitation
applies to any old image-only edit whose source file was not retained. The
release therefore treats the current plotting modules and the checked-in
Figure 4 assets as authoritative, and records historical raster edits here
instead of pretending that a missing source can be reconstructed.

Run the new helper on a retained submission directory with:

```bash
python scripts/postprocess_submission_figures.py audit path/to/Figures --output outputs/figure_submission_audit.json
```

The helper reports pixel dimensions, embedded DPI metadata, SHA256, and the
presence of SVG/PDF companions. It does not alter scientific data.
