# Manuscript-to-code coverage

The selected Chinese manuscript is identified by SHA256 in `evidence/manuscript_identity.json`. This mapping follows that revision, not earlier numbering, earlier `Full` models, or later exploratory field corrections. No manuscript text was edited during this release preparation.

## Experimental evidence

| Manuscript | Required evidence | Implementation and retained evidence |
|---|---|---|
| 2.1; Table 1 | 19 density subtypes, geometry and density constraints | `dataset_generation/v017/geometry.py`, ordinary/pair builders, `config_v023.json`; subtype arrays |
| 2.2 | Grid, tensor ordering, units, forward model | `scripts/build_operator.py`, `code/v024_inversion/physics.py`, operator specification, independent prism tests |
| 2.3 | Separable/shape acceptance gates, pair differences | Both pair builders, `v023_shape_optimizer.py`, full dataset audit; independent Figure 3 kept separate |
| 2.4; Table 2 | 25,000/2,500/10,710 models, disjoint splits, online/frozen noise | Final assembly/freezer, 170-file checksum manifest, seed registry, `data.py`, `noise.py` |
| 3.1–3.4; Table 3 | Role transform, TARF, GLIN, one-step GDC; SC/MC controls | `model.py`, `complementarity.py`, `residual_transformer.py`, `physics.py`; 27 registered recipes |
| 3.5 | Two-stage freezing, loss calibration, initialization, epoch-zero selection | `trainer.py`, `staged_refinement.py`, `losses.py`; `scripts/calibrate_physics.py`, calibration JSON |
| 3.6 | SAE/NRMSE/foreground error, observation consistency, pair difference and SD | `metrics.py`, `evaluation.py`, `statistics.py`; full sample/pair evaluation |
| 4.1–4.2.1; Table 4 | Four core configurations × three seeds | Prefixes 0001/1001/0101/1101; overall seed metrics and selected checkpoints |
| 4.2.2; Table 5 | Equal additional 50 epochs; FT versus frozen GDC | 0001C/1001C versus 0101/1101, matching first-stage weights, epoch-zero logs |
| 4.2.3; Table 6 | Obs-only versus residual control, validation/test | Obs-only versus 1101; explicit observation-only flag, control seed metrics |
| 4.3; Table 7 | SC/MC/Base/TARF-GDC pair recovery | T0P/B0P/0001/1101; pair seed metrics; complete A–B difference vectors |
| 4.4 | Fixed ordinary/shape A/B/deep reconstruction examples | Fixed prediction arrays, sample selections and truth-derived cuts; no new case selection |
| 4.5 | Same 200-model classical comparison | `scripts/classical.py`, `code/classical_solvers/`, training calibration, subset IDs, warm start, final densities and stopping records |
| 4.6; Table 8 | All eight main configurations on 10,710 locked-test models | Registered checkpoints; five-realization evaluation; source seed metrics and pair metrics |
| 5.1–5.2 | Six field components, frozen inference, all-seed fit, geometry | `code/field_preprocessing.py`, `reproduce.py field`; authorized raw TXT and three frozen predictions |
| 6 | Interpretation and limitations | No additional unreported experimental branch is required; avoid treating field fit as underground density accuracy |

`scripts/evaluate_all.py` runs validation and locked-test evaluation for all 27 registered models, including Obs-only. It skips only matching, completed summaries. It does not train models or change checkpoint selection.

`scripts/check_release.py` checks 56 non-duplicated printed mean/SD entries from Tables 4–8 at their displayed precision. Repeated entries in Tables 5/6 refer to the same core results. `scripts/verify_replays.py` additionally compares all 200 classical source models, three neural seeds on that subset, and optionally full validation/field replays. Table 1 is a subtype catalogue, Table 2 is a count table, and Table 3 is an architecture contract rather than independent training experiments.

## Final main figures

| Figure | Runner group | Plotting source | Scientific role / important boundary |
|---|---|---|---|
| 1 | 01-02 | `figures/dataset_figures.py` | All 19 subtype examples; fixed density arrays |
| 2 | 01-02 | same | Two fixed models and all six clean responses |
| 3 | 03 | `figures/independent_example.py`, rotation helpers | Independent V022 illustration, rigid horizontal 22.5° rotation of the tensor/model/receivers; not a performance result |
| 4 | 04 | `figures/architecture.py` | Code-derived architecture and training schematic; no AI-generated bitmap |
| 5 | 05 | `figures/training.py` | All recorded seeds/epochs, sample-SD bands, epoch zero for follow-up validation |
| 6–7 | 06-07 | `figures/pairs.py`, pair helpers | Fixed separable/shape pairs; SC/MC/Base/TARF-GDC, no reselection |
| 8 | 08 | `figures/spatial.py` | Four fixed reconstruction cases; full model identity and preserved color-scale reference |
| 9 | 09 | `figures/sections.py` | Final 0.05 g/cm³ common main-body contours; all density pixels retained, evaluation threshold remains 0.005 |
| 10–11 | 10-11 | `figures/field_and_classical.py` | Synthetic classical matched cases, not Vinton; original converged L2/IRLS arrays |
| 12 | 12 | `figures/best_field_seed.py` | Display seed 1107 selected by all-six-component mean R²; all 1,089 receivers |
| 13 | 13 | `figures/field_metrics.py` | Final two-panel, TARF-GDC-only all-seed NRMS/amplitude plot; not the obsolete three-panel comparison |
| 14 | 14 | `figures/field_and_classical.py` | Same frozen seed 1107, peak-based cuts, thresholded 3D display and unmasked slices |

Figures retain their existing question, panel roles, transforms and fixed cases. This is exact scientific-source reuse with path adaptation, not a figure redesign. Replots were checked against the actual manuscript-linked PNGs. Figures 3 and 5–14 are pixel-identical in the tested environment; Figures 1/2 differ in raster dimensions by one pixel, and Figure 4 is not pixel-identical. All 14 replayed PDFs passed the 5 pt glyph-floor audit. The source arrays and scientific formulas were not changed to force bitmap equality.

## Issues found outside executable code

The selected manuscript mentions “Table 9” for field geometry, but its actual numbered tables end at Table 8. The underlying geometry is available and recomputable; this is a dangling manuscript cross-reference, not missing experimental data. The author should decide whether to restore the geometry table or remove/replace the reference. This code-focused task did not silently edit the manuscript or its Word version.

The data/code availability and author-declaration placeholders still need author-approved publication details. Local paths are not public repository URLs.
