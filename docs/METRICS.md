# Reported metrics and statistical scope

These tables summarize existing frozen experiments; they are not new training or evaluation runs. Values were recomputed from the public per-seed CSVs using the arithmetic mean and sample SD across seeds 1107, 1108, and 1109 (`ddof=1`). The source CSVs and numerical model settings are unchanged.

## Reconstruction metrics on the held-out test set

Source: [overall_seed_metrics.csv](../evidence/overall_seed_metrics.csv), filtered to `test_locked` (10,710 models), checked against [overall_mean_sd.csv](../evidence/overall_mean_sd.csv). SAE uses density-contrast units; NRMSE, FG-NRMSE, and Physics NRMS are dimensionless.

| Configuration | SAE | NRMSE | FG-NRMSE | Physics NRMS |
|---|---:|---:|---:|---:|
| SC | 139.174 ± 8.089 | 0.3475 ± 0.0104 | 0.3107 ± 0.0065 | 0.2557 ± 0.0187 |
| MC | 150.788 ± 2.084 | 0.3862 ± 0.0085 | 0.3600 ± 0.0081 | 0.2765 ± 0.0120 |
| Base | 128.105 ± 6.939 | 0.3698 ± 0.0224 | 0.3426 ± 0.0207 | 0.2488 ± 0.0289 |
| Base-TARF | 106.492 ± 5.352 | 0.3237 ± 0.0107 | 0.2952 ± 0.0072 | 0.2140 ± 0.0239 |
| Base-GDC | 91.714 ± 4.129 | 0.2946 ± 0.0121 | 0.2694 ± 0.0120 | 0.1288 ± 0.0139 |
| TARF-GDC | 79.161 ± 2.841 | 0.2678 ± 0.0031 | 0.2425 ± 0.0016 | 0.1143 ± 0.0090 |
| Base-FT | 127.688 ± 6.688 | 0.3663 ± 0.0197 | 0.3425 ± 0.0212 | 0.2440 ± 0.0242 |
| Base-TARF-FT | 106.178 ± 5.412 | 0.3235 ± 0.0108 | 0.2938 ± 0.0087 | 0.2159 ± 0.0226 |

Validation results for the same metrics are already available in those CSVs and manuscript Table 4. Current Table 8 can include all four metrics without rerunning the experiments.

## Pair-difference metrics on the held-out test set

Source: [pair_seed_metrics.csv](../evidence/pair_seed_metrics.csv), checked against [pair_mean_sd.csv](../evidence/pair_mean_sd.csv). `separable` denotes distinguishable pairs (1,427 test pairs); `shape` denotes shape-ambiguous pairs (1,428 test pairs). Validation pairs remain separate from these test results.

| Pair class | Configuration | Difference NRMSE | Difference cosine similarity |
|---|---|---:|---:|
| Distinguishable pairs | SC | 1.2207 ± 0.0355 | 0.2828 ± 0.0162 |
| Distinguishable pairs | MC | 1.1916 ± 0.0100 | 0.2991 ± 0.0048 |
| Distinguishable pairs | Base | 1.1592 ± 0.0356 | 0.2930 ± 0.0138 |
| Distinguishable pairs | TARF-GDC | 1.1034 ± 0.0078 | 0.3652 ± 0.0050 |
| Shape-ambiguous pairs | SC | 1.1122 ± 0.0100 | 0.0051 ± 0.0013 |
| Shape-ambiguous pairs | MC | 1.0233 ± 0.0107 | 0.2259 ± 0.0281 |
| Shape-ambiguous pairs | Base | 1.0360 ± 0.0104 | 0.1922 ± 0.0095 |
| Shape-ambiguous pairs | TARF-GDC | 0.9715 ± 0.0122 | 0.2965 ± 0.0192 |

These results support relative improvements among the tested configurations. Difference NRMSE above one and cosine similarity well below one do not establish complete recovery of inter-model differences or resolution of inversion non-uniqueness.

## Why different comparisons report different metrics

The core ablations and held-out comparisons report overall density error, normalized density error, foreground error, and response fit. Continued-training and Obs-only controls emphasize SAE and Physics NRMS to separate model reconstruction from observation fit. Pair comparisons use full A–B difference fields, not only plotted slices. Public [control_seed_metrics.csv](../evidence/control_seed_metrics.csv) additionally contains pair metrics, but does not contain foreground NRMSE for Obs-only; missing values must not be inferred from SAE.

Within a model or pair, metrics are first averaged over the five fixed noise realizations; they are then averaged across models or pairs within each training seed. The displayed SD is across training seeds, not across noise realizations. Noise-realization SD cannot be recovered from these aggregate CSVs. Calculating it requires the original per-realization rows, if retained, with noise IDs 0–4 and consistent sample/pair identities. The presence of a reaggregation audit file alone does not supply that variance.

## Field diagnostics

For each frozen seed, field R² and response NRMS are averaged equally over six components. Across-seed means and sample SDs use all three seeds. Response NRMS divides each component’s residual RMS by its observed RMS; it differs from five-role-channel Physics NRMS. These are observation-fit metrics, not errors against known subsurface density.

The plotting code selects seed 1107 for Figs. 12 and 13 by the largest six-component mean R² among three frozen predictions. Checkpoints were selected on synthetic validation data before field inference. No field fine-tuning or new training-epoch selection is performed by the plotting step. The [field-diagnostics command](USER_GUIDE.md#7-figures) exports all-seed and per-component CSVs when authorized field outputs are supplied. This documentation update does not claim a new field replay or publish missing per-seed values.

Display aliases are interpreted by configuration ID: 0101 is Base-GDC, 1101 is TARF-GDC, 0001C is Base-FT, and 1001C is Base-TARF-FT. A legacy “Full” label without matching provenance must not be treated as the current architecture.
