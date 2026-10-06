# Verification and limits

Verification was performed in the recorded Python 3.8.10 / PyTorch 2.4.1 CUDA environment. Machine-readable results are retained in `evidence/verification/`. The original research files, dataset and checkpoints were not overwritten.

| Check | Observed result |
|---|---|
| Frozen dataset checksums | 170/170 source files match; no split ID/pair overlap |
| Full scientific dataset audit | All 38,210 endpoints, all 10,005 pair gates, noise and independent physics checks PASS |
| Checkpoint prediction replay | All 27 models load strictly; 16 fixed validation r0 samples/model; four primary metrics exactly match the historical batch values |
| Training routes | One optimization step for each of nine model types; GDC backbone remains frozen; 58,849 correction parameters |
| Complete evaluation route | TARF-GDC seed 1107, all 2,500 validation models, 650 pairs, all five noise realizations; aggregate/pair metrics match historical summaries (CSV precision tolerance) |
| Field inference | All three frozen seeds; density and observed/predicted arrays exactly match historical arrays |
| Classical source metrics | 200 models each for original L2/IRLS and three neural seeds, independently recomputed in float64 and matched |
| Classical solver execution | L2 and warm-start IRLS each run on one fixed model and meet stopping criteria; not a full 200-model solver rerun |
| L2 parameter calibration | All 12 training models × 12 beta candidates rerun; every calibration score exactly matches the stored reference; beta 3000 selected |
| Dataset generators | Ordinary 8 endpoints, separable 4 pairs and shape 2 pairs generated successfully; not a regeneration of the frozen dataset |
| Operator rebuild | SimPEG reconstruction produces the exact historical G6 SHA256 |
| Preparation | All-training normalization exactly matches; operator derivatives and calibration replay pass |
| Paper tables | 56 non-duplicated mean/SD entries checked; seed SD uses ddof=1 |
| Figures | All 14 main figure routes rendered; final Figure 13 separately recovered; PDF text audit passed |
| Unit tests | 27 tests passed, including original dataset tests, English-comment checks and a data-independent CPU demo |

Not performed: training all 27 models for their full 50/100-epoch schedules; complete validation/test reruns for every seed; re-solving all 200 classical models; byte-identical regeneration of the multi-stage synthetic dataset; installation in a fresh OS/container; public hosting or GitHub upload. These are not implied by a successful smoke test.

The code tree was also copied to a standalone ASCII-named temporary directory. Its data-independent CPU demo and paper-table checks passed there using the installed environment; this is a path-portability check, not a fresh dependency installation. The manuscript changed during preparation. Its final fingerprint, current printed-table checks, figure links and comparison with an existing manuscript snapshot are recorded in `evidence/verification/manuscript_recheck.json`; the original initial fingerprint remains in `evidence/manuscript_identity.json`.

Exact replay of historical weights and exact retraining are different claims. CUDA kernels, device architecture, driver and dependency changes can alter trajectories. The code preserves the original BF16/FP32 boundaries, seeds, deterministic resize and data/noise contracts; historical hashes and numerical references allow differences to be detected.

## Changes made for release

- Centralized explicit asset paths and selected-model identities; removed workstation-bound formal-experiment bookkeeping from public execution.
- Retained original model/physics/loss/metric numerical routines; implemented Obs-only as an explicit flag instead of the historical runtime patch.
- Strengthened strict same-seed stage-one loading, shape/dtype checks, normalization checks and frozen-state verification.
- Preserved epoch-zero eligibility and training-only loss calibration.
- Recovered the exact forward-operator generator, IRLS initial solution, interrupted-worker recovery and final plotting sources.
- Translated non-English comments/docstrings; preserved numerical expressions and data.
- Disabled automatic deletion of existing smoke-generation directories; audit output cannot overwrite the frozen dataset report.
- Protected existing freeze manifests and completed worker splits; new-dataset freezing accepts a successful external audit report.
- Isolated drafts, old drivers, caches and test outputs outside the proposed GitHub tree. A beta selected in a separate exploratory validation exercise was excluded from the manuscript comparison tables.

Legacy optional model branches remain inside the shared architecture implementation for state-layout provenance, but are not supported release experiments. Only the 27 frozen recipes are public training/evaluation contracts; `legacy full`, depth/SAB/Tikhonov variants and field posthoc refinements must not be substituted for TARF-GDC.
