# User guide

All paths below are relative to the repository root. Use a new output directory for each run. Set `TARF_GDC_ASSETS` only where shown; the main CLI takes an explicit `--assets` argument.

## 1. Verify supplied assets

```bash
python scripts/check_release.py --assets ../TARF-GDC-assets --predictions --output outputs/release_check.json
TARF_GDC_ASSETS=../TARF-GDC-assets python dataset_generation/audit_dataset_v023.py --dataset ../TARF-GDC-assets/dataset --report-output outputs/data_audit.json
TARF_GDC_ASSETS=../TARF-GDC-assets python -m pytest tests dataset_generation/tests_v023 -q
```

The first check hashes all 170 frozen data files and replays 16 fixed r0 validation samples for every registered checkpoint. The second recomputes all endpoint contracts, all 10,005 pair gates, frozen-noise and independent prism checks. It writes its report outside the frozen dataset.

## 2. Train

```bash
# One-step diagnostic; it does not produce a manuscript result.
python reproduce.py train --assets ../TARF-GDC-assets --model 1101_s1107 --smoke --output outputs/smoke

# First stage: 50 epochs.
python reproduce.py train --assets ../TARF-GDC-assets --model 1001_s1107 --output outputs/rerun

# Second stage: 50 epochs from YOUR corresponding first-stage best checkpoint.
python reproduce.py train --assets ../TARF-GDC-assets --model 1101_s1107 --stage1-checkpoint outputs/rerun/1001_s1107/checkpoints/best.pt --output outputs/rerun
```

Repeat for seeds 1108/1109 and the configurations in the README. Train Base and Base-TARF first, then their matching correction/FT controls. Without `--stage1-checkpoint`, follow-up training deliberately starts from the **distributed historical** same-seed first-stage checkpoint. Supply your newly trained checkpoint when reproducing the complete training chain. `--resume` resumes the same resolved configuration; it does not allow changing the experiment midway.

GDC freezes the initial predictor, including dropout behavior, and trains 58,849 correction parameters. FT keeps the entire initial predictor trainable. Epoch 0 is an eligible follow-up checkpoint. AdamW uses learning rate and weight decay 1e-4, batch size 16, gradient clipping 1, warmup fraction 0.05 and dropout 0.05. See each JSON recipe for the complete settings. Validation SAE on fixed realization r0 selects checkpoints. Test data must not select a checkpoint. BF16 applies to the network; input preparation and the physical calculations keep their original precision boundaries.

## 3. Evaluate

```bash
python reproduce.py evaluate --assets ../TARF-GDC-assets --model 1101_s1107 --split validation_iid --output outputs/val
python reproduce.py evaluate --assets ../TARF-GDC-assets --model 1101_s1107 --split test_locked --output outputs/test
```

By default this entry point evaluates hash-registered historical checkpoints. To evaluate your complete rerun, add `--checkpoint outputs/rerun/1101_s1107/checkpoints/best.pt`. The loader checks seed, architecture, loss, noise, task and normalization and refuses smoke checkpoints. The output records the actual checkpoint hash and whether it was the historical checkpoint. Only pass trusted checkpoint files. The evaluator requires all 2,500/10,710 endpoints and 650/2,855 pairs for each of five frozen realizations. First average noise within each model/pair, then average examples within each seed; manuscript error bars are sample SD across three seeds (`ddof=1`). Pair metrics use the full predicted A–B density difference, not only displayed slices.

## 4. Recompute physical assets

```bash
python -m pip install -r requirements-operator.txt
python scripts/build_operator.py --output outputs/operator/g6.npy
python scripts/prepare_assets.py --assets ../TARF-GDC-assets --output outputs/prepared --operators
python scripts/calibrate_physics.py --assets ../TARF-GDC-assets --output outputs/loss_calibration.json
```

The first command reconstructs the exact 6,534 × 16,384 float32 matrix. Preparation hashes the frozen data, creates a training cache, computes normalization from **all 25,000 training endpoints**, and checks exact numerical equality against the reference. `--operators` also rebuilds sensitivity/backprojection scales and the historical optional Tikhonov matrices; the latter are not used by the manuscript configurations. `--cache-all` explicitly adds validation/test caches. The main runtime can read compressed shards without these caches. Derivatives require several additional GB. Do not replace frozen references merely because a new environment produces slightly different values; investigate the differences first.

Loss calibration uses four fixed training batches of eight examples, the original 0.2 target ratio and median aggregation. Expected weight: `1.9504151332365432e-07`.

## 5. Classical inversion

```bash
python scripts/classical.py --assets ../TARF-GDC-assets --method L2 --output outputs/L2
python scripts/classical.py --assets ../TARF-GDC-assets --method IRLS --output outputs/IRLS
# Optional original training-only parameter search (potentially expensive).
python scripts/classical.py --assets ../TARF-GDC-assets --method IRLS --calibrate --output outputs/IRLS_calibration
```

The manuscript uses the same 200 fixed validation r0 models: 100 ordinary and 50 complete shape pairs. L2 beta is 3000; IRLS beta is 0.3. Parameter calibration uses 12 training models, not validation or test selection. IRLS continues from the bundled historical short-budget initial solution; cold starts and beta 0.5477225575 belong to different experiments. Inner limit 6,000, relative KKT tolerance 1e-5, at most 80 outer iterations **per continuation round**, outer-change tolerance 1e-3, epsilon 0.01, smoothness 1. An exhausted iteration limit is not reported as convergence. The driver raises if convergence is not achieved. `--limit 1` is diagnostic only. Floating-point batch grouping can affect convergence trajectories; original converged density arrays remain the exact source of the manuscript comparison.

## 6. Vinton application

```bash
python reproduce.py field --assets ../TARF-GDC-assets --field-directory /path/to/authorized_six_component_txt --output outputs/vinton
```

Required files: `Txx.txt`, `Txy.txt`, `Txz.txt`, `Tyy.txt`, `Tyz.txt`, `Tzz.txt`; comma-delimited, 1,681 rows × 5 columns. Columns 0/1 are local x/y metres on the 0–4,000 m grid; column 3 is the component in s^-2. Values are converted to Eötvös (×1e9), bilinearly resampled 41×41 → 33×33; Txz/Tyz signs are reversed for z-up. All three TARF-GDC models remain frozen. There is no field fine-tuning, posthoc correction or density-ground-truth accuracy claim. Seed 1107 was selected only for display by six-component mean R²; all seeds remain in the statistics. Author confirmation of data rights is required; the field files are not in the GitHub code directory.

## 7. Figures

```bash
python scripts/render_figures.py --assets ../TARF-GDC-assets --group 06-07 --output outputs/fig06_07
```

Available groups: `01-02`, `03`, `04`, `05`, `06-07`, `08`, `09`, `10-11`, `12`, `13`, `14`. The source arrays are copied into each fresh output directory so plot helpers cannot overwrite the immutable bundle. For groups 05 and 06-07 install the system font **Noto Sans CJK JP** used by the original Chinese manuscript. Group 13 is the final two-panel TARF-GDC-only plot. Group 08 additionally emits an older section plot; **use group 09** for final manuscript Figure 9. Replotting does not regenerate network predictions; evaluate/train separately when testing new weights.

For groups 12–14, supply `--field-source outputs/vinton` (the output of `reproduce.py field`) or another explicitly authorized processed field directory. Restricted raw/source field arrays are kept outside the distributable asset bundle. The release's group 14 entry point exports only Figure 14; the obsolete field-comparison panels have been removed from that code path.
