# TARF-GDC

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reproducibility candidate for three-dimensional density-contrast inversion from full-tensor gravity gradients, using Tensor-Aware Role Fusion (TARF) and Gated Density Correction (GDC).

This release implements the frozen-initial-network, one-step GDC manuscript version. It is **not** an earlier model called `Full`. The release contains nine configurations, three training seeds, dataset-generation code, training/evaluation, classical inversion, field preprocessing, figure-generation code and verification tests.

**Publication status:** public release accompanying the Computers & Geosciences manuscript "TARF-GDC: three-dimensional density-contrast inversion from full-tensor gravity gradients". Repository: <https://github.com/xiangjianbin/TARF-GDC>. First public release: 2026. The frozen dataset, forward operator, trained weights and supporting source arrays are maintained separately from the code. The trained model weights are planned for public release on Zenodo. The record DOI, version, included checkpoints and license will be added after the deposit is published; until then, this repository does not claim that the weights are publicly downloadable. See [docs/DATA.md](docs/DATA.md) for the current asset status.

## Contact

Repository and software enquiries: Jianbin Xiang (<xiangjianbin25@mails.ucas.ac.cn>).

## License

This code is released under the [MIT License](LICENSE). Copyright (c) 2026 Jianbin Xiang, Xin Wang, Xingping Hu, Yuchen Xu (National Institute of Natural Hazards, Ministry of Emergency Management of China). The separately maintained data, model weights and third-party field data are **not** covered by this license; their terms are described in [docs/DATA.md](docs/DATA.md).

## Citation

If you use this code, please cite the manuscript and this repository. Machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff).

## Quick start without the research dataset

Tested on Python 3.8.10. Create an isolated environment, install the CPU build of PyTorch, then the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python scripts/demo.py --output outputs/demo
python -m pytest tests -q
```

The demo generates an analytical prism example, adds reproducible noise, performs one optimization step and checks a checkpoint round trip. It requires no downloaded data, operator or pretrained weights. It is a diagnostic example, **not a reproduction of the paper's accuracy**. Results are written to a new output directory.

## Reproduce the paper with the external assets

Place the separately distributed asset bundle beside this repository:

```text
TARF-GDC/                 code, configs, documentation and small reference tables
TARF-GDC-assets/          frozen data, operator, checkpoints and source arrays
```

```bash
python reproduce.py list
python scripts/check_release.py --assets ../TARF-GDC-assets --predictions --output outputs/check.json
python reproduce.py evaluate --assets ../TARF-GDC-assets --model 1101_s1107 --split validation_iid --output outputs/validation_s1107
```

Use CUDA/BF16 for the historical prediction comparison. For a CUDA installation, install `torch==2.4.1` from the `https://download.pytorch.org/whl/cu121` index instead of the CPU index. The recorded platform is Ubuntu 20.04.6, Python 3.8.10, PyTorch 2.4.1, CUDA 12.1, cuDNN 9.1, NVIDIA RTX 5000 Ada (32 GB), and Intel Xeon E5-2620 v4. CPU execution is supported for diagnostics, but is not claimed to reproduce CUDA/BF16 values bit for bit. Other hardware/software combinations have not been certified.

## Documentation

- [Step-by-step user guide](docs/USER_GUIDE.md): training order, evaluation, calibration, classical and field runs.
- [Data and generation](docs/DATA.md): schema, units, seeds, noise, hashes, asset availability and historical staging.
- [Manuscript coverage](docs/EXPERIMENT_MAP.md): every experimental section and Figures 1–14.
- [Verification scope](docs/VERIFICATION.md): what was executed, what was compared, and what was not rerun.
- [Publication checklist](docs/PUBLICATION_CHECKLIST.md): current license, attribution and external-asset release status.

Run commands from this repository root. Never point output arguments into the frozen asset directory. Keep original checkpoints and datasets unchanged. Historical checkpoints contain Python metadata: deserialize only trusted, hash-verified files. This repository does not download or execute arbitrary checkpoints.

## Model identifiers

| Prefix | Manuscript configuration | Initial checkpoint for follow-up training |
|---|---|---|
| `T0P` | SC | None |
| `B0P` | MC | None |
| `0001` | Base | None |
| `1001` | Base-TARF | None |
| `0101` | Base-Correction | Same-seed Base |
| `1101` | TARF-GDC | Same-seed Base-TARF |
| `0001C` | Base-FT | Same-seed Base |
| `1001C` | Base-TARF-FT | Same-seed Base-TARF |
| `Obs-only` | Observation-only correction control | Same-seed Base-TARF |

Each prefix has `_s1107`, `_s1108`, and `_s1109`. The registry records all 27 checkpoint hashes and selected epochs. Reruns are labelled `release` or `smoke`, never retroactively labelled as the original formal experiment.
