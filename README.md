# TARF-GDC

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reproducibility candidate for three-dimensional density-contrast inversion from full-tensor gravity gradients, using Tensor-Aware Role Fusion (TARF) and Gated Density Correction (GDC).

This release implements the frozen-initial-network, one-step GDC manuscript version. It is **not** an earlier model called `Full`. The release contains nine configurations, three training seeds, dataset-generation code, training/evaluation, classical inversion, field preprocessing, figure-generation code and verification tests.

**Publication status:** The source code accompanies the Computers & Geosciences manuscript "TARF-GDC: A deep learning network for 3D density contrast inversion from full tensor gravity gradients". The frozen synthetic dataset, fixed forward operator, 27 trained checkpoints, and supporting assets are publicly available as [Zenodo assets v1.0.0](https://doi.org/10.5281/zenodo.23235197), published on 8 October 2026. Source code uses the MIT License; the Zenodo assets use CC BY 4.0. Raw third-party Vinton observations are excluded. See [asset contents](docs/DATA.md) and [download and verification instructions](docs/DOWNLOAD.md). The asset version does not denote a GitHub source-code tag.

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

Download, reassemble, and verify the [Zenodo asset archives](docs/DOWNLOAD.md), then place the extracted asset directory beside this repository:

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
- [Download assets](docs/DOWNLOAD.md): the published DOI, split archives, checksums, and licenses.
- [Reported metrics](docs/METRICS.md): validation/test coverage, complete held-out metrics, and statistical interpretation.
- [Data and generation](docs/DATA.md): schema, units, seeds, noise, hashes, asset availability and historical staging.
- [Manuscript coverage](docs/EXPERIMENT_MAP.md): current experimental sections and Figures 1–13, plus an unnumbered field diagnostic.
- [Verification scope](docs/VERIFICATION.md): what was executed, what was compared, and what was not rerun.

Run commands from this repository root. Never point output arguments into the frozen asset directory. Keep original checkpoints and datasets unchanged. Historical checkpoints contain Python metadata: deserialize only trusted, hash-verified files. This repository does not download or execute arbitrary checkpoints.

## Model identifiers

| Prefix | Manuscript configuration | Initial checkpoint for follow-up training |
|---|---|---|
| `T0P` | SC | None |
| `B0P` | MC | None |
| `0001` | Base | None |
| `1001` | Base-TARF | None |
| `0101` | Base-GDC | Same-seed Base |
| `1101` | TARF-GDC | Same-seed Base-TARF |
| `0001C` | Base-FT | Same-seed Base |
| `1001C` | Base-TARF-FT | Same-seed Base-TARF |
| `Obs-only` | Observation-only correction control | Same-seed Base-TARF |

Display names are mapped by registered configuration ID: `0101` is Base-GDC and `1101` is TARF-GDC. Historical CSV labels `Base-correction` and `Full` refer to these configurations only when their IDs confirm that provenance. Original evidence files, checkpoint IDs, and hashes are retained.

Each prefix has `_s1107`, `_s1108`, and `_s1109`. The registry records all 27 checkpoint hashes and selected epochs. Reruns are labelled `release` or `smoke`, never retroactively labelled as the original formal experiment.
