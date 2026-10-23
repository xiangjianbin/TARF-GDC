# Data, assets and generation history

The frozen synthetic dataset is V023, independent of V022. V022 is used only for the explicitly disclosed illustrative Figure 3. It is not part of training, validation, test or performance statistics.

| Split | Ordinary endpoints | Separable pairs | Shape pairs | Total endpoints |
|---|---:|---:|---:|---:|
| train | 12,000 | 3,250 | 3,250 | 25,000 |
| validation_iid | 1,200 | 325 | 325 | 2,500 |
| test_locked | 5,000 | 1,427 | 1,428 | 10,710 |

The asset directory contains:

```text
dataset/                 NPZ shards, records/pairs JSONL, indices, manifest, audit, SHA256SUMS
operator/g6.npy          frozen float32 forward matrix
checkpoints/             27 registered public-release checkpoints, identities in the code registry
evidence/classical/      converged L2/IRLS densities and the required IRLS initial solution
evidence/statistics/     retained statistical source arrays
figure_data/             synthetic figure sources and explicitly labelled legacy color-scale source
```

`density_zyx`: (16,32,32), g/cm³; stored deep-to-shallow. Cell sizes (x,y,z)=(125,125,62.5) m, origin=(0,0,-1000) m; depth of storage index k is `(15.5-k)*62.5` m. `clean_d6`: (6,33,33), Eötvös, ordered Txx,Txy,Txz,Tyy,Tyz,Tzz. G6 maps flattened z-y-x densities into **receiver-major** six-component observations. The five roles are q=(2Tzz-Txx-Tyy)/3, Txz, Tyz, (Txx-Tyy)/2, Txy. Do not confuse the channel-major normalized classical system with the raw receiver-major matrix.

Training noise is deterministically regenerated from sample ID and epoch, with amplitude scale uniformly sampled from [0.75,1.5]. Validation/test have five stored independent-white-noise realizations; component sigmas are (0.425,0.31,0.70,0.425,0.70,0.74) E and evaluation amplitude scale is 1. The two endpoints of a pair have independent noise. The support threshold for evaluation is 0.005 g/cm³. The 0.05 g/cm³ display threshold in Figures 8/9 is not an evaluation-mask change.

Synthetic freeze: 170 hashed files, approximately 6.69 GB decimal. SHA256 of `dataset/SHA256SUMS`: `33004e5d73c610cbc529a8615352407a6b8f1f3771ebd436357770a89b70fbfd`. G6 SHA256: `544d1ceeb63f7da3e83be606e613fb5c3ff6cbe11b7585e1f85c281d71346b0c`. Checkpoint hashes are in `assets/model_registry.json`.

The frozen V023 dataset, `operator/g6.npy`, all 27 registered public-release checkpoints, and supporting evidence are available as [Zenodo assets v1.0.0](https://doi.org/10.5281/zenodo.23235197), published on 8 October 2026 under CC BY 4.0. See [DOWNLOAD.md](DOWNLOAD.md) for all 18 archive parts, checksum verification, reassembly, and extraction. The approximately 4.45 GB download is separate from the source repository and from the extracted dataset size quoted above. The version number refers to the asset deposit, not a source-code tag.

## Regeneration code and its limits

`dataset_generation/` contains ordinary generation, separable-pair generation, shape-pair optimization, worker merging, interrupted-worker salvage, final assembly, auditing and freezing. `v017/` is an inherited implementation dependency, not a request to use the V017 dataset. Configurations and registered seeds are retained.

For a small **new** dataset-generation diagnostic:

```bash
TARF_GDC_ASSETS=../TARF-GDC-assets python dataset_generation/build_ordinary_v023.py --split train --seed 2026091501 --n 8 --output generation_work/ordinary_demo --device cuda
TARF_GDC_ASSETS=../TARF-GDC-assets python dataset_generation/build_pairs_v023.py --mode smoke --n-pairs 4 --output generation_work/separable_demo --device cuda
TARF_GDC_ASSETS=../TARF-GDC-assets python dataset_generation/build_shape_pairs_v023.py --mode smoke --n-pairs 2 --output generation_work/shape_demo --device cuda
```

The historical final dataset was assembled from multiple staged batches, including interrupted-worker checkpoints, salvaged accepted pairs, top-ups and deficit rounds. A single fresh seed invocation is **not promised to recreate it byte for byte**. Exact manuscript evaluation uses the distributed frozen dataset. Shape pairs in the final set are all tier 1 (Jaccard ≤0.7); 355 tier-2 pairs were excluded, and hash-ordered surplus trimming removed 3/2/8 pairs from train/validation/test. Those exclusions and IDs are retained in the frozen metadata.

For a newly assembled full-size dataset, write its scientific audit outside the dataset and pass that report to `freeze_dataset_v023.py --dataset /path/to/new/dataset --audit-report /path/to/PASS_report.json`. Existing checksum/freeze manifests cannot be overwritten. Worker salvage must operate on a copy of an unfinished worker directory; it refuses completed splits. These safety changes do not alter the frozen supplied dataset.

Seed families: ordinary 2026091501–03; original separable 2026091201–13; separable top-ups 2026091511–13; initial shape 2026091311 and workers 2026091401+w×1000; shape top-up bases 2026091521/1531/1541; deficit bases 2026091551/1561/1581/1591/1621/1641/1661 (+w×1000). Merge shuffle 2026091471; final IDs 2026091601–03 and final shuffle 2026091571. See the JSON seed registry and per-endpoint provenance for exact assignments.

The 10% deep-targeting policy is not an assertion that exactly 10% of every track is deep. Actual ≥600 m support-centroid fractions differ; shape pairs have no deep examples under their acceptance gates. Training separable data include a previously generated cohort. These distribution limitations must not be hidden by regenerating or filtering the test set.

## Rights and availability

The repository source code uses the MIT License. The synthetic data, forward operator, trained checkpoints, and supporting assets in Zenodo v1.0.0 use CC BY 4.0; see `LICENSE_DATA.md` in the record. Public checkpoint copies contain portable metadata; the supplied sanitization report records numerical-content equivalence and public hashes.

Raw Vinton observations are third-party data, excluded from the GitHub repository and the Zenodo asset bundle. Access remains subject to the original provider’s terms. Pass authorized raw files to `reproduce.py field`, then supply its output to `scripts/render_figures.py --field-source`. Providing preprocessing code does not itself grant access to these observations. The synthetic assets and the data-independent prism demo do not require the restricted raw field data.
