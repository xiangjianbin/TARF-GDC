# Publication checklist — release status

This repository is the public code release accompanying the Computers & Geosciences manuscript. The Computers & Geosciences software instructions call for a public repository at submission, an explicit license, English documentation/comments, installation/dependency information, and sufficient reproducibility material (or justified data restrictions with a runnable example). The authors should recheck the [official journal instructions](https://shop.elsevier.com/journals/computers-and-geosciences/0098-3004) when submitting.

Resolved for this release:

- **License:** the code tree is released under the MIT License; see [LICENSE](../LICENSE). Software, model-weight, synthetic-data and third-party field-data permissions remain distinct: the MIT license covers this code tree only.
- **Repository:** <https://github.com/xiangjianbin/TARF-GDC> (public). First public release year: 2026. Upload **only this `TARF-GDC` directory**, not its parent research/submission folder.
- **Contact:** Jianbin Xiang (<xiangjianbin25@mails.ucas.ac.cn>).
- **Citation metadata:** [CITATION.cff](../CITATION.cff) records the software title, authors, year, license, contact email and repository URL. A weights-only Zenodo DOI belongs in the data documentation, not in the software DOI field.
- **Comments/docstrings:** English throughout the Python tree; this is enforced by `tests/test_release.py::test_all_comments_and_docstrings_english`.

Still required from the authors (outside this code tree):

- Publish the planned Zenodo record for the trained weights. Record its DOI, version, included checkpoints and license in the README and data guide after the record is live; do not describe the weights as publicly available before then.
- Decide how the frozen synthetic dataset and G6 will be made available. Preserve their directory layout and hashes. If an asset cannot be shared, document the specific reason and provide a runnable alternative where possible. Do not put the 8+ GB research archive into ordinary Git history.
- Do not upload the restricted Vinton working directory until its source, permission and redistribution conditions are confirmed. It is outside both the code tree and the distributable synthetic asset bundle. If redistribution is prohibited, provide truthful source and access instructions and retain the runnable synthetic example.
- Update the manuscript's data and code statements after the Zenodo DOI and the dataset/operator access decision are final, and resolve the dangling field-geometry Table 9 reference.
- Test the downloaded release in a new environment before representing it as independently installed/reproduced.

Original research and process archives remain outside the code tree. They are recoverable, but may contain obsolete paths, incomplete runs, intermediate claims or private metadata; they should not be uploaded wholesale.
