# Publication checklist — release status

This repository is the public code release accompanying the Computers & Geosciences manuscript. The Computers & Geosciences software instructions call for a public repository at submission, an explicit license, English documentation/comments, installation/dependency information, and sufficient reproducibility material (or justified data restrictions with a runnable example). The authors should recheck the [official journal instructions](https://shop.elsevier.com/journals/computers-and-geosciences/0098-3004) when submitting.

Resolved for this release:

- **License:** the code tree is released under the MIT License; see [LICENSE](../LICENSE). Software, model-weight, synthetic-data and third-party field-data permissions remain distinct: the MIT license covers this code tree only.
- **Repository:** <https://github.com/xiangjianbin/TARF-GDC> (public). First public release year: 2026. Upload **only this `TARF-GDC` directory**, not its parent research/submission folder.
- **Citation metadata:** [CITATION.cff](../CITATION.cff) records the software title, authors, year, license and repository URL; add the archive DOI there once minted.
- **Comments/docstrings:** English throughout the Python tree; this is enforced by `tests/test_release.py::test_all_comments_and_docstrings_english`.

Still required from the authors (outside this code tree):

- Host synthetic data, G6, 27 checkpoints and required source arrays separately; preserve their directory layout and hashes. Add actual versioned download links/DOI and an asset license to the README/data guide. Do not put an 8+ GB research archive into ordinary Git history.
- Do not upload the separately isolated `PRIVATE_Vinton_DO_NOT_UPLOAD` directory until Vinton source/permission/redistribution conditions are confirmed. It is outside both the code tree and the distributable asset bundle. If redistribution is prohibited, provide truthful access instructions and retain the runnable synthetic example.
- Replace the manuscript's code/data availability placeholders with the released URL/version/license, and resolve the dangling field-geometry Table 9 reference.
- Test the downloaded release in a new environment before representing it as independently installed/reproduced.

Local upload preparation:

```bash
git init
git add README.md LICENSE CITATION.cff requirements.txt requirements-operator.txt .gitignore SHA256SUMS reproduce.py code configs assets dataset_generation scripts tests figures docs evidence
git status --short
git diff --cached --stat
```

Then create the commit and connect the author's remote.

Original research and process archives remain outside the code tree. They are recoverable, but may contain obsolete paths, incomplete runs, intermediate claims or private metadata; they should not be uploaded wholesale.
