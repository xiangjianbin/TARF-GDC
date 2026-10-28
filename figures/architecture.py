"""Export the audited final Figure 4 artwork.

Figure 4 is an editable PowerPoint/SVG schematic rather than a Matplotlib
plot. The final manuscript PNG was produced from that artwork after the
TARF-GDC colours and labels were revised. The checked-in assets are therefore
the authoritative release source for all three submission formats.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from figure_runtime import OUTPUT

HERE = OUTPUT
ASSET_DIR = Path(__file__).resolve().parent / "assets"
ASSET_NAMES = ("Figure04.svg", "Figure04.pdf", "Figure04.png")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    missing = [name for name in ASSET_NAMES if not (ASSET_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing audited Figure 4 assets: " + ", ".join(missing)
        )

    outputs = {}
    for name in ASSET_NAMES:
        source = ASSET_DIR / name
        destination = HERE / name
        shutil.copy2(source, destination)
        outputs[name] = {
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
            "source": str(source),
        }

    provenance_path = ASSET_DIR / "Figure04_provenance.json"
    provenance = {}
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance.update({
        "generator": "figures/architecture.py",
        "asset_source": str(ASSET_DIR),
        "outputs": outputs,
        "formats_share_one_audited_source": True,
    })
    (HERE / "Figure04_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Exported audited Figure 4 assets", flush=True)


if __name__ in {"__main__", "release_figure"}:
    main()
