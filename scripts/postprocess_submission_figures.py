#!/usr/bin/env python3
"""Apply the retained Figure 1 title-band edit and audit submission figures.

The historical Figure 1 edit was a presentation-only raster operation. It
replaced nine titles inside their original title bands and did not touch the
data panels. The current plotting source already emits the same terminology;
this command remains available for an archived pre-edit PNG or for a fresh
submission audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import numpy as np


LABELS = (
    ("Ellipsoid", "Ellipsoid"),
    ("Superellipsoid", "Superellipsoid"),
    ("Box", "Cuboid"),
    ("Cylinder", "Cylinder"),
    ("Lens", "Lens"),
    ("Frustum", "Truncated cone"),
    ("Dipping slab", "Inclined slab"),
    ("Dike", "Dike"),
    ("Wedge", "Wedge"),
    ("Faulted body", "Fault-displaced body"),
    ("Curved layer", "Curved layer"),
    ("GRF level set", "Gaussian random field level set"),
    ("Harmonic level set", "Harmonic level set"),
    ("Blob union", "Union of blobs"),
    ("Body with flank", "Main body with wings"),
    ("Notched body", "Notched body"),
    ("Weak appendage", "Weak appendage"),
    ("Density gradient", "Density gradient body"),
    ("Zoned density", "Zoned density body"),
)
MIN_SINGLE_COLUMN_PIXELS = 1063
MIN_FULL_PAGE_PIXELS = 2244


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_font(requested: Path | None) -> Path:
    candidates = []
    if requested:
        candidates.append(requested)
    candidates.extend(
        [
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("No usable TrueType font was found")


def rewrite_figure1(source: Path, destination: Path, audit_path: Path, font_path: Path | None) -> None:
    """Replace only the historical Figure 1 title bands and write 600 dpi."""
    image = Image.open(source).convert("RGB")
    before = np.asarray(image).copy()
    draw = ImageDraw.Draw(image)
    font_file = choose_font(font_path)
    scale_x, scale_y = image.width / 1408.0, image.height / 1770.0
    x0 = [81, 424, 766, 1107]
    y0 = [50, 380, 709, 1039, 1369]
    boxes = []
    for index, (old, new) in enumerate(LABELS):
        column, row = index % 4, index // 4
        x = round(x0[column] * scale_x)
        y = round(y0[row] * scale_y)
        right = round((x0[column] + 294) * scale_x)
        bottom = round((y0[row] + 31) * scale_y)
        box = (x - round(8 * scale_x), y, right, bottom)
        boxes.append(box)
        draw.rectangle(box, fill="white")
        size = 56
        while True:
            font = ImageFont.truetype(str(font_file), size)
            if draw.textbbox((0, 0), new, font=font)[2] <= right - x - 6 or size <= 6:
                break
            size -= 1
        draw.text((x + 4, y + 12), new, font=font, fill="black", anchor="lt")

    after = np.asarray(image)
    allowed = np.zeros(before.shape[:2], dtype=bool)
    for left, top, right, bottom in boxes:
        allowed[top : bottom + 1, left : right + 1] = True
    changed = np.any(before != after, axis=2)
    if np.any(changed & ~allowed):
        raise AssertionError("Figure 1 post-processing changed pixels outside title bands")

    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, dpi=(600, 600))
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(
            {
                "source": str(source),
                "output": str(destination),
                "sha256": sha256(destination),
                "size": list(image.size),
                "dpi": [600, 600],
                "only_title_bands_changed": True,
                "changes": [
                    {"panel": chr(97 + i), "before": old, "after": new}
                    for i, (old, new) in enumerate(LABELS)
                    if old != new
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def figure_path(directory: Path, index: int) -> Path | None:
    for name in (f"fig{index:02d}.png", f"Figure_{index}.png", f"Figure{index:02d}.png"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def audit_figures(directory: Path, output: Path) -> None:
    """Audit PNG dimensions, metadata and available vector companions."""
    rows = []
    errors = []
    for index in range(1, 14):
        path = figure_path(directory, index)
        if path is None:
            errors.append(f"missing Figure {index} PNG")
            continue
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (None, None))
            width, height = image.size
            row = {
                "figure": index,
                "path": str(path),
                "width_px": width,
                "height_px": height,
                "dpi_x": dpi[0],
                "dpi_y": dpi[1],
                "sha256": sha256(path),
                "single_column_width_pass": width >= MIN_SINGLE_COLUMN_PIXELS,
                "full_page_width_pass": width >= MIN_FULL_PAGE_PIXELS,
                "svg_present": path.with_suffix(".svg").is_file(),
                "pdf_present": path.with_suffix(".pdf").is_file(),
            }
            if width < MIN_SINGLE_COLUMN_PIXELS:
                errors.append(f"Figure {index} is narrower than {MIN_SINGLE_COLUMN_PIXELS}px")
            rows.append(row)
    report = {"directory": str(directory), "figures": rows, "errors": errors, "passed": not errors}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise SystemExit("; ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    edit = sub.add_parser("figure1", help="rewrite Figure 1 title bands")
    edit.add_argument("source", type=Path)
    edit.add_argument("destination", type=Path)
    edit.add_argument("--audit", type=Path, default=Path("figure1_label_audit.json"))
    edit.add_argument("--font", type=Path)
    audit = sub.add_parser("audit", help="audit Figure 1--13 PNG files")
    audit.add_argument("directory", type=Path)
    audit.add_argument("--output", type=Path, default=Path("figure_submission_audit.json"))
    args = parser.parse_args()
    if args.command == "figure1":
        rewrite_figure1(args.source, args.destination, args.audit, args.font)
    else:
        audit_figures(args.directory, args.output)


if __name__ == "__main__":
    main()
