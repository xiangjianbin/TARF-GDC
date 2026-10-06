#!/usr/bin/env python3
"""Freeze a newly generated dataset; never rewrite an existing freeze."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from v017.config import sha256_file, write_json


ROOT = Path(__file__).resolve().parent


def freeze(args: argparse.Namespace) -> Path:
    dataset = Path(args.dataset).resolve()
    sums_path = dataset / "SHA256SUMS"
    if sums_path.exists() or (dataset / "freeze_manifest.json").exists():
        raise FileExistsError("An existing dataset freeze must not be overwritten")
    audit_source = args.audit_report or dataset / "audit_report.json"
    audit = json.loads(Path(audit_source).read_text(encoding="utf-8"))
    if audit.get("verdict") != "PASS" or audit.get("failures"):
        raise ValueError("Only a successfully audited dataset can be frozen")
    if args.audit_report:
        if (dataset / "audit_report.json").exists():
            raise FileExistsError("The newly generated dataset already contains an audit report")
        write_json(dataset / "audit_report.json", audit)
    lines: list[str] = []
    total_bytes = 0
    n_files = 0
    for path in sorted(dataset.rglob("*")):
        if not path.is_file() or path.name in ("SHA256SUMS", "freeze_manifest.json"):
            continue
        digest = sha256_file(path)
        lines.append(f"{digest}  {path.relative_to(dataset)}")
        total_bytes += path.stat().st_size
        n_files += 1
        if n_files % 200 == 0:
            print(f"hashed {n_files} files", flush=True)
    sums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    audit = json.loads((dataset / "audit_report.json").read_text(encoding="utf-8"))
    manifest = {
        "dataset_id": "V023_STANDALONE_FULL",
        "frozen_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "file_count": n_files,
        "total_bytes": total_bytes,
        "sha256sums_file": "SHA256SUMS",
        "sha256sums_sha256": sha256_file(sums_path),
        "audit_verdict": audit.get("verdict"),
        "audit_report": "audit_report.json",
        "splits": {
            split_dir.name: {
                "endpoints": json.loads(
                    (dataset / f"summary_{split_dir.name}.json").read_text(encoding="utf-8")
                )["total_endpoints"],
            }
            for split_dir in sorted(dataset.iterdir())
            if split_dir.is_dir()
        },
    }
    write_json(dataset / "freeze_manifest.json", manifest)
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the final V023 dataset")
    parser.add_argument("--dataset", type=Path, default=ROOT / ".." / "dataset")
    parser.add_argument("--audit-report", type=Path,
                        help="Successful external audit report for a NEW generated dataset")
    return parser.parse_args()


if __name__ == "__main__":
    print(freeze(parse_args()))
