from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = read_json(config_path)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(PROJECT_ROOT)
    # Explicit asset-root override; no workstation-specific fallback.
    assets = os.environ.get('TARF_GDC_ASSETS')
    if assets:
        config['operator']['g6_path'] = str(Path(assets).resolve() / 'operator/g6.npy')
        config['operator']['manifest_path'] = str(PROJECT_ROOT.parent / 'assets/operator_specification.json')
    return config


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256_file(path: str | Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any, length: int = 20) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:length]
