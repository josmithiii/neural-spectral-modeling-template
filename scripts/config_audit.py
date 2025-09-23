"""Hydra configuration hygiene auditor.

Checks that defaults in train/eval/experiment configs reference existing YAML files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf

CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs"


def _normalize_group(key: str) -> Optional[Tuple[Path, bool]]:
    """Convert a Hydra default key into a configs/ subpath.

    Returns a tuple of (path, optional_flag) or None when the key should be ignored.
    """
    key = key.strip()
    if key in {"_self_", "hparams_search", "experiment", "debug"}:
        return None
    optional = False
    if key.startswith("override "):
        key = key.split("override ", 1)[1]
    if key.startswith("optional "):
        optional = True
        key = key.split("optional ", 1)[1]
    if key.startswith("/"):
        key = key[1:]
    if key.endswith(".yaml"):
        key = key[:-5]
    if not key:
        return None
    return CONFIG_ROOT / key, optional


def _candidates(group_path: Path, value: str) -> List[Path]:
    """Return candidate file paths for a group/value pair."""
    candidates: List[Path] = []
    if value is None:
        return candidates
    value = str(value).strip()
    if value.lower() in {"null", "none"}:
        return candidates
    if not value:
        return candidates
    if value.endswith(".yaml"):
        candidates.append(group_path / value)
    else:
        candidates.append(group_path / f"{value}.yaml")
    if group_path.is_file():
        candidates.append(group_path)
    else:
        candidates.append(group_path.with_suffix(".yaml"))
    return candidates


def _check_defaults(cfg: DictConfig, source: Path) -> List[str]:
    missing: List[str] = []
    defaults = cfg.get("defaults", [])
    for entry in defaults:
        if not isinstance(entry, DictConfig):
            continue
        for raw_key, value in entry.items():
            result = _normalize_group(str(raw_key))
            if result is None:
                continue
            group_path, optional = result
            if optional:
                continue
            candidates = _candidates(group_path, value)
            if not candidates:
                continue
            for candidate in candidates:
                if candidate.is_file():
                    break
            else:
                missing.append(f"{source}: '{raw_key}: {value}' -> missing file near {group_path}")
    return missing


def audit(paths: Iterable[Path]) -> int:
    missing: List[str] = []
    for path in paths:
        cfg = OmegaConf.load(path)
        missing.extend(_check_defaults(cfg, path))
    if missing:
        print("✖ Configuration issues found:")
        for item in missing:
            print(f"  - {item}")
        return 1
    print("✓ All referenced Hydra configs resolved successfully.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=None,
        help="Explicit config files to audit (defaults to train/eval + experiment configs).",
    )
    args = parser.parse_args()

    targets: List[Path]
    if args.paths:
        targets = [p for p in args.paths if p.exists()]
    else:
        experiment_cfgs = sorted((CONFIG_ROOT / "experiment").glob("*.yaml"))
        targets = [CONFIG_ROOT / "train.yaml", CONFIG_ROOT / "eval.yaml"] + experiment_cfgs
    exit(audit(targets))


if __name__ == "__main__":
    main()
