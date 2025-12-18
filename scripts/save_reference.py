#!/usr/bin/env python3
"""Save a training checkpoint as a reference model.

Usage:
    # Simplified mode (timestamp from evaluator):
    python scripts/save_reference.py 2025-11-04_15-30-45_experiment_name

    # Legacy mode (explicit paths):
    python scripts/save_reference.py model_name --run-dir logs/train/runs/2025-10-21_11-10-32
"""

import argparse
import re
import shutil
import sys
import yaml
from pathlib import Path
from typing import List, Optional

RUNS_DIR = Path("logs/train/runs")
REFERENCE_DIR = Path("checkpoints/reference")
# Matches: timestamp_experiment (e.g., 2025-12-18_03-41-00_wah_cnn_tiny_regression)
TIMESTAMP_EXPERIMENT_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})(?:_(.+))?$")
# Legacy pattern for old directories with just timestamp
TIMESTAMP_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$")


def get_experiment_name(run_dir: Path) -> Optional[str]:
    """Extract experiment name from hydra config."""
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        return None

    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
            return cfg.get("experiment")
    except Exception:
        pass
    return None


def find_best_checkpoint(run_dir: Path) -> Optional[Path]:
    """Find the best checkpoint in a run directory."""
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None

    # Prefer the latest epoch checkpoint
    epoch_ckpts = sorted(ckpt_dir.glob("epoch_*.ckpt"), reverse=True)
    if epoch_ckpts:
        return epoch_ckpts[0]

    # Fall back to last.ckpt
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        return last_ckpt

    return None


def list_recent_runs(n: int = 5) -> List[Path]:
    """List the n most recent run directories."""
    if not RUNS_DIR.exists():
        return []
    runs = sorted(RUNS_DIR.glob("20*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[:n]


def parse_run_dir_name(dir_name: str) -> tuple[str, Optional[str]]:
    """Parse run directory name into (timestamp, experiment_name).

    Handles both new format (2025-12-18_03-41-00_wah_cnn_tiny) and
    legacy format (2025-12-18_03-41-00).
    """
    match = TIMESTAMP_EXPERIMENT_PATTERN.match(dir_name)
    if match:
        return match.group(1), match.group(2)
    return dir_name, None


def save_reference(name: Optional[str] = None, run_dir: Optional[Path] = None) -> int:
    """Save a checkpoint as a reference model."""
    # If no name provided, use most recent run
    if name is None:
        recent = list_recent_runs(1)
        if not recent:
            print("Error: No runs found in logs/train/runs/")
            return 1
        run_dir = recent[0]

        # Parse directory name - new format already includes experiment name
        timestamp, experiment_from_dir = parse_run_dir_name(run_dir.name)

        # Use experiment from directory name, or fall back to hydra config (for legacy dirs)
        experiment = experiment_from_dir or get_experiment_name(run_dir)

        if experiment:
            name = f"{timestamp}_{experiment}"
        else:
            name = timestamp

        print(f"Using most recent run: {run_dir.name}")
        print(f"Target name: {name}")

    # Determine run directory
    if run_dir is None:
        # Try to match as new format first (timestamp_experiment)
        match = TIMESTAMP_EXPERIMENT_PATTERN.match(name)
        if not match:
            print(f"Error: NAME must start with YYYY-MM-DD_HH-MM-SS or specify --run-dir")
            print(f"\nRecent runs:")
            for run in list_recent_runs():
                print(f"  {run.name}")
            return 1

        # Try new format directory first, fall back to timestamp-only (legacy)
        run_dir = RUNS_DIR / name
        if not run_dir.exists():
            timestamp = match.group(1)
            run_dir = RUNS_DIR / timestamp

    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        print(f"\nRecent runs:")
        for run in list_recent_runs():
            print(f"  {run.name}")
        return 1

    # Find checkpoint
    ckpt = find_best_checkpoint(run_dir)
    if ckpt is None:
        print(f"Error: No checkpoint found in {run_dir}/checkpoints/")
        return 1

    # Create reference directory and copy
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    dest = REFERENCE_DIR / f"{name}.ckpt"

    print(f"Source: {ckpt}")
    print(f"Dest:   {dest}")
    shutil.copy2(ckpt, dest)

    print(f"\n✅ Saved reference checkpoint: {dest}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Save a training checkpoint as a reference model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 2025-11-04_15-30-45_my_experiment
  %(prog)s my_model_v1 --run-dir logs/train/runs/2025-10-21_11-10-32
        """,
    )
    parser.add_argument(
        "name",
        nargs="?",
        default=None,
        help="Reference name (default: most recent run timestamp)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Explicit run directory (default: inferred from timestamp in name)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List recent runs and exit",
    )

    args = parser.parse_args()

    if args.list:
        print("Recent runs:")
        for run in list_recent_runs(10):
            ckpt = find_best_checkpoint(run)
            status = f"✓ {ckpt.name}" if ckpt else "✗ no checkpoint"
            print(f"  {run.name}  {status}")
        return 0

    return save_reference(args.name, args.run_dir)


if __name__ == "__main__":
    sys.exit(main())
