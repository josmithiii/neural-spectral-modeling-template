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
from pathlib import Path
from typing import List, Optional

RUNS_DIR = Path("logs/train/runs")
REFERENCE_DIR = Path("checkpoints/reference")
TIMESTAMP_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")


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


def save_reference(name: Optional[str] = None, run_dir: Optional[Path] = None) -> int:
    """Save a checkpoint as a reference model."""
    # If no name provided, use most recent run
    if name is None:
        recent = list_recent_runs(1)
        if not recent:
            print("Error: No runs found in logs/train/runs/")
            return 1
        run_dir = recent[0]
        name = run_dir.name
        print(f"Using most recent run: {name}")

    # Determine run directory
    if run_dir is None:
        match = TIMESTAMP_PATTERN.match(name)
        if not match:
            print(f"Error: NAME must start with YYYY-MM-DD_HH-MM-SS or specify --run-dir")
            print(f"\nRecent runs:")
            for run in list_recent_runs():
                print(f"  {run.name}")
            return 1
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
