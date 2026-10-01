#!/usr/bin/env python3
"""Save a training checkpoint as a reference model.

Usage:
    # Simplified mode (timestamp from evaluator):
    python scripts/save_reference.py 2025-11-04_15-30-45_experiment_name

    # Legacy mode (explicit paths):
    python scripts/save_reference.py model_name --run-dir logs/train/runs/2025-10-21_11-10-32

    # Make an older reference checkpoint self-contained for audio_reconstruction_eval.py:
    python scripts/save_reference.py --embed checkpoints/reference/NAME.ckpt --run-dir logs/train/runs/RUN

The saved checkpoint carries the run's dataset-configured model/data config
(hyper_parameters.run_config), so it can be rebuilt exactly without its run directory.
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
    """Find the best (monitored-metric) checkpoint in a run directory.

    The newest epoch_*.ckpt is NOT necessarily the best (save_top_k > 1), so use the
    best.ckpt symlink written by train.py, or for older runs the best_model_path that
    ModelCheckpoint recorded in last.ckpt.
    """
    import torch

    ckpt_dir = run_dir / "checkpoints"
    best_link = ckpt_dir / "best.ckpt"
    if best_link.exists():
        return best_link.resolve()
    last_ckpt = ckpt_dir / "last.ckpt"
    if not last_ckpt.exists():
        return None
    state = torch.load(last_ckpt, map_location="cpu", weights_only=False)
    for key, cb_state in state.get("callbacks", {}).items():
        best = cb_state.get("best_model_path") if key.startswith("ModelCheckpoint") else None
        if best:
            # Recorded path may be relative to another cwd; the file lives in ckpt_dir
            candidate = ckpt_dir / Path(best).name
            if candidate.exists():
                return candidate
    return None


def embed_run_config(ckpt_path: Path, run_dir: Path) -> None:
    """Store the run's dataset-configured model/data config in the checkpoint hparams.

    audio_reconstruction_eval.py rebuilds models from this ``run_config`` (or the
    run's .hydra dir, which a copied reference checkpoint no longer sits next to).
    """
    import torch
    from omegaconf import OmegaConf

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.train import configure_vimh_run_config

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = state.setdefault("hyper_parameters", {})
    if "run_config" in hparams:
        return
    hydra_cfg = run_dir / ".hydra" / "config.yaml"
    if not hydra_cfg.exists():
        raise FileNotFoundError(f"No run_config in {ckpt_path} and no Hydra config {hydra_cfg}")
    cfg = OmegaConf.load(hydra_cfg)
    configure_vimh_run_config(cfg)
    hparams["run_config"] = {
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "data": OmegaConf.to_container(cfg.data, resolve=True),
    }
    torch.save(state, ckpt_path)
    print(f"Embedded run_config from {hydra_cfg}")


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
    embed_run_config(dest, run_dir)

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
    parser.add_argument(
        "--embed",
        type=Path,
        help="Embed run_config (from --run-dir) into an existing reference checkpoint and exit",
    )

    args = parser.parse_args()

    if args.embed:
        if args.run_dir is None:
            parser.error("--embed requires --run-dir")
        embed_run_config(args.embed, args.run_dir)
        return 0

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
