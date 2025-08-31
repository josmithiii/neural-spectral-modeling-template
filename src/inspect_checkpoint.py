#!/usr/bin/env python3
"""
Checkpoint Inspection Utility for Neural Spectral Modeling

Prints detailed metadata from PyTorch Lightning checkpoints including:
- Training configuration (Hydra config)
- Model architecture and parameters
- Dataset information
- Training metrics and hyperparameters

Usage:
    python src/inspect_checkpoint.py path/to/checkpoint.ckpt
    python src/inspect_checkpoint.py  # Auto-discover latest checkpoint
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import json

import torch
import rootutils

# Setup root path
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def find_latest_checkpoint(base_dir: str = "logs/train") -> Optional[str]:
    """Find the latest checkpoint from training runs."""
    import glob
    import os
    
    patterns = [
        f"{base_dir}/runs/*/checkpoints/best.ckpt",
        f"{base_dir}/runs/*/checkpoints/last.ckpt", 
        f"{base_dir}/runs/*/checkpoints/*.ckpt",
        f"{base_dir}/*/checkpoints/best.ckpt",
        f"{base_dir}/*/checkpoints/last.ckpt",
        f"{base_dir}/*/checkpoints/*.ckpt"
    ]
    
    latest_checkpoint = None
    latest_time = 0
    
    for pattern in patterns:
        checkpoints = glob.glob(pattern)
        for ckpt_path in checkpoints:
            try:
                mtime = os.path.getmtime(ckpt_path)
                if mtime > latest_time:
                    latest_time = mtime
                    latest_checkpoint = ckpt_path
            except OSError:
                continue
    
    return latest_checkpoint


def print_dict_structure(data: Dict[str, Any], prefix: str = "", max_depth: int = 3, current_depth: int = 0) -> None:
    """Print dictionary structure with indentation."""
    if current_depth >= max_depth:
        return
        
    for key, value in data.items():
        print(f"{prefix}{key}: {type(value).__name__}")
        
        if isinstance(value, dict) and current_depth < max_depth - 1:
            print_dict_structure(value, prefix + "  ", max_depth, current_depth + 1)
        elif isinstance(value, (list, tuple)) and len(value) > 0:
            print(f"{prefix}  [{len(value)} items] - first item: {type(value[0]).__name__}")
            if isinstance(value[0], dict) and current_depth < max_depth - 1:
                print(f"{prefix}  First item structure:")
                print_dict_structure(value[0], prefix + "    ", max_depth, current_depth + 2)


def format_tensor_info(tensor: torch.Tensor) -> str:
    """Format tensor information for display."""
    return f"shape={list(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"


def inspect_checkpoint(ckpt_path: str) -> None:
    """Inspect and print checkpoint metadata."""
    if not Path(ckpt_path).exists():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    
    print(f"🔍 Inspecting checkpoint: {ckpt_path}")
    print("=" * 80)
    
    try:
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        # Basic checkpoint info
        print(f"\n📁 CHECKPOINT OVERVIEW")
        print("-" * 40)
        print(f"File size: {Path(ckpt_path).stat().st_size / 1024 / 1024:.2f} MB")
        print(f"Keys in checkpoint: {list(checkpoint.keys())}")
        
        # Training metadata
        if 'epoch' in checkpoint:
            print(f"Training epoch: {checkpoint['epoch']}")
        if 'global_step' in checkpoint:
            print(f"Global step: {checkpoint['global_step']}")
            
        # Model state dict info
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"\n🧠 MODEL STATE DICT")
            print("-" * 40)
            print(f"Number of parameters: {len(state_dict)}")
            
            # Group parameters by component
            param_groups = {}
            total_params = 0
            for name, tensor in state_dict.items():
                component = name.split('.')[0] if '.' in name else name
                if component not in param_groups:
                    param_groups[component] = []
                param_groups[component].append((name, tensor))
                total_params += tensor.numel()
            
            print(f"Total parameters: {total_params:,}")
            print(f"Parameter groups:")
            for component, params in param_groups.items():
                group_params = sum(tensor.numel() for _, tensor in params)
                print(f"  {component}: {group_params:,} parameters ({len(params)} tensors)")
            
            # Show key model parameters
            print(f"\nKey model parameters:")
            for name, tensor in list(state_dict.items())[:10]:
                print(f"  {name}: {format_tensor_info(tensor)}")
            if len(state_dict) > 10:
                print(f"  ... and {len(state_dict) - 10} more")
        
        # Hyperparameters
        if 'hyper_parameters' in checkpoint:
            hyper_params = checkpoint['hyper_parameters']
            print(f"\n⚙️  MODEL HYPERPARAMETERS")
            print("-" * 40)
            print(f"Hyperparameter keys: {list(hyper_params.keys())}")
            
            # Print important hyperparameters
            important_keys = ['_target_', 'optimizer', 'scheduler', 'net', 'compile', 'output_mode']
            for key in important_keys:
                if key in hyper_params:
                    value = hyper_params[key]
                    if isinstance(value, dict):
                        print(f"{key}:")
                        print_dict_structure(value, "  ", max_depth=2)
                    else:
                        print(f"{key}: {value}")
            
            # Look for synthesis-related parameters
            synth_keys = [k for k in hyper_params.keys() if 'synth' in k.lower() or 'param' in k.lower()]
            if synth_keys:
                print(f"\nSynthesis-related hyperparameters:")
                for key in synth_keys:
                    print(f"  {key}: {hyper_params[key]}")
        
        # Training configuration (Hydra config)
        config_keys = ['cfg', 'config', 'hydra_cfg']
        config_data = None
        for key in config_keys:
            if key in checkpoint:
                config_data = checkpoint[key]
                break
        
        if config_data:
            print(f"\n🔧 TRAINING CONFIGURATION")
            print("-" * 40)
            print(f"Configuration structure:")
            print_dict_structure(config_data, max_depth=3)
            
            # Extract key configuration sections
            if isinstance(config_data, dict):
                important_sections = ['data', 'model', 'trainer', 'experiment']
                for section in important_sections:
                    if section in config_data:
                        print(f"\n{section.upper()} CONFIG:")
                        section_data = config_data[section]
                        if isinstance(section_data, dict):
                            for key, value in section_data.items():
                                if isinstance(value, dict) and len(value) > 5:
                                    print(f"  {key}: {{...}} ({len(value)} keys)")
                                else:
                                    print(f"  {key}: {value}")
                        else:
                            print(f"  {section_data}")
        else:
            print(f"\n⚠️  No training configuration found in checkpoint")
        
        # Lightning module info
        if 'pytorch-lightning_version' in checkpoint:
            print(f"\n⚡ LIGHTNING INFO")
            print("-" * 40)
            print(f"PyTorch Lightning version: {checkpoint['pytorch-lightning_version']}")
        
        # Optimizer state
        if 'optimizer_states' in checkpoint:
            opt_states = checkpoint['optimizer_states']
            print(f"\n🎯 OPTIMIZER STATE")
            print("-" * 40)
            print(f"Number of optimizer states: {len(opt_states)}")
            if opt_states:
                first_state = opt_states[0]
                if 'param_groups' in first_state:
                    print(f"Parameter groups: {len(first_state['param_groups'])}")
                    if first_state['param_groups']:
                        pg = first_state['param_groups'][0]
                        print(f"Learning rate: {pg.get('lr', 'unknown')}")
        
        # Scheduler state
        if 'lr_schedulers' in checkpoint:
            print(f"Learning rate schedulers: {len(checkpoint['lr_schedulers'])}")
        
        # Look for dataset/data-related info
        dataset_hints = []
        if config_data and isinstance(config_data, dict):
            if 'data' in config_data:
                dataset_hints.append(f"Data config: {config_data['data']}")
        
        if hyper_params:
            data_keys = [k for k in hyper_params.keys() if 'data' in k.lower() or 'dataset' in k.lower()]
            for key in data_keys:
                dataset_hints.append(f"{key}: {hyper_params[key]}")
        
        if dataset_hints:
            print(f"\n📊 DATASET HINTS")
            print("-" * 40)
            for hint in dataset_hints:
                print(f"  {hint}")
        
        print(f"\n✅ Checkpoint inspection complete")
        
    except Exception as e:
        print(f"❌ Error inspecting checkpoint: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Inspect PyTorch Lightning checkpoint metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python src/inspect_checkpoint.py logs/train/runs/2024-01-01_12-00-00/checkpoints/best.ckpt
    python src/inspect_checkpoint.py  # Auto-discover latest checkpoint
        """
    )
    parser.add_argument(
        'checkpoint_path',
        nargs='?',
        help='Path to checkpoint file (if not provided, auto-discovers latest)'
    )
    
    args = parser.parse_args()
    
    ckpt_path = args.checkpoint_path
    if not ckpt_path:
        print("🔍 Auto-discovering latest checkpoint...")
        ckpt_path = find_latest_checkpoint()
        if not ckpt_path:
            print("❌ No checkpoints found in logs/train/")
            print("💡 Please provide checkpoint path or ensure you have trained models")
            sys.exit(1)
        print(f"📍 Found: {ckpt_path}")
    
    inspect_checkpoint(ckpt_path)


if __name__ == "__main__":
    main()