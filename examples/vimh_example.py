#!/usr/bin/env python3
"""
VIMH Example: Variable Image MultiHead Training

This example demonstrates how to train a model on the VIMH dataset format,
which supports variable image dimensions and multiple continuous parameters.
"""

import torch
from pathlib import Path
import sys

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from data.vimh_datamodule import VIMHDataModule
from models.simple_cnn_lit import SimpleCNNLitModule


def main():
    """Run VIMH training example."""
    print("🎵 VIMH (Variable Image MultiHead) Training Example")
    print("=" * 60)
    
    # Check if we have any VIMH datasets
    data_dir = Path("data-vimh")
    if not data_dir.exists():
        print(f"❌ VIMH data directory not found: {data_dir}")
        print("   Please create VIMH datasets first using the conversion tools.")
        print("   Example: make vimh-stk  # Convert STK dataset to VIMH")
        return
    
    # Find first available VIMH dataset
    vimh_datasets = list(data_dir.glob("vimh-*"))
    if not vimh_datasets:
        print(f"❌ No VIMH datasets found in {data_dir}")
        print("   Please create VIMH datasets first using the conversion tools.")
        return
    
    dataset_path = vimh_datasets[0]
    print(f"📁 Using dataset: {dataset_path.name}")
    
    # Initialize data module
    print("🔧 Setting up VIMH data module...")
    dm = VIMHDataModule(
        data_dir=str(dataset_path),
        batch_size=32,
        num_workers=0,  # Use 0 for MPS compatibility
    )
    
    # Setup data module
    dm.setup()
    dataset_info = dm.get_dataset_info()
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   Image shape: {dataset_info['image_shape']}")
    print(f"   Heads config: {dataset_info['heads_config']}")
    print(f"   Training samples: {dataset_info['num_train_samples']}")
    print(f"   Test samples: {dataset_info['num_test_samples']}")
    
    # Get a sample batch
    train_loader = dm.train_dataloader()
    batch_images, batch_labels = next(iter(train_loader))
    
    print(f"📊 Sample batch:")
    print(f"   Images shape: {batch_images.shape}")
    print(f"   Label heads: {list(batch_labels.keys())}")
    for head_name, labels in batch_labels.items():
        print(f"   {head_name}: {labels.shape} (range: {labels.min()}-{labels.max()})")
    
    # Create model
    print("🤖 Creating multihead CNN model...")
    
    # Extract image dimensions
    channels, height, width = dataset_info['image_shape']
    heads_config = dataset_info['heads_config']
    
    model = SimpleCNNLitModule(
        input_channels=channels,
        input_height=height,
        input_width=width,
        heads_config=heads_config,
        learning_rate=0.001
    )
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Quick training demonstration
    print("🚀 Running quick training demonstration...")
    
    import lightning as L
    
    # Use MPS if available on Mac, otherwise CPU
    if torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
    else:
        accelerator = "cpu"
        devices = 1
    
    trainer = L.Trainer(
        max_epochs=2,
        limit_train_batches=10,  # Only 10 batches for demo
        limit_val_batches=5,     # Only 5 validation batches
        accelerator=accelerator,
        devices=devices,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True
    )
    
    print(f"   Using device: {accelerator}")
    print(f"   Training for 2 epochs with limited batches...")
    
    # Train model
    trainer.fit(model, dm)
    
    # Test model
    print("📈 Running test evaluation...")
    trainer.test(model, dm)
    
    print("🎉 VIMH example completed successfully!")
    print("\nNext steps:")
    print("- Try full training: python src/train.py experiment=vimh_cnn_16kdss")
    print("- Experiment with different architectures")
    print("- Adjust hyperparameters for your specific dataset")


if __name__ == "__main__":
    main()