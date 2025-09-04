#!/usr/bin/env python3
"""
Quick test script for VIMH Results Viewer core functionality.
Tests model loading, dataset loading, and inference without GUI.
"""

import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.vimh_datamodule import VIMHDataModule
from src.models.vimh_lit_module import VIMHLitModule
from src.models.components.simple_cnn import SimpleCNN

def test_viewer_core_functionality():
    """Test the core components of the VIMH viewer."""
    
    print("Testing VIMH Results Viewer Core Functionality")
    print("=" * 50)
    
    # Test dataset loading
    print("1. Loading dataset...")
    dataset_path = "data/vimh-32x32x1_8000Hz_1p0s_256dss_simple_4p"
    
    try:
        datamodule = VIMHDataModule(
            data_dir=dataset_path,
            batch_size=1,
            num_workers=0
        )
        datamodule.setup()
        
        dataset_info = datamodule.get_dataset_info()
        print(f"✓ Dataset loaded: {dataset_info['num_test_samples']} test samples")
        print(f"✓ Image shape: {dataset_info['image_shape']}")
        print(f"✓ Parameter heads: {list(dataset_info['heads_config'].keys())}")
        
        # Get a test sample
        test_dataset = datamodule.data_test
        image, true_labels, aux = test_dataset[0]
        print(f"✓ Sample shape: {image.shape}")
        print(f"✓ True labels: {true_labels}")
        
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False
    
    # Test model loading
    print("\n2. Loading model...")
    checkpoint_path = "logs/train/runs/2025-08-29_20-10-24/checkpoints/epoch_000.ckpt"
    
    try:
        # Create network architecture
        net = SimpleCNN(
            input_channels=dataset_info['image_shape'][0],
            conv1_channels=64,
            conv2_channels=128, 
            fc_hidden=512,
            heads_config=dataset_info['heads_config'],
            dropout=0.5,
            input_size=dataset_info['image_shape'][1]
        )
        
        # Create dummy optimizer/scheduler
        dummy_optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        dummy_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(dummy_optimizer)
        
        # Load model
        model = VIMHLitModule.load_from_checkpoint(
            checkpoint_path,
            net=net,
            optimizer=dummy_optimizer,
            scheduler=dummy_scheduler,
            strict=False
        )
        model.eval()
        model = model.cpu()
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Model heads: {list(model.net.heads_config.keys())}")
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test inference
    print("\n3. Testing inference...")
    
    try:
        with torch.no_grad():
            # Add batch dimension
            image_batch = image.unsqueeze(0)
            
            # Run inference
            predictions = model(image_batch)
            
            print(f"✓ Inference successful")
            
            # Process predictions
            for head_name, logits in predictions.items():
                probs = torch.nn.functional.softmax(logits.squeeze(0), dim=0)
                pred_class = torch.argmax(probs).item()
                confidence = probs[pred_class].item()
                true_val = true_labels[head_name]
                
                correct = "✓" if pred_class == true_val else "✗"
                print(f"  {head_name}: {correct} Pred={pred_class}, True={true_val}, Conf={confidence:.3f}")
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 50)
    print("✓ All core functionality tests passed!")
    print("\nTo launch the full viewer, run:")
    print(f"python vimh_results_viewer.py --dataset {dataset_path} --checkpoint {checkpoint_path}")
    
    return True

if __name__ == "__main__":
    success = test_viewer_core_functionality()
    sys.exit(0 if success else 1)