#!/usr/bin/env python3
"""
Audio Reconstruction Evaluation Example

This example demonstrates how to use the audio reconstruction evaluator to:
1. Load a trained model and test dataset
2. Compare true vs predicted synthesis parameters  
3. Listen to original and reconstructed audio
4. Analyze synthesis quality with metrics and visualizations

Prerequisites:
- A trained checkpoint (from training)
- A VIMH test dataset
- Optional: soundfile for audio export (pip install soundfile)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.audio_reconstruction_eval import AudioReconstructionEvaluator
from src.data.vimh_datamodule import VIMHDataModule  
from src.models.multihead_module import MultiheadLitModule
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

def basic_evaluation_example():
    """
    Basic evaluation example - evaluate a few samples with plots.
    """
    print("🎵 Basic Audio Reconstruction Evaluation Example")
    print("=" * 60)
    
    # Configuration - adjust these paths for your setup
    checkpoint_path = "logs/train/runs/2024-01-01_12-00-00/checkpoints/best.ckpt"  # Your checkpoint
    dataset_path = "data/vimh-32x32x1_8000Hz_1p0s_16384dss_simple_2p"  # Your dataset
    
    # Check if files exist
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please update the checkpoint_path variable with your trained model")
        return
        
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please update the dataset_path variable with your dataset")
        return
    
    # Load datamodule manually (alternative to Hydra)
    print("📁 Loading dataset...")
    datamodule = VIMHDataModule(
        data_dir=dataset_path,
        batch_size=32,
        num_workers=4,
        pin_memory=False
    )
    
    # Load model - you'll need to match your model architecture
    print("🤖 Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load checkpoint to get model config
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # You might need to reconstruct the model based on your saved config
    # This is a simplified example - adjust based on your model
    # Example: from src.models.cnn_64k import CNN64k
    # net = CNN64k()  # Initialize with appropriate parameters
    
    # For now, use a placeholder - you'll need to match your actual model
    print("⚠️  This example needs to be adapted to your specific model architecture")
    print("Please modify the model loading section to match your trained model")
    return
    
    model = MultiheadLitModule(
        net=net,
        optimizer=None,  # Not needed for inference
        scheduler=None,  # Not needed for inference
        output_mode="regression"  # or "classification" based on your training
    )
    
    # Load the trained weights
    model.load_state_dict(checkpoint["state_dict"])
    
    print("🔬 Creating evaluator...")
    evaluator = AudioReconstructionEvaluator(model, datamodule, device)
    
    # Evaluate a few samples
    print("🎯 Evaluating samples...")
    
    for sample_idx in range(min(3, len(evaluator.test_dataset))):
        print(f"\n--- Sample {sample_idx} ---")
        
        # Run comprehensive evaluation
        results = evaluator.evaluate_sample(
            sample_idx=sample_idx,
            plot=True,  # Show plots
            save_audio=True,  # Save audio files
            output_dir=f"audio_eval_sample_{sample_idx}"
        )
        
        # Print parameter comparison
        print("Parameter Comparison:")
        for param_name, error_info in results["param_errors"].items():
            true_val = error_info["true"]
            pred_val = error_info["predicted"]
            abs_err = error_info["absolute_error"]
            rel_err = error_info["relative_error"] * 100
            
            print(f"  {param_name}:")
            print(f"    True: {true_val:.4f}")
            print(f"    Predicted: {pred_val:.4f}")
            print(f"    Error: {abs_err:.4f} ({rel_err:.2f}%)")
        
        # Print audio quality metrics
        print("Audio Quality:")
        metrics = results["audio_metrics"]
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  SNR: {metrics['snr_db']:.2f} dB")
        print(f"  Correlation: {metrics['correlation']:.4f}")
        
        print(f"✅ Audio files saved to: audio_eval_sample_{sample_idx}/")


def interactive_evaluation_example():
    """
    Interactive evaluation example using the GUI widget.
    """
    print("🎮 Interactive Audio Reconstruction Evaluation")
    print("=" * 60)
    print("This will launch an interactive widget for browsing samples")
    print("Use the slider to navigate samples, and buttons to play audio")
    
    # This would use the same setup as above, but launch the interactive widget
    # See the basic example for loading setup
    
    print("To run interactively, use:")
    print("python src/audio_reconstruction_eval.py ckpt_path=YOUR_CHECKPOINT interactive=true")


def hydra_evaluation_example():
    """
    Example using Hydra configuration (recommended approach).
    """
    print("⚙️  Hydra Configuration Example")
    print("=" * 60)
    
    print("1. Auto-discovery (simplest - uses latest checkpoint + interactive mode):")
    print("   python src/audio_reconstruction_eval.py")
    
    print("\n2. Batch evaluation mode:")
    print("   python src/audio_reconstruction_eval.py interactive=false")
    
    print("\n3. Specific checkpoint:")
    print("   python src/audio_reconstruction_eval.py ckpt_path=path/to/checkpoint.ckpt")
    
    print("\n4. Custom configuration:")
    print("   python src/audio_reconstruction_eval.py \\")
    print("     data=vimh_256dss \\")
    print("     num_samples=10 \\")
    print("     save_audio=true \\")
    print("     output_dir=my_audio_results")
    
    print("\n5. Different model architecture:")
    print("   python src/audio_reconstruction_eval.py \\")
    print("     model=vit_micro \\")
    print("     data=vimh_stk")


def batch_evaluation_example():
    """
    Example of evaluating multiple checkpoints or datasets.
    """
    print("📊 Batch Evaluation Example")
    print("=" * 60)
    
    # List of checkpoints to evaluate
    checkpoints = [
        "logs/train/cnn_model/checkpoints/best.ckpt",
        "logs/train/vit_model/checkpoints/best.ckpt", 
        "logs/train/other_model/checkpoints/best.ckpt"
    ]
    
    # List of datasets to test on
    datasets = [
        "data/vimh-32x32x1_8000Hz_1p0s_16384dss_simple_2p",
        "data/vimh-32x32x1_8000Hz_1p0s_256dss_simple_2p"
    ]
    
    print("This example would evaluate each checkpoint on each dataset:")
    
    for checkpoint in checkpoints:
        for dataset in datasets:
            ckpt_name = Path(checkpoint).parent.parent.name
            dataset_name = Path(dataset).name
            
            print(f"\n📈 Evaluating {ckpt_name} on {dataset_name}")
            print(f"Command: python src/audio_reconstruction_eval.py \\")
            print(f"  ckpt_path={checkpoint} \\")
            print(f"  data.data_dir={dataset} \\")
            print(f"  output_dir=results/{ckpt_name}_{dataset_name} \\")
            print(f"  num_samples=20")
    
    print("\n💡 You could automate this with a shell script or Python loop!")


def analysis_tips():
    """
    Tips for analyzing audio reconstruction results.
    """
    print("💡 Analysis Tips")
    print("=" * 60)
    
    print("🎯 What to look for in results:")
    print("  • Parameter accuracy - how close are predicted vs true values?")
    print("  • Audio similarity - do waveforms look similar?") 
    print("  • Spectral similarity - do spectrograms match?")
    print("  • Perceptual quality - does it sound similar?")
    
    print("\n📊 Key metrics:")
    print("  • RMSE: Lower is better (0 = perfect)")
    print("  • SNR (dB): Higher is better (>20 dB is good)")
    print("  • Correlation: Closer to 1.0 is better")
    print("  • Parameter errors: Check relative errors especially")
    
    print("\n🔧 Troubleshooting:")
    print("  • High parameter errors → Model needs more training")
    print("  • Audio sounds different but params are close → Check synthesis")
    print("  • Spectrograms differ → Check STFT/mel settings match")
    print("  • Poor correlations → May need different model architecture")
    
    print("\n🎵 Listening tests:")
    print("  • Use the interactive widget to browse many samples")
    print("  • Save audio files and do blind listening tests")
    print("  • Check if certain parameter ranges work better")
    print("  • Look for systematic biases in predictions")


if __name__ == "__main__":
    print("🎵 Audio Reconstruction Evaluation Examples")
    print("=" * 60)
    
    print("\nAvailable examples:")
    print("1. basic_evaluation_example() - Simple evaluation with plots")
    print("2. interactive_evaluation_example() - GUI widget demo")  
    print("3. hydra_evaluation_example() - Hydra configuration examples")
    print("4. batch_evaluation_example() - Multiple model evaluation")
    print("5. analysis_tips() - Tips for interpreting results")
    
    print("\nRunning analysis tips:")
    analysis_tips()
    
    print("\nTo run other examples, call the functions directly:")
    print("python -c 'from examples.audio_reconstruction_example import *; basic_evaluation_example()'")
