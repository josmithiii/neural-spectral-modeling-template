#!/usr/bin/env python3
"""
VIMH Results Viewer for Neural Spectral Modeling

Interactive viewer for examining training results, model predictions, and parameter accuracy.
Displays spectrograms, parameter predictions, and per-head classification metrics.
"""

import os
import sys
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, Slider
import torch
import torch.nn.functional as F
from lightning import LightningModule

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import using full module paths
from src.data.vimh_datamodule import VIMHDataModule  
from src.data.vimh_dataset import VIMHDataset
from src.models.multihead_module import MultiheadLitModule


class VIMHResultsViewer:
    """Interactive viewer for VIMH training results and model predictions."""
    
    def __init__(self, 
                 dataset_path: str = None,
                 checkpoint_path: str = None,
                 logs_dir: str = "logs/train/runs"):
        """
        Initialize the VIMH Results Viewer.
        
        Args:
            dataset_path: Path to VIMH dataset directory
            checkpoint_path: Path to model checkpoint file  
            logs_dir: Directory containing training run logs
        """
        self.logs_dir = logs_dir
        self.dataset_path = dataset_path
        self.checkpoint_path = checkpoint_path
        
        # Find available datasets and checkpoints
        self.available_datasets = self._find_datasets()
        self.available_checkpoints = self._find_checkpoints()
        
        if not self.available_datasets:
            raise ValueError(f"No VIMH datasets found. Expected format: data/vimh-*")
            
        if not self.available_checkpoints:
            raise ValueError(f"No model checkpoints found in {logs_dir}")
        
        # Set defaults if not provided
        if not self.dataset_path:
            self.dataset_path = self.available_datasets[0]
        if not self.checkpoint_path:
            self.checkpoint_path = self.available_checkpoints[0]
            
        print(f"Using dataset: {self.dataset_path}")
        print(f"Using checkpoint: {self.checkpoint_path}")
        
        # Initialize state
        self.current_dataset_idx = self.available_datasets.index(self.dataset_path)
        self.current_checkpoint_idx = self.available_checkpoints.index(self.checkpoint_path)
        self.current_sample_idx = 0
        
        # Load initial dataset and model
        self._load_dataset()
        self._load_model()
        
        # Setup matplotlib
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.canvas.manager.set_window_title('VIMH Results Viewer')
        
        # Create layout and widgets
        self._setup_layout()
        self._setup_widgets()
        self._connect_events()
        
        # Initial display
        self.update_display()
    
    def _find_datasets(self) -> List[str]:
        """Find available VIMH datasets."""
        datasets = []
        data_dir = Path("data")
        if data_dir.exists():
            for item in data_dir.iterdir():
                if item.is_dir() and item.name.startswith("vimh-"):
                    datasets.append(str(item))
        
        # Sort by name for consistent ordering
        datasets.sort()
        return datasets
    
    def _find_checkpoints(self) -> List[str]:
        """Find available model checkpoints from training runs."""
        checkpoints = []
        logs_path = Path(self.logs_dir)
        
        if logs_path.exists():
            # Look for checkpoint files in all runs
            for run_dir in logs_path.glob("*/"):
                checkpoint_dir = run_dir / "checkpoints"
                if checkpoint_dir.exists():
                    # Look for best checkpoint first, then last, then any .ckpt
                    for ckpt_name in ["epoch_000.ckpt", "last.ckpt", "*.ckpt"]:
                        ckpt_files = list(checkpoint_dir.glob(ckpt_name))
                        if ckpt_files:
                            checkpoints.extend([str(f) for f in ckpt_files])
                            break
        
        # Remove duplicates and sort by modification time (newest first)
        checkpoints = list(set(checkpoints))
        checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return checkpoints
    
    def _load_dataset(self):
        """Load the current VIMH dataset."""
        print(f"Loading dataset: {self.dataset_path}")
        
        # Create data module
        self.datamodule = VIMHDataModule(
            data_dir=self.dataset_path,
            batch_size=1,  # Single sample for viewing
            num_workers=0
        )
        
        # Setup the datamodule to load metadata
        self.datamodule.setup()
        
        # Get dataset info
        dataset_info = self.datamodule.get_dataset_info()
        print(f"Dataset info: {dataset_info}")
        
        # Store test dataset for sample access
        self.test_dataset = self.datamodule.data_test
        self.val_dataset = self.datamodule.data_val
        
        # Get metadata
        self.image_shape = self.datamodule.image_shape
        self.heads_config = self.datamodule.heads_config
        self.param_bounds = self.datamodule.param_bounds
        
        # Reset sample index
        self.current_sample_idx = 0
        self.max_samples = len(self.test_dataset) if self.test_dataset else 0
        
        print(f"Loaded {self.max_samples} test samples")
        print(f"Image shape: {self.image_shape}")
        print(f"Parameter heads: {list(self.heads_config.keys())}")
    
    def _load_model(self):
        """Load the model from checkpoint."""
        print(f"Loading checkpoint: {self.checkpoint_path}")
        
        try:
            # Import the network architecture
            from src.models.components.simple_cnn import SimpleCNN
            
            # Create network with current dataset configuration
            # Use reasonable defaults for CNN architecture
            net = SimpleCNN(
                input_channels=self.image_shape[0],  # Number of channels
                conv1_channels=64,
                conv2_channels=128, 
                fc_hidden=512,
                heads_config=self.heads_config,  # Use dataset heads config
                dropout=0.5,
                input_size=self.image_shape[1]  # Height (assuming square images)
            )
            
            # Create dummy optimizer and scheduler for loading
            dummy_optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
            dummy_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(dummy_optimizer)
            
            # Load the complete model with our network (strict=False to handle head mismatches)
            self.model = MultiheadLitModule.load_from_checkpoint(
                self.checkpoint_path,
                net=net,
                optimizer=dummy_optimizer,
                scheduler=dummy_scheduler,
                strict=False
            )
            self.model.eval()
            
            # Move model to CPU for inference (consistent with viewer data)
            self.model = self.model.cpu()
            
            # Update heads config from loaded model
            if hasattr(self.model, 'net') and hasattr(self.model.net, 'heads_config'):
                self.model_heads_config = self.model.net.heads_config
            else:
                self.model_heads_config = self.heads_config
            
            print(f"Model loaded successfully")
            print(f"Model heads: {list(self.model_heads_config.keys())}")
            
            # Verify heads match dataset
            dataset_heads = set(self.heads_config.keys())
            model_heads = set(self.model_heads_config.keys())
            
            if dataset_heads != model_heads:
                print(f"WARNING: Dataset heads {dataset_heads} don't match model heads {model_heads}")
                # Use the intersection for inference
                self.inference_heads = dataset_heads & model_heads
                print(f"Will use common heads for inference: {self.inference_heads}")
            else:
                self.inference_heads = dataset_heads
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _setup_layout(self):
        """Setup the matplotlib layout."""
        # Create grid layout
        gs = GridSpec(3, 4, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # Main spectrogram display (top half)
        self.ax_spec = self.fig.add_subplot(gs[0:2, 0:3])
        
        # Parameter comparison (top right)
        self.ax_params = self.fig.add_subplot(gs[0, 3])
        
        # Accuracy metrics (middle right)  
        self.ax_accuracy = self.fig.add_subplot(gs[1, 3])
        
        # Controls (bottom)
        self.ax_controls = self.fig.add_subplot(gs[2, :])
        self.ax_controls.set_xlim(0, 1)
        self.ax_controls.set_ylim(0, 1)
        self.ax_controls.axis('off')
    
    def _setup_widgets(self):
        """Setup interactive widgets."""
        # Sample navigation
        ax_prev = plt.axes([0.1, 0.02, 0.1, 0.04])
        ax_next = plt.axes([0.21, 0.02, 0.1, 0.04])
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        
        # Sample slider
        ax_slider = plt.axes([0.35, 0.02, 0.3, 0.03])
        self.sample_slider = Slider(
            ax_slider, 'Sample', 
            0, max(0, self.max_samples - 1),
            valinit=0, valfmt='%d'
        )
        
        # Dataset/Model selection buttons
        ax_dataset = plt.axes([0.7, 0.02, 0.1, 0.04])
        ax_model = plt.axes([0.81, 0.02, 0.1, 0.04])
        self.btn_dataset = Button(ax_dataset, 'Dataset')
        self.btn_model = Button(ax_model, 'Model')
        
        # Status text
        self.status_text = self.ax_controls.text(
            0.02, 0.8, "", transform=self.ax_controls.transAxes,
            fontsize=10, verticalalignment='top'
        )
    
    def _connect_events(self):
        """Connect widget events."""
        self.btn_prev.on_clicked(self._prev_sample)
        self.btn_next.on_clicked(self._next_sample)
        self.sample_slider.on_changed(self._change_sample)
        self.btn_dataset.on_clicked(self._cycle_dataset)
        self.btn_model.on_clicked(self._cycle_model)
    
    def _prev_sample(self, event):
        """Navigate to previous sample."""
        if self.current_sample_idx > 0:
            self.current_sample_idx -= 1
            self.sample_slider.set_val(self.current_sample_idx)
            self.update_display()
    
    def _next_sample(self, event):
        """Navigate to next sample."""
        if self.current_sample_idx < self.max_samples - 1:
            self.current_sample_idx += 1
            self.sample_slider.set_val(self.current_sample_idx)
            self.update_display()
    
    def _change_sample(self, val):
        """Handle sample slider change."""
        new_idx = int(val)
        if new_idx != self.current_sample_idx:
            self.current_sample_idx = new_idx
            self.update_display()
    
    def _cycle_dataset(self, event):
        """Cycle to next available dataset."""
        self.current_dataset_idx = (self.current_dataset_idx + 1) % len(self.available_datasets)
        self.dataset_path = self.available_datasets[self.current_dataset_idx]
        self._load_dataset()
        self._load_model()  # Reload model with new dataset config
        self._update_slider_range()
        self.update_display()
    
    def _cycle_model(self, event):
        """Cycle to next available model checkpoint."""
        self.current_checkpoint_idx = (self.current_checkpoint_idx + 1) % len(self.available_checkpoints)
        self.checkpoint_path = self.available_checkpoints[self.current_checkpoint_idx]
        self._load_model()
        self.update_display()
    
    def _update_slider_range(self):
        """Update sample slider range when dataset changes."""
        if hasattr(self, 'sample_slider'):
            self.sample_slider.set_val(0)
            self.sample_slider.valmax = max(0, self.max_samples - 1)
            self.current_sample_idx = 0
    
    def _get_current_sample(self) -> Tuple[torch.Tensor, Dict[str, int], Optional[torch.Tensor]]:
        """Get the current sample from the dataset."""
        if self.current_sample_idx >= len(self.test_dataset):
            # Fallback to validation dataset if test is smaller
            dataset = self.val_dataset if self.val_dataset else self.test_dataset
            idx = self.current_sample_idx % len(dataset)
        else:
            dataset = self.test_dataset
            idx = self.current_sample_idx
            
        return dataset[idx]
    
    def _run_inference(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run model inference on input image."""
        with torch.no_grad():
            # Add batch dimension
            image_batch = image.unsqueeze(0)
            
            # Run forward pass
            predictions = self.model(image_batch)
            
            # Remove batch dimension and get class predictions
            results = {}
            for head_name, logits in predictions.items():
                # Get class probabilities and predictions
                probs = F.softmax(logits.squeeze(0), dim=0)
                pred_class = torch.argmax(probs)
                results[head_name] = {
                    'predicted_class': pred_class.item(),
                    'confidence': probs[pred_class].item(),
                    'probabilities': probs
                }
            
            return results
    
    def update_display(self):
        """Update the entire display with current sample."""
        try:
            # Get current sample
            image, true_labels, aux_features = self._get_current_sample()
            
            # Run inference
            predictions = self._run_inference(image)
            
            # Update spectrogram display
            self._update_spectrogram(image)
            
            # Update parameter comparison
            self._update_parameters(true_labels, predictions)
            
            # Update accuracy display
            self._update_accuracy(true_labels, predictions)
            
            # Update status
            self._update_status(true_labels, predictions)
            
            # Redraw
            self.fig.canvas.draw()
            
        except Exception as e:
            print(f"Error updating display: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_spectrogram(self, image: torch.Tensor):
        """Update the spectrogram display."""
        self.ax_spec.clear()
        
        # Convert tensor to numpy and handle different channel formats
        if image.dim() == 3:
            # Remove channel dimension if single channel
            if image.shape[0] == 1:
                img_data = image.squeeze(0).numpy()
            else:
                # For multi-channel, take first channel
                img_data = image[0].numpy()
        else:
            img_data = image.numpy()
        
        # Display spectrogram
        im = self.ax_spec.imshow(
            img_data, 
            aspect='auto', 
            origin='lower',
            cmap='viridis'
        )
        
        self.ax_spec.set_title(f'Input Spectrogram - Sample {self.current_sample_idx}')
        self.ax_spec.set_xlabel('Time')
        self.ax_spec.set_ylabel('Frequency')
        
        # Add colorbar if not already present
        if not hasattr(self, '_spectrogram_colorbar'):
            self._spectrogram_colorbar = plt.colorbar(im, ax=self.ax_spec, shrink=0.8)
    
    def _update_parameters(self, true_labels: Dict[str, int], predictions: Dict[str, Dict]):
        """Update parameter comparison display."""
        self.ax_params.clear()
        
        param_names = list(true_labels.keys())
        if not param_names:
            return
        
        y_pos = np.arange(len(param_names))
        
        # Get true and predicted values (normalized to 0-255 range)
        true_vals = [true_labels[name] for name in param_names]
        pred_vals = [predictions[name]['predicted_class'] for name in param_names]
        confidences = [predictions[name]['confidence'] for name in param_names]
        
        # Create horizontal bar chart
        bar_height = 0.35
        bars1 = self.ax_params.barh(y_pos - bar_height/2, true_vals, bar_height, 
                                   label='True', alpha=0.7, color='lightblue')
        bars2 = self.ax_params.barh(y_pos + bar_height/2, pred_vals, bar_height,
                                   label='Predicted', alpha=0.7, color='orange')
        
        # Add confidence as bar color intensity
        for i, (bar, conf) in enumerate(zip(bars2, confidences)):
            bar.set_alpha(0.5 + 0.5 * conf)  # Higher confidence = more opaque
        
        self.ax_params.set_yticks(y_pos)
        self.ax_params.set_yticklabels([name.replace('_', ' ').title() for name in param_names])
        self.ax_params.set_xlabel('Parameter Value (0-255)')
        self.ax_params.set_title('Parameter Comparison')
        self.ax_params.legend(fontsize=8)
        self.ax_params.grid(True, alpha=0.3)
    
    def _update_accuracy(self, true_labels: Dict[str, int], predictions: Dict[str, Dict]):
        """Update accuracy metrics display."""
        self.ax_accuracy.clear()
        
        param_names = list(true_labels.keys())
        if not param_names:
            return
        
        # Calculate per-parameter accuracy (1 if correct, 0 if wrong)
        accuracies = []
        confidences = []
        
        for name in param_names:
            true_val = true_labels[name]
            pred_val = predictions[name]['predicted_class']
            conf = predictions[name]['confidence']
            
            accuracies.append(1.0 if true_val == pred_val else 0.0)
            confidences.append(conf)
        
        y_pos = np.arange(len(param_names))
        
        # Create accuracy bars with confidence-based coloring
        colors = ['green' if acc == 1.0 else 'red' for acc in accuracies]
        bars = self.ax_accuracy.barh(y_pos, confidences, color=colors, alpha=0.7)
        
        # Add accuracy indicators
        for i, (acc, conf) in enumerate(zip(accuracies, confidences)):
            symbol = '✓' if acc == 1.0 else '✗'
            self.ax_accuracy.text(conf + 0.02, i, f'{symbol} {conf:.2f}', 
                                va='center', fontsize=10)
        
        self.ax_accuracy.set_yticks(y_pos)
        self.ax_accuracy.set_yticklabels([name.replace('_', ' ').title() for name in param_names])
        self.ax_accuracy.set_xlabel('Prediction Confidence')
        self.ax_accuracy.set_title('Per-Head Accuracy')
        self.ax_accuracy.set_xlim(0, 1.1)
        self.ax_accuracy.grid(True, alpha=0.3)
    
    def _update_status(self, true_labels: Dict[str, int], predictions: Dict[str, Dict]):
        """Update status information."""
        dataset_name = Path(self.dataset_path).name
        checkpoint_name = Path(self.checkpoint_path).name
        
        # Calculate overall accuracy
        correct = sum(1 for name in true_labels.keys() 
                     if true_labels[name] == predictions[name]['predicted_class'])
        total = len(true_labels)
        overall_accuracy = correct / total if total > 0 else 0
        
        # Calculate average confidence
        avg_confidence = np.mean([predictions[name]['confidence'] 
                                for name in true_labels.keys()]) if total > 0 else 0
        
        status = f"""Dataset: {dataset_name}
Model: {checkpoint_name}
Sample: {self.current_sample_idx + 1} / {self.max_samples}
Overall Accuracy: {overall_accuracy:.1%} ({correct}/{total})
Avg Confidence: {avg_confidence:.2f}
Image Shape: {self.image_shape}"""
        
        self.status_text.set_text(status)


def main():
    """Main function to launch the VIMH Results Viewer."""
    parser = argparse.ArgumentParser(description='VIMH Results Viewer')
    parser.add_argument('--dataset', type=str, help='Path to VIMH dataset directory')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint file')
    parser.add_argument('--logs-dir', type=str, default='logs/train/runs',
                       help='Directory containing training run logs')
    
    args = parser.parse_args()
    
    try:
        viewer = VIMHResultsViewer(
            dataset_path=args.dataset,
            checkpoint_path=args.checkpoint, 
            logs_dir=args.logs_dir
        )
        
        print("\nVIMH Results Viewer Controls:")
        print("- Previous/Next buttons: Navigate samples")
        print("- Sample slider: Jump to specific sample")
        print("- Dataset button: Cycle through available datasets") 
        print("- Model button: Cycle through available checkpoints")
        print("\nClose the window to exit.")
        
        plt.show()
        
    except Exception as e:
        print(f"Error launching viewer: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()