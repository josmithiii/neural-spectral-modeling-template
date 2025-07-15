#!/usr/bin/env python3
"""
VIMH Training Example

This example demonstrates the complete VIMH (Variable Image MultiHead) training pipeline,
from data loading to model training and evaluation. It showcases both programmatic and
config-based approaches to multihead neural network training.

Usage:
    python examples/vimh_training.py  # Basic training with default dataset
    python examples/vimh_training.py --data-dir data-vimh/vimh-32x32_8000Hz_1p0s_256dss_stk_1p
    python examples/vimh_training.py --demo  # Quick demo mode
    python examples/vimh_training.py --analyze-only --checkpoint path/to/model.ckpt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data.vimh_datamodule import VIMHDataModule
from src.models.multihead_module import MultiheadLitModule
from src.models.components.simple_cnn import SimpleCNN

console = Console()

def visualize_batch(batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]],
                   num_samples: int = 4,
                   save_path: Optional[str] = None) -> None:
    """Visualize a batch of VIMH data with labels.

    Args:
        batch: Tuple of (images, labels) from dataloader
        num_samples: Number of samples to visualize
        save_path: Optional path to save the visualization
    """
    images, labels = batch
    batch_size = images.size(0)
    num_samples = min(num_samples, batch_size)

    # Create subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i in range(num_samples):
        ax = axes[i]

        # Display image
        image = images[i]
        if image.size(0) == 1:  # Grayscale
            ax.imshow(image.squeeze(), cmap='gray')
        else:  # RGB
            ax.imshow(image.permute(1, 2, 0))

        # Create title with label information
        title_parts = []
        for head_name, head_values in labels.items():
            value = head_values[i].item()
            title_parts.append(f"{head_name}: {value}")

        ax.set_title(f"Sample {i+1}\n" + "\n".join(title_parts), fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        console.print(f"Visualization saved to {save_path}")
    else:
        plt.show()

def plot_parameter_distributions(datamodule: VIMHDataModule,
                               save_path: Optional[str] = None) -> None:
    """Plot distribution of parameter values in dataset.

    Args:
        datamodule: VIMH data module
        save_path: Optional path to save the plot
    """
    # Get a sample of data for analysis
    train_loader = datamodule.train_dataloader()
    all_labels = {head: [] for head in datamodule.num_classes.keys()}

    # Collect labels from first few batches
    for batch_idx, (_, labels) in enumerate(train_loader):
        if batch_idx >= 10:  # Limit to first 10 batches for speed
            break
        for head_name, head_values in labels.items():
            all_labels[head_name].extend(head_values.cpu().numpy())

    # Create histograms
    num_heads = len(all_labels)
    fig, axes = plt.subplots(1, num_heads, figsize=(5 * num_heads, 4))
    if num_heads == 1:
        axes = [axes]

    for idx, (head_name, values) in enumerate(all_labels.items()):
        axes[idx].hist(values, bins=50, alpha=0.7, color=f'C{idx}')
        axes[idx].set_title(f'{head_name} Distribution')
        axes[idx].set_xlabel('Quantized Value (0-255)')
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        console.print(f"Parameter distributions saved to {save_path}")
    else:
        plt.show()

def inspect_dataset(datamodule: VIMHDataModule) -> Dict[str, Any]:
    """Inspect VIMH dataset and return comprehensive information.

    Args:
        datamodule: VIMH data module

    Returns:
        Dictionary containing dataset information
    """
    with console.status("Inspecting dataset..."):
        datamodule.setup()

        # Get basic information
        dataset_info = datamodule.get_dataset_info()

        # Get sample batch for detailed analysis
        train_loader = datamodule.train_dataloader()
        sample_batch = next(iter(train_loader))
        images, labels = sample_batch

        # Analyze image properties
        image_stats = {
            'shape': tuple(images.shape),
            'dtype': str(images.dtype),
            'min_value': float(images.min()),
            'max_value': float(images.max()),
            'mean': float(images.mean()),
            'std': float(images.std())
        }

        # Analyze label properties
        label_stats = {}
        for head_name, head_values in labels.items():
            label_stats[head_name] = {
                'min': int(head_values.min()),
                'max': int(head_values.max()),
                'unique_values': len(torch.unique(head_values)),
                'dtype': str(head_values.dtype)
            }

        inspection_results = {
            'dataset_info': dataset_info,
            'image_stats': image_stats,
            'label_stats': label_stats,
            'num_classes': datamodule.num_classes,
            'image_shape': datamodule.image_shape,
            'batch_size': datamodule.hparams.batch_size,
            'train_size': len(datamodule.data_train),
            'val_size': len(datamodule.data_val),
            'test_size': len(datamodule.data_test)
        }

        return inspection_results

def create_model_from_dataset(datamodule: VIMHDataModule,
                            model_type: str = "simple_cnn") -> MultiheadLitModule:
    """Create and configure model from dataset properties.

    Args:
        datamodule: VIMH data module
        model_type: Type of model to create

    Returns:
        Configured multihead model
    """
    if model_type == "simple_cnn":
        net = SimpleCNN(
            input_channels=datamodule.image_shape[0],
            heads_config=datamodule.num_classes
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create Lightning module with auto-configuration
    model = MultiheadLitModule(
        net=net,
        optimizer=torch.optim.Adam,
        scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1
        ),
        auto_configure_from_dataset=True
    )

    return model

def train_model(model: MultiheadLitModule,
                datamodule: VIMHDataModule,
                max_epochs: int = 50,
                demo_mode: bool = False) -> Trainer:
    """Train the model using PyTorch Lightning.

    Args:
        model: Multihead model to train
        datamodule: VIMH data module
        max_epochs: Maximum number of training epochs
        demo_mode: If True, use fewer epochs for quick demo

    Returns:
        Trained Lightning trainer
    """
    if demo_mode:
        max_epochs = min(max_epochs, 5)
        console.print(f"[yellow]Demo mode: reducing epochs to {max_epochs}[/yellow]")

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="val/acc_best",
            mode="max",
            save_top_k=1,
            filename="best-{epoch:02d}-{val/acc_best:.3f}"
        ),
        EarlyStopping(
            monitor="val/acc_best",
            mode="max",
            patience=10,
            verbose=True
        )
    ]

    # Setup logger
    logger = CSVLogger("logs", name="vimh_training")

    # Create trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator="cpu",  # Use CPU to avoid MPS issues
        devices=1,
        log_every_n_steps=10,
        val_check_interval=1.0,
        enable_progress_bar=True
    )

    # Train model
    console.print("[green]Starting training...[/green]")
    trainer.fit(model, datamodule)

    return trainer

def analyze_performance(model: MultiheadLitModule,
                       datamodule: VIMHDataModule,
                       trainer: Trainer) -> Dict[str, Any]:
    """Analyze model performance across different heads.

    Args:
        model: Trained multihead model
        datamodule: VIMH data module
        trainer: Lightning trainer

    Returns:
        Performance analysis results
    """
    console.print("[blue]Analyzing model performance...[/blue]")

    # Test the model
    test_results = trainer.test(model, datamodule)

    # Get per-head performance from the test results
    performance_analysis = {
        'test_results': test_results[0],
        'model_summary': {
            'total_params': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        }
    }

    # Extract per-head accuracies
    head_accuracies = {}
    for key, value in test_results[0].items():
        if key.startswith('test/') and key.endswith('_acc'):
            head_name = key.replace('test/', '').replace('_acc', '')
            head_accuracies[head_name] = float(value)

    performance_analysis['head_accuracies'] = head_accuracies

    return performance_analysis

def generate_confusion_matrices(model: MultiheadLitModule,
                              datamodule: VIMHDataModule,
                              save_path: Optional[str] = None) -> None:
    """Generate confusion matrices for each head.

    Args:
        model: Trained multihead model
        datamodule: VIMH data module
        save_path: Optional path to save the matrices
    """
    console.print("[blue]Generating confusion matrices...[/blue]")

    model.eval()
    device = next(model.parameters()).device

    # Collect predictions and targets
    all_predictions = {head: [] for head in datamodule.num_classes.keys()}
    all_targets = {head: [] for head in datamodule.num_classes.keys()}

    test_loader = datamodule.test_dataloader()
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            images = images.to(device)

            # Get predictions
            predictions = model(images)

            # Store predictions and targets
            for head_name in datamodule.num_classes.keys():
                pred = predictions[head_name].argmax(dim=1).cpu()
                target = labels[head_name].cpu()

                all_predictions[head_name].extend(pred.tolist())
                all_targets[head_name].extend(target.tolist())

    # Create confusion matrices
    num_heads = len(all_predictions)
    fig, axes = plt.subplots(1, num_heads, figsize=(6 * num_heads, 5))
    if num_heads == 1:
        axes = [axes]

    for idx, head_name in enumerate(all_predictions.keys()):
        try:
            from sklearn.metrics import confusion_matrix
            import seaborn as sns

            cm = confusion_matrix(all_targets[head_name], all_predictions[head_name])

            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            # Plot with seaborn
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=axes[idx])

        except ImportError:
            # Fallback to matplotlib if sklearn/seaborn not available
            from collections import Counter

            # Create simple confusion matrix visualization
            max_class = max(max(all_targets[head_name]), max(all_predictions[head_name]))
            cm = np.zeros((max_class + 1, max_class + 1))

            for true_val, pred_val in zip(all_targets[head_name], all_predictions[head_name]):
                cm[true_val, pred_val] += 1

            # Normalize
            cm_normalized = cm / cm.sum(axis=1)[:, np.newaxis]

            # Plot with matplotlib
            im = axes[idx].imshow(cm_normalized, cmap='Blues')
            axes[idx].figure.colorbar(im, ax=axes[idx])

        axes[idx].set_title(f'{head_name} Confusion Matrix')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        console.print(f"Confusion matrices saved to {save_path}")
    else:
        plt.show()

def display_results_table(inspection_results: Dict[str, Any],
                         performance_analysis: Dict[str, Any]) -> None:
    """Display comprehensive results in a formatted table.

    Args:
        inspection_results: Dataset inspection results
        performance_analysis: Model performance analysis
    """
    # Dataset Information Table
    table = Table(title="VIMH Dataset Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Dataset Name", inspection_results['dataset_info'].get('dataset_name', 'Unknown'))
    table.add_row("Image Shape", str(inspection_results['image_shape']))
    table.add_row("Train Samples", str(inspection_results['train_size']))
    table.add_row("Val Samples", str(inspection_results['val_size']))
    table.add_row("Test Samples", str(inspection_results['test_size']))
    table.add_row("Batch Size", str(inspection_results['batch_size']))

    for head_name, num_classes in inspection_results['num_classes'].items():
        table.add_row(f"{head_name} Classes", str(num_classes))

    console.print(table)

    # Performance Results Table
    perf_table = Table(title="Model Performance")
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Value", style="magenta")

    perf_table.add_row("Total Parameters", f"{performance_analysis['model_summary']['total_params']:,}")
    perf_table.add_row("Model Size (MB)", f"{performance_analysis['model_summary']['model_size_mb']:.2f}")
    perf_table.add_row("Test Loss", f"{performance_analysis['test_results']['test/loss']:.4f}")

    for head_name, accuracy in performance_analysis['head_accuracies'].items():
        perf_table.add_row(f"{head_name} Accuracy", f"{accuracy:.4f}")

    console.print(perf_table)

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="VIMH Training Example")
    parser.add_argument("--data-dir", type=str,
                       default="data-vimh/vimh-32x32x3_8000Hz_1p0s_16384dss_resonarium_2p",
                       help="Path to VIMH dataset directory")
    parser.add_argument("--max-epochs", type=int, default=100,
                       help="Maximum number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Batch size for training")
    parser.add_argument("--demo", action="store_true",
                       help="Run in demo mode with fewer epochs")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze existing model")
    parser.add_argument("--checkpoint", type=str,
                       help="Path to model checkpoint for analysis")
    parser.add_argument("--export-results", type=str,
                       help="Export results to JSON file")
    parser.add_argument("--save-plots", action="store_true",
                       help="Save plots to files instead of showing")

    args = parser.parse_args()

    # Validate arguments
    if args.analyze_only and not args.checkpoint:
        console.print("[red]Error: --analyze-only requires --checkpoint[/red]")
        return

    if not Path(args.data_dir).exists():
        console.print(f"[red]Error: Dataset directory {args.data_dir} does not exist[/red]")
        return

    console.print(f"[green]🚀 VIMH Training Example[/green]")
    console.print(f"Dataset: {args.data_dir}")

    # Create data module
    datamodule = VIMHDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0  # Avoid multiprocessing issues
    )

    # Inspect dataset
    inspection_results = inspect_dataset(datamodule)

    # Visualize sample data
    train_loader = datamodule.train_dataloader()
    sample_batch = next(iter(train_loader))

    if args.save_plots:
        visualize_batch(sample_batch, save_path="sample_data.png")
        plot_parameter_distributions(datamodule, save_path="parameter_distributions.png")
    else:
        console.print("[blue]Displaying sample data...[/blue]")
        visualize_batch(sample_batch)
        plot_parameter_distributions(datamodule)

    if args.analyze_only:
        # Load existing model for analysis
        model = MultiheadLitModule.load_from_checkpoint(args.checkpoint)
        trainer = Trainer(accelerator="cpu", devices=1)
        performance_analysis = analyze_performance(model, datamodule, trainer)
    else:
        # Create and train model
        model = create_model_from_dataset(datamodule)
        trainer = train_model(model, datamodule, args.max_epochs, args.demo)
        performance_analysis = analyze_performance(model, datamodule, trainer)

        # Generate confusion matrices
        if args.save_plots:
            generate_confusion_matrices(model, datamodule, save_path="confusion_matrices.png")
        else:
            generate_confusion_matrices(model, datamodule)

    # Display results
    display_results_table(inspection_results, performance_analysis)

    # Export results if requested
    if args.export_results:
        export_data = {
            'dataset_info': inspection_results,
            'performance': performance_analysis,
            'config': {
                'data_dir': args.data_dir,
                'max_epochs': args.max_epochs,
                'batch_size': args.batch_size,
                'demo_mode': args.demo
            }
        }

        with open(args.export_results, 'w') as f:
            json.dump(export_data, f, indent=2)

        console.print(f"[green]Results exported to {args.export_results}[/green]")

    console.print("[green]✅ Training example completed![/green]")

if __name__ == "__main__":
    main()
