#!/usr/bin/env python3
"""
Analyze and visualize loss functions from VIMH ordinal regression training.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def load_metrics(csv_path: str) -> pd.DataFrame:
    """Load and clean metrics CSV data."""
    df = pd.read_csv(csv_path)

    # Filter out rows with missing epoch data (intermediate logging steps)
    df = df[df['epoch'].notna()].copy()

    # Group by epoch and combine train/val metrics
    # Each epoch has multiple rows - one with train metrics, one with val metrics
    metrics_by_epoch = []

    for epoch in sorted(df['epoch'].unique()):
        epoch_data = df[df['epoch'] == epoch]

        # Combine all non-null values for this epoch
        combined = {'epoch': epoch}
        for col in df.columns:
            if col != 'epoch':
                non_null_values = epoch_data[col].dropna()
                if len(non_null_values) > 0:
                    combined[col] = non_null_values.iloc[-1]  # Take last non-null value

        metrics_by_epoch.append(combined)

    result_df = pd.DataFrame(metrics_by_epoch)

    # Sort by epoch for proper plotting
    result_df = result_df.sort_values('epoch').reset_index(drop=True)

    return result_df

def plot_loss_analysis(df: pd.DataFrame, save_path: str = None):
    """Create comprehensive loss analysis plots."""

    # Filter out test data (last row) for training plots
    train_data = df[df['train/loss'].notna()].copy()

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('VIMH Ordinal Regression Loss Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Training vs Validation Loss
    ax1 = axes[0, 0]
    ax1.plot(train_data['epoch'], train_data['train/loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(train_data['epoch'], train_data['val/loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training vs Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add annotations for key points
    min_train_loss = train_data['train/loss'].min()
    min_epoch = train_data.loc[train_data['train/loss'].idxmin(), 'epoch']
    ax1.annotate(f'Min Train Loss: {min_train_loss:.3f}',
                xy=(min_epoch, min_train_loss),
                xytext=(min_epoch + 2, min_train_loss + 0.01),
                arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7))

    # Plot 2: Individual Head Accuracies
    ax2 = axes[0, 1]
    ax2.plot(train_data['epoch'], train_data['train/note_number_acc'] * 100, 'g-', label='Note Number (Train)', linewidth=2)
    ax2.plot(train_data['epoch'], train_data['train/note_velocity_acc'] * 100, 'orange', label='Note Velocity (Train)', linewidth=2)
    ax2.plot(train_data['epoch'], train_data['val/note_number_acc'] * 100, 'g--', label='Note Number (Val)', linewidth=2)
    ax2.plot(train_data['epoch'], train_data['val/note_velocity_acc'] * 100, '--', color='orange', label='Note Velocity (Val)', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Multi-Head Accuracy Trends')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add random baseline
    ax2.axhline(y=100/256, color='red', linestyle=':', alpha=0.7, label='Random Baseline (0.39%)')

    # Plot 3: Learning Rate Schedule (if available)
    ax3 = axes[1, 0]
    if 'lr-Adam' in train_data.columns:
        lr_data = train_data['lr-Adam'].ffill()  # Forward fill missing values
        ax3.plot(train_data['epoch'], lr_data, 'purple', linewidth=2, marker='o', markersize=4)
        ax3.set_yscale('log')

        # Annotate LR reduction
        lr_change_epoch = train_data[train_data['lr-Adam'] == 0.0005].iloc[0]['epoch'] if len(train_data[train_data['lr-Adam'] == 0.0005]) > 0 else None
        if lr_change_epoch:
            ax3.annotate('LR Reduction',
                        xy=(lr_change_epoch, 0.0005),
                        xytext=(lr_change_epoch + 2, 0.0007),
                        arrowprops=dict(arrowstyle='->', color='purple', alpha=0.7))
    else:
        # Show loss improvement rate instead
        loss_improvement = train_data['train/loss'].iloc[0] - train_data['train/loss']
        ax3.plot(train_data['epoch'], loss_improvement, 'purple', linewidth=2, marker='o', markersize=4)
        ax3.set_ylabel('Loss Improvement')
        ax3.set_title('Training Loss Improvement')
        ax3.annotate('From manual analysis: LR 0.001→0.0005 at epoch ~12',
                    xy=(0.5, 0.5), xycoords='axes fraction', ha='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate' if 'lr-Adam' in train_data.columns else 'Loss Improvement')
    ax3.set_title('Learning Rate Schedule' if 'lr-Adam' in train_data.columns else 'Training Loss Improvement')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Loss Convergence Analysis
    ax4 = axes[1, 1]

    # Calculate loss difference between train and val
    loss_gap = train_data['val/loss'] - train_data['train/loss']
    ax4.plot(train_data['epoch'], loss_gap, 'red', linewidth=2, label='Val - Train Loss')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss Difference')
    ax4.set_title('Generalization Gap (Val - Train Loss)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Add text box with key statistics
    stats_text = f"""Training Summary:
Final Train Loss: {train_data['train/loss'].iloc[-1]:.3f}
Final Val Loss: {train_data['val/loss'].iloc[-1]:.3f}
Loss Improvement: {train_data['train/loss'].iloc[0] - train_data['train/loss'].iloc[-1]:.3f}
Final Note Acc: {train_data['val/note_number_acc'].iloc[-1]*100:.2f}%
Final Velocity Acc: {train_data['val/note_velocity_acc'].iloc[-1]*100:.2f}%"""

    ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss analysis plot saved to: {save_path}")

    plt.show()

def analyze_loss_behavior(df: pd.DataFrame):
    """Analyze loss behavior and print insights."""
    print("=== VIMH Ordinal Regression Loss Analysis ===\n")

    # Filter out test data (last row) for training analysis
    train_data = df[df['train/loss'].notna()].copy()

    # Basic statistics
    print("📊 Loss Statistics:")
    print(f"  Initial Training Loss: {train_data['train/loss'].iloc[0]:.4f}")
    print(f"  Final Training Loss: {train_data['train/loss'].iloc[-1]:.4f}")
    print(f"  Training Loss Improvement: {train_data['train/loss'].iloc[0] - train_data['train/loss'].iloc[-1]:.4f}")
    print(f"  Final Validation Loss: {train_data['val/loss'].iloc[-1]:.4f}")
    print(f"  Generalization Gap: {train_data['val/loss'].iloc[-1] - train_data['train/loss'].iloc[-1]:.4f}")

    # Convergence analysis
    print("\n📈 Convergence Analysis:")
    train_loss_std = train_data['train/loss'].std()
    val_loss_std = train_data['val/loss'].std()
    print(f"  Training Loss Std Dev: {train_loss_std:.4f}")
    print(f"  Validation Loss Std Dev: {val_loss_std:.4f}")

    if train_loss_std > val_loss_std:
        print("  ✅ Training loss shows more variation (good learning)")
    else:
        print("  ⚠️  Validation loss more variable than training (possible overfitting)")

    # Accuracy analysis
    print("\n🎯 Accuracy Analysis:")
    final_note_acc = train_data['val/note_number_acc'].iloc[-1] * 100
    final_vel_acc = train_data['val/note_velocity_acc'].iloc[-1] * 100
    random_baseline = 100 / 256

    print(f"  Final Note Number Accuracy: {final_note_acc:.3f}%")
    print(f"  Final Note Velocity Accuracy: {final_vel_acc:.3f}%")
    print(f"  Random Baseline: {random_baseline:.3f}%")

    if final_note_acc > random_baseline:
        print(f"  ✅ Note number {final_note_acc/random_baseline:.1f}x above random")
    if final_vel_acc > random_baseline:
        print(f"  ✅ Note velocity {final_vel_acc/random_baseline:.1f}x above random")

    # Learning rate analysis
    print("\n📚 Learning Rate Analysis:")
    if 'lr-Adam' in df.columns:
        initial_lr = df['lr-Adam'].iloc[0]
        final_lr = df['lr-Adam'].ffill().iloc[-1]
        print(f"  Initial LR: {initial_lr}")
        print(f"  Final LR: {final_lr}")
        if final_lr < initial_lr:
            print(f"  📉 LR reduced by {initial_lr/final_lr:.1f}x during training")
    else:
        print("  📉 LR data not available in processed metrics")
        print("  From manual analysis: LR reduced from 0.001 to 0.0005 around epoch 12")

    # Ordinal regression insights
    print("\n🔍 Ordinal Regression Insights:")
    print("  • Loss values are now in perceptual units (parameter range units)")
    print("  • For 2-unit parameter ranges: loss ~0.05 means 0.05 perceptual units off")
    print("  • Continuous predictions enable distance-aware learning")
    print("  • Low accuracy expected for 256-class quantized continuous parameters")
    print("  • Steady loss improvement indicates ordinal regression is working")
    print("  • Same learning rate works across all parameters (perceptual units)")

def main():
    parser = argparse.ArgumentParser(description='Analyze VIMH ordinal regression loss')
    parser.add_argument('--csv_path', type=str,
                       default='logs/train/runs/2025-07-15_04-30-06/csv/cnn_16kdss_ordinal/version_0/metrics.csv',
                       help='Path to metrics CSV file')
    parser.add_argument('--save_path', type=str,
                       default='vimh_loss_analysis.png',
                       help='Path to save the plot')

    args = parser.parse_args()

    # Load data
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return

    df = load_metrics(str(csv_path))
    print(f"Loaded {len(df)} epochs of training data")
    print(f"Available columns: {list(df.columns)}")

    # Create visualization
    plot_loss_analysis(df, args.save_path)

    # Analyze behavior
    analyze_loss_behavior(df)

if __name__ == "__main__":
    main()
