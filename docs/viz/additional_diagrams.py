#!/usr/bin/env python3
"""Generate additional architecture diagrams for presentation slides."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def create_convnext_diagram(output_dir="viz/diagrams"):
    """Create ConvNeXt-V2 architecture diagram."""
    print("Creating ConvNeXt-V2 architecture diagram...")

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    # ConvNeXt components
    components = [
        ("Input\n28×28×1", 0, 3, 'lightblue', 1.5),
        ("Stem\n4×4 conv", 2, 3, 'lightgreen', 1.5),
        ("Stage 1\n7×7 DWConv\n+ GRN", 4, 3, 'lightcoral', 2),
        ("Stage 2\nDownsample", 7, 3, 'lightyellow', 1.5),
        ("Stage 3\n7×7 DWConv\n+ GRN", 9, 3, 'lightcoral', 2),
        ("Stage 4\nDownsample", 12, 3, 'lightyellow', 1.5),
        ("Global\nAvgPool", 14.5, 3, 'lightgray', 1.5),
        ("Output\n→ 10", 16.5, 3, 'orange', 1.5)
    ]

    for name, x, y, color, width in components:
        rect = patches.Rectangle((x, y-0.7), width, 1.4, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x+width/2, y, name, ha='center', va='center', fontsize=9, fontweight='bold')

        if x < 16:
            ax.arrow(x+width, y, 0.4, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

    ax.set_xlim(-0.5, 19)
    ax.set_ylim(1.5, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('ConvNeXt-V2 Architecture with Global Response Normalization', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    output_path = Path(output_dir) / "convnext_architecture.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"ConvNeXt architecture diagram saved to {output_path}")
    plt.close()

def create_vit_diagram(output_dir="viz/diagrams"):
    """Create Vision Transformer architecture diagram."""
    print("Creating ViT architecture diagram...")

    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    # ViT components
    components = [
        ("Input\n28×28×1", 0, 4, 'lightblue', 1.5),
        ("Patch\nEmbedding\n7×7 patches", 2, 4, 'lightgreen', 2),
        ("Position\nEmbedding", 5, 4, 'lightyellow', 1.5),
        ("Transformer\nBlock 1\nMSA + MLP", 7.5, 4, 'lightcoral', 2.5),
        ("Transformer\nBlock 2\nMSA + MLP", 11, 4, 'lightcoral', 2.5),
        ("Global\nAvg Pool", 14.5, 4, 'lightgray', 1.5),
        ("Output\n→ 10", 16.5, 4, 'orange', 1.5)
    ]

    for name, x, y, color, width in components:
        rect = patches.Rectangle((x, y-0.8), width, 1.6, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x+width/2, y, name, ha='center', va='center', fontsize=9, fontweight='bold')

        if x < 16:
            ax.arrow(x+width, y, 0.4, 0, head_width=0.15, head_length=0.15, fc='black', ec='black')

    # Add attention visualization
    ax.text(9, 2, 'Multi-Head\nSelf-Attention', ha='center', va='center', fontsize=8, style='italic')
    ax.text(12.75, 2, 'Multi-Head\nSelf-Attention', ha='center', va='center', fontsize=8, style='italic')

    ax.set_xlim(-0.5, 19)
    ax.set_ylim(1, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Vision Transformer (ViT) Architecture', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    output_path = Path(output_dir) / "vit_architecture.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"ViT architecture diagram saved to {output_path}")
    plt.close()

def create_efficientnet_diagram(output_dir="viz/diagrams"):
    """Create EfficientNet architecture diagram."""
    print("Creating EfficientNet architecture diagram...")

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    components = [
        ("Input\n28×28×1", 0, 3, 'lightblue', 1.5),
        ("Stem\nConv", 2, 3, 'lightgreen', 1.5),
        ("MBConv\nBlock 1", 4, 3, 'lightcoral', 1.8),
        ("MBConv\nBlock 2\n+ SE", 6.3, 3, 'lightcoral', 1.8),
        ("MBConv\nBlock 3\n+ SE", 8.6, 3, 'lightcoral', 1.8),
        ("MBConv\nBlock 4\n+ SE", 10.9, 3, 'lightcoral', 1.8),
        ("Global\nAvgPool", 13.2, 3, 'lightgray', 1.5),
        ("Output\n→ 10", 15.2, 3, 'orange', 1.5)
    ]

    for name, x, y, color, width in components:
        rect = patches.Rectangle((x, y-0.7), width, 1.4, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x+width/2, y, name, ha='center', va='center', fontsize=9, fontweight='bold')

        if x < 15:
            ax.arrow(x+width, y, 0.15, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

    # Add compound scaling note
    ax.text(8.5, 1.5, 'Compound Scaling: Depth × Width × Resolution', ha='center',
            va='center', fontsize=10, style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))

    ax.set_xlim(-0.5, 18)
    ax.set_ylim(1, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('EfficientNet Architecture with Mobile Optimization', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    output_path = Path(output_dir) / "efficientnet_architecture.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"EfficientNet architecture diagram saved to {output_path}")
    plt.close()

def create_multihead_diagram(output_dir="viz/diagrams"):
    """Create Multi-Head architecture diagram."""
    print("Creating Multi-Head architecture diagram...")

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Shared backbone
    backbone = [
        ("Input\nImage", 5, 7, 'lightblue', 2),
        ("Feature\nExtractor\n(CNN/ViT)", 5, 5, 'lightgreen', 2),
        ("Shared\nFeatures", 5, 3, 'lightyellow', 2)
    ]

    for name, x, y, color, width in backbone:
        rect = patches.Rectangle((x, y-0.5), width, 1, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x+width/2, y, name, ha='center', va='center', fontsize=10, fontweight='bold')

        if y > 3:
            ax.arrow(x+width/2, y-0.5, 0, -0.4, head_width=0.15, head_length=0.1, fc='black', ec='black')

    # Multiple heads
    heads = [
        ("Head 1\nDigit\nClassification", 1, 1, 'lightcoral', 2),
        ("Head 2\nThickness\nRegression", 5, 1, 'lightpink', 2),
        ("Head 3\nStyle\nClassification", 9, 1, 'lightcyan', 2)
    ]

    for name, x, y, color, width in heads:
        rect = patches.Rectangle((x, y-0.5), width, 1, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x+width/2, y, name, ha='center', va='center', fontsize=9, fontweight='bold')

        # Arrow from shared features to each head
        ax.arrow(6, 2.5, x+width/2-6, y+0.5-2.5, head_width=0.1, head_length=0.1, fc='gray', ec='gray')

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Multi-Head Neural Network Architecture', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    output_path = Path(output_dir) / "multihead_architecture.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Multi-Head architecture diagram saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    output_dir = "viz/diagrams"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    create_convnext_diagram(output_dir)
    create_vit_diagram(output_dir)
    create_efficientnet_diagram(output_dir)
    create_multihead_diagram(output_dir)

    print("\nAll additional architecture diagrams generated successfully!")
