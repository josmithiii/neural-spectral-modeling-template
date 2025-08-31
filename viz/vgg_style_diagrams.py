#!/usr/bin/env python3
"""
Generate VGG-16 style architecture diagrams in EPS format for NSMT models.

Creates clean, publication-quality diagrams showing convolutional blocks,
pooling layers, and fully connected layers with actual parameter counts
and dimensions from the trained models.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from typing import List, Tuple, Dict, Any
import argparse
from pathlib import Path


class VGGStyleDiagram:
    """Generate VGG-16 style architecture diagrams."""
    
    def __init__(self, figsize: Tuple[int, int] = (14, 8)):
        self.figsize = figsize
        self.colors = {
            'conv': '#4A90E2',      # Blue for conv layers
            'pool': '#E85D75',      # Pink/Red for pooling  
            'fc': '#7ED321',        # Green for fully connected
            'input': '#9013FE',     # Purple for input
            'output': '#50E3C2'     # Teal for output
        }
        
    def create_3d_block(self, ax, x: float, y: float, width: float, height: float, 
                       depth: float, color: str, alpha: float = 0.9) -> None:
        """Create a clean 3D-style block using matplotlib patches."""
        # Main face - clean rectangle
        main_face = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor='black',
            linewidth=2,
            alpha=alpha
        )
        ax.add_patch(main_face)
        
        # Simple 3D effect with subtle shadows (light from lower-left)
        if depth > 0:
            # Right shadow (goes up and right - flipped to match top shadow)
            shadow_color = '#CCCCCC'  # Light gray shadow
            right_shadow = patches.Polygon(
                [(x + width, y), 
                 (x + width + depth*0.3, y + depth*0.3),
                 (x + width + depth*0.3, y + height + depth*0.3),
                 (x + width, y + height)],
                facecolor=shadow_color,
                edgecolor='gray',
                linewidth=1,
                alpha=0.6
            )
            ax.add_patch(right_shadow)
            
            # Top shadow (goes up and right)
            top_shadow = patches.Polygon(
                [(x, y + height),
                 (x + depth*0.3, y + height + depth*0.3),
                 (x + width + depth*0.3, y + height + depth*0.3),
                 (x + width, y + height)],
                facecolor=shadow_color,
                edgecolor='gray', 
                linewidth=1,
                alpha=0.6
            )
            ax.add_patch(top_shadow)

    def add_arrow(self, ax, start_x: float, end_x: float, y: float, 
                  color: str = '#2ECC71', width: float = 0.3) -> None:
        """Add arrow between blocks."""
        arrow = patches.FancyArrowPatch(
            (start_x, y), (end_x, y),
            arrowstyle='->', 
            mutation_scale=20,
            color=color,
            linewidth=3,
            alpha=0.8
        )
        ax.add_patch(arrow)

    def add_text_label(self, ax, x: float, y: float, text: str, 
                      fontsize: int = 10, fontweight: str = 'normal') -> None:
        """Add text label with background."""
        ax.text(x, y, text, fontsize=fontsize, fontweight=fontweight,
                ha='center', va='center', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='gray', alpha=0.9))

    def create_cnn_diagram(self, model_name: str, architecture: Dict[str, Any], 
                          save_path: str) -> None:
        """Create CNN architecture diagram in VGG-16 style."""
        
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Title
        ax.text(10, 9.5, f'{model_name.upper()} Architecture', 
                fontsize=16, fontweight='bold', ha='center')
        ax.text(10, 9, f"Parameters: {architecture['total_params']:,}", 
                fontsize=12, ha='center', style='italic')
        
        current_x = 1
        block_y = 5
        
        # Input block
        input_shape = architecture['input_shape']
        self.create_3d_block(ax, current_x, block_y-1, 1.2, 2, 0.3, 
                            self.colors['input'])
        self.add_text_label(ax, current_x+0.6, block_y, 'Input\nLayer')
        self.add_text_label(ax, current_x+0.6, block_y-1.5, 
                           f"{input_shape[1]}×{input_shape[2]}×{input_shape[3]}")
        current_x += 2.5
        
        # Convolutional blocks
        for i, conv_block in enumerate(architecture['conv_blocks']):
            # Add arrow
            self.add_arrow(ax, current_x-1, current_x, block_y)
            
            # Conv block
            channels = conv_block['out_channels']
            block_width = min(2.0, max(0.8, channels/50))  # Scale with channels
            block_height = min(2.5, max(1.5, channels/40))
            
            self.create_3d_block(ax, current_x, block_y-block_height/2, 
                               block_width, block_height, 0.4, 
                               self.colors['conv'])
            
            # Labels
            self.add_text_label(ax, current_x+block_width/2, block_y+0.3, 
                               f'Conv{i+1}')
            self.add_text_label(ax, current_x+block_width/2, block_y-0.3, 
                               f'{channels} filters\n3×3, pad=1')
            
            if conv_block.get('pool', False):
                # Add pooling layer
                pool_x = current_x + block_width + 0.3
                self.create_3d_block(ax, pool_x, block_y-0.5, 0.6, 1, 0.2, 
                                   self.colors['pool'])
                self.add_text_label(ax, pool_x+0.3, block_y, 'Pool\n2×2')
                current_x = pool_x + 1.2
            else:
                current_x += block_width + 0.5
        
        # Feature extraction
        self.add_arrow(ax, current_x-0.5, current_x+0.5, block_y)
        self.create_3d_block(ax, current_x+0.5, block_y-1, 1, 2, 0.2, 
                           self.colors['pool'])
        self.add_text_label(ax, current_x+1, block_y, 'Adaptive\nAvgPool')
        current_x += 2.5
        
        # Fully connected layers
        for i, fc_layer in enumerate(architecture['fc_layers']):
            self.add_arrow(ax, current_x-1, current_x, block_y)
            
            fc_width = 1.2
            fc_height = 2.0 if i == 0 else 1.5  # First FC layer bigger
            
            self.create_3d_block(ax, current_x, block_y-fc_height/2, 
                               fc_width, fc_height, 0.3, 
                               self.colors['fc'])
            
            layer_name = f"FC{i+1}" if i < len(architecture['fc_layers'])-1 else "Output"
            self.add_text_label(ax, current_x+fc_width/2, block_y+0.3, layer_name)
            self.add_text_label(ax, current_x+fc_width/2, block_y-0.3, 
                               f"{fc_layer['out_features']} units")
            
            if fc_layer.get('activation'):
                self.add_text_label(ax, current_x+fc_width/2, block_y-0.8, 
                                   fc_layer['activation'])
            
            current_x += fc_width + 1.2
        
        # Save as EPS
        plt.tight_layout()
        plt.savefig(save_path, format='eps', dpi=300, bbox_inches='tight', 
                   pad_inches=0.2)
        plt.savefig(save_path.replace('.eps', '.png'), format='png', dpi=300, 
                   bbox_inches='tight', pad_inches=0.2)
        plt.close()

    def create_vit_diagram(self, model_name: str, architecture: Dict[str, Any], 
                          save_path: str) -> None:
        """Create ViT architecture diagram."""
        
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Title
        ax.text(10, 9.5, f'{model_name.upper()} Vision Transformer', 
                fontsize=16, fontweight='bold', ha='center')
        ax.text(10, 9, f"Parameters: {architecture['total_params']:,}", 
                fontsize=12, ha='center', style='italic')
        
        current_x = 1
        block_y = 5
        
        # Input + Patch Embedding
        self.create_3d_block(ax, current_x, block_y-1, 1.5, 2, 0.3, 
                           self.colors['input'])
        self.add_text_label(ax, current_x+0.75, block_y, 'Patch\nEmbedding')
        current_x += 2.5
        
        # Transformer blocks
        num_layers = architecture['num_transformer_layers']
        for i in range(min(num_layers, 3)):  # Show max 3 blocks visually
            self.add_arrow(ax, current_x-1, current_x, block_y)
            
            # Transformer block
            self.create_3d_block(ax, current_x, block_y-1.2, 2, 2.4, 0.4, 
                               self.colors['conv'])
            
            self.add_text_label(ax, current_x+1, block_y+0.5, f'Transformer\nBlock {i+1}')
            self.add_text_label(ax, current_x+1, block_y-0.2, 'Self-Attention')
            self.add_text_label(ax, current_x+1, block_y-0.8, 'Feed Forward')
            
            current_x += 3
            
            if i == 1 and num_layers > 3:  # Show ellipsis
                ax.text(current_x-0.5, block_y, '...', fontsize=20, 
                       ha='center', va='center')
                current_x += 1
        
        # Classification head
        self.add_arrow(ax, current_x-1.5, current_x, block_y)
        self.create_3d_block(ax, current_x, block_y-0.8, 1.5, 1.6, 0.3, 
                           self.colors['fc'])
        self.add_text_label(ax, current_x+0.75, block_y, 'Classification\nHead')
        
        # Save as EPS
        plt.tight_layout()
        plt.savefig(save_path, format='eps', dpi=300, bbox_inches='tight', 
                   pad_inches=0.2)
        plt.savefig(save_path.replace('.eps', '.png'), format='png', dpi=300, 
                   bbox_inches='tight', pad_inches=0.2)
        plt.close()


def parse_architecture_from_text() -> Dict[str, Dict[str, Any]]:
    """Parse architecture details from the all-architectures.txt output."""
    
    architectures = {
        'cnn_micro': {
            'total_params': 9850,
            'input_shape': (1, 1, 32, 32),
            'conv_blocks': [
                {'out_channels': 8, 'pool': True},
                {'out_channels': 16, 'pool': True}
            ],
            'fc_layers': [
                {'out_features': 32, 'activation': 'ReLU'},
                {'out_features': 10, 'activation': None}
            ]
        },
        'cnn_tiny': {
            'total_params': 38378,
            'input_shape': (1, 1, 32, 32),
            'conv_blocks': [
                {'out_channels': 16, 'pool': True},
                {'out_channels': 32, 'pool': True}
            ],
            'fc_layers': [
                {'out_features': 64, 'activation': 'ReLU'},
                {'out_features': 10, 'activation': None}
            ]
        },
        'cnn_64k': {
            'total_params': 1129098,
            'input_shape': (1, 1, 32, 32),
            'conv_blocks': [
                {'out_channels': 64, 'pool': True},
                {'out_channels': 128, 'pool': True}
            ],
            'fc_layers': [
                {'out_features': 512, 'activation': 'ReLU'},
                {'out_features': 10, 'activation': None}
            ]
        },
        'vit_micro': {
            'total_params': 21162,
            'input_shape': (1, 1, 32, 32),
            'num_transformer_layers': 2,
            'embed_dim': 32,
            'patch_size': 8
        },
        'vit_tiny': {
            'total_params': 110602,
            'input_shape': (1, 1, 32, 32),
            'num_transformer_layers': 3,
            'embed_dim': 64,
            'patch_size': 4
        }
    }
    
    return architectures


def main():
    """Generate VGG-style architecture diagrams."""
    parser = argparse.ArgumentParser(description='Generate VGG-style architecture diagrams')
    parser.add_argument('--output-dir', default='diagrams/vgg_style',
                       help='Output directory for EPS diagrams')
    parser.add_argument('--models', nargs='+', 
                       choices=['cnn_micro', 'cnn_tiny', 'cnn_64k', 'vit_micro', 'vit_tiny'],
                       default=['cnn_micro', 'cnn_tiny', 'cnn_64k', 'vit_micro', 'vit_tiny'],
                       help='Models to generate diagrams for')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse architectures
    architectures = parse_architecture_from_text()
    
    # Create diagram generator
    diagram_gen = VGGStyleDiagram()
    
    print(f"Generating VGG-style architecture diagrams...")
    
    for model_name in args.models:
        if model_name not in architectures:
            print(f"Warning: {model_name} not found in architectures")
            continue
            
        arch = architectures[model_name]
        output_path = output_dir / f"{model_name}_vgg_style.eps"
        
        print(f"  Generating {model_name}...")
        
        if model_name.startswith('vit'):
            diagram_gen.create_vit_diagram(model_name, arch, str(output_path))
        else:
            diagram_gen.create_cnn_diagram(model_name, arch, str(output_path))
        
        print(f"    Saved: {output_path}")
        print(f"    PNG:   {str(output_path).replace('.eps', '.png')}")
    
    print(f"\nGenerated diagrams in: {output_dir}")
    print("Files are ready for PowerDot presentations!")


if __name__ == '__main__':
    main()