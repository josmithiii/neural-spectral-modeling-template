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
import torch
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

# Set up project root and imports
import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


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
    
    def create_multihead_outputs(self, ax, start_x: float, block_y: float, 
                                num_heads: int, head_names: list, arrow_gap: float) -> float:
        """Create multiple output heads arranged vertically."""
        if num_heads <= 1:
            return start_x
            
        head_height = 0.8
        total_height = head_height * min(3, num_heads) + 0.4 * max(0, min(3, num_heads) - 1)
        start_y = block_y + total_height/2 - head_height/2
        
        # Show up to 3 heads, then ellipsis
        heads_to_show = min(3, num_heads)
        
        for i in range(heads_to_show):
            y_pos = start_y - i * (head_height + 0.4)
            
            # Create small output block
            self.create_3d_block(ax, start_x, y_pos - head_height/2, 1.2, head_height, 0.2, 
                               self.colors['output'])
            
            if i < len(head_names):
                head_name = head_names[i]
                # Truncate long names
                if len(head_name) > 8:
                    head_name = head_name[:8] + "..."
                self.add_text_label(ax, start_x + 0.6, y_pos, head_name, fontsize=8)
            
            if i == 1 and num_heads > 3:
                # Add ellipsis
                ax.text(start_x + 0.6, y_pos - (head_height + 0.4), '⋮', fontsize=16, 
                       ha='center', va='center')
                break
        
        return start_x + 1.2

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
        arrow_gap = 0.2  # Consistent gap before and after arrows (reduced)
        arrow_length = 0.8  # Consistent arrow length (reduced)
        
        # Input block
        input_shape = architecture['input_shape']
        self.create_3d_block(ax, current_x, block_y-1, 1.2, 2, 0.3, 
                            self.colors['input'])
        self.add_text_label(ax, current_x+0.6, block_y, 'Input\nLayer')
        self.add_text_label(ax, current_x+0.6, block_y-1.5, 
                           f"{input_shape[1]}×{input_shape[2]}×{input_shape[3]}")
        current_x += 1.2 + arrow_gap  # Move past block width + gap
        
        # Track spatial dimensions through the network
        current_height, current_width = architecture['input_shape'][2], architecture['input_shape'][3]
        
        # Convolutional blocks
        for i, conv_block in enumerate(architecture['conv_blocks']):
            # Add arrow with consistent spacing
            self.add_arrow(ax, current_x, current_x + arrow_length, block_y)
            current_x += arrow_length + arrow_gap
            
            # Conv block
            channels = conv_block['out_channels']
            block_width = min(2.0, max(0.8, channels/50))  # Scale with channels
            block_height = min(2.5, max(1.5, channels/40))
            
            self.create_3d_block(ax, current_x, block_y-block_height/2, 
                               block_width, block_height, 0.4, 
                               self.colors['conv'])
            
            # Labels with more details
            kernel_size = conv_block.get('kernel_size', 3)
            stride = conv_block.get('stride', 1)
            padding = conv_block.get('padding', 1)
            params = conv_block.get('params', 0)
            
            # Update spatial dimensions after convolution
            current_height = (current_height + 2*padding - kernel_size) // stride + 1
            current_width = (current_width + 2*padding - kernel_size) // stride + 1
            
            self.add_text_label(ax, current_x+block_width/2, block_y+0.5, 
                               f'Conv{i+1}')
            self.add_text_label(ax, current_x+block_width/2, block_y+0.1, 
                               f'{channels} filters')
            self.add_text_label(ax, current_x+block_width/2, block_y-0.3, 
                               f'{kernel_size}×{kernel_size}, s={stride}')
            if params > 0:
                self.add_text_label(ax, current_x+block_width/2, block_y-0.7, 
                                   f'{params:,} params')
            
            # Add spatial dimensions below the block
            self.add_text_label(ax, current_x+block_width/2, block_y-block_height/2-0.6, 
                               f'{channels}×{current_height}×{current_width}')
            
            current_x += block_width + arrow_gap
            
            if conv_block.get('pool', False):
                # Add pooling layer with consistent spacing
                self.add_arrow(ax, current_x, current_x + arrow_length, block_y)
                current_x += arrow_length + arrow_gap
                
                # Update spatial dimensions after pooling (2x2 with stride 2)
                current_height = current_height // 2
                current_width = current_width // 2
                
                pool_width = 0.6
                self.create_3d_block(ax, current_x, block_y-0.5, pool_width, 1, 0.2, 
                                   self.colors['pool'])
                self.add_text_label(ax, current_x + pool_width/2, block_y+0.2, 'Pool\n2×2')
                
                # Add spatial dimensions below pooling
                self.add_text_label(ax, current_x + pool_width/2, block_y-0.8, 
                                   f'{channels}×{current_height}×{current_width}')
                current_x += pool_width + arrow_gap
        
        # Feature extraction
        self.add_arrow(ax, current_x, current_x + arrow_length, block_y)
        current_x += arrow_length + arrow_gap
        
        # AdaptiveAvgPool reduces to fixed output size (typically 1x1 or 4x4)
        adaptive_out_size = 1  # Most common case for classification
        flattened_features = channels * adaptive_out_size * adaptive_out_size
        
        feature_width = 1.0
        self.create_3d_block(ax, current_x, block_y-1, feature_width, 2, 0.2, 
                           self.colors['pool'])
        self.add_text_label(ax, current_x + feature_width/2, block_y+0.3, 'Adaptive\nAvgPool')
        self.add_text_label(ax, current_x + feature_width/2, block_y-0.3, 'Output: 1×1 or 4×4')
        
        # Add output dimensions below
        self.add_text_label(ax, current_x + feature_width/2, block_y-1.3, 
                           f'{flattened_features} features')
        
        # Check for auxiliary inputs
        has_auxiliary = architecture.get('has_auxiliary', False)
        aux_input_size = architecture.get('auxiliary_input_size', 0)
        aux_hidden_size = architecture.get('auxiliary_hidden_size', 32)
        
        # Check if auxiliary should be shown (either configured or placeholder)
        show_auxiliary = has_auxiliary and (isinstance(aux_input_size, str) or aux_input_size > 0)
        
        if show_auxiliary:
            # Show auxiliary input path
            aux_start_x = current_x + feature_width/2
            aux_y_offset = -2.5  # Position below the main path
            
            # Auxiliary input block
            aux_input_width = 0.8
            self.create_3d_block(ax, aux_start_x - aux_input_width/2, block_y + aux_y_offset - 0.4, 
                               aux_input_width, 0.8, 0.15, '#9013FE')  # Purple like input
            self.add_text_label(ax, aux_start_x, block_y + aux_y_offset, 'Aux Input', fontsize=8)
            # Handle placeholder values
            if isinstance(aux_input_size, str):
                size_label = f'{aux_input_size}D'
            else:
                size_label = f'{aux_input_size}D'
            self.add_text_label(ax, aux_start_x, block_y + aux_y_offset - 0.6, 
                               size_label, fontsize=7)
            
            # Auxiliary MLP
            aux_mlp_x = aux_start_x + 1.5
            self.create_3d_block(ax, aux_mlp_x - 0.6, block_y + aux_y_offset - 0.4, 
                               1.2, 0.8, 0.15, '#7ED321')  # Green like FC
            self.add_text_label(ax, aux_mlp_x, block_y + aux_y_offset, 'Aux MLP', fontsize=8)
            self.add_text_label(ax, aux_mlp_x, block_y + aux_y_offset - 0.6, 
                               f'→{aux_hidden_size}D', fontsize=7)
            
            # Arrow from aux input to aux MLP
            self.add_arrow(ax, aux_start_x + aux_input_width/2, aux_mlp_x - 0.6, 
                          block_y + aux_y_offset, '#2ECC71', width=0.2)
        
        current_x += feature_width + arrow_gap
        
        # Check if this is a multihead model
        is_multihead = architecture.get('is_multihead', False)
        heads_info = architecture.get('heads_info', {})
        
        if is_multihead:
            # For multihead models, show shared FC layer, then multiple heads
            shared_fc = architecture['fc_layers'][0] if architecture['fc_layers'] else None
            
            if shared_fc:
                # Shared features layer
                self.add_arrow(ax, current_x, current_x + arrow_length, block_y)
                current_x += arrow_length + arrow_gap
                
                fc_width = 1.2
                fc_height = 2.0
                
                self.create_3d_block(ax, current_x, block_y-fc_height/2, 
                                   fc_width, fc_height, 0.3, 
                                   self.colors['fc'])
                
                # Check if auxiliary features are concatenated here
                has_auxiliary = architecture.get('has_auxiliary', False)
                aux_input_size = architecture.get('auxiliary_input_size', 0)
                aux_hidden_size = architecture.get('auxiliary_hidden_size', 32)
                show_auxiliary = has_auxiliary and (isinstance(aux_input_size, str) or aux_input_size > 0)
                
                if show_auxiliary:
                    self.add_text_label(ax, current_x+fc_width/2, block_y+0.7, 'Shared\nFeatures')
                    self.add_text_label(ax, current_x+fc_width/2, block_y+0.2, 
                                       f"{shared_fc['out_features']} units")
                    self.add_text_label(ax, current_x+fc_width/2, block_y-0.2, '(concat with aux)')
                    
                    # Draw concatenation arrow from aux MLP
                    aux_mlp_x = current_x - 2.5  # Approximate position of aux MLP
                    concat_y = block_y - 1.0  # Lower connection point
                    
                    # Curved arrow from aux MLP to shared features  
                    self.add_arrow(ax, aux_mlp_x + 0.6, current_x, concat_y, '#FF6B6B', width=0.2)
                    self.add_text_label(ax, (aux_mlp_x + current_x)/2, concat_y - 0.3, 
                                       'concat', fontsize=7)
                else:
                    self.add_text_label(ax, current_x+fc_width/2, block_y+0.5, 'Shared\nFeatures')
                    self.add_text_label(ax, current_x+fc_width/2, block_y+0.1, 
                                       f"{shared_fc['out_features']} units")
                
                if shared_fc.get('params', 0) > 0:
                    self.add_text_label(ax, current_x+fc_width/2, block_y-0.5, 
                                       f"{shared_fc['params']:,} params")
                
                current_x += fc_width + arrow_gap
            
            # Multiple output heads
            if heads_info:
                self.add_arrow(ax, current_x, current_x + arrow_length, block_y)
                current_x += arrow_length + arrow_gap
                
                num_heads = len(heads_info)
                head_names = list(heads_info.keys())
                
                # Always use the multihead visualization for ordinal models to show the structure
                if num_heads >= 1:
                    if num_heads == 1:
                        # For single head, create vertically symmetric blocks around the arrow level
                        head_name, num_classes = next(iter(heads_info.items()))
                        
                        head_height = 0.7
                        gap = 0.3  # Gap between blocks
                        
                        # Upper block (main output head) - centered above arrow level
                        upper_y = block_y + gap/2 + head_height/2
                        self.create_3d_block(ax, current_x, upper_y - head_height/2, 1.2, head_height, 0.2, 
                                           self.colors['output'])
                        self.add_text_label(ax, current_x + 0.6, upper_y, head_name, fontsize=9)
                        
                        # Ellipsis centered on the arrow level
                        self.add_text_label(ax, current_x + 0.6, block_y, '⋮', fontsize=14)
                        
                        # Lower block (potential heads) - centered below arrow level  
                        lower_y = block_y - gap/2 - head_height/2
                        self.create_3d_block(ax, current_x, lower_y - head_height/2, 1.2, head_height, 0.1, 
                                           self.colors['output'], alpha=0.4)
                        
                        # Add output type info below the lower block
                        output_mode = architecture.get('output_mode', 'classification')
                        if output_mode == 'regression':
                            label = f"{num_classes} params (floats)"
                        else:
                            if num_classes == 1:
                                label = "1 logit (binary)"
                            else:
                                label = f"{num_classes} logits (floats)"
                        
                        self.add_text_label(ax, current_x + 0.6, lower_y - head_height/2 - 0.4, 
                                           label, fontsize=8)
                        
                        current_x += 1.2
                    else:
                        # Multiple heads - use original logic
                        self.create_multihead_outputs(ax, current_x, block_y, num_heads, head_names, arrow_gap)
                        
                        # Add summary info with output type
                        total_outputs = sum(heads_info.values())
                        output_mode = architecture.get('output_mode', 'classification')
                        
                        if output_mode == 'regression':
                            summary = f"{num_heads} heads\n{total_outputs} total params"
                        else:
                            summary = f"{num_heads} heads\n{total_outputs} total logits"
                        
                        self.add_text_label(ax, current_x + 0.6, block_y-2, summary, fontsize=8)
                        current_x += 1.2
        else:
            # Traditional single-head CNN
            for i, fc_layer in enumerate(architecture['fc_layers']):
                self.add_arrow(ax, current_x, current_x + arrow_length, block_y)
                current_x += arrow_length + arrow_gap
                
                # Make output block more compact
                is_output = i == len(architecture['fc_layers']) - 1
                fc_width = 1.0 if is_output else 1.2  # Narrower output block
                fc_height = 2.0 if i == 0 else 1.5  # First FC layer bigger
                
                self.create_3d_block(ax, current_x, block_y-fc_height/2, 
                                   fc_width, fc_height, 0.3, 
                                   self.colors['fc'])
                
                layer_name = f"FC{i+1}" if not is_output else "Output"
                self.add_text_label(ax, current_x+fc_width/2, block_y+0.5, layer_name)
                self.add_text_label(ax, current_x+fc_width/2, block_y+0.1, 
                                   f"{fc_layer['out_features']} units")
                
                # Add parameter count if available
                if fc_layer.get('params', 0) > 0:
                    self.add_text_label(ax, current_x+fc_width/2, block_y-0.3, 
                                       f"{fc_layer['params']:,} params")
                
                if fc_layer.get('activation'):
                    self.add_text_label(ax, current_x+fc_width/2, block_y-0.7, 
                                       fc_layer['activation'])
                
                # Add output dimensions below the block for final layer
                if is_output:
                    output_mode = architecture.get('output_mode', 'classification')
                    num_outputs = fc_layer['out_features']
                    
                    if output_mode == 'regression':
                        param_names = architecture.get('parameter_names', [])
                        if len(param_names) == num_outputs:
                            label = f"{num_outputs} params (floats)"
                        else:
                            label = f"{num_outputs} values (floats)"
                    else:
                        # Classification mode
                        if num_outputs == 1:
                            label = "1 logit (binary)"
                        else:
                            label = f"{num_outputs} logits (floats)"
                    
                    self.add_text_label(ax, current_x+fc_width/2, block_y-fc_height/2-0.6, label)
                
                current_x += fc_width + arrow_gap
        
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
        self.add_text_label(ax, current_x+0.75, block_y+0.3, 'Patch\nEmbedding')
        
        # Add patch embedding details
        patch_size = architecture.get('patch_size', 4)
        embed_dim = architecture.get('embed_dim', 64)
        self.add_text_label(ax, current_x+0.75, block_y-0.3, f'{patch_size}×{patch_size} patches')
        self.add_text_label(ax, current_x+0.75, block_y-0.7, f'{embed_dim}D embed')
        current_x += 2.5
        
        # Transformer blocks
        num_layers = architecture['num_transformer_layers']
        for i in range(min(num_layers, 3)):  # Show max 3 blocks visually
            self.add_arrow(ax, current_x-1, current_x, block_y)
            
            # Transformer block
            self.create_3d_block(ax, current_x, block_y-1.2, 2, 2.4, 0.4, 
                               self.colors['conv'])
            
            self.add_text_label(ax, current_x+1, block_y+0.7, f'Transformer\nBlock {i+1}')
            
            # Add architectural details
            num_heads = architecture.get('num_heads', 4)
            embed_dim = architecture.get('embed_dim', 64)
            self.add_text_label(ax, current_x+1, block_y+0.1, f'{num_heads} heads')
            self.add_text_label(ax, current_x+1, block_y-0.3, 'Self-Attention')
            self.add_text_label(ax, current_x+1, block_y-0.7, f'FFN {embed_dim*4}→{embed_dim}')
            
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


def parse_architecture_from_config(config_name: str) -> Dict[str, Any]:
    """Parse architecture details from Hydra config."""
    
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    # Get absolute path to configs directory
    config_dir = str(Path(__file__).parent.parent / "configs")
    
    try:
        # Initialize Hydra with the configs directory
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            # Load the specific model config
            cfg = compose(config_name="train.yaml", overrides=[f"model={config_name}"])
            
            # Extract dynamic input parameters from model config
            net_config = cfg.model.net
            
            # Determine input channels (CNN uses input_channels, ViT uses n_channels)
            if hasattr(net_config, 'input_channels'):
                input_channels = net_config.input_channels
            elif hasattr(net_config, 'n_channels'):
                input_channels = net_config.n_channels
            else:
                input_channels = 1  # fallback
            
            # Determine input size (CNN uses input_size, ViT uses image_size)
            if hasattr(net_config, 'input_size'):
                input_size = net_config.input_size
            elif hasattr(net_config, 'image_size'):
                input_size = net_config.image_size
            else:
                input_size = 32  # fallback to VIMH default
            
            # Create dynamic input shape (batch_size=1, channels, height, width)
            input_shape = (1, input_channels, input_size, input_size)
            
            # Instantiate the model to get actual architecture details
            model = hydra.utils.instantiate(cfg.model)
            total_params = sum(p.numel() for p in model.parameters())
            
            # Analyze model architecture
            architecture = {
                'total_params': total_params,
                'input_shape': input_shape,
                'model_type': 'vit' if 'vit' in config_name.lower() else 'cnn'
            }
            
            if architecture['model_type'] == 'vit':
                # ViT-specific parsing
                architecture['num_transformer_layers'] = getattr(net_config, 'depth', 3)
                architecture['embed_dim'] = getattr(net_config, 'embed_dim', 64)
                architecture['patch_size'] = getattr(net_config, 'patch_size', 4)
                architecture['num_heads'] = getattr(net_config, 'num_heads', 4)
                architecture['mlp_ratio'] = getattr(net_config, 'mlp_ratio', 4.0)
            else:
                # CNN-specific parsing
                conv_blocks = []
                fc_layers = []
                
                # Extract conv block info from model structure
                if hasattr(model.net, 'conv_layers'):
                    conv_layers = model.net.conv_layers
                    for i, layer in enumerate(conv_layers):
                        if hasattr(layer, 'out_channels'):
                            # Try to get detailed layer parameters
                            kernel_size = getattr(layer, 'kernel_size', (3, 3))
                            if isinstance(kernel_size, tuple):
                                kernel_size = kernel_size[0]
                            stride = getattr(layer, 'stride', (1, 1))
                            if isinstance(stride, tuple):
                                stride = stride[0]
                            padding = getattr(layer, 'padding', (1, 1))
                            if isinstance(padding, tuple):
                                padding = padding[0]
                            
                            # Calculate layer parameters
                            layer_params = sum(p.numel() for p in layer.parameters()) if hasattr(layer, 'parameters') else 0
                            
                            conv_blocks.append({
                                'out_channels': layer.out_channels,
                                'kernel_size': kernel_size,
                                'stride': stride,
                                'padding': padding,
                                'params': layer_params,
                                'pool': True  # Assume pooling for VGG-style
                            })
                
                # Check for multihead structure first
                # Consider it multihead if it has heads_config with multiple heads OR if it's already multihead
                has_heads_config = hasattr(model.net, 'heads_config') and model.net.heads_config
                actual_multihead = hasattr(model.net, 'is_multihead') and model.net.is_multihead
                
                # For ordinal models, even single head should be treated as multihead-style if configured
                is_multihead_style = (has_heads_config and len(model.net.heads_config) >= 1) or actual_multihead
                is_multihead = actual_multihead
                heads_info = {}
                
                if has_heads_config and 'ordinal' in config_name.lower():
                    # For ordinal models, treat as multihead-style even with single head
                    heads_info = dict(model.net.heads_config)
                    # Improve generic head names
                    heads_to_rename = {}
                    for old_name, classes in heads_info.items():
                        if old_name.startswith('synth_param'):
                            heads_to_rename[old_name] = 'Out Heads'
                    
                    # Apply renaming
                    for old_name, new_name in heads_to_rename.items():
                        heads_info[new_name] = heads_info.pop(old_name)
                    
                    is_multihead = True  # Force multihead visualization for ordinal
                elif is_multihead and hasattr(model.net, 'heads'):
                    # Extract multihead information
                    for head_name, head_module in model.net.heads.items():
                        if hasattr(head_module, '__iter__'):
                            # Sequential head (regression mode)
                            for layer in head_module:
                                if hasattr(layer, 'out_features'):
                                    heads_info[head_name] = layer.out_features
                                    break
                        elif hasattr(head_module, 'out_features'):
                            # Single linear layer head
                            heads_info[head_name] = head_module.out_features
                    
                    # Extract shared features layer
                    if hasattr(model.net, 'shared_features'):
                        shared_features = model.net.shared_features
                        # Find the linear layer in shared_features
                        for layer in shared_features:
                            if hasattr(layer, 'out_features'):
                                layer_params = sum(p.numel() for p in layer.parameters()) if hasattr(layer, 'parameters') else 0
                                fc_layers.append({
                                    'out_features': layer.out_features,
                                    'params': layer_params,
                                    'activation': 'ReLU'
                                })
                                break
                else:
                    # Extract traditional FC layer info
                    if hasattr(model.net, 'classifier') or hasattr(model.net, 'fc'):
                        classifier = getattr(model.net, 'classifier', getattr(model.net, 'fc', None))
                        if classifier:
                            # Handle both single layer and sequential layers
                            if hasattr(classifier, '__iter__') and not hasattr(classifier, 'out_features'):
                                # It's a Sequential/ModuleList - iterate through layers
                                for layer in classifier:
                                    if hasattr(layer, 'out_features'):
                                        activation = 'ReLU' if hasattr(layer, 'relu') or 'ReLU' in str(layer) else None
                                        layer_params = sum(p.numel() for p in layer.parameters()) if hasattr(layer, 'parameters') else 0
                                        fc_layers.append({
                                            'out_features': layer.out_features,
                                            'params': layer_params,
                                            'activation': activation
                                        })
                            elif hasattr(classifier, 'out_features'):
                                # It's a single Linear layer
                                layer_params = sum(p.numel() for p in classifier.parameters()) if hasattr(classifier, 'parameters') else 0
                                fc_layers.append({
                                    'out_features': classifier.out_features,
                                    'params': layer_params,
                                    'activation': None
                                })
                
                # Fallback to reasonable defaults if extraction fails
                if not conv_blocks:
                    if '64k' in config_name:
                        conv_blocks = [{'out_channels': 64, 'pool': True}, {'out_channels': 128, 'pool': True}]
                    elif 'tiny' in config_name:
                        conv_blocks = [{'out_channels': 16, 'pool': True}, {'out_channels': 32, 'pool': True}]
                    else:  # micro
                        conv_blocks = [{'out_channels': 8, 'pool': True}, {'out_channels': 16, 'pool': True}]
                
                if not fc_layers:
                    # Get output dimensions from model config
                    output_dim = getattr(net_config, 'num_classes', 10)
                    if '64k' in config_name:
                        fc_layers = [{'out_features': 512, 'activation': 'ReLU'}, {'out_features': output_dim, 'activation': None}]
                    elif 'tiny' in config_name:
                        fc_layers = [{'out_features': 64, 'activation': 'ReLU'}, {'out_features': output_dim, 'activation': None}]
                    else:  # micro
                        fc_layers = [{'out_features': 32, 'activation': 'ReLU'}, {'out_features': output_dim, 'activation': None}]
                
                # Check for auxiliary inputs
                has_auxiliary_attr = hasattr(model.net, 'auxiliary_input_size')
                aux_input_size = getattr(model.net, 'auxiliary_input_size', 0)
                aux_hidden_size = getattr(model.net, 'auxiliary_hidden_size', 32)
                
                # For auxiliary models, show visualization even if not auto-configured yet
                if 'auxiliary' in config_name.lower() and has_auxiliary_attr:
                    has_auxiliary = True
                    # Use placeholder values if not configured
                    if aux_input_size == 0:
                        aux_input_size = "N"  # Placeholder for auto-configuration
                else:
                    has_auxiliary = has_auxiliary_attr and aux_input_size > 0
                
                # Detect output mode
                output_mode = getattr(model.net, 'output_mode', 'classification')
                parameter_names = getattr(model.net, 'parameter_names', [])
                
                architecture['conv_blocks'] = conv_blocks
                architecture['fc_layers'] = fc_layers
                architecture['is_multihead'] = is_multihead
                architecture['heads_info'] = heads_info
                architecture['has_auxiliary'] = has_auxiliary
                architecture['auxiliary_input_size'] = aux_input_size
                architecture['auxiliary_hidden_size'] = aux_hidden_size
                architecture['output_mode'] = output_mode
                architecture['parameter_names'] = parameter_names
            
            return architecture
            
    except Exception as e:
        print(f"Error parsing config {config_name}: {e}")
        return None


def get_available_models() -> List[str]:
    """Get list of available model configs."""
    config_path = Path("configs/model")
    if config_path.exists():
        return [config_file.stem for config_file in config_path.glob("*.yaml")]
    return []


def main():
    """Generate VGG-style architecture diagrams."""
    # Get available models dynamically
    available_models = get_available_models()
    
    parser = argparse.ArgumentParser(description='Generate VGG-style architecture diagrams')
    parser.add_argument('--output-dir', default='diagrams/vgg_style',
                       help='Output directory for EPS diagrams')
    parser.add_argument('--models', nargs='+', 
                       choices=available_models,
                       default=available_models,
                       help='Models to generate diagrams for')
    parser.add_argument('--list', action='store_true',
                       help='List available model configs')
    
    args = parser.parse_args()
    
    if args.list:
        print("Available model configs:")
        for model in available_models:
            print(f"  {model}")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create diagram generator
    diagram_gen = VGGStyleDiagram()
    
    print(f"Generating VGG-style architecture diagrams for {len(args.models)} models...")
    
    for i, model_name in enumerate(args.models, 1):
        print(f"[{i}/{len(args.models)}] Processing {model_name}...")
        
        # Parse architecture from config
        arch = parse_architecture_from_config(model_name)
        if arch is None:
            print(f"  Skipped: Failed to parse config")
            continue
            
        output_path = output_dir / f"{model_name}_vgg_style.eps"
        
        try:
            if arch['model_type'] == 'vit':
                diagram_gen.create_vit_diagram(model_name, arch, str(output_path))
            else:
                diagram_gen.create_cnn_diagram(model_name, arch, str(output_path))
            
            print(f"  ✓ Generated: {output_path}")
            print(f"  ✓ PNG:       {str(output_path).replace('.eps', '.png')}")
            print(f"  ✓ Parameters: {arch['total_params']:,}")
            
        except Exception as e:
            print(f"  ✗ Error generating diagram: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Generated diagrams in: {output_dir}")
    print("Files are ready for PowerDot presentations!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()