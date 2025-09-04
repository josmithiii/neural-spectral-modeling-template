"""
Architecture Utilities for Neural Spectral Modeling Template

This module provides shared utilities for architecture metadata extraction,
storage, and model configuration. It centralizes logic that was previously
duplicated across training and evaluation scripts.
"""

from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

from lightning import LightningDataModule, LightningModule

log = logging.getLogger(__name__)


@dataclass
class ArchitectureMetadata:
    """Container for model architecture metadata."""
    architecture_type: str
    input_channels: int = 1
    conv1_channels: int = 16
    conv2_channels: int = 32
    fc_hidden: int = 64
    dropout: float = 0.5
    input_size: int = 32
    embed_dim: int = 64
    n_layers: int = 6
    n_attention_heads: int = 4
    patch_size: int = 4
    image_size: Tuple[int, int] = (28, 28)
    forward_mul: float = 2.0
    use_torch_layers: bool = False


class ArchitectureMetadataExtractor:
    """Extracts and stores architecture metadata from trained models."""

    def __init__(self):
        """Initialize the architecture metadata extractor."""
        self.default_values = {
            'dropout': 0.5,
            'patch_size': 4,
            'embed_dim': 64,
            'input_channels': 1,
            'conv1_channels': 16,
            'conv2_channels': 32,
            'fc_hidden': 64,
            'input_size': 32,
            'n_layers': 6,
            'n_attention_heads': 4,
            'forward_mul': 2.0,
            'use_torch_layers': False
        }

    def extract_and_store_metadata(self, model: LightningModule, datamodule: LightningDataModule) -> Dict[str, Any]:
        """
        Extract architecture metadata from model and dataset, then store in model hparams.

        Args:
            model: The Lightning model to extract metadata from
            datamodule: The datamodule containing dataset metadata

        Returns:
            Dictionary containing the extracted architecture metadata
        """
        if not hasattr(model, 'net'):
            log.warning("Model does not have 'net' attribute, skipping architecture metadata extraction")
            return {}

        # Get dataset metadata if available
        dataset_metadata = self._get_dataset_metadata(datamodule)

        # Extract architecture metadata based on model type
        architecture_metadata = self._extract_architecture_metadata(model, dataset_metadata)

        # Store in model hparams for checkpoint saving
        if architecture_metadata:
            model.hparams['architecture_metadata'] = architecture_metadata
            log.info(f"Stored architecture metadata: {architecture_metadata}")

        return architecture_metadata

    def _get_dataset_metadata(self, datamodule: LightningDataModule) -> Dict[str, Any]:
        """Get dataset metadata from datamodule."""
        dataset_metadata = {}
        if hasattr(datamodule, 'get_dataset_info'):
            try:
                # Ensure datamodule is set up
                if not hasattr(datamodule, 'data_train') or datamodule.data_train is None:
                    datamodule.setup("fit")
                dataset_metadata = datamodule.get_dataset_info()
                log.info(f"Retrieved dataset metadata: {list(dataset_metadata.keys())}")
            except Exception as e:
                log.warning(f"Could not get dataset metadata: {e}")
        return dataset_metadata

    def _extract_architecture_metadata(self, model: LightningModule, dataset_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract architecture metadata based on model type."""
        net = model.net

        # Detect ViT architecture
        if hasattr(net, 'embedding') and hasattr(net.embedding, 'pos_embedding'):
            return self._extract_vit_metadata(model, dataset_metadata)

        # Detect CNN architecture
        elif hasattr(net, 'conv_layers') or hasattr(net, 'conv1'):
            return self._extract_cnn_metadata(model, dataset_metadata)

        else:
            log.warning("Could not detect model architecture type")
            return {}

    def _extract_vit_metadata(self, model: LightningModule, dataset_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata for Vision Transformer models."""
        metadata = {'type': 'ViT'}
        net = model.net

        # Try to get parameters from model hparams first
        model_hparams = model.hparams
        if hasattr(model, 'net') and hasattr(model_hparams, 'net'):
            net_hparams = model_hparams.net

            # Extract parameters from the ViT model config
            metadata.update({
                'embed_dim': getattr(net_hparams, 'embed_dim', self.default_values['embed_dim']),
                'n_layers': getattr(net_hparams, 'n_layers', self.default_values['n_layers']),
                'n_attention_heads': getattr(net_hparams, 'n_attention_heads', self.default_values['n_attention_heads']),
                'patch_size': getattr(net_hparams, 'patch_size', self.default_values['patch_size']),
                'image_size': getattr(net_hparams, 'image_size', [28, 28]),
                'input_channels': getattr(net_hparams, 'n_channels', self.default_values['input_channels']),
                'forward_mul': getattr(net_hparams, 'forward_mul', self.default_values['forward_mul']),
                'dropout': getattr(net_hparams, 'dropout', self.default_values['dropout']),
                'use_torch_layers': getattr(net_hparams, 'use_torch_layers', self.default_values['use_torch_layers'])
            })
        else:
            # Fallback - try to infer from model structure and dataset
            log.warning("Could not access ViT hyperparameters, using inference from model")
            metadata.update(self._infer_vit_from_structure(net, dataset_metadata))

        return metadata

    def _extract_cnn_metadata(self, model: LightningModule, dataset_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata for CNN models."""
        metadata = {'type': 'CNN'}
        net = model.net

        # Try to get parameters from model hparams first
        model_hparams = model.hparams
        if hasattr(model, 'net') and hasattr(model_hparams, 'net'):
            net_hparams = model_hparams.net

            # Extract parameters from the CNN model config
            metadata.update({
                'input_channels': getattr(net_hparams, 'input_channels', self.default_values['input_channels']),
                'conv1_channels': getattr(net_hparams, 'conv1_channels', self.default_values['conv1_channels']),
                'conv2_channels': getattr(net_hparams, 'conv2_channels', self.default_values['conv2_channels']),
                'fc_hidden': getattr(net_hparams, 'fc_hidden', self.default_values['fc_hidden']),
                'dropout': getattr(net_hparams, 'dropout', self.default_values['dropout']),
            })

            # Get input size from dataset metadata or hparams
            if dataset_metadata and 'height' in dataset_metadata and 'width' in dataset_metadata:
                metadata['input_size'] = max(dataset_metadata['height'], dataset_metadata['width'])
            else:
                metadata['input_size'] = getattr(net_hparams, 'input_size', self.default_values['input_size'])
        else:
            # Fallback - try to infer from model structure
            log.warning("Could not access CNN hyperparameters, using inference from model")
            metadata.update(self._infer_cnn_from_structure(net, dataset_metadata))

        return metadata

    def _infer_vit_from_structure(self, net, dataset_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Infer ViT parameters from model structure."""
        metadata = {
            'embed_dim': self.default_values['embed_dim'],
            'patch_size': self.default_values['patch_size'],
            'image_size': self.default_values['input_size'],
            'n_layers': self.default_values['n_layers'],
            'n_attention_heads': self.default_values['n_attention_heads'],
            'input_channels': self.default_values['input_channels']
        }

        # Try to infer from position embedding dimensions
        if hasattr(net, 'embedding') and hasattr(net.embedding, 'pos_embedding'):
            pos_emb_shape = net.embedding.pos_embedding.shape
            num_patches = pos_emb_shape[1]  # Second dimension is number of patches
            embed_dim = pos_emb_shape[2]    # Third dimension is embedding dimension
            metadata['embed_dim'] = embed_dim

            # Infer image size and patch size from dataset metadata and num_patches
            if dataset_metadata and 'height' in dataset_metadata and 'width' in dataset_metadata:
                height = dataset_metadata['height']
                width = dataset_metadata['width']
                log.info(f"Dataset dimensions: {height}x{width}, num_patches: {num_patches}")

                # Try common patch sizes to find the right one
                found = False
                for ps in [2, 4, 6, 8, 12, 16]:
                    if (height % ps == 0) and (width % ps == 0):
                        expected_patches = (height // ps) * (width // ps)
                        if expected_patches == num_patches:
                            metadata['patch_size'] = ps
                            metadata['image_size'] = [height, width]
                            log.info(f"Found match: patch_size={ps}, image_size=[{height}, {width}]")
                            found = True
                            break

                if not found:
                    log.warning(f"Could not find patch size for {height}x{width} with {num_patches} patches, using defaults")

        metadata['n_layers'] = len(net.encoder) if hasattr(net, 'encoder') else self.default_values['n_layers']
        return metadata

    def _infer_cnn_from_structure(self, net, dataset_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Infer CNN parameters from model structure."""
        metadata = {
            'input_channels': self.default_values['input_channels'],
            'conv1_channels': self.default_values['conv1_channels'],
            'conv2_channels': self.default_values['conv2_channels'],
            'fc_hidden': self.default_values['fc_hidden'],
            'dropout': self.default_values['dropout']
        }

        # Try to infer from actual layer weights
        if hasattr(net, 'conv_layers') and len(net.conv_layers) > 0:
            # Get input channels from first conv layer
            first_conv = net.conv_layers[0]
            if hasattr(first_conv, 'weight'):
                conv_weight = first_conv.weight
                metadata['input_channels'] = conv_weight.shape[1]  # Input channels
                metadata['conv1_channels'] = conv_weight.shape[0]  # Output channels

            # Get second conv layer channels if available
            if len(net.conv_layers) > 4:  # Common pattern: Conv->ReLU->MaxPool->Conv
                second_conv = net.conv_layers[4]
                if hasattr(second_conv, 'weight'):
                    metadata['conv2_channels'] = second_conv.weight.shape[0]

        # Try to infer FC hidden size
        if hasattr(net, 'shared_features') and len(net.shared_features) > 0:
            # Look for Linear layers in shared_features
            for layer in net.shared_features:
                if hasattr(layer, 'weight') and len(layer.weight.shape) == 2:
                    metadata['fc_hidden'] = layer.weight.shape[0]
                    break

        # Get input size from dataset metadata
        if dataset_metadata and 'height' in dataset_metadata and 'width' in dataset_metadata:
            metadata['input_size'] = max(dataset_metadata['height'], dataset_metadata['width'])
        else:
            metadata['input_size'] = self.default_values['input_size']

        return metadata


def create_spectrogram_transforms() -> Tuple[Dict[str, Any], str]:
    """
    Create appropriate transforms for spectrogram datasets.

    Returns:
        Tuple of (transforms_kwargs, log_message)
    """
    from torchvision.transforms import transforms

    # Create spectrogram-appropriate transforms
    spectrogram_train_transform = transforms.Compose([
        # For now, just use identity transform - we can add spectrogram-specific augmentations later
        # No normalization either to preserve the original spectrogram values for audio reconstruction
    ])

    spectrogram_val_test_transform = transforms.Compose([
        # Identity transform for validation and test - no modifications
    ])

    datamodule_kwargs = {
        'train_transform': spectrogram_train_transform,
        'val_transform': spectrogram_val_test_transform,
        'test_transform': spectrogram_val_test_transform
    }

    log_message = "Using spectrogram-appropriate transforms (no image augmentations)"
    return datamodule_kwargs, log_message


def configure_vimh_model(model, datamodule, cfg) -> None:
    """
    Configure model for VIMH datasets with auto-configuration from metadata.

    Args:
        model: The Lightning model to configure
        datamodule: The datamodule with VIMH data
        cfg: Hydra configuration
    """
    try:
        from src.utils.vimh_utils import get_parameter_names_from_metadata, get_heads_config_from_metadata, get_image_dimensions_from_metadata

        # Auto-configure input channels from dataset metadata
        if hasattr(model, 'net') and hasattr(model.net, 'input_channels'):
            height, width, channels = get_image_dimensions_from_metadata(cfg.data.data_dir)
            if model.net.input_channels != channels:
                log.info(f"Auto-configuring network input channels: {model.net.input_channels} -> {channels}")
                model.net.input_channels = channels

        parameter_names = get_parameter_names_from_metadata(cfg.data.data_dir)
        if parameter_names and hasattr(model, 'net'):
            log.info(f"Configuring model with parameter names from dataset: {parameter_names}")

            # Configure model based on output mode
            if hasattr(model, 'output_mode') and model.output_mode == 'regression':
                # For regression mode, use parameter_names
                model.net.parameter_names = parameter_names

                # Auto-configure regression loss functions with parameter ranges
                if hasattr(model, 'criteria'):
                    _configure_regression_criteria(model, cfg.data.data_dir, parameter_names)

            else:
                # For classification/ordinal mode, use heads_config
                heads_config = get_heads_config_from_metadata(cfg.data.data_dir)
                model.net.heads_config = heads_config

            # Auto-configure loss_weights (equal weight for all parameters)
            if not hasattr(model, 'loss_weights') or not model.loss_weights or len(model.loss_weights) == 0:
                model.loss_weights = {name: 1.0 for name in parameter_names}
                log.info(f"Auto-configured loss_weights: {model.loss_weights}")

        # Auto-configure auxiliary input size based on dataset auxiliary features
        if (hasattr(cfg.data, 'auxiliary_features') and cfg.data.auxiliary_features and
            hasattr(model, 'net') and hasattr(model.net, 'auxiliary_input_size')):

            # Calculate auxiliary input size based on feature types
            auxiliary_input_size = 0
            for feature_type in cfg.data.auxiliary_features:
                if feature_type == "decay_time":
                    auxiliary_input_size += 1  # decay_time extracts 1 feature
                # Add other feature types as needed in the future

            if auxiliary_input_size > 0:
                old_size = getattr(model.net, 'auxiliary_input_size', 0)
                model.net.auxiliary_input_size = auxiliary_input_size
                log.info(f"Auto-configured auxiliary_input_size: {old_size} -> {auxiliary_input_size} (based on {cfg.data.auxiliary_features})")
    except Exception as e:
        log.warning(f"Failed to auto-configure model from dataset metadata: {e}")


def _configure_regression_criteria(model, data_dir: str, parameter_names: list) -> None:
    """Configure regression criteria with proper parameter ranges."""
    try:
        from src.models.losses import NormalizedRegressionLoss
        from src.utils.vimh_utils import load_vimh_metadata

        # Get parameter ranges from metadata
        from src.utils.vimh_utils import get_parameter_ranges_from_metadata
        param_ranges = get_parameter_ranges_from_metadata(data_dir)

        # Configure criteria for each parameter
        if not hasattr(model, 'criteria') or not model.criteria:
            model.criteria = {}

        for param_name in parameter_names:
            if param_name in param_ranges:
                param_range = param_ranges[param_name]
                model.criteria[param_name] = NormalizedRegressionLoss(param_range=param_range)
                log.info(f"Configured {param_name} regression loss with range: {param_range}")
            else:
                # Fallback if range not available - get from full metadata
                metadata = load_vimh_metadata(data_dir)
                if 'parameter_mappings' in metadata and param_name in metadata['parameter_mappings']:
                    param_info = metadata['parameter_mappings'][param_name]
                    if 'min' in param_info and 'max' in param_info:
                        param_range = (param_info['min'], param_info['max'])
                        model.criteria[param_name] = NormalizedRegressionLoss(param_range=param_range)
                        log.info(f"Configured {param_name} regression loss with range: {param_range}")
                    else:
                        log.warning(f"No min/max range found for {param_name}, using default (0,1)")
                        model.criteria[param_name] = NormalizedRegressionLoss(param_range=(0.0, 1.0))
                else:
                    # Fallback if no mapping available
                    log.warning(f"No parameter mapping found for {param_name}, using default (0,1)")
                    model.criteria[param_name] = NormalizedRegressionLoss(param_range=(0.0, 1.0))

        # Update multihead flag
        if hasattr(model, 'is_multihead'):
            model.is_multihead = len(model.criteria) > 1

    except Exception as e:
        log.error(f"Failed to configure regression criteria: {e}")
        raise


