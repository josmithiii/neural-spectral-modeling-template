"""
Audio Reconstruction Evaluation for Neural Spectral Modeling

This script evaluates trained models by comparing original and reconstructed audio:
1. Loads a trained checkpoint and test dataset
2. Runs inference to get predicted synthesis parameters
3. Synthesizes audio using both true and predicted parameters
4. Provides side-by-side comparisons with audio playback and visual analysis
5. Computes perceptual metrics for synthesis quality assessment

USAGE:
    # Basic usage - auto-discovers latest checkpoint
    python src/audio_reconstruction_eval.py

    # Evaluate specific checkpoint
    python src/audio_reconstruction_eval.py ckpt_path=path/to/checkpoint.ckpt

    # Evaluate reference checkpoint
    python src/audio_reconstruction_eval.py ckpt_path=checkpoints/reference/cnn_medium_dsaw_3p_baseline_v1.ckpt

    # Batch mode (non-interactive) evaluating 10 samples
    python src/audio_reconstruction_eval.py ckpt_path=checkpoints/reference/cnn_medium_dsaw_3p_baseline_v1.ckpt interactive=false num_samples=10

    # Evaluate without saving audio files
    python src/audio_reconstruction_eval.py ckpt_path=checkpoints/reference/convnext_medium_wah_avix_v1.ckpt save_audio=false

CONFIGURATION OPTIONS (set via command line: key=value):

    ckpt_path: null (default)
        Path to checkpoint file (.ckpt). If null, auto-discovers latest checkpoint from logs/train/
        Examples:
            ckpt_path=checkpoints/reference/cnn_medium_dsaw_3p_baseline_v1.ckpt
            ckpt_path=logs/train/runs/2025-10-26_12-00-00/checkpoints/best.ckpt

    num_samples: 5 (default)
        Number of test samples to evaluate. Set higher for batch evaluation.
        Examples: num_samples=1, num_samples=10, num_samples=100

    interactive: true (default)
        Launch interactive matplotlib widget for browsing samples.
        Set to false for batch processing mode with summary statistics.
        Examples: interactive=true, interactive=false

    save_audio: true (default)
        Save reconstructed audio files to disk (WAV format).
        Examples: save_audio=true, save_audio=false

    output_dir: "audio_eval_results" (default)
        Directory where audio files and figures are saved.
        Examples: output_dir=audio_eval_results, output_dir=results/eval_run1

    enable_playback: true (default)
        Enable audio playback in interactive mode (requires sounddevice or IPython).
        Examples: enable_playback=true, enable_playback=false

    compute_spectral_metrics: true (default)
        Compute frequency-domain metrics (spectrogram comparisons).

    compute_perceptual_metrics: true (default)
        Compute perceptual audio quality metrics (SNR, correlation, cross-correlation).

    data: vimh (default)
        Data module config to use (usually auto-configured from checkpoint).
        Override only if you need to evaluate on a different dataset.

INTERACTIVE MODE FEATURES:
    - Navigate samples: Left/Right arrow keys, Prev/Next buttons, or slider
    - Play audio: Click "Play True"/"Play Pred" buttons
    - Save outputs: Click "Save PNG", "Save True WAV", "Save Pred WAV" buttons
    - Copy info: Click "Copy Info" to copy evaluation details to clipboard
    - Keyboard shortcuts: s=save figure, t=save true WAV, p=save pred WAV

OUTPUT FILES:
    audio_eval_results/
        ├── ae_{run_name}_{timestamp}.png          # Evaluation figure
        ├── ae_{run_name}_sample-{idx}_true_{timestamp}.wav   # True audio
        └── ae_{run_name}_sample-{idx}_pred_{timestamp}.wav   # Predicted audio

METRICS COMPUTED:
    - MSE: Mean squared error between waveforms
    - SNR: Signal-to-noise ratio (dB)
    - Correlation: Pearson correlation coefficient
    - Max XCorr: Maximum cross-correlation (time-aligned similarity)
    - Max XCorr Lag: Time alignment offset in samples/milliseconds
    - MSS Distance: Multi-Scale Spectral distance (PNP-compatible perceptual metric)

EXAMPLE WORKFLOWS:

    1. Quick evaluation of latest trained model:
       python src/audio_reconstruction_eval.py

    2. Evaluate reference baseline on 3-parameter detuning task:
       python src/audio_reconstruction_eval.py ckpt_path=checkpoints/reference/cnn_medium_dsaw_3p_baseline_v1.ckpt

    3. Batch evaluation of 100 samples (non-interactive):
       python src/audio_reconstruction_eval.py ckpt_path=checkpoints/reference/cnn_medium_dsaw_3p_baseline_v1.ckpt interactive=false num_samples=100

    4. Evaluate wah model with auxiliary features:
       python src/audio_reconstruction_eval.py ckpt_path=checkpoints/reference/convnext_medium_wah_avix_aux_v1.ckpt

NOTES:
    - The script automatically uses the exact dataset that was used during training
    - Dataset directory and all hyperparameters are loaded from the checkpoint
    - For Mac users: Audio playback uses afplay by default (most reliable in CLI)
    - Evaluation requires the training dataset to exist at its original location
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch
import torch.nn.functional as F
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from matplotlib.widgets import Button, CheckButtons, TextBox
from omegaconf import DictConfig, OmegaConf

# Register custom resolver to flatten path separators in experiment names
# Must match the resolver in train.py for hydra config compatibility
OmegaConf.register_new_resolver("flatten", lambda x: x.replace("/", "_") if x else x)

try:
    import IPython.display as ipd

    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False
    ipd = None
from scipy.signal import correlate
from scipy.stats import pearsonr

# Setup root path
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from src.utils.display_utils import (
    get_metric_color,
    plot_audio_metrics,
    plot_parameter_comparison,
    prepare_spectrogram_for_display,
)
from src.utils.synth_utils import SimpleSawSynth, PercussionSynth, SpectrogramProcessor, check_params

log = RankedLogger(__name__, rank_zero_only=True)


# Custom exceptions for better error handling
class CheckpointError(Exception):
    """Base exception for checkpoint-related errors."""

    pass


class ArchitectureInferenceError(CheckpointError):
    """Raised when model architecture cannot be inferred from checkpoint."""

    pass


class MissingCheckpointDataError(CheckpointError):
    """Raised when required data is missing from checkpoint."""

    pass


@dataclass
class ArchitectureConfig:
    """Configuration for reconstructed model architecture."""

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
    # ConvNeXt-specific parameters
    dims: Optional[Tuple[int, ...]] = None
    depths: Optional[Tuple[int, ...]] = None
    kernel_size: int = 3
    drop_path_rate: float = 0.1
    head_init_scale: float = 1.0
    in_chans: int = 1


class CheckpointArchitectureReconstructor:
    """Centralized class for reconstructing model architecture from checkpoints."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize reconstructor with configuration.

        Args:
            config: Configuration dictionary with default values and settings
        """
        self.config = config or {}
        self.default_dropout = self.config.get("default_dropout", 0.5)
        self.default_patch_size = self.config.get("default_patch_size", 4)
        self.default_embed_dim = self.config.get("default_embed_dim", 64)

    def reconstruct_from_checkpoint(
        self,
        checkpoint_path: str,
        heads_config: Dict[str, int],
        dataset_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.nn.Module, ArchitectureConfig]:
        """Reconstruct model architecture from checkpoint with stored metadata or inference.

        Args:
            checkpoint_path: Path to the checkpoint file
            heads_config: Configuration of model heads (parameter names -> num_classes)
            dataset_metadata: Optional dataset metadata for architecture inference

        Returns:
            Tuple of (reconstructed_network, architecture_config)

        Raises:
            ArchitectureInferenceError: When architecture cannot be determined
            MissingCheckpointDataError: When required checkpoint data is missing
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint {checkpoint_path}: {e}")

        hyper_parameters = checkpoint.get("hyper_parameters", {})
        architecture_metadata = hyper_parameters.get("architecture_metadata", {})
        output_mode = hyper_parameters.get("output_mode", "classification")
        parameter_names = None
        if dataset_metadata and "parameter_names" in dataset_metadata:
            parameter_names = list(dataset_metadata.get("parameter_names", []))

        # Try stored metadata first (preferred method)
        if architecture_metadata and architecture_metadata.get("type"):
            return self._reconstruct_from_metadata(
                architecture_metadata,
                heads_config,
                output_mode=output_mode,
                parameter_names=parameter_names,
            )

        # Fallback to inference from weights
        log.warning(
            "No architecture metadata found in checkpoint, attempting inference from weights"
        )
        return self._reconstruct_from_weights(
            checkpoint["state_dict"],
            heads_config,
            dataset_metadata,
            output_mode=output_mode,
            parameter_names=parameter_names,
        )

    def _reconstruct_from_metadata(
        self,
        metadata: Dict[str, Any],
        heads_config: Dict[str, int],
        output_mode: str = "classification",
        parameter_names: Optional[List[str]] = None,
    ) -> Tuple[torch.nn.Module, ArchitectureConfig]:
        """Reconstruct model using stored architecture metadata."""
        arch_type = metadata["type"]

        if arch_type == "ViT":
            return self._create_vit_from_metadata(metadata, heads_config, output_mode=output_mode)
        elif arch_type == "CNN":
            return self._create_cnn_from_metadata(
                metadata,
                heads_config,
                output_mode=output_mode,
                parameter_names=parameter_names,
            )
        else:
            raise ArchitectureInferenceError(f"Unknown architecture type in metadata: {arch_type}")

    def _create_vit_from_metadata(
        self,
        metadata: Dict[str, Any],
        heads_config: Dict[str, int],
        output_mode: str = "classification",
    ) -> Tuple[torch.nn.Module, ArchitectureConfig]:
        """Create ViT model from stored metadata."""
        from src.models.components.vision_transformer import VisionTransformer

        config = ArchitectureConfig(
            architecture_type="ViT",
            input_channels=metadata.get("input_channels", 1),
            embed_dim=metadata.get("embed_dim", self.default_embed_dim),
            n_layers=metadata.get("n_layers", 6),
            n_attention_heads=metadata.get("n_attention_heads", 4),
            patch_size=metadata.get("patch_size", self.default_patch_size),
            image_size=tuple(metadata.get("image_size", [28, 28])),
            forward_mul=int(metadata.get("forward_mul", 2)),
            dropout=metadata.get("dropout", 0.1),
            use_torch_layers=metadata.get("use_torch_layers", False),
        )

        net = VisionTransformer(
            image_size=config.image_size,
            patch_size=config.patch_size,
            n_channels=config.input_channels,
            embed_dim=config.embed_dim,
            n_layers=config.n_layers,
            n_attention_heads=config.n_attention_heads,
            heads_config=heads_config,
            forward_mul=config.forward_mul,
            dropout=config.dropout,
            use_torch_layers=config.use_torch_layers,
        )

        return net, config

    def _create_cnn_from_metadata(
        self,
        metadata: Dict[str, Any],
        heads_config: Dict[str, int],
        output_mode: str = "classification",
        parameter_names: Optional[List[str]] = None,
    ) -> Tuple[torch.nn.Module, ArchitectureConfig]:
        """Create CNN model from stored metadata."""
        from src.models.components.simple_cnn import SimpleCNN

        config = ArchitectureConfig(
            architecture_type="CNN",
            input_channels=metadata.get("input_channels", 1),
            conv1_channels=metadata.get("conv1_channels", 16),
            conv2_channels=metadata.get("conv2_channels", 32),
            fc_hidden=metadata.get("fc_hidden", 64),
            dropout=metadata.get("dropout", self.default_dropout),
            input_size=metadata.get("input_size", 32),
        )

        net = SimpleCNN(
            input_channels=config.input_channels,
            conv1_channels=config.conv1_channels,
            conv2_channels=config.conv2_channels,
            fc_hidden=config.fc_hidden,
            heads_config=heads_config,
            dropout=config.dropout,
            input_size=config.input_size,
            output_mode=output_mode,
            parameter_names=parameter_names if output_mode == "regression" else None,
        )

        return net, config

    def _reconstruct_from_weights(
        self,
        state_dict: Dict[str, torch.Tensor],
        heads_config: Dict[str, int],
        dataset_metadata: Optional[Dict[str, Any]] = None,
        output_mode: str = "classification",
        parameter_names: Optional[List[str]] = None,
    ) -> Tuple[torch.nn.Module, ArchitectureConfig]:
        """Reconstruct model by inferring architecture from weights (fallback method)."""
        if "net.embedding.pos_embedding" in state_dict:
            return self._infer_vit_architecture(
                state_dict, heads_config, dataset_metadata, output_mode, parameter_names
            )
        elif any(key.startswith("net.backbone.features") for key in state_dict.keys()):
            return self._infer_effnet_architecture(
                state_dict, heads_config, dataset_metadata, output_mode, parameter_names
            )
        elif any(key.startswith("net.downsample_layers") for key in state_dict.keys()):
            return self._infer_convnext_architecture(
                state_dict, heads_config, dataset_metadata, output_mode, parameter_names
            )
        elif any(
            key.startswith("net.conv_layers.0.weight")
            or key.startswith("net.features.0.weight")
            or key.startswith("net.conv1.weight")
            or key.startswith("encoder.feature_extractor")
            for key in state_dict.keys()
        ):
            return self._infer_cnn_architecture(
                state_dict, heads_config, dataset_metadata, output_mode, parameter_names
            )
        elif "net.feature_extractor.0.weight" in state_dict:
            # Check if it's MLP (2D weights) or CNN (4D weights)
            first_weight = state_dict["net.feature_extractor.0.weight"]
            if len(first_weight.shape) == 2:
                return self._infer_mlp_architecture(
                    state_dict, heads_config, dataset_metadata, output_mode, parameter_names
                )
            else:
                return self._infer_cnn_architecture(
                    state_dict, heads_config, dataset_metadata, output_mode, parameter_names
                )
        else:
            raise ArchitectureInferenceError(
                "Cannot determine model architecture from checkpoint weights. "
                "Expected ViT (net.embedding.pos_embedding), EfficientNet (net.backbone.features), "
                "ConvNeXt (net.downsample_layers), "
                "CNN (net.conv_layers/net.features/net.conv1/encoder.feature_extractor), "
                "or MLP (net.feature_extractor with 2D weights)."
            )

    def _infer_vit_architecture(
        self,
        state_dict: Dict[str, torch.Tensor],
        heads_config: Dict[str, int],
        dataset_metadata: Optional[Dict[str, Any]] = None,
        output_mode: str = "classification",
        parameter_names: Optional[List[str]] = None,
    ) -> Tuple[torch.nn.Module, ArchitectureConfig]:
        """Infer ViT architecture from weights (not recommended - use metadata instead)."""
        log.error("ViT architecture inference from weights is unreliable and not supported.")
        log.error("Please ensure your checkpoint contains architecture metadata.")
        raise ArchitectureInferenceError(
            "Cannot reliably infer ViT architecture from checkpoint weights alone. "
            "ViT architecture must be stored in checkpoint metadata."
        )

    def _infer_cnn_architecture(
        self,
        state_dict: Dict[str, torch.Tensor],
        heads_config: Dict[str, int],
        dataset_metadata: Optional[Dict[str, Any]] = None,
        output_mode: str = "classification",
        parameter_names: Optional[List[str]] = None,
    ) -> Tuple[torch.nn.Module, ArchitectureConfig]:
        """Infer CNN architecture from weights."""
        if not dataset_metadata:
            raise MissingCheckpointDataError(
                "CNN architecture inference requires dataset metadata"
            )

        required_keys = ["height", "width"]
        if not all(key in dataset_metadata for key in required_keys):
            raise MissingCheckpointDataError(
                f"CNN inference requires dataset metadata with: {required_keys}"
            )

        # Extract CNN parameters with validation
        cnn_params = self._extract_cnn_parameters(state_dict, dataset_metadata)

        from src.models.components.simple_cnn import SimpleCNN

        net = SimpleCNN(
            input_channels=cnn_params["input_channels"],
            conv1_channels=cnn_params["conv1_channels"],
            conv2_channels=cnn_params["conv2_channels"],
            fc_hidden=cnn_params["fc_hidden"],
            heads_config=heads_config,
            dropout=self.default_dropout,
            input_size=cnn_params["input_size"],
            output_mode=output_mode,
            parameter_names=parameter_names if output_mode == "regression" else None,
        )

        config = ArchitectureConfig(
            architecture_type="CNN", **cnn_params, dropout=self.default_dropout
        )

        return net, config

    def _extract_cnn_parameters(
        self, state_dict: Dict[str, torch.Tensor], dataset_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract CNN parameters from state dict with validation."""
        params = {}

        # Validate required weights exist
        required_weights = ["net.conv_layers.0.weight", "net.conv_layers.4.weight"]
        missing_weights = [key for key in required_weights if key not in state_dict]
        if missing_weights:
            raise MissingCheckpointDataError(
                f"CNN checkpoint missing required weights: {missing_weights}"
            )

        # Extract conv1 parameters
        conv1_weight = state_dict["net.conv_layers.0.weight"]
        params["input_channels"] = conv1_weight.shape[1]  # Input channels
        params["conv1_channels"] = conv1_weight.shape[0]  # Output channels

        # Extract conv2 parameters
        conv2_weight = state_dict["net.conv_layers.4.weight"]
        params["conv2_channels"] = conv2_weight.shape[0]

        # Extract FC hidden size
        fc_candidates = ["net.shared_features.1.weight", "net.shared_features.0.weight"]
        fc_weight = None
        for candidate in fc_candidates:
            if candidate in state_dict:
                fc_weight = state_dict[candidate]
                break

        if fc_weight is None:
            raise MissingCheckpointDataError(f"CNN checkpoint missing FC weights: {fc_candidates}")

        params["fc_hidden"] = fc_weight.shape[0]

        # Get input size from dataset
        params["input_size"] = max(dataset_metadata["height"], dataset_metadata["width"])

        return params

    def _infer_convnext_architecture(
        self,
        state_dict: Dict[str, torch.Tensor],
        heads_config: Dict[str, int],
        dataset_metadata: Optional[Dict[str, Any]] = None,
        output_mode: str = "classification",
        parameter_names: Optional[List[str]] = None,
    ) -> Tuple[torch.nn.Module, ArchitectureConfig]:
        """Infer ConvNeXt architecture from weights."""
        if not dataset_metadata:
            raise MissingCheckpointDataError(
                "ConvNeXt architecture inference requires dataset metadata"
            )

        required_keys = ["height", "width"]
        if not all(key in dataset_metadata for key in required_keys):
            raise MissingCheckpointDataError(
                f"ConvNeXt inference requires dataset metadata with: {required_keys}"
            )

        # Extract ConvNeXt parameters
        convnext_params = self._extract_convnext_parameters(state_dict, dataset_metadata)

        from src.models.components.convnext_v2 import ConvNeXtV2

        net = ConvNeXtV2(
            input_size=convnext_params["input_size"],
            in_chans=convnext_params["in_chans"],
            heads_config=heads_config,
            depths=convnext_params["depths"],
            dims=convnext_params["dims"],
            drop_path_rate=convnext_params.get("drop_path_rate", 0.1),
            head_init_scale=convnext_params.get("head_init_scale", 1.0),
            kernel_size=convnext_params.get("kernel_size", 3),
            output_mode=output_mode,
            parameter_names=parameter_names if output_mode == "regression" else None,
        )

        config = ArchitectureConfig(
            architecture_type="ConvNeXt",
            **convnext_params,
        )

        return net, config

    def _extract_convnext_parameters(
        self, state_dict: Dict[str, torch.Tensor], dataset_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract ConvNeXt parameters from state dict."""
        params = {}

        # Extract dims from downsample_layers
        dims = []
        for stage_idx in range(4):
            # Look for the LayerNorm weight in each downsample layer
            key = f"net.downsample_layers.{stage_idx}.1.weight"
            if key in state_dict:
                dims.append(state_dict[key].shape[0])
            else:
                # Fallback: try to get from stage weights
                stage_key = f"net.downsample_layers.{stage_idx}.0.weight"
                if stage_key in state_dict:
                    dims.append(state_dict[stage_key].shape[0])

        if len(dims) != 4:
            raise MissingCheckpointDataError(
                f"ConvNeXt checkpoint missing downsample_layers (found {len(dims)}/4 stages)"
            )

        params["dims"] = tuple(dims)

        # Extract depths by counting blocks in each stage
        depths = []
        for stage_idx in range(4):
            stage_blocks = set()
            for key in state_dict.keys():
                if key.startswith(f"net.stages.{stage_idx}."):
                    parts = key.split(".")
                    if len(parts) > 3 and parts[3].isdigit():
                        stage_blocks.add(int(parts[3]))
            depths.append(len(stage_blocks) if stage_blocks else 0)

        if sum(depths) == 0:
            raise MissingCheckpointDataError("ConvNeXt checkpoint missing stage blocks")

        params["depths"] = tuple(depths)

        # Extract input channels from first downsample layer
        first_conv_key = "net.downsample_layers.0.0.weight"
        if first_conv_key in state_dict:
            params["in_chans"] = state_dict[first_conv_key].shape[1]
        else:
            params["in_chans"] = 1  # Default for spectrograms

        # Get input size from dataset
        params["input_size"] = max(dataset_metadata["height"], dataset_metadata["width"])

        # Default values for other parameters
        params["drop_path_rate"] = 0.1
        params["head_init_scale"] = 1.0
        params["kernel_size"] = 3

        return params

    def _infer_effnet_architecture(
        self,
        state_dict: Dict[str, torch.Tensor],
        heads_config: Dict[str, int],
        dataset_metadata: Optional[Dict[str, Any]] = None,
        output_mode: str = "classification",
        parameter_names: Optional[List[str]] = None,
    ) -> Tuple[torch.nn.Module, ArchitectureConfig]:
        """Infer EfficientNet architecture from weights."""
        if not dataset_metadata:
            raise MissingCheckpointDataError(
                "EfficientNet architecture inference requires dataset metadata"
            )

        required_keys = ["height", "width"]
        if not all(key in dataset_metadata for key in required_keys):
            raise MissingCheckpointDataError(
                f"EfficientNet inference requires dataset metadata with: {required_keys}"
            )

        # Extract EfficientNet parameters
        effnet_params = self._extract_effnet_parameters(state_dict, dataset_metadata)

        from src.models.components.effnet import EffNet

        net = EffNet(
            input_channels=effnet_params["input_channels"],
            eff_type=effnet_params["eff_type"],
            heads_config=heads_config,
            output_mode=output_mode,
            parameter_names=parameter_names if output_mode == "regression" else None,
            pretrained=False,  # Loading from checkpoint, not ImageNet
        )

        config = ArchitectureConfig(
            architecture_type="EfficientNet",
            input_channels=effnet_params["input_channels"],
        )

        return net, config

    def _extract_effnet_parameters(
        self, state_dict: Dict[str, torch.Tensor], dataset_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract EfficientNet parameters from state dict."""
        params = {}

        # Extract input channels from input_conv or backbone first layer
        input_conv_key = "net.input_conv.weight"
        backbone_first_key = "net.backbone.features.0.0.weight"

        if input_conv_key in state_dict:
            params["input_channels"] = state_dict[input_conv_key].shape[1]
        elif backbone_first_key in state_dict:
            params["input_channels"] = state_dict[backbone_first_key].shape[1]
        else:
            params["input_channels"] = 1  # Default for spectrograms

        # Infer EfficientNet type from backbone structure
        # Check number of features in the last backbone layer
        last_conv_key = "net.backbone.features.8.0.weight"
        if last_conv_key in state_dict:
            last_conv_channels = state_dict[last_conv_key].shape[0]
            # EfficientNet-B0 has 1280 channels in final conv
            # This is a heuristic - different variants have different channel counts
            if last_conv_channels == 1280:
                params["eff_type"] = "b0"
            else:
                # Default to b0 if we can't determine
                log.warning(f"Could not determine exact EfficientNet variant from weights (last_conv_channels={last_conv_channels}), defaulting to b0")
                params["eff_type"] = "b0"
        else:
            log.warning("Could not find backbone final layer, defaulting to EfficientNet-B0")
            params["eff_type"] = "b0"

        return params


def find_latest_checkpoint(base_dir: str = "logs/train") -> Optional[str]:
    """
    Find the latest best checkpoint from training runs.

    Args:
        base_dir: Base directory to search for training logs

    Returns:
        Path to the latest best checkpoint, or None if not found
    """
    import glob
    from pathlib import Path

    # Look for checkpoint patterns in training logs
    patterns = [
        f"{base_dir}/runs/*/checkpoints/best.ckpt",
        f"{base_dir}/runs/*/checkpoints/last.ckpt",
        f"{base_dir}/runs/*/checkpoints/*.ckpt",
        f"{base_dir}/*/checkpoints/best.ckpt",
        f"{base_dir}/*/checkpoints/last.ckpt",
        f"{base_dir}/*/checkpoints/*.ckpt",
    ]

    latest_checkpoint = None
    latest_time = 0

    for pattern in patterns:
        checkpoints = glob.glob(pattern)
        for ckpt_path in checkpoints:
            try:
                # Get modification time
                mtime = os.path.getmtime(ckpt_path)
                if mtime > latest_time:
                    latest_time = mtime
                    latest_checkpoint = ckpt_path
            except OSError:
                continue

    if latest_checkpoint:
        log.info(f"Auto-discovered checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    else:
        log.warning(f"No checkpoints found in {base_dir}")
        return None


class AudioReconstructionEvaluator:
    """Evaluates model predictions by reconstructing and comparing audio."""

    def __init__(
        self,
        model: LightningModule,
        datamodule: LightningDataModule,
        device: str = "cpu",
        ckpt_path: Optional[str] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            model: Trained PyTorch Lightning model
            datamodule: Data module with test dataset
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.model.eval()
        self.datamodule = datamodule
        self.device = device
        self.ckpt_path = ckpt_path
        try:
            from pathlib import Path as _P

            if ckpt_path:
                cp = _P(ckpt_path)
                # logs/train/runs/<run>/checkpoints/<file>.ckpt → run directory two levels up
                self.run_dir = cp.parent.parent
                self.run_name = self.run_dir.name
            else:
                self.run_dir = None
                self.run_name = ""
        except Exception:
            self.run_dir = None
            self.run_name = ""

        # Setup both train and test datasets
        self.datamodule.setup("fit")  # Sets up train dataset
        self.train_dataset = self.datamodule.data_train
        self.datamodule.setup("test")  # Sets up test dataset
        self.test_dataset = self.datamodule.data_test

        # Get dataset metadata from datamodule (includes saved checkpoint metadata)
        self.dataset_info = self.datamodule.get_dataset_info()
        self.sample_rate = self.dataset_info.get("sample_rate", 8000)
        self.duration = self.dataset_info.get("duration", 1.0)
        self.dataset_name = self.dataset_info.get("dataset_name", "")

        # Extract training steps and experiment name from checkpoint if available
        self.training_steps = None
        self.total_epochs = None
        self.experiment_name = None
        if ckpt_path:
            try:
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                # Try different keys that might contain step/epoch information
                self.training_steps = (
                    checkpoint.get("global_step") or
                    checkpoint.get("step") or
                    checkpoint.get("trainer", {}).get("global_step")
                )
                self.total_epochs = (
                    checkpoint.get("epoch") or
                    checkpoint.get("trainer", {}).get("current_epoch")
                )
                # Extract experiment name from hyperparameters
                hyper_params = checkpoint.get("hyper_parameters", {})
                self.experiment_name = hyper_params.get("experiment_name", None)
                if self.experiment_name:
                    log.info(f"Loaded experiment name from checkpoint: {self.experiment_name}")
            except Exception as e:
                log.warning(f"Could not extract training metadata from checkpoint: {e}")

        # Initialize synthesizer - read synth_type from metadata
        synth_type_metadata = self.dataset_info.get("synth_type", None)

        # Try checkpoint metadata if not in dataset
        if not synth_type_metadata and ckpt_path:
            try:
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                arch_metadata = checkpoint.get("hyper_parameters", {}).get("architecture_metadata", {})
                synth_type_metadata = arch_metadata.get("synth_type", None)
                if synth_type_metadata:
                    log.info(f"Read synth_type from checkpoint: {synth_type_metadata}")
            except Exception as e:
                log.warning(f"Could not read synth_type from checkpoint: {e}")

        if not synth_type_metadata:
            log.error("Dataset missing synth_type in metadata; aborting.")
            sys.exit(1)

        # Map synth_type to synthesizer class
        if synth_type_metadata in ("percussion", "perc"):
            self.synth = PercussionSynth(sample_rate=self.sample_rate)
            self.synth_type = 'percussion'
            log.info(f"Using PercussionSynth (synth_type={synth_type_metadata})")
        else:  # "simple", "saw", or any other type defaults to SimpleSawSynth
            self.synth = SimpleSawSynth(sample_rate=self.sample_rate)
            self.synth_type = 'saw'
            log.info(f"Using SimpleSawSynth (synth_type={synth_type_metadata})")

        # Initialize spectrogram processor for visualization
        stft_config = self.dataset_info.get("spectrogram_config", {"type": "stft"})
        mel_config = self.dataset_info.get("mel_config", {})
        height = self.dataset_info.get("height", 32)
        width = self.dataset_info.get("width", 32)

        # Ensure stft_config has required fields
        if "type" not in stft_config:
            stft_config["type"] = "stft"

        self.spec_processor = SpectrogramProcessor(
            sample_rate=self.sample_rate,
            height=height,
            width=width,
            stft_config=stft_config,
            mel_config=mel_config,
        )

        # Initialize MSS loss for perceptual audio evaluation (PNP-compatible)
        try:
            import auraloss.freq
            self.mss_loss = auraloss.freq.MultiResolutionSTFTLoss()
            self.mss_implementation = "auraloss"
            log.info("Initialized auraloss MultiResolutionSTFTLoss (PNP-compatible configuration)")
            log.info("  FFT sizes: [1024, 2048, 512]")
            log.info("  Hop sizes: [120, 240, 50]")
            log.info("  Window lengths: [600, 1200, 240]")
            log.info("  Loss components: spectral convergence + log magnitude")
        except ImportError:
            log.warning("auraloss not installed, falling back to custom MSS implementation")
            log.warning("NOTE: Results will NOT be directly comparable to PNP paper")
            from src.models.losses import MultiScaleSpectralLoss
            self.mss_loss = MultiScaleSpectralLoss(
                max_n_fft=2048,
                num_scales=6,
                p=1.0
            )
            self.mss_implementation = "custom"
            log.info("Initialized custom MultiScaleSpectralLoss for audio evaluation")

        # Debug: Print spectrogram processor configuration (can be removed)
        # log.info(f"SpectrogramProcessor config:")
        # log.info(f"  Sample rate: {self.sample_rate} Hz")
        # log.info(f"  Dimensions: {height}x{width}")
        # log.info(f"  STFT config: {stft_config}")
        # log.info(f"  MEL config: {mel_config}")

        # Parameter mappings for denormalization (no fallbacks)
        self.param_mappings = self.dataset_info.get("parameter_mappings")
        dataset_param_names = self.dataset_info.get("parameter_names")
        if not self.param_mappings or not dataset_param_names:
            print("Dataset is missing parameter_mappings or parameter_names; aborting.")
            sys.exit(1)

        # Use model's heads to determine which parameters to evaluate
        # This handles cases where some parameters are auxiliary inputs, not prediction targets
        if not (hasattr(self.model, "net") and hasattr(self.model.net, "heads_config")):
            log.error("Model missing net.heads_config - cannot determine prediction targets")
            sys.exit(1)

        self.param_names = list(self.model.net.heads_config.keys())
        log.info(f"Model heads config: {self.model.net.heads_config}")

        log.info(f"Initialized evaluator with {len(self.test_dataset)} test samples")
        log.info(f"Sample rate: {self.sample_rate} Hz, Duration: {self.duration}s")
        log.info(f"Available parameters (dataset): {dataset_param_names}")
        log.info(f"Evaluating parameters (model outputs): {self.param_names}")

        # Debug dataset metadata
        log.info(f"Dataset info keys: {list(self.dataset_info.keys())}")
        if self.param_mappings:
            log.info(f"Parameter mappings: {list(self.param_mappings.keys())}")

        # Check if model output mode matches expectations
        if hasattr(self.model, "output_mode"):
            log.info(f"Model output mode: {self.model.output_mode}")

    def check_prediction_diversity(self, n_samples: int = 20) -> Dict[str, Any]:
        """Run a quick pass over the first N test samples and summarize
        prediction diversity for each head/parameter.

        Returns a dict with per-parameter stats and prints a concise report.
        Does not exit; this is a diagnostic for model quality, not config.
        """
        n = min(n_samples, len(self.test_dataset))
        if n == 0:
            print("No test samples available for diversity check.")
            return {}

        # Storage per parameter
        values: Dict[str, List[float]] = {name: [] for name in self.param_names}

        with torch.no_grad():
            for i in range(n):
                sample = self.test_dataset[i]
                image_tensor = sample[0] if isinstance(sample, (list, tuple)) else sample
                image_batch = image_tensor.unsqueeze(0).to(self.device)
                preds = self.model(image_batch)

                if not isinstance(preds, dict):
                    print("Diversity check expects multihead dict outputs; aborting.")
                    sys.exit(1)

                # Enforce head-name alignment
                for head in preds.keys():
                    if head not in self.param_names:
                        print(f"Unexpected model head '{head}' during diversity check; aborting.")
                        sys.exit(1)

                for head_name, logits in preds.items():
                    if logits.numel() > 1:  # Multiple logits = classification
                        v = int(torch.argmax(logits, dim=-1).item())
                    else:  # Single logit = regression
                        v = float(logits.squeeze().item())
                    values[head_name].append(v)

        # Compute simple stats
        report: Dict[str, Any] = {}
        print("\nPrediction diversity over", n, "sample(s):")
        for name in self.param_names:
            vals = values.get(name, [])
            if vals and isinstance(vals[0], float):  # Regression values
                std = float(np.std(vals)) if vals else 0.0
                report[name] = {
                    "std": std,
                    "min": float(min(vals)) if vals else None,
                    "max": float(max(vals)) if vals else None,
                }
                status = "⚠️ low variance" if std < 1e-3 else "ok"
                print(
                    f"  - {name}: std={std:.6f} [{status}] range=({report[name]['min']}, {report[name]['max']})"
                )
            else:  # Classification values
                uniques = sorted(list(set(vals))) if vals else []
                report[name] = {"unique": len(uniques), "values": uniques[:15]}
                status = "⚠️ degenerate" if len(uniques) <= 1 else "ok"
                preview = ", ".join(map(str, report[name]["values"])) + (
                    " …" if len(uniques) > 15 else ""
                )
                print(f"  - {name}: {len(uniques)} unique [{status}] → [{preview}]")

        return report

    def denormalize_parameters(
        self, predicted_params: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Denormalize predicted parameters from [0,1] or class indices to actual parameter values.

        Args:
            predicted_params: Dictionary of predicted parameter tensors

        Returns:
            Dictionary of denormalized parameter values
        """
        denorm_params = {}

        for param_name, pred_tensor in predicted_params.items():
            if param_name not in self.param_mappings:
                print(f"Parameter '{param_name}' missing from parameter_mappings; aborting.")
                sys.exit(1)

            mapping = self.param_mappings[param_name]
            if "min" not in mapping or "max" not in mapping:
                print(f"Parameter mapping for '{param_name}' missing 'min'/'max'; aborting.")
                sys.exit(1)
            param_min = mapping["min"]
            param_max = mapping["max"]

            # Handle different prediction formats
            # Determine if this was a classification or regression output
            is_regression = (
                isinstance(pred_tensor, torch.Tensor)
                and pred_tensor.numel() == 1
                and pred_tensor.dtype.is_floating_point
            )

            if is_regression:
                # Direct regression output – models emit sigmoid-activated [0,1]
                # Be robust to tiny numerical drift and clamp to [0,1].
                try:
                    normalized_value = float(pred_tensor.item())
                except Exception:
                    normalized_value = float(pred_tensor)
                # Safety clamp
                if normalized_value < 0.0:
                    normalized_value = 0.0
                elif normalized_value > 1.0:
                    normalized_value = 1.0
            else:
                # Classification - convert class index to normalized value
                if "num_classes" in mapping:
                    num_classes = mapping["num_classes"]
                else:
                    # Allow deriving num_classes from min/max/step if present
                    if "step" in mapping and mapping["step"] not in (None, 0):
                        step = float(mapping["step"])
                        num_floats = (param_max - param_min) / step
                        # Round to nearest int within tolerance
                        num_steps = int(round(num_floats))
                        if abs(num_floats - num_steps) > 1e-3:
                            print(
                                f"Warning: parameter '{param_name}' (max-min)/step = {num_floats} not integer; rounding to {num_steps}"
                            )
                        num_classes = int(num_steps) + 1
                        mapping["num_classes"] = num_classes  # cache for later
                    else:
                        print(
                            f"Parameter mapping for '{param_name}' missing 'num_classes' and 'step'; aborting."
                        )
                        sys.exit(1)
                # Allow plain numbers or 0-D tensors
                try:
                    class_idx = int(pred_tensor.item())
                except Exception:
                    class_idx = int(pred_tensor)
                normalized_value = class_idx / (num_classes - 1)

            # Denormalize to actual parameter range
            actual_value = param_min + normalized_value * (param_max - param_min)
            denorm_params[param_name] = actual_value

        return denorm_params

    def get_true_parameters(self, sample_idx: int, dataset=None) -> Dict[str, float]:
        """
        Get the true synthesis parameters for a dataset sample.

        Args:
            sample_idx: Index of the sample in the dataset
            dataset: Dataset to use (defaults to test_dataset if None)

        Returns:
            Dictionary of true parameter values
        """
        # Use provided dataset or default to test dataset
        if dataset is None:
            dataset = self.test_dataset

        # Get metadata for this sample
        sample_metadata = dataset._get_sample_metadata(sample_idx)

        # Debug: Print metadata structure for first few samples
        if sample_idx < 3:
            print(f"🔍 Sample {sample_idx} metadata keys: {list(sample_metadata.keys())}")
            for key, value in sample_metadata.items():
                if isinstance(value, dict) and "actual_value" in value:
                    print(f"   {key}: actual_value = {value['actual_value']}")

        # Extract actual parameter values
        true_params = {}
        for param_name in self.param_names:
            param_info_key = f"{param_name}_info"
            if param_info_key in sample_metadata:
                actual_value = sample_metadata[param_info_key]["actual_value"]
                true_params[param_name] = actual_value
            else:
                raise ValueError(
                    f"Could not find true value for parameter {param_name} in sample metadata"
                )

        return true_params

    def run_inference(
        self, sample_idx: int, dataset=None
    ) -> Tuple[Dict[str, float], Dict[str, float], torch.Tensor]:
        """
        Run model inference on a dataset sample.

        Args:
            sample_idx: Index of the sample to evaluate
            dataset: Dataset to use (defaults to test_dataset if None)

        Returns:
            Tuple of (predicted_params, true_params, input_spectrogram)
        """
        # Use provided dataset or default to test dataset
        if dataset is None:
            dataset = self.test_dataset

        # Check for model performance issues on first few samples
        if sample_idx < 3 and hasattr(self, "_check_model_predictions"):
            self._check_model_predictions = False  # Only show once
            print("🔍 Checking model prediction quality...")

        # Track predictions to detect if model always predicts the same values
        if not hasattr(self, "_prediction_tracker"):
            self._prediction_tracker = {}
            self._samples_checked = 0
        # Get the sample from specified dataset
        sample_data = dataset[sample_idx]
        if len(sample_data) == 2:
            image_tensor, labels_dict = sample_data
        else:
            image_tensor, labels_dict = sample_data[0], sample_data[1]

        # Debug: Print image tensor info for first few samples (can be removed)
        # if sample_idx < 2:
        #     print(f"🔍 Raw dataset sample {sample_idx}:")
        #     print(f"   image_tensor shape: {image_tensor.shape}")
        #     print(f"   image_tensor range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
        #     print(f"   image_tensor type: {type(image_tensor)}")
        #     if hasattr(image_tensor, 'dtype'):
        #         print(f"   image_tensor dtype: {image_tensor.dtype}")

        # Run inference
        with torch.no_grad():
            image_batch = image_tensor.unsqueeze(0).to(self.device)
            raw_predictions = self.model(image_batch)

            # Debug: Print raw predictions to see if they're changing (can be removed)
            # if isinstance(raw_predictions, dict):
            #     for head_name, logits in raw_predictions.items():
            #         print(f"   🔍 Sample {sample_idx}, {head_name}: raw_logits shape={logits.shape}, "
            #              f"first few values={logits.flatten()[:3].tolist()}")
            # else:
            #     print(f"   🔍 Sample {sample_idx}: raw_predictions shape={raw_predictions.shape}, "
            #          f"first few values={raw_predictions.flatten()[:3].tolist()}")

            # Process predictions based on model type
            if not isinstance(raw_predictions, dict):
                print("Model is expected to be multihead and return a dict of logits; aborting.")
                sys.exit(1)

            # Enforce that model heads match dataset parameter names
            model_heads = list(raw_predictions.keys())
            missing = [p for p in self.param_names if p not in model_heads]
            extra = [h for h in model_heads if h not in self.param_names]
            if missing or extra:
                print(f"Head/parameter name mismatch. Missing: {missing}, Extra: {extra}")
                sys.exit(1)

            # Process predictions based on actual model head configuration
            # Check if heads have multiple outputs (classification) or single output (regression)
            processed_predictions = {}
            for head_name, logits in raw_predictions.items():
                if logits.numel() > 1:  # Multiple logits = classification
                    processed_predictions[head_name] = torch.argmax(logits, dim=-1).squeeze(0)
                else:  # Single logit = regression
                    processed_predictions[head_name] = logits.squeeze(0)

        # Denormalize predictions
        predicted_params = self.denormalize_parameters(processed_predictions)

        # Track predictions to detect issues
        self._samples_checked += 1
        for param_name, value in predicted_params.items():
            if param_name not in self._prediction_tracker:
                self._prediction_tracker[param_name] = []
            self._prediction_tracker[param_name].append(value)

        # Check for prediction issues after a few samples
        if self._samples_checked == 5:
            for param_name, values in self._prediction_tracker.items():
                if all(abs(v - values[0]) < 1e-6 for v in values):
                    print(
                        f"⚠️  WARNING: Model always predicts the same value for {param_name}: {values[0]:.4f}"
                    )
                    print(
                        f"   This suggests the model was not trained properly or weights are incorrect."
                    )
                    print(
                        f"   Expected behavior: predictions should vary across different input samples."
                    )

        # Get true parameters
        true_params = self.get_true_parameters(sample_idx, dataset)

        return predicted_params, true_params, image_tensor

    def synthesize_audio(self, params: Dict[str, float]) -> np.ndarray:
        """
        Synthesize audio using the given parameters.

        Args:
            params: Dictionary of synthesis parameters

        Returns:
            Generated audio array
        """
        # Start with a copy to avoid modifying original
        complete_params = params.copy()

        # Add fixed parameters from dataset metadata to ensure identical synthesis
        if hasattr(self, "dataset_info") and "fixed_parameters" in self.dataset_info:
            fixed_params = self.dataset_info["fixed_parameters"]
            for param_name, param_info in fixed_params.items():
                if param_name not in complete_params:
                    complete_params[param_name] = param_info["value"]
                    log.debug(f"Added fixed parameter {param_name} = {param_info['value']}")

        # Set duration from dataset info
        complete_params["duration"] = self.duration

        # Check for required parameters based on synth type
        if self.synth_type == 'percussion':
            check_params(complete_params, "omega", "tau", "p", "D", "alpha", "duration")
        else:  # saw synth
            check_params(complete_params, "note_number", "note_velocity", "duration", "log10_decay_time")

        # Generate audio - fail fast, no fallbacks
        audio = self.synth.generate_audio(complete_params)
        return audio

    def compute_audio_metrics(
        self, true_audio: np.ndarray, pred_audio: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute perceptual metrics between true and predicted audio.

        Args:
            true_audio: Original audio
            pred_audio: Reconstructed audio

        Returns:
            Dictionary of computed metrics
        """
        metrics = {}

        # Ensure same length
        min_len = min(len(true_audio), len(pred_audio))
        true_audio = true_audio[:min_len]
        pred_audio = pred_audio[:min_len]

        # Mean Absolute Error (MAE)
        metrics["mae"] = float(np.mean(np.abs(true_audio - pred_audio)))

        # Mean Squared Error (MSE)
        metrics["mse"] = float(np.mean((true_audio - pred_audio) ** 2))

        # Signal-to-Noise Ratio
        signal_power = np.mean(true_audio**2)
        noise_power = np.mean((true_audio - pred_audio) ** 2)
        if noise_power > 0:
            metrics["snr_db"] = float(10 * np.log10(signal_power / noise_power))
        else:
            metrics["snr_db"] = float("inf")

        # Pearson correlation
        if np.std(true_audio) > 0 and np.std(pred_audio) > 0:
            correlation, _ = pearsonr(true_audio, pred_audio)
            metrics["correlation"] = float(correlation)
        else:
            metrics["correlation"] = 0.0

        # Cross-correlation maximum (time alignment)
        if len(true_audio) > 1 and len(pred_audio) > 1:
            xcorr = correlate(true_audio, pred_audio, mode="full")
            abs_xcorr = np.abs(xcorr)
            peak_idx = int(np.argmax(abs_xcorr))

            denom = np.linalg.norm(true_audio) * np.linalg.norm(pred_audio)
            if denom > 0:
                metrics["max_xcorr"] = float(abs_xcorr[peak_idx] / denom)
            else:
                metrics["max_xcorr"] = 0.0

            lag_samples = peak_idx - (len(true_audio) - 1)
            metrics["max_xcorr_lag_samples"] = float(lag_samples)
        else:
            metrics["max_xcorr"] = 0.0
            metrics["max_xcorr_lag_samples"] = 0.0

        # Multi-Scale Spectral (MSS) distance
        if hasattr(self, 'mss_loss'):
            # Convert numpy arrays to torch tensors
            # auraloss expects [batch, channels, time], custom expects [batch, time]
            true_tensor = torch.from_numpy(true_audio).unsqueeze(0).float()
            pred_tensor = torch.from_numpy(pred_audio).unsqueeze(0).float()

            # Add channel dimension for auraloss (PNP-compatible format)
            if hasattr(self, 'mss_implementation') and self.mss_implementation == "auraloss":
                true_tensor = true_tensor.unsqueeze(1)  # [batch, 1, time]
                pred_tensor = pred_tensor.unsqueeze(1)  # [batch, 1, time]

            # Move to same device as MSS loss
            device = next(self.mss_loss.parameters()).device if list(self.mss_loss.parameters()) else torch.device('cpu')
            true_tensor = true_tensor.to(device)
            pred_tensor = pred_tensor.to(device)

            # Compute MSS distance (no gradients needed)
            with torch.no_grad():
                mss_distance = self.mss_loss(pred_tensor, true_tensor)
                metrics["mss_distance"] = float(mss_distance.item())

        return metrics

    def evaluate_sample(
        self,
        sample_idx: int,
        plot: bool = True,
        save_audio: bool = False,
        output_dir: Optional[str] = None,
        dataset=None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single sample with complete analysis.

        Args:
            sample_idx: Index of sample to evaluate
            plot: Whether to create visualization plots
            save_audio: Whether to save audio files
            output_dir: Directory to save outputs
            dataset: Dataset to use (defaults to test_dataset if None)

        Returns:
            Dictionary containing all evaluation results
        """
        log.info(f"Evaluating sample {sample_idx}")

        # Run inference
        predicted_params, true_params, input_spec = self.run_inference(sample_idx, dataset)

        # Synthesize audio
        true_audio = self.synthesize_audio(true_params)
        pred_audio = self.synthesize_audio(predicted_params)

        # Compute metrics
        audio_metrics = self.compute_audio_metrics(true_audio, pred_audio)

        # Parameter errors
        param_errors = {}
        for param_name in true_params:
            if param_name in predicted_params:
                true_val = true_params[param_name]
                pred_val = predicted_params[param_name]
                param_errors[param_name] = {
                    "true": true_val,
                    "predicted": pred_val,
                    "absolute_error": abs(pred_val - true_val),
                    "relative_error": (
                        abs(pred_val - true_val) / abs(true_val) if true_val != 0 else float("inf")
                    ),
                }

        results = {
            "sample_idx": sample_idx,
            "predicted_params": predicted_params,
            "true_params": true_params,
            "param_errors": param_errors,
            "audio_metrics": audio_metrics,
            "true_audio": true_audio,
            "pred_audio": pred_audio,
            "input_spectrogram": input_spec.numpy(),
        }

        if plot:
            self.plot_comparison(results)

        if save_audio and output_dir:
            self.save_audio_files(results, output_dir)

        return results

    def plot_comparison(self, results: Dict[str, Any]) -> None:
        """
        Create comprehensive visualization of evaluation results.

        Args:
            results: Evaluation results from evaluate_sample()
        """
        fig = plt.figure(figsize=(16, 12))

        # Store results for click handlers
        self._current_results = results

        # Get data
        true_audio = results["true_audio"]
        pred_audio = results["pred_audio"]
        input_spec = results["input_spectrogram"]

        # Prepare spectrogram for display (extract first channel if multi-channel)
        input_spec = prepare_spectrogram_for_display(input_spec, debug=True)

        # Time axis
        t = np.arange(len(true_audio)) / self.sample_rate

        # 1. Input spectrogram (from dataset)
        plt.subplot(3, 3, 1)
        plt.imshow(input_spec, aspect="auto", origin="lower", cmap="viridis")
        plt.title(f"Input Spectrogram (Sample {results['sample_idx']})")
        plt.xlabel("Time")
        plt.ylabel("Frequency")

        # 2. True audio waveform
        plt.subplot(3, 3, 2)
        plt.plot(t, true_audio, "b-", alpha=0.7, label="True")
        plt.title("True Audio Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)

        # 3. Predicted audio waveform
        plt.subplot(3, 3, 3)
        plt.plot(t, pred_audio, "r-", alpha=0.7, label="Predicted")
        plt.title("Predicted Audio Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)

        # 4. Waveform overlay
        plt.subplot(3, 3, 4)
        plt.plot(t, true_audio, "b-", alpha=0.7, label="True")
        plt.plot(t, pred_audio, "r--", alpha=0.7, label="Predicted")
        plt.title("Waveform Comparison")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 5. True audio spectrogram (re-synthesized)
        plt.subplot(3, 3, 5)
        # Use the same parameters that were used for original dataset generation
        true_spec, _, _ = self.spec_processor.audio_to_spectrogram(
            results["true_params"], true_audio
        )

        # The spectrograms should now match since we're using the same parameters and processing

        plt.imshow(true_spec, aspect="auto", origin="lower", cmap="viridis")
        plt.title("True Synthesis Spectrogram")
        plt.xlabel("Time")
        plt.ylabel("Frequency")

        # 6. Predicted audio spectrogram
        plt.subplot(3, 3, 6)
        pred_spec, _, _ = self.spec_processor.audio_to_spectrogram({}, pred_audio)
        plt.imshow(pred_spec, aspect="auto", origin="lower", cmap="viridis")
        plt.title("Predicted Synthesis Spectrogram")
        plt.xlabel("Time")
        plt.ylabel("Frequency")

        # 7. Parameter comparison (normalized 0-1)
        ax = plt.subplot(3, 3, 7)
        param_names = list(results["param_errors"].keys())
        true_vals = [results["param_errors"][p]["true"] for p in param_names]
        pred_vals = [results["param_errors"][p]["predicted"] for p in param_names]
        plot_parameter_comparison(
            ax, param_names, true_vals, pred_vals, self.param_mappings
        )

        # 8. Audio metrics
        ax = plt.subplot(3, 3, 8)
        metrics = results["audio_metrics"]
        plot_audio_metrics(ax, metrics, self.sample_rate)

        # 9. Error difference waveform
        plt.subplot(3, 3, 9)
        min_len = min(len(true_audio), len(pred_audio))
        error = true_audio[:min_len] - pred_audio[:min_len]
        plt.plot(t[:min_len], error, "g-", alpha=0.7)
        plt.title("Prediction Error (True - Predicted)")
        plt.xlabel("Time (s)")
        plt.ylabel("Error")
        plt.grid(True, alpha=0.3)

        # Add click event handlers for audio playback
        def on_click(event):
            if event.inaxes is None:
                return

            # Get the subplot that was clicked
            ax = event.inaxes
            title = ax.get_title().lower()

            try:
                if IPYTHON_AVAILABLE:
                    from IPython.display import display

                    sample_rate = self.sample_rate

                    # Determine which audio to play based on the clicked plot
                    if "true" in title and "waveform" in title:
                        audio = results["true_audio"]
                        print(f"🎵 Playing true audio waveform (Sample {results['sample_idx']})")
                    elif "predicted" in title and "waveform" in title:
                        audio = results["pred_audio"]
                        print(
                            f"🎵 Playing predicted audio waveform (Sample {results['sample_idx']})"
                        )
                    elif "true" in title and "spectrogram" in title:
                        audio = results["true_audio"]
                        print(
                            f"🎵 Playing true audio from spectrogram (Sample {results['sample_idx']})"
                        )
                    elif "predicted" in title and "spectrogram" in title:
                        audio = results["pred_audio"]
                        print(
                            f"🎵 Playing predicted audio from spectrogram (Sample {results['sample_idx']})"
                        )
                    elif "comparison" in title:
                        # For comparison plot, play true audio
                        audio = results["true_audio"]
                        print(
                            f"🎵 Playing true audio from comparison (Sample {results['sample_idx']})"
                        )
                    else:
                        print(f"💡 Click on waveform or spectrogram plots to play audio")
                        return

                    display(ipd.Audio(audio, rate=sample_rate))
                else:
                    print(f"🎵 Audio playback requires IPython/Jupyter environment")
                    print(f"💡 Tip: Run in Jupyter notebook for audio playback")
            except Exception as e:
                print(f"⚠️  Error playing audio: {e}")

        # Connect the click event
        fig.canvas.mpl_connect("button_press_event", on_click)

        plt.tight_layout()
        plt.show()

        # Add usage instructions
        print(f"💡 Click on waveform or spectrogram plots to play audio!")
        print(f"   - True audio plots (blue): Play original audio")
        print(f"   - Predicted audio plots (red): Play reconstructed audio")

        # Print summary
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY - Sample {results['sample_idx']}")
        print(f"{'='*60}")

        print("\nParameter Comparison:")
        for param_name, error_info in results["param_errors"].items():
            print(f"  {param_name}:")
            print(f"    True: {error_info['true']:.4f}")
            print(f"    Predicted: {error_info['predicted']:.4f}")
            print(f"    Absolute Error: {error_info['absolute_error']:.4f}")
            print(f"    Relative Error: {error_info['relative_error']*100:.2f}%")

        print(f"\nAudio Quality Metrics:")
        for metric_name, value in results["audio_metrics"].items():
            if np.isfinite(value):
                print(f"  {metric_name}: {value:.6f}")
            else:
                print(f"  {metric_name}: {value}")

    def save_audio_files(self, results: Dict[str, Any], output_dir: str) -> None:
        """
        Save true and predicted audio files.

        Args:
            results: Evaluation results
            output_dir: Directory to save files
        """
        try:
            import soundfile as sf
        except ImportError:
            log.warning("soundfile not available, saving as numpy arrays instead")
            import numpy as np

        os.makedirs(output_dir, exist_ok=True)

        sample_idx = results["sample_idx"]

        # Save audio files
        true_path = os.path.join(output_dir, f"sample_{sample_idx}_true")
        pred_path = os.path.join(output_dir, f"sample_{sample_idx}_pred")

        try:
            # Try to save as WAV if soundfile is available
            sf.write(true_path + ".wav", results["true_audio"], self.sample_rate)
            sf.write(pred_path + ".wav", results["pred_audio"], self.sample_rate)
        except NameError:
            # Fallback to numpy arrays
            np.save(true_path + ".npy", results["true_audio"])
            np.save(pred_path + ".npy", results["pred_audio"])

        # Save metadata
        metadata = {
            "sample_idx": sample_idx,
            "predicted_params": results["predicted_params"],
            "true_params": results["true_params"],
            "param_errors": results["param_errors"],
            "audio_metrics": results["audio_metrics"],
        }

        metadata_path = os.path.join(output_dir, f"sample_{sample_idx}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        log.info(f"Saved audio files to {output_dir}")


class InteractiveAudioEvaluator:
    """Interactive widget for browsing and evaluating multiple samples."""

    def __init__(self, evaluator: AudioReconstructionEvaluator):
        """
        Initialize interactive evaluator.

        Args:
            evaluator: AudioReconstructionEvaluator instance
        """
        self.evaluator = evaluator
        self.current_sample = 0
        self.current_results = None
        self.use_test_set = True  # Start with test set by default

        # Create interactive widget
        self.create_widget()

    def get_current_dataset(self):
        """Get the currently selected dataset (train or test)."""
        if self.use_test_set:
            return self.evaluator.test_dataset
        else:
            return self.evaluator.train_dataset

    def get_dataset_label(self):
        """Get a label for the current dataset."""
        return "Test" if self.use_test_set else "Train"

    def create_widget(self):
        """Create the interactive matplotlib widget."""
        # Create figure and explicit layout: 3x3 plot grid on top, control strip below
        self.fig = plt.figure(figsize=(18, 11))
        outer_gs = self.fig.add_gridspec(
            nrows=4, ncols=3, height_ratios=[1.0, 1.0, 0.6, 0.18], hspace=0.35, wspace=0.3
        )

        # Status line (top-left corner). Monospace for compact alignment.
        self.status_text = self.fig.text(
            0.01,
            0.985,
            "",
            fontsize=9,
            va="top",
            ha="left",
            family="monospace",
        )

        # Axes matrix for the plots (3 rows of plots, 1 row of controls)
        import numpy as _np

        self.axes = _np.empty((3, 3), dtype=object)
        for r in range(3):
            for c in range(3):
                self.axes[r, c] = self.fig.add_subplot(outer_gs[r, c])

        # Controls row spanning all columns
        # Bottom controls: text input + checkbox + load button + 8 buttons
        controls_gs = outer_gs[3, :].subgridspec(
            1,
            11,
            width_ratios=[4.5, 0.8, 0.8, 1, 1, 1, 1, 1, 1, 1, 1],
        )

        # Sample number text input occupies most of the bottom strip
        ax_sample_input = self.fig.add_subplot(controls_gs[0, 0])
        # Checkbox for train/test toggle
        ax_checkbox = self.fig.add_subplot(controls_gs[0, 1])
        # Load checkpoint button
        ax_load_ckpt = self.fig.add_subplot(controls_gs[0, 2])
        # Buttons arranged to the right
        ax_prev = self.fig.add_subplot(controls_gs[0, 3])
        ax_next = self.fig.add_subplot(controls_gs[0, 4])
        ax_play_true = self.fig.add_subplot(controls_gs[0, 5])
        ax_play_pred = self.fig.add_subplot(controls_gs[0, 6])
        ax_copy = self.fig.add_subplot(controls_gs[0, 7])
        ax_save = self.fig.add_subplot(controls_gs[0, 8])
        ax_wav_true = self.fig.add_subplot(controls_gs[0, 9])
        ax_wav_pred = self.fig.add_subplot(controls_gs[0, 10])

        # Reduce visual clutter in control axes
        for ax in [
            ax_sample_input,
            ax_checkbox,
            ax_load_ckpt,
            ax_prev,
            ax_next,
            ax_play_true,
            ax_play_pred,
            ax_copy,
            ax_save,
            ax_wav_true,
            ax_wav_pred,
        ]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        # Create interactive widgets
        dataset = self.get_current_dataset()
        self.sample_textbox = TextBox(
            ax_sample_input, "Sample", initial="0", label_pad=0.01
        )
        self.sample_textbox.on_submit(self.update_sample)
        self.max_sample = len(dataset) - 1

        # Checkbox for train/test toggle
        self.check_test = CheckButtons(ax_checkbox, ["Test"], [self.use_test_set])
        self.check_test.on_clicked(self.toggle_dataset)

        # Load checkpoint button
        self.btn_load_ckpt = Button(ax_load_ckpt, "Load\nCkpt")
        self.btn_load_ckpt.on_clicked(self.load_checkpoint)

        self.btn_prev = Button(ax_prev, "Prev")
        self.btn_next = Button(ax_next, "Next")
        self.btn_play_true = Button(ax_play_true, "Play True")
        self.btn_play_pred = Button(ax_play_pred, "Play Pred")
        self.btn_copy = Button(ax_copy, "Copy Info")
        self.btn_save = Button(ax_save, "Save PNG")
        self.btn_wav_true = Button(ax_wav_true, "Save True WAV")
        self.btn_wav_pred = Button(ax_wav_pred, "Save Pred WAV")

        self.btn_prev.on_clicked(self.prev_sample)
        self.btn_next.on_clicked(self.next_sample)
        self.btn_play_true.on_clicked(self.play_true_audio)
        self.btn_play_pred.on_clicked(self.play_pred_audio)
        self.btn_copy.on_clicked(self.copy_status_info)
        self.btn_save.on_clicked(self.save_figure)
        self.btn_wav_true.on_clicked(self.save_true_wav)
        self.btn_wav_pred.on_clicked(self.save_pred_wav)

        # Add keyboard navigation
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        # Add parameter plot click navigation (for AVIX datasets)
        self.fig.canvas.mpl_connect("button_press_event", self.on_param_plot_click)

        # Initial update
        self.update_display()

        # Quick diagnostic: check prediction diversity across several samples
        try:
            self.evaluator.check_prediction_diversity(n_samples=32)
        except SystemExit:
            raise
        except Exception as e:
            print(f"Prediction diversity check failed: {e}")

        # Print usage instructions
        print("🎵 Interactive Audio Evaluator")
        print("💡 Navigation:")
        print("   • Left/Right arrow keys: Navigate between samples")
        print("   • Click 'Prev'/'Next' buttons: Navigate between samples")
        print("   • Type sample number in text field and press Enter: Jump to specific sample")
        print("   • Click 'Test' checkbox: Toggle between Test and Train datasets")
        print("   • Click 'Load Ckpt' button: Choose a checkpoint from reference folder")
        print("   • Click 'Play True'/'Play Pred': Play audio for current sample")
        print("   • Click 'Copy Info': Copy header info to clipboard (or print)")
        if self.evaluator.dataset_info.get('is_avix_dataset'):
            print("   • [AVIX] Click on parameter bars to navigate to that parameter value")

    def load_checkpoint(self, event):
        """List available checkpoints and provide command to restart with selected one."""
        from pathlib import Path
        import glob
        import datetime

        print("\n" + "="*70)
        print("📁 AVAILABLE CHECKPOINTS")
        print("="*70)

        # Search for checkpoints in reference directory and logs
        ref_dir = Path("checkpoints/reference")
        logs_dir = Path("logs/train")

        checkpoints = []

        # Reference checkpoints
        if ref_dir.exists():
            ref_ckpts = sorted(ref_dir.glob("*.ckpt"))
            if ref_ckpts:
                print(f"\n📌 Reference checkpoints ({len(ref_ckpts)}):")
                for i, ckpt in enumerate(ref_ckpts, 1):
                    print(f"   {i}. {ckpt.name}")
                    checkpoints.append(str(ckpt))

        # Recent training checkpoints
        if logs_dir.exists():
            recent_ckpts = []
            for pattern in ["runs/*/checkpoints/best.ckpt", "runs/*/checkpoints/last.ckpt"]:
                recent_ckpts.extend(glob.glob(str(logs_dir / pattern)))

            # Sort by modification time, most recent first
            recent_ckpts = sorted(recent_ckpts, key=lambda x: Path(x).stat().st_mtime, reverse=True)[:10]

            if recent_ckpts:
                print(f"\n🕐 Recent training checkpoints (last 10):")
                for i, ckpt in enumerate(recent_ckpts, 1):
                    ckpt_path = Path(ckpt)
                    run_name = ckpt_path.parent.parent.name
                    ckpt_type = ckpt_path.name

                    # Try to extract experiment name from checkpoint
                    experiment_name = None
                    try:
                        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                        hyper_params = checkpoint.get("hyper_parameters", {})
                        experiment_name = hyper_params.get("experiment_name", None)
                    except Exception:
                        pass

                    # Format: YYYY-MM-DD_HH-MM-SS_<experiment> or YYYY-MM-DD_HH-MM-SS if no experiment
                    if experiment_name:
                        display_name = f"{run_name}_{experiment_name}"
                    else:
                        display_name = run_name

                    print(f"   {len(checkpoints)+i}. {display_name} [{ckpt_type}]")
                    checkpoints.append(str(ckpt))

        if not checkpoints:
            print("\n⚠️  No checkpoints found")
            print("   Searched in: checkpoints/reference/ and logs/train/runs/")
            print("="*70)
            return

        print("\n" + "="*70)
        print("💡 To load a checkpoint, type a number and press Enter:")
        print("   Or type a full path to a .ckpt file")
        print("   Or press Enter alone to cancel")
        print("="*70)

        # Get user input from console
        try:
            selection = input("\nEnter selection: ").strip()

            if not selection:
                print("❌ Cancelled")
                print("="*70)
                return

            # Try to parse as a number
            try:
                idx = int(selection)
                if 1 <= idx <= len(checkpoints):
                    ckpt_path = checkpoints[idx - 1]
                else:
                    print(f"❌ Invalid number. Please choose 1-{len(checkpoints)}")
                    print("="*70)
                    return
            except ValueError:
                # Not a number - treat as a path
                ckpt_path = selection
                if not Path(ckpt_path).exists():
                    print(f"❌ File not found: {ckpt_path}")
                    print("="*70)
                    return

            print(f"\n📂 Selected checkpoint: {ckpt_path}")

            # Build restart command
            cmd = f"python src/audio_reconstruction_eval.py ckpt_path={ckpt_path}"

            # Try to copy to clipboard
            copied = False
            try:
                import pyperclip
                pyperclip.copy(cmd)
                copied = True
            except ImportError:
                pass

            print("\n" + "="*70)
            print("💡 To load this checkpoint, please restart with:")
            print(f"   {cmd}")
            if copied:
                print("\n✅ Command copied to clipboard!")
            else:
                print("\n   (Install pyperclip to enable clipboard copy)")
            print("\n   Close this window and run the command above.")
            print("="*70)

        except EOFError:
            print("\n❌ Input cancelled")
            print("="*70)
        except KeyboardInterrupt:
            print("\n❌ Interrupted")
            print("="*70)
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            print("="*70)

    def on_param_plot_click(self, event):
        """Handle clicks on parameter plot for AVIX navigation."""
        # Debug: print all clicks
        print(f"\n🖱️  Click detected: button={event.button}, inaxes={event.inaxes is not None}")

        # Only works for AVIX datasets on train set
        is_avix = self.evaluator.dataset_info.get('is_avix_dataset', False)
        print(f"   Is AVIX dataset: {is_avix}")

        if not is_avix:
            print("   → Ignoring: Not an AVIX dataset")
            return

        if not self.use_test_set:  # Train set
            dataset_to_use = self.evaluator.train_dataset
            use_nearest_neighbor = False
            print(f"   Using train set ({len(dataset_to_use)} samples) - grid navigation")
        else:  # Test set
            dataset_to_use = self.evaluator.test_dataset
            use_nearest_neighbor = True
            print(f"   Using test set ({len(dataset_to_use)} samples) - nearest neighbor search")

        # Check if click was on the parameter plot (axes[1, 1])
        if event.inaxes is None:
            print("   → Ignoring: Click outside axes")
            return

        if event.inaxes != self.axes[1, 1]:
            # Show which subplot was clicked
            for r in range(2):
                for c in range(3):
                    if event.inaxes == self.axes[r, c]:
                        print(f"   → Ignoring: Click on axes[{r},{c}], not parameter plot [1,1]")
                        return
            print(f"   → Ignoring: Click on unknown axes")
            return

        print(f"   ✓ Click on parameter plot!")

        # Get click coordinates
        x_click = event.xdata
        y_click = event.ydata

        if x_click is None or y_click is None:
            return

        # Check if we have parameter info stored
        if not hasattr(self, '_param_names_for_click') or not hasattr(self, '_param_x_positions'):
            return

        param_names = self._param_names_for_click
        x_positions = self._param_x_positions

        # Find nearest parameter (x position)
        # Each bar is centered at an integer x position
        param_idx = int(round(x_click))
        if param_idx < 0 or param_idx >= len(param_names):
            return

        param_name = param_names[param_idx]

        # Get parameter mapping
        mapping = self.evaluator.param_mappings.get(param_name)
        if not mapping:
            print(f"⚠️  No mapping found for parameter '{param_name}'")
            return

        pmin = mapping.get('min', 0.0)
        pmax = mapping.get('max', 1.0)
        step = mapping.get('step')

        if step is None or step <= 0:
            print(f"⚠️  Invalid step for parameter '{param_name}'")
            return

        # Map y-coordinate (0-1 normalized) to actual parameter value
        # y_click is already in normalized 0-1 space
        y_clamped = max(0.0, min(1.0, y_click))
        actual_value = pmin + y_clamped * (pmax - pmin)

        # Round to nearest step
        num_steps_from_min = round((actual_value - pmin) / step)
        rounded_value = pmin + num_steps_from_min * step
        rounded_value = max(pmin, min(pmax, rounded_value))

        print(f"\n🖱️  Clicked on '{param_name}' at y={y_click:.3f}")
        print(f"   Mapped to: {actual_value:.4f} → Rounded to: {rounded_value:.4f}")

        # Build target parameter values for all parameters
        target_params = {}
        for pname in param_names:
            pmapping = self.evaluator.param_mappings.get(pname)
            if not pmapping:
                continue

            p_min = pmapping['min']
            p_max = pmapping['max']

            if pname == param_name:
                # Use our clicked/rounded value
                target_params[pname] = rounded_value
            else:
                # Use current true value from displayed sample
                target_params[pname] = self.current_results['true_params'].get(pname, p_min)

        if use_nearest_neighbor:
            # Test set: Find nearest neighbor by parameter distance
            print(f"   Searching for nearest test sample...")

            best_sample_idx = None
            best_distance = float('inf')

            for sample_idx in range(len(dataset_to_use)):
                # Get true parameters for this test sample
                sample_params = self.evaluator.get_true_parameters(sample_idx, dataset_to_use)

                # Calculate normalized Euclidean distance
                distance = 0.0
                for pname in param_names:
                    pmapping = self.evaluator.param_mappings.get(pname)
                    if not pmapping:
                        continue

                    p_min = pmapping['min']
                    p_max = pmapping['max']
                    p_range = max(1e-12, p_max - p_min)

                    # Normalize both values to [0, 1] for fair distance calculation
                    target_norm = (target_params[pname] - p_min) / p_range
                    sample_norm = (sample_params[pname] - p_min) / p_range

                    distance += (target_norm - sample_norm) ** 2

                distance = np.sqrt(distance)

                if distance < best_distance:
                    best_distance = distance
                    best_sample_idx = sample_idx

            if best_sample_idx is not None:
                sample_idx = best_sample_idx
                print(f"   Found nearest sample: index={sample_idx}, distance={best_distance:.4f}")
            else:
                print(f"⚠️  No nearest sample found")
                return
        else:
            # Train set: Calculate AVIX grid index
            # For AVIX, parameters are ordered and the rightmost varies fastest
            param_indices = {}

            for pname in param_names:
                pmapping = self.evaluator.param_mappings.get(pname)
                if not pmapping:
                    continue

                p_min = pmapping['min']
                p_max = pmapping['max']
                p_step = pmapping['step']

                # Calculate step index for this parameter
                p_value = target_params[pname]
                idx = round((p_value - p_min) / p_step)
                num_steps = round((p_max - p_min) / p_step)
                idx = max(0, min(num_steps, idx))
                param_indices[pname] = (idx, num_steps + 1)

            # Calculate flat sample index using AVIX ordering
            # Rightmost parameter varies fastest
            sample_idx = 0
            stride = 1

            for pname in reversed(param_names):  # Rightmost first
                if pname not in param_indices:
                    print(f"⚠️  Missing parameter '{pname}' in indices")
                    return
                idx, num_classes = param_indices[pname]
                sample_idx += idx * stride
                stride *= num_classes

            print(f"   AVIX sample index: {sample_idx}")
            print(f"   Parameter combination: {', '.join(f'{p}[{param_indices[p][0]}/{param_indices[p][1]-1}]' for p in param_names)}")

        # Navigate to this sample
        if 0 <= sample_idx < len(dataset_to_use):
            self.current_sample = sample_idx
            self.sample_textbox.set_val(str(self.current_sample))
            self.update_display()
        else:
            print(f"⚠️  Calculated sample index {sample_idx} out of range [0, {len(dataset_to_use)-1}]")

    def on_key_press(self, event):
        """Handle keyboard navigation events."""
        if event.key == "left":
            # Left arrow key - go to previous sample
            self.prev_sample(None)
        elif event.key == "right":
            # Right arrow key - go to next sample
            self.next_sample(None)
        elif event.key.lower() == "s":
            # Save figure
            self.save_figure(None)
        elif event.key.lower() == "t":
            # Save true WAV
            self.save_true_wav(None)
        elif event.key.lower() == "p":
            # Save predicted WAV
            self.save_pred_wav(None)

    def update_sample(self, text):
        """Update current sample from text input."""
        try:
            sample_num = int(text)
            # Clamp to valid range
            sample_num = max(0, min(sample_num, self.max_sample))
            self.current_sample = sample_num
            # Update text box to show clamped value if it was out of range
            self.sample_textbox.set_val(str(self.current_sample))
            self.update_display()
        except ValueError:
            # Invalid input - reset to current sample
            self.sample_textbox.set_val(str(self.current_sample))
            print(f"⚠️ Invalid sample number: '{text}'. Please enter a number between 0 and {self.max_sample}.")

    def prev_sample(self, event):
        """Go to previous sample."""
        old_sample = self.current_sample
        self.current_sample = max(0, self.current_sample - 1)
        print(f"Prev: {old_sample} -> {self.current_sample}")
        self.sample_textbox.set_val(str(self.current_sample))
        self.update_display()

    def next_sample(self, event):
        """Go to next sample."""
        old_sample = self.current_sample
        self.current_sample = min(self.max_sample, self.current_sample + 1)
        print(f"Next: {old_sample} -> {self.current_sample}")
        self.sample_textbox.set_val(str(self.current_sample))
        self.update_display()

    def toggle_dataset(self, label):
        """Toggle between train and test datasets."""
        # Toggle the flag
        self.use_test_set = not self.use_test_set
        dataset = self.get_current_dataset()

        print(f"🔀 Switched to {self.get_dataset_label()} set ({len(dataset)} samples)")

        # Reset to first sample
        self.current_sample = 0
        self.max_sample = len(dataset) - 1

        # Update text box
        self.sample_textbox.set_val("0")
        self.update_display()

    def update_display(self):
        """Update all plots for current sample."""
        dataset = self.get_current_dataset()
        dataset_label = self.get_dataset_label()
        print(f"🔄 Updating display for {dataset_label} sample {self.current_sample}")

        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()

        # Evaluate current sample from the selected dataset
        self.current_results = self.evaluator.evaluate_sample(
            self.current_sample, plot=False, dataset=dataset
        )

        # Update status line with model/eval info
        try:
            self.status_text.set_text(self._build_status_string())
            # Optional secondary line for run/dataset/ckpt details
            detail = self._build_status_detail_string()

            # Determine color based on training steps (red if undertrained)
            text_color = "black"
            e = self.evaluator
            if hasattr(e, 'training_steps') and e.training_steps is not None:
                if e.training_steps < 100:
                    text_color = "red"

            if not hasattr(self, "status_text2"):
                self.status_text2 = self.fig.text(
                    0.01,
                    0.965,
                    detail,
                    fontsize=9,
                    va="top",
                    ha="left",
                    family="monospace",
                    color=text_color,
                )
            else:
                self.status_text2.set_text(detail)
                self.status_text2.set_color(text_color)
        except Exception:
            # Avoid UI breakage on any unexpected attribute
            pass

        # Debug: Print some identifying info about this sample
        if self.current_results:
            print(
                f"   Sample {self.current_sample} - True audio range: [{self.current_results['true_audio'].min():.3f}, {self.current_results['true_audio'].max():.3f}]"
            )
            print(
                f"   Sample {self.current_sample} - Pred audio range: [{self.current_results['pred_audio'].min():.3f}, {self.current_results['pred_audio'].max():.3f}]"
            )

            # Print parameter values to check if they're actually changing
            true_params = self.current_results["true_params"]
            pred_params = self.current_results["predicted_params"]

            # Just show a few key parameters to see if they vary
            key_params = ["note_number", "log10_filter_cutoff_hz", "filter_resonance"]
            for param in key_params:
                if param in true_params and param in pred_params:
                    print(
                        f"   {param}: true={true_params[param]:.3f}, pred={pred_params[param]:.3f}"
                    )

            # Also print ALL true params for a couple samples to see the full picture
            if self.current_sample < 3:
                print(f"   All true params for sample {self.current_sample}: {true_params}")
                print(f"   All pred params for sample {self.current_sample}: {pred_params}")

        # Get data
        true_audio = self.current_results["true_audio"]
        pred_audio = self.current_results["pred_audio"]
        input_spec = self.current_results["input_spectrogram"]

        # Prepare spectrogram for display (extract first channel if multi-channel)
        input_spec = prepare_spectrogram_for_display(input_spec, debug=True)

        t = np.arange(len(true_audio)) / self.evaluator.sample_rate

        # Row 0: Input spectrogram, Waveform overlay, True spectrogram
        self.axes[0, 0].imshow(input_spec, aspect="auto", origin="lower", cmap="viridis")
        self.axes[0, 0].set_title(
            f"Input Spectrogram ({dataset_label} Sample {self.current_sample})"
        )

        self.axes[0, 1].plot(t, true_audio, "b-", alpha=0.7, label="True")
        self.axes[0, 1].plot(t, pred_audio, "r--", alpha=0.7, label="Predicted")
        self.axes[0, 1].set_title("Audio Waveforms")
        self.axes[0, 1].legend()
        self.axes[0, 1].grid(True, alpha=0.3)

        true_params = self.current_results["true_params"]
        true_spec, _, _ = self.evaluator.spec_processor.audio_to_spectrogram(
            true_params, true_audio
        )
        self.axes[0, 2].imshow(true_spec, aspect="auto", origin="lower", cmap="viridis")
        self.axes[0, 2].set_title("True Audio Spectrogram")

        # Row 1: Predicted spectrogram, Parameters, Metrics
        pred_spec, _, _ = self.evaluator.spec_processor.audio_to_spectrogram({}, pred_audio)
        self.axes[1, 0].imshow(pred_spec, aspect="auto", origin="lower", cmap="viridis")
        self.axes[1, 0].set_title("Predicted Audio Spectrogram")

        param_names = list(self.current_results["param_errors"].keys())
        if param_names:
            true_vals = [self.current_results["param_errors"][p]["true"] for p in param_names]
            pred_vals = [self.current_results["param_errors"][p]["predicted"] for p in param_names]

            title_suffix = (
                "[AVIX: Click to navigate]"
                if self.evaluator.dataset_info.get("is_avix_dataset")
                else ""
            )

            plot_parameter_comparison(
                self.axes[1, 1],
                param_names,
                true_vals,
                pred_vals,
                self.evaluator.param_mappings,
                title_suffix=title_suffix,
            )

            self._param_names_for_click = param_names
            self._param_x_positions = np.arange(len(param_names))

        metrics = self.current_results["audio_metrics"]
        plot_audio_metrics(self.axes[1, 2], metrics, self.evaluator.sample_rate)

        # Row 2: Error waveform (centered, spanning leftmost position)
        min_len = min(len(true_audio), len(pred_audio))
        error = true_audio[:min_len] - pred_audio[:min_len]
        self.axes[2, 0].plot(t[:min_len], error, "g-", alpha=0.7)
        self.axes[2, 0].set_title("Prediction Error (True - Predicted)")
        self.axes[2, 0].set_xlabel("Time (s)")
        self.axes[2, 0].set_ylabel("Error")
        self.axes[2, 0].grid(True, alpha=0.3)

        # Hide unused plots in row 2
        self.axes[2, 1].axis('off')
        self.axes[2, 2].axis('off')

        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # Force figure to update in interactive backends
        plt.pause(0.01)

    def _build_status_string(self) -> str:
        """Assemble a one-line status string for the UI header.

        Shows: output mode, head sizes, and whether regression head weights were loaded.
        """
        m = getattr(self.evaluator, "model", None)
        if m is None:
            return ""

        mode = getattr(m, "output_mode", "?")
        # Head sizes from the network config
        heads = {}
        if hasattr(m, "net") and hasattr(m.net, "heads_config") and m.net.heads_config:
            heads = dict(m.net.heads_config)
        head_str = ", ".join(f"{k}:{v}" for k, v in heads.items()) if heads else "-"

        # Weight loading info populated during reconstruction
        info = getattr(m, "_ae_weight_load_info", {}) or {}
        compat = info.get("compatible", "-")
        incompat = info.get("incompatible", "-")
        heads_loaded = "-"
        try:
            loaded_keys = info.get("loaded_keys", [])
            heads_loaded = "yes" if any(".heads." in k for k in loaded_keys) else "no"
        except Exception:
            pass

        return (
            f"mode={mode} | heads[{head_str}] | weights: compat={compat}, "
            f"skip={incompat}, regression_loaded={heads_loaded}"
        )

    def _build_status_detail_string(self) -> str:
        """Second header line with run/dataset/arch summary."""
        e = self.evaluator
        dataset = getattr(e, "dataset_name", "")
        sr = getattr(e, "sample_rate", None)
        dur = getattr(e, "duration", None)
        dims = (
            f"{e.dataset_info.get('height','?')}x{e.dataset_info.get('width','?')}x{e.dataset_info.get('channels','?')}"
            if hasattr(e, "dataset_info")
            else "?x?x?"
        )
        run = getattr(e, "run_name", "")
        experiment = getattr(e, "experiment_name", None)
        ckpt = getattr(e, "ckpt_path", "")
        ckpt_short = ckpt
        try:
            if ckpt and len(ckpt) > 64:
                ckpt_short = "…" + ckpt[-63:]
        except Exception:
            pass
        net_name = (
            type(e.model.net).__name__ if hasattr(e.model, "net") else type(e.model).__name__
        )
        # Param count
        try:
            import numpy as _np  # noqa: F401

            params = sum(p.numel() for p in e.model.parameters())
        except Exception:
            params = None
        params_str = f"{params:,}" if params is not None else "?"
        sr_str = f"{sr}" if sr is not None else "?"
        dur_str = f"{dur}" if dur is not None else "?"

        # Training progress info with warning for undertrained models
        training_info = ""
        if hasattr(e, 'training_steps') and e.training_steps is not None:
            steps = e.training_steps
            steps_str = f"steps={steps}"
            # Add epoch info if available
            if hasattr(e, 'total_epochs') and e.total_epochs is not None:
                steps_str += f" epoch={e.total_epochs}"
            training_info = f" | {steps_str}"

        # Build experiment info string (prioritize experiment name over run name)
        if experiment:
            exp_str = f"experiment={experiment}"
        else:
            exp_str = f"run={run or '-'}"

        return (
            f"{exp_str} | dataset={dataset or '-'} ({dims}) | sr={sr_str} Hz, dur={dur_str}s | "
            f"arch={net_name} params={params_str}{training_info} | ckpt={ckpt_short or '-'}"
        )

    def _build_metrics_string(self) -> str:
        """Format current audio metrics for clipboard export."""
        results = getattr(self, "current_results", None)
        if not results:
            return ""

        metrics = results.get("audio_metrics", {})
        if not metrics:
            return ""

        metric_order = ["snr_db", "correlation", "max_xcorr", "max_xcorr_lag_samples", "mse"]
        labels = {
            "snr_db": "Waveform SNR",
            "correlation": "Correlation",
            "max_xcorr": "Max XCorr",
            "max_xcorr_lag_samples": "Max XCorr Lag",
            "mse": "MSE",
        }

        parts = []
        for name in metric_order:
            value = metrics.get(name)
            if value is None or not np.isfinite(value):
                continue

            if name == "snr_db":
                parts.append(f"{labels[name]}={value:.2f} dB")
            elif name in ["correlation", "max_xcorr"]:
                parts.append(f"{labels[name]}={value:.4f}")
            elif name == "max_xcorr_lag_samples":
                lag_samples = int(round(value))
                sr = getattr(self.evaluator, "sample_rate", None)
                lag_ms = (
                    (lag_samples / sr) * 1000 if sr and sr > 0 else 0.0
                )
                parts.append(
                    f"{labels[name]}={lag_samples:+d} samples ({lag_ms:+.3f} ms)"
                )
            elif name == "mse":
                parts.append(f"{labels[name]}={value:.8f}")

        if not parts:
            return ""

        return "Audio metrics: " + " | ".join(parts)

    def copy_status_info(self, event):
        """Copy header lines to clipboard, with fallbacks to stdout and file."""
        try:
            segments = [self._build_status_string(), self._build_status_detail_string()]
            metrics_line = self._build_metrics_string()
            if metrics_line:
                segments.append(metrics_line)
            text = "\n".join(seg for seg in segments if seg)
        except Exception as _e:  # noqa: F841
            text = ""
        success = False
        # Try pyperclip
        try:
            import pyperclip

            pyperclip.copy(text)
            success = True
        except Exception:
            pass
        # Try tkinter as a fallback
        if not success:
            try:
                import tkinter as tk

                r = tk.Tk()
                r.withdraw()
                r.clipboard_clear()
                r.clipboard_append(text)
                r.update()  # now it stays on the clipboard after the window is closed
                r.destroy()
                success = True
            except Exception:
                pass
        # Fallback to printing and saving to a file
        if not success:
            print("[AE] Clipboard unavailable. Header info:\n" + text)
            try:
                import os
                from pathlib import Path

                out_dir = Path("logs/audio_eval")
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / "clipboard.txt", "w") as f:
                    f.write(text)
                print(f"[AE] Saved header info to {out_dir / 'clipboard.txt'}")
            except Exception:
                pass

    def save_figure(self, event):
        """Save current figure to PNG under audio_eval_results/ with timestamp."""
        try:
            import time
            from pathlib import Path

            out_dir = Path("audio_eval_results")
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d-%H%M%S")
            run = getattr(self.evaluator, "run_name", "run") or "run"
            fname = out_dir / f"ae_{run}_{ts}.png"
            self.fig.savefig(str(fname), dpi=150)
            print(f"[AE] Saved figure to {fname}")
        except Exception as e:
            print(f"[AE] Save figure failed: {e}")

    def _save_wav(self, audio, sr: int, kind: str) -> None:
        """Save numpy audio array to WAV (16-bit PCM) in audio_eval_results."""
        from pathlib import Path
        import time
        import numpy as _np

        out_dir = Path("audio_eval_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        run = getattr(self.evaluator, "run_name", "run") or "run"
        idx = self.current_results.get("sample_idx", 0) if self.current_results else 0
        fname = out_dir / f"ae_{run}_sample-{idx:04d}_{kind}_{ts}.wav"

        # Prefer soundfile, else scipy
        try:
            import soundfile as sf

            sf.write(str(fname), _np.asarray(audio, dtype=_np.float32), sr)
        except Exception:
            from scipy.io import wavfile as _wav

            x = _np.asarray(audio, dtype=_np.float32)
            x = _np.clip(x, -1.0, 1.0)
            x16 = (x * 32767.0).astype(_np.int16)
            _wav.write(str(fname), sr, x16)

        print(f"[AE] Saved {kind} WAV to {fname}")

    def save_true_wav(self, event):
        if not self.current_results:
            print("[AE] No data to save")
            return
        self._save_wav(self.current_results["true_audio"], self.evaluator.sample_rate, "true")

    def save_pred_wav(self, event):
        if not self.current_results:
            print("[AE] No data to save")
            return
        self._save_wav(self.current_results["pred_audio"], self.evaluator.sample_rate, "pred")

    def play_true_audio(self, event):
        """Play true audio with robust playback options."""
        if not self.current_results:
            print("⚠️  No audio data available")
            return

        try:
            sample_rate = self.evaluator.sample_rate
            audio = self.current_results["true_audio"]
            sample_idx = self.current_results["sample_idx"]

            print(f"🎵 Playing true audio (Sample {sample_idx})")
            self._play_audio_array(audio, sample_rate)

        except Exception as e:
            print(f"⚠️  Error playing true audio: {e}")

    def play_pred_audio(self, event):
        """Play predicted audio with robust playback options."""
        if not self.current_results:
            print("⚠️  No audio data available")
            return

        try:
            sample_rate = self.evaluator.sample_rate
            audio = self.current_results["pred_audio"]
            sample_idx = self.current_results["sample_idx"]

            print(f"🎵 Playing predicted audio (Sample {sample_idx})")
            self._play_audio_array(audio, sample_rate)

        except Exception as e:
            print(f"⚠️  Error playing predicted audio: {e}")

    def _play_audio_array(self, audio, sample_rate: int) -> None:
        """Centralized robust playback with OS-specific fallbacks."""
        import platform, subprocess, tempfile, numpy as _np

        system = platform.system()
        audio = _np.asarray(audio, dtype=_np.float32)

        # macOS: afplay first (often most reliable from CLI/Matplotlib)
        if system == "Darwin":
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    try:
                        import soundfile as sf

                        sf.write(tmp_path, audio, sample_rate)
                    except Exception:
                        from scipy.io import wavfile as _wav

                        _wav.write(tmp_path, sample_rate, (audio * 32767).astype(_np.int16))
                    subprocess.run(["afplay", tmp_path], check=True)
                    print("   ✓ Played via afplay")
                    return
                finally:
                    import os, time as _t

                    _t.sleep(0.25)
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
            except Exception as e:
                print(f"   macOS afplay failed: {e}")

        # Generic: sounddevice
        try:
            import sounddevice as sd

            sd.play(audio, sample_rate)
            sd.wait()
            print("   ✓ Played via sounddevice")
            return
        except Exception as e:
            print(f"   sounddevice not available or failed: {e}")

        # Jupyter/IPython
        if IPYTHON_AVAILABLE:
            try:
                from IPython.display import display

                display(ipd.Audio(audio, rate=sample_rate))
                print("   ✓ Displayed via Jupyter")
                return
            except Exception as e:
                print(f"   Jupyter playback failed: {e}")

        # Linux/Windows system players
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                try:
                    import soundfile as sf

                    sf.write(tmp_path, audio, sample_rate)
                except Exception:
                    from scipy.io import wavfile as _wav

                    _wav.write(tmp_path, sample_rate, (audio * 32767).astype(_np.int16))
                if system == "Linux":
                    subprocess.run(["aplay", tmp_path], check=True)
                elif system == "Windows":
                    import os

                    os.startfile(tmp_path)
                else:
                    raise RuntimeError("No suitable system player")
                print("   ✓ Played via system player")
                return
            finally:
                import os, time as _t

                _t.sleep(0.25)
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        except Exception as e:
            print(f"   Fallback system player failed: {e}")


@task_wrapper
def evaluate_audio_reconstruction(cfg: DictConfig) -> Tuple[Dict[str, Any], Optional[None]]:
    """
    Main evaluation function for audio reconstruction.

    Args:
        cfg: Hydra configuration

    Returns:
        Tuple of (evaluation results dictionary, None)
    """
    try:
        # Setup and validation
        ckpt_path = _resolve_checkpoint_path(cfg)
        device = _determine_device()
        datamodule = _setup_datamodule(cfg, ckpt_path)
        model = _load_model_from_checkpoint(ckpt_path, datamodule, device)

        # Create evaluator and run evaluation
        evaluator = AudioReconstructionEvaluator(model, datamodule, device, ckpt_path=ckpt_path)
        return _run_evaluation(cfg, evaluator, ckpt_path)

    except CheckpointError as e:
        log.error(f"Checkpoint error: {e}")
        raise
    except Exception as e:
        log.error(f"Unexpected error during evaluation: {e}")
        raise


def _resolve_checkpoint_path(cfg: DictConfig) -> str:
    """Resolve checkpoint path from config or auto-discovery."""
    ckpt_path = cfg.get("ckpt_path")
    if not ckpt_path:
        ckpt_path = find_latest_checkpoint()
        if not ckpt_path:
            raise CheckpointError(
                "No checkpoint path provided and no checkpoints found. "
                "Please either:\n"
                "1. Provide ckpt_path=path/to/checkpoint.ckpt, or\n"
                "2. Ensure you have trained checkpoints in logs/train/"
            )

    if not os.path.exists(ckpt_path):
        raise CheckpointError(f"Checkpoint not found: {ckpt_path}")

    log.info(f"Using checkpoint: {ckpt_path}")
    return ckpt_path


def _determine_device() -> str:
    """Determine the appropriate device for evaluation."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _setup_datamodule(cfg: DictConfig, ckpt_path: str) -> LightningDataModule:
    """Setup and configure the datamodule for evaluation."""
    # Load checkpoint to extract dataset metadata
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vimh_data = checkpoint.get("VIMHDataModule", {})
    dataset_metadata = vimh_data.get("dataset_metadata", {})

    # Override config with checkpoint's datamodule hyperparameters
    datamodule_hparams = checkpoint.get("datamodule_hyper_parameters", {})
    if not datamodule_hparams:
        log.error("Checkpoint missing datamodule_hyper_parameters - cannot determine training dataset")
        sys.exit(1)

    # CRITICAL: Use the exact data_dir from training to ensure parameter consistency
    if "data_dir" not in datamodule_hparams:
        log.error("Checkpoint missing data_dir in datamodule_hyper_parameters")
        sys.exit(1)

    trained_data_dir = datamodule_hparams["data_dir"]
    if not os.path.exists(trained_data_dir):
        log.error(
            f"Training dataset directory not found: {trained_data_dir}\n"
            f"Evaluation requires the exact dataset used during training.\n"
            f"Please ensure the dataset exists at the specified path."
        )
        sys.exit(1)

    log.info(f"Using training dataset directory: {trained_data_dir}")
    cfg.data.data_dir = trained_data_dir

    if datamodule_hparams:
        log.info("Applying checkpoint's datamodule hyperparameters to ensure consistency")

        # Temporarily disable struct mode to allow adding new keys
        from omegaconf import OmegaConf
        OmegaConf.set_struct(cfg.data, False)
        try:
            # Critical: preserve auxiliary_features from training
            if "auxiliary_features" in datamodule_hparams:
                aux_features = datamodule_hparams["auxiliary_features"]
                cfg.data.auxiliary_features = aux_features
                log.info(f"Using auxiliary_features from checkpoint: {aux_features}")

            # Also preserve other critical hyperparameters
            if "label_mode" in datamodule_hparams:
                cfg.data.label_mode = datamodule_hparams["label_mode"]
                log.info(f"Using label_mode from checkpoint: {datamodule_hparams['label_mode']}")

            if "target_width" in datamodule_hparams:
                cfg.data.target_width = datamodule_hparams["target_width"]
        finally:
            OmegaConf.set_struct(cfg.data, True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")

    # Use identity transforms for spectrograms
    from torchvision.transforms import transforms

    identity_transform = transforms.Compose([])

    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data,
        train_transform=identity_transform,
        val_transform=identity_transform,
        test_transform=identity_transform,
    )

    # Inject saved metadata if available
    if dataset_metadata and "parameter_names" in dataset_metadata:
        log.info(
            f"Found saved dataset metadata with parameters: {dataset_metadata['parameter_names']}"
        )
        datamodule._saved_dataset_metadata = dataset_metadata
        log.info("Injected saved metadata into datamodule")

    # Setup the datamodule
    datamodule.setup("fit")
    return datamodule


def _load_from_hydra_config(ckpt_path: str, device: str) -> LightningModule:
    """Load model by reconstructing from Hydra config file.

    This is the standard practice for Hydra-based projects where the full
    configuration is saved alongside the checkpoint in .hydra/config.yaml.
    """
    from pathlib import Path
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    # Find .hydra/config.yaml relative to checkpoint
    ckpt_dir = Path(ckpt_path).parent.parent  # checkpoints/last.ckpt -> run_dir
    hydra_config_path = ckpt_dir / ".hydra" / "config.yaml"

    if not hydra_config_path.exists():
        raise FileNotFoundError(f"Hydra config not found: {hydra_config_path}")

    log.info(f"Loading model from Hydra config: {hydra_config_path}")

    # Load checkpoint to extract heads configuration
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]

    # Extract heads config from checkpoint state_dict
    heads_config = {}
    for key in state_dict.keys():
        if key.startswith("net.heads.") and key.endswith(".weight"):
            # Extract head name: net.heads.log10_decay_time.0.weight -> log10_decay_time
            parts = key.split(".")
            if len(parts) >= 4:
                head_name = parts[2]
                # Get number of outputs from weight shape
                num_outputs = state_dict[key].shape[0]
                heads_config[head_name] = num_outputs

    log.info(f"Extracted heads config from checkpoint: {heads_config}")

    # Load configuration
    cfg = OmegaConf.load(hydra_config_path)

    # Update net config with actual heads from checkpoint
    if heads_config:
        cfg.model.net.heads_config = heads_config
        # Pass output_mode and parameter_names to net for proper head creation
        if hasattr(cfg.model, 'output_mode'):
            cfg.model.net.output_mode = cfg.model.output_mode
        # Extract parameter names from heads_config
        cfg.model.net.parameter_names = list(heads_config.keys())

    # Instantiate model using Hydra (this creates net, optimizer, scheduler from config)
    model = instantiate(cfg.model)

    # Load checkpoint weights into the model
    model.load_state_dict(state_dict)

    return model


def _load_model_from_checkpoint(
    ckpt_path: str, datamodule: LightningDataModule, device: str
) -> LightningModule:
    """Load model from checkpoint with proper architecture reconstruction."""
    log.info(f"Loading model from checkpoint: {ckpt_path}")

    # Step 1: Try Lightning's built-in checkpoint loading (fast path)
    try:
        from src.models.vimh_lit_module import VIMHLitModule

        model: LightningModule = VIMHLitModule.load_from_checkpoint(
            ckpt_path, map_location=device, strict=False
        )
        log.info("Successfully loaded VIMH model using Lightning's built-in method")
        return model
    except Exception as e:
        log.warning(f"Lightning checkpoint loading failed: {e}")

    # Step 2: Try loading from Hydra config file (reliable, standard practice)
    try:
        model = _load_from_hydra_config(ckpt_path, device)
        log.info("Successfully loaded model from Hydra config")
        return model
    except Exception as e:
        log.warning(f"Hydra config loading failed: {e}")
        log.info("Attempting weight-based architecture inference (last resort)...")

    # Step 3: Manual reconstruction by inferring from weights (brittle fallback)
    return _reconstruct_model_manually(ckpt_path, datamodule, device)


def _reconstruct_model_manually(
    ckpt_path: str, datamodule: LightningDataModule, device: str
) -> LightningModule:
    """Manually reconstruct model when Lightning's method fails."""
    # Get heads configuration from datamodule (for fallback only)
    if not hasattr(datamodule, "data_train") or datamodule.data_train is None:
        raise CheckpointError("Cannot reconstruct model: datamodule not properly configured")

    dataset_heads_config = datamodule.data_train.get_heads_config()

    # Get dataset metadata for architecture inference
    dataset_metadata = getattr(datamodule, "_saved_dataset_metadata", {})

    # Use the new architecture reconstructor
    reconstructor_config = {
        "default_dropout": 0.5,
        "default_patch_size": 4,
        "default_embed_dim": 64,
    }
    reconstructor = CheckpointArchitectureReconstructor(reconstructor_config)

    # Load checkpoint for hyperparameters first
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    output_mode = hyper_parameters.get("output_mode", "classification")

    # Extract heads config directly from checkpoint state_dict (CORRECT approach)
    # This ensures we only use heads that actually exist in the trained model
    actual_heads_config = {}
    state_dict = checkpoint.get("state_dict", {})

    # Find all head weight keys in the checkpoint
    head_weight_keys = [k for k in state_dict.keys() if k.startswith("net.heads.") and k.endswith(".0.weight")]

    for weight_key in head_weight_keys:
        # Extract head name from key: "net.heads.{head_name}.0.weight"
        head_name = weight_key.split(".")[2]
        weight_tensor = state_dict[weight_key]
        actual_output_size = weight_tensor.shape[0]  # First dimension is output size
        actual_heads_config[head_name] = actual_output_size
        log.info(f"Found head '{head_name}' with {actual_output_size} outputs in checkpoint")

    if not actual_heads_config:
        log.error("No heads found in checkpoint state_dict - checkpoint may be corrupted")
        sys.exit(1)

    log.info(f"Reconstructing with actual checkpoint heads config: {actual_heads_config}")

    try:
        net, arch_config = reconstructor.reconstruct_from_checkpoint(
            ckpt_path, actual_heads_config, dataset_metadata
        )
        log.info(f"Successfully reconstructed {arch_config.architecture_type} model")

        # Create the complete model
        from src.models.vimh_lit_module import VIMHLitModule

        # Create criteria based on the actual head architecture
        criteria = {}
        for head_name, num_outputs in actual_heads_config.items():
            if num_outputs == 1:
                # Single output = regression
                from torch.nn import MSELoss

                criteria[head_name] = MSELoss()
            else:
                # Multiple outputs = classification
                from torch.nn import CrossEntropyLoss

                criteria[head_name] = CrossEntropyLoss()

        log.info(
            f"Using criteria for heads: {[(k, type(v).__name__) for k, v in criteria.items()]}"
        )

        model = VIMHLitModule(
            net=net,
            optimizer=hyper_parameters.get("optimizer"),
            scheduler=None,
            loss_weights=hyper_parameters.get("loss_weights", {}),
            compile=hyper_parameters.get("compile", False),
            criteria=criteria,
            auto_configure_from_dataset=False,
            output_mode=output_mode,
        )

        # Load compatible weights
        _load_compatible_weights(model, checkpoint["state_dict"])

        return model

    except (ArchitectureInferenceError, MissingCheckpointDataError) as e:
        log.error(f"Model reconstruction failed: {e}")
        raise CheckpointError(f"Could not reconstruct model from checkpoint: {e}")


def _load_compatible_weights(
    model: LightningModule, checkpoint_state_dict: Dict[str, torch.Tensor]
) -> None:
    """Load only compatible weights from checkpoint state dict."""
    model_state_dict = model.state_dict()

    compatible_weights = {}
    incompatible_weights = []

    for key, value in checkpoint_state_dict.items():
        if key in model_state_dict:
            if model_state_dict[key].shape == value.shape:
                compatible_weights[key] = value
            else:
                incompatible_weights.append(
                    f"{key} (checkpoint: {value.shape}, model: {model_state_dict[key].shape})"
                )
        else:
            incompatible_weights.append(f"{key} (missing in model)")

    # Debug: Show some model keys for troubleshooting
    log.debug(f"Model state dict has {len(model_state_dict)} keys")
    head_keys = [k for k in model_state_dict.keys() if "head" in k.lower()]
    if head_keys:
        log.debug(f"Model head keys: {head_keys}")
    else:
        log.warning("No head keys found in model state dict!")

    # Report incompatible weights
    if incompatible_weights:
        log.warning(f"Found {len(incompatible_weights)} incompatible/missing weights:")
        for weight_info in incompatible_weights[:5]:  # Show first 5
            log.warning(f"  - {weight_info}")
        if len(incompatible_weights) > 5:
            log.warning(f"  ... and {len(incompatible_weights) - 5} more")

    # Load compatible weights
    if compatible_weights:
        model.load_state_dict(compatible_weights, strict=False)
        log.info(f"Loaded {len(compatible_weights)} compatible weights from checkpoint")
    else:
        raise CheckpointError("No compatible weights found in checkpoint")

    # Expose load summary for the interactive UI
    try:
        model._ae_weight_load_info = {
            "compatible": len(compatible_weights),
            "incompatible": len(incompatible_weights),
            "loaded_keys": list(compatible_weights.keys())[:50],  # cap for brevity
        }
    except Exception:
        pass


def _print_evaluation_header(ckpt_path: str, evaluator: AudioReconstructionEvaluator) -> None:
    """Print evaluation header with checkpoint and dataset information."""
    import os
    from pathlib import Path

    print("\n" + "=" * 80)
    print("🎵 AUDIO RECONSTRUCTION EVALUATION")
    print("=" * 80)

    # Checkpoint path and basic info
    ckpt_name = Path(ckpt_path).name
    ckpt_dir = Path(ckpt_path).parent
    print(f"📁 Checkpoint: {ckpt_name}")
    print(f"   Path: {ckpt_path}")

    # Extract training run information from path
    # Format: logs/train/runs/YYYY-MM-DD_HH-MM-SS/checkpoints/best.ckpt
    path_parts = Path(ckpt_path).parts
    run_info = None
    for i, part in enumerate(path_parts):
        if part == "runs" and i + 1 < len(path_parts):
            run_info = path_parts[i + 1]
            break

    if run_info:
        print(f"🏃 Training Run: {run_info}")

    # Load checkpoint to get additional metadata
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Training dataset information
        vimh_data = checkpoint.get("VIMHDataModule", {})
        dataset_metadata = vimh_data.get("dataset_metadata", {})

        if dataset_metadata:
            dataset_name = dataset_metadata.get("dataset_name", "Unknown")
            print(f"📊 Training Dataset: {dataset_name}")

            # Dataset statistics
            if "num_samples" in dataset_metadata:
                print(f"   Samples: {dataset_metadata['num_samples']}")
            if "sample_rate" in dataset_metadata:
                print(f"   Sample Rate: {dataset_metadata['sample_rate']} Hz")
            if "duration" in dataset_metadata:
                print(f"   Duration: {dataset_metadata['duration']} s")
            if "parameter_names" in dataset_metadata:
                param_names = dataset_metadata["parameter_names"]
                print(f"   Parameters: {', '.join(param_names)} ({len(param_names)} total)")

        # Model information from hyperparameters
        hyper_params = checkpoint.get("hyper_parameters", {})
        if hyper_params:
            output_mode = hyper_params.get("output_mode", "Unknown")
            print(f"🤖 Model Output Mode: {output_mode}")

            # Architecture metadata if available
            arch_metadata = hyper_params.get("architecture_metadata", {})
            if arch_metadata:
                arch_type = arch_metadata.get("type", "Unknown")
                print(f"   Architecture: {arch_type}")

                # Architecture-specific details
                if arch_type == "CNN":
                    channels = arch_metadata.get("conv1_channels", "?")
                    fc_hidden = arch_metadata.get("fc_hidden", "?")
                    print(f"   Details: {channels} conv channels, {fc_hidden} FC hidden")
                elif arch_type == "ViT":
                    embed_dim = arch_metadata.get("embed_dim", "?")
                    n_layers = arch_metadata.get("n_layers", "?")
                    n_heads = arch_metadata.get("n_attention_heads", "?")
                    print(f"   Details: {embed_dim}d embed, {n_layers} layers, {n_heads} heads")

    except Exception as e:
        log.warning(f"Could not read checkpoint metadata: {e}")

    # Current evaluation dataset info
    dataset_info = evaluator.dataset_info
    test_size = len(evaluator.test_dataset)
    print(f"🧪 Evaluation Dataset: {test_size} test samples")

    print("=" * 80 + "\n")


def _run_evaluation(
    cfg: DictConfig, evaluator: AudioReconstructionEvaluator, ckpt_path: str
) -> Tuple[Dict[str, Any], Optional[None]]:
    """Run the actual evaluation based on configuration."""
    # Extract and display checkpoint information
    _print_evaluation_header(ckpt_path, evaluator)

    num_samples = cfg.get("num_samples", 5)
    interactive = cfg.get("interactive", False)
    save_audio = cfg.get("save_audio", False)
    output_dir = cfg.get("output_dir", "audio_eval_results")

    if interactive:
        log.info("Launching interactive evaluator...")
        interactive_eval = InteractiveAudioEvaluator(evaluator)
        plt.show()
        return {"message": "Interactive evaluation launched"}, None
    else:
        return _run_batch_evaluation(evaluator, num_samples, save_audio, output_dir)


def _run_batch_evaluation(
    evaluator: AudioReconstructionEvaluator, num_samples: int, save_audio: bool, output_dir: str
) -> Tuple[Dict[str, Any], Optional[None]]:
    """Run batch evaluation on multiple samples."""
    log.info(f"Evaluating {num_samples} samples...")
    results = []

    for i in range(min(num_samples, len(evaluator.test_dataset))):
        log.info(f"Evaluating sample {i+1}/{num_samples}")
        result = evaluator.evaluate_sample(
            i, plot=False, save_audio=save_audio, output_dir=output_dir
        )
        results.append(result)

    # Compute aggregate statistics
    aggregate_metrics = _compute_aggregate_metrics(results)

    log.info("Aggregate metrics:")
    for metric, value in aggregate_metrics.items():
        log.info(f"  {metric}: {value:.6f}")

    # Print publication-ready results table
    _print_results_table(aggregate_metrics, len(results))

    return {
        "individual_results": results,
        "aggregate_metrics": aggregate_metrics,
        "num_samples_evaluated": len(results),
    }, None


def _compute_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute aggregate metrics from individual evaluation results."""
    if not results:
        return {}

    all_metrics = [r["audio_metrics"] for r in results]
    aggregate_metrics = {}

    for metric_name in all_metrics[0].keys():
        values = [m[metric_name] for m in all_metrics if np.isfinite(m[metric_name])]
        if values:
            aggregate_metrics[f"mean_{metric_name}"] = float(np.mean(values))
            aggregate_metrics[f"std_{metric_name}"] = float(np.std(values))

    return aggregate_metrics


def _print_results_table(aggregate_metrics: Dict[str, float], num_samples: int) -> None:
    """Print publication-ready results table comparing to PNP baseline.

    Args:
        aggregate_metrics: Dictionary with mean_ and std_ prefixed metrics
        num_samples: Number of samples evaluated
    """
    print("\n" + "="*80)
    print("📊 PUBLICATION-READY RESULTS TABLE (PNP Format)")
    print("="*80)

    # Extract mean ± std for key metrics
    def get_stat(name: str) -> tuple:
        """Get (mean, std) for a metric, return (None, None) if missing."""
        mean_key = f"mean_{name}"
        std_key = f"std_{name}"
        if mean_key in aggregate_metrics and std_key in aggregate_metrics:
            return (aggregate_metrics[mean_key], aggregate_metrics[std_key])
        return (None, None)

    mae_mean, mae_std = get_stat("mae")
    mse_mean, mse_std = get_stat("mse")
    mss_mean, mss_std = get_stat("mss_distance")
    snr_mean, snr_std = get_stat("snr_db")
    corr_mean, corr_std = get_stat("correlation")

    print(f"\nEvaluated on {num_samples} test samples")
    print(f"\n{'Metric':<25} {'Mean ± Std':<25} {'Format for Paper'}")
    print("-" * 80)

    if mss_mean is not None:
        print(f"{'MSS Distance':<25} {mss_mean:>8.2f} ± {mss_std:<8.2f}    {mss_mean:.2f} ± {mss_std:.2f}")

    if mae_mean is not None:
        print(f"{'MAE (waveform)':<25} {mae_mean:>8.6f} ± {mae_std:<8.6f}    {mae_mean:.6f} ± {mae_std:.6f}")

    if mse_mean is not None:
        print(f"{'MSE (waveform)':<25} {mse_mean:>8.6f} ± {mse_std:<8.6f}    {mse_mean:.6f} ± {mse_std:.6f}")

    if snr_mean is not None:
        print(f"{'SNR (dB)':<25} {snr_mean:>8.2f} ± {snr_std:<8.2f}    {snr_mean:.2f} ± {snr_std:.2f}")

    if corr_mean is not None:
        print(f"{'Correlation':<25} {corr_mean:>8.4f} ± {corr_std:<8.4f}    {corr_mean:.4f} ± {corr_std:.4f}")

    print("\n" + "="*80)
    print("💡 LaTeX table format:")
    print("="*80)

    # Generate LaTeX table row
    latex_parts = []
    if mss_mean is not None:
        latex_parts.append(f"${mss_mean:.2f} \\pm {mss_std:.2f}$")
    if mae_mean is not None:
        latex_parts.append(f"${mae_mean:.6f} \\pm {mae_std:.6f}$")
    if mse_mean is not None:
        latex_parts.append(f"${mse_mean:.6f} \\pm {mse_std:.6f}$")

    if latex_parts:
        print("NSMT (this work) & " + " & ".join(latex_parts) + " \\\\")

    print("\n" + "="*80)
    print("📋 Comparison with PNP baseline (from paper):")
    print("="*80)
    print("Method                   MSS Distance")
    print("-" * 40)
    print(f"PNP (Han et al. 2024)    0.78 ± 0.03")
    if mss_mean is not None:
        print(f"NSMT (this work)         {mss_mean:.2f} ± {mss_std:.2f}")
        improvement = ((0.78 - mss_mean) / 0.78) * 100
        print(f"\nImprovement: {improvement:+.1f}% {'(better)' if improvement > 0 else '(worse)'}")
    print("="*80 + "\n")


def print_simple_help():
    """Print simple, user-friendly help message."""
    print("""
🎵 Audio Reconstruction Evaluation - Simple Usage Guide

BASIC USAGE:
    # Auto-discover latest checkpoint
    python src/audio_reconstruction_eval.py

    # Evaluate specific checkpoint
    python src/audio_reconstruction_eval.py ckpt_path=path/to/checkpoint.ckpt

REFERENCE CHECKPOINTS (in checkpoints/reference/):
    # 3-parameter detuning baseline
    python src/audio_reconstruction_eval.py \\
        ckpt_path=checkpoints/reference/cnn_medium_dsaw_3p_baseline_v1.ckpt

    # Wah model with auxiliary features
    python src/audio_reconstruction_eval.py \\
        ckpt_path=checkpoints/reference/convnext_medium_wah_avix_aux_v1.ckpt

    # Wah model (AVIX only)
    python src/audio_reconstruction_eval.py \\
        ckpt_path=checkpoints/reference/convnext_medium_wah_avix_v1.ckpt

COMMON OPTIONS:
    ckpt_path=PATH              Path to checkpoint file (default: auto-discover latest)
    num_samples=N               Number of samples to evaluate (default: 5)
    interactive=true|false      Interactive widget vs batch mode (default: true)
    save_audio=true|false       Save WAV files (default: true)
    output_dir=PATH             Output directory (default: audio_eval_results)

EXAMPLES:
    # Batch mode, 100 samples
    python src/audio_reconstruction_eval.py \\
        ckpt_path=checkpoints/reference/cnn_medium_dsaw_3p_baseline_v1.ckpt \\
        interactive=false num_samples=100

    # Quick check of 1 sample, no audio saving
    python src/audio_reconstruction_eval.py \\
        ckpt_path=checkpoints/reference/cnn_medium_dsaw_3p_baseline_v1.ckpt \\
        num_samples=1 save_audio=false

INTERACTIVE MODE:
    - Navigate: Left/Right arrow keys, Prev/Next buttons, slider
    - Play audio: "Play True" / "Play Pred" buttons
    - Save: "Save PNG", "Save True WAV", "Save Pred WAV" buttons
    - Keyboard: s=save figure, t=save true WAV, p=save pred WAV

MORE INFO:
    --help-hydra                Show full Hydra configuration details
    --help-full                 Show comprehensive documentation

    python -c "import src.audio_reconstruction_eval; help(src.audio_reconstruction_eval)"
        View detailed documentation with all options and examples
""")


@hydra.main(version_base="1.3", config_path="../configs", config_name="audio_eval")
def main(cfg: DictConfig) -> None:
    """Main entry point for audio reconstruction evaluation."""

    # Show usage info
    if cfg.get("ckpt_path") is None:
        log.info("🎵 Audio Reconstruction Evaluation")
        log.info("Auto-discovering latest checkpoint...")
        log.info("")
        log.info("💡 Tip: Run with ckpt_path=path/to/checkpoint.ckpt to use specific checkpoint")
        log.info("   Example: ckpt_path=checkpoints/reference/cnn_medium_dsaw_3p_baseline_v1.ckpt")
        log.info("   Use --help for simple usage guide")
        log.info("")

    extras(cfg)
    evaluate_audio_reconstruction(cfg)


if __name__ == "__main__":
    # Intercept help flags before Hydra processes them
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        print_simple_help()
        sys.exit(0)
    elif len(sys.argv) > 1 and sys.argv[1] in ["--help-full"]:
        print(__doc__)
        sys.exit(0)
    elif len(sys.argv) > 1 and sys.argv[1] in ["--help-hydra"]:
        # Let Hydra handle this by continuing to main()
        # But remove our custom flag so Hydra sees --help
        sys.argv[1] = "--help"

    main()
