"""
Audio Reconstruction Evaluation for Neural Spectral Modeling

This script evaluates trained models by comparing original and reconstructed audio:
1. Loads a trained checkpoint and test dataset
2. Runs inference to get predicted synthesis parameters
3. Synthesizes audio using both true and predicted parameters
4. Provides side-by-side comparisons with audio playback and visual analysis
5. Computes perceptual metrics for synthesis quality assessment

Usage:
    python src/audio_reconstruction_eval.py ckpt_path=/path/to/checkpoint.ckpt data=vimh_16kdss
"""

from typing import Any, Dict, List, Tuple, Optional
import os
import sys
from pathlib import Path
import json

import hydra
import rootutils
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
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
from src.utils.synth_utils import SimpleSawSynth, SpectrogramProcessor, check_params

log = RankedLogger(__name__, rank_zero_only=True)


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
        f"{base_dir}/*/checkpoints/*.ckpt"
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
    
    def __init__(self, 
                 model: LightningModule, 
                 datamodule: LightningDataModule,
                 device: str = "cpu"):
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
        
        # Setup test dataset
        self.datamodule.setup("test")
        self.test_dataset = self.datamodule.data_test
        
        # Get dataset metadata from datamodule (includes saved checkpoint metadata)
        self.dataset_info = self.datamodule.get_dataset_info()
        self.sample_rate = self.dataset_info.get("sample_rate", 8000)
        self.duration = self.dataset_info.get("duration", 1.0)
        
        # Initialize synthesizer
        self.synth = SimpleSawSynth(sample_rate=self.sample_rate)
        
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
            mel_config=mel_config
        )
        
        # Debug: Print spectrogram processor configuration (can be removed)
        # log.info(f"SpectrogramProcessor config:")
        # log.info(f"  Sample rate: {self.sample_rate} Hz") 
        # log.info(f"  Dimensions: {height}x{width}")
        # log.info(f"  STFT config: {stft_config}")
        # log.info(f"  MEL config: {mel_config}")
        
        # Parameter mappings for denormalization (no fallbacks)
        self.param_mappings = self.dataset_info.get("parameter_mappings")
        self.param_names = self.dataset_info.get("parameter_names")
        if not self.param_mappings or not self.param_names:
            print("Dataset is missing parameter_mappings or parameter_names; aborting.")
            sys.exit(1)
        
        log.info(f"Initialized evaluator with {len(self.test_dataset)} test samples")
        log.info(f"Sample rate: {self.sample_rate} Hz, Duration: {self.duration}s")
        log.info(f"Available parameters: {self.param_names}")
        
        # Debug dataset metadata
        log.info(f"Dataset info keys: {list(self.dataset_info.keys())}")
        if self.param_mappings:
            log.info(f"Parameter mappings: {list(self.param_mappings.keys())}")
        
        # Check if model output mode matches expectations
        if hasattr(self.model, 'output_mode'):
            log.info(f"Model output mode: {self.model.output_mode}")
        if hasattr(self.model, 'net') and hasattr(self.model.net, 'heads_config'):
            log.info(f"Model heads config: {self.model.net.heads_config}")

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
                    if getattr(self.model, 'output_mode', 'classification') == 'regression':
                        v = float(logits.squeeze().item())
                    else:
                        v = int(torch.argmax(logits, dim=-1).item())
                    values[head_name].append(v)

        # Compute simple stats
        report: Dict[str, Any] = {}
        print("\nPrediction diversity over", n, "sample(s):")
        for name in self.param_names:
            vals = values.get(name, [])
            if getattr(self.model, 'output_mode', 'classification') == 'regression':
                std = float(np.std(vals)) if vals else 0.0
                report[name] = {"std": std, "min": float(min(vals)) if vals else None, "max": float(max(vals)) if vals else None}
                status = "⚠️ low variance" if std < 1e-3 else "ok"
                print(f"  - {name}: std={std:.6f} [{status}] range=({report[name]['min']}, {report[name]['max']})")
            else:
                uniques = sorted(list(set(vals))) if vals else []
                report[name] = {"unique": len(uniques), "values": uniques[:15]}
                status = "⚠️ degenerate" if len(uniques) <= 1 else "ok"
                preview = ", ".join(map(str, report[name]["values"])) + (" …" if len(uniques) > 15 else "")
                print(f"  - {name}: {len(uniques)} unique [{status}] → [{preview}]")

        return report
    
    def denormalize_parameters(self, predicted_params: Dict[str, torch.Tensor]) -> Dict[str, float]:
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
            if getattr(self.model, 'output_mode', 'classification') == "regression":
                # Direct regression output (should be in [0,1])
                normalized_value = float(pred_tensor.item())
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
                            print(f"Warning: parameter '{param_name}' (max-min)/step = {num_floats} not integer; rounding to {num_steps}")
                        num_classes = int(num_steps) + 1
                        mapping["num_classes"] = num_classes  # cache for later
                    else:
                        print(f"Parameter mapping for '{param_name}' missing 'num_classes' and 'step'; aborting.")
                        sys.exit(1)
                class_idx = int(pred_tensor.item())
                normalized_value = class_idx / (num_classes - 1)
                
            # Denormalize to actual parameter range
            actual_value = param_min + normalized_value * (param_max - param_min)
            denorm_params[param_name] = actual_value
            
        return denorm_params
    
    def get_true_parameters(self, sample_idx: int) -> Dict[str, float]:
        """
        Get the true synthesis parameters for a dataset sample.
        
        Args:
            sample_idx: Index of the sample in the test dataset
            
        Returns:
            Dictionary of true parameter values
        """
        # Get metadata for this sample
        sample_metadata = self.test_dataset._get_sample_metadata(sample_idx)
        
        # Debug: Print metadata structure for first few samples
        if sample_idx < 3:
            print(f"🔍 Sample {sample_idx} metadata keys: {list(sample_metadata.keys())}")
            for key, value in sample_metadata.items():
                if isinstance(value, dict) and 'actual_value' in value:
                    print(f"   {key}: actual_value = {value['actual_value']}")
        
        # Extract actual parameter values
        true_params = {}
        for param_name in self.param_names:
            param_info_key = f"{param_name}_info"
            if param_info_key in sample_metadata:
                actual_value = sample_metadata[param_info_key]["actual_value"]
                true_params[param_name] = actual_value
            else:
                raise ValueError(f"Could not find true value for parameter {param_name} in sample metadata")
                
        return true_params
    
    def run_inference(self, sample_idx: int) -> Tuple[Dict[str, float], Dict[str, float], torch.Tensor]:
        """
        Run model inference on a test sample.
        
        Args:
            sample_idx: Index of the sample to evaluate
            
        Returns:
            Tuple of (predicted_params, true_params, input_spectrogram)
        """
        # Check for model performance issues on first few samples
        if sample_idx < 3 and hasattr(self, '_check_model_predictions'):
            self._check_model_predictions = False  # Only show once
            print("🔍 Checking model prediction quality...")
            
        # Track predictions to detect if model always predicts the same values
        if not hasattr(self, '_prediction_tracker'):
            self._prediction_tracker = {}
            self._samples_checked = 0
        # Get the test sample
        sample_data = self.test_dataset[sample_idx]
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

            processed_predictions = {
                head_name: torch.argmax(logits, dim=-1).squeeze(0)
                for head_name, logits in raw_predictions.items()
            }
        
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
                    print(f"⚠️  WARNING: Model always predicts the same value for {param_name}: {values[0]:.4f}")
                    print(f"   This suggests the model was not trained properly or weights are incorrect.")
                    print(f"   Expected behavior: predictions should vary across different input samples.")
        
        # Get true parameters
        true_params = self.get_true_parameters(sample_idx)
        
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
        if hasattr(self, 'dataset_info') and 'fixed_parameters' in self.dataset_info:
            fixed_params = self.dataset_info['fixed_parameters']
            for param_name, param_info in fixed_params.items():
                if param_name not in complete_params:
                    complete_params[param_name] = param_info['value']
                    log.debug(f"Added fixed parameter {param_name} = {param_info['value']}")
        
        # Set duration from dataset info 
        complete_params["duration"] = self.duration
        
        # Debug: Print complete parameter set for verification (can be disabled)
        # log.info(f"Complete synthesis parameters: {complete_params}")
        
        # Check that we have required parameters
        check_params(complete_params, "note_number", "note_velocity", "duration", "log10_decay_time")
        
        # Generate audio - fail fast, no fallbacks
        audio = self.synth.generate_audio(complete_params)
        return audio
    
    def compute_audio_metrics(self, true_audio: np.ndarray, pred_audio: np.ndarray) -> Dict[str, float]:
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
        
        # Mean Squared Error
        metrics["mse"] = float(np.mean((true_audio - pred_audio) ** 2))
        
        # Root Mean Squared Error
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
        
        # Signal-to-Noise Ratio
        signal_power = np.mean(true_audio ** 2)
        noise_power = np.mean((true_audio - pred_audio) ** 2)
        if noise_power > 0:
            metrics["snr_db"] = float(10 * np.log10(signal_power / noise_power))
        else:
            metrics["snr_db"] = float('inf')
        
        # Pearson correlation
        if np.std(true_audio) > 0 and np.std(pred_audio) > 0:
            correlation, _ = pearsonr(true_audio, pred_audio)
            metrics["correlation"] = float(correlation)
        else:
            metrics["correlation"] = 0.0
        
        # Cross-correlation maximum (time alignment)
        if len(true_audio) > 1 and len(pred_audio) > 1:
            xcorr = correlate(true_audio, pred_audio, mode='full')
            metrics["max_xcorr"] = float(np.max(np.abs(xcorr)) / (np.linalg.norm(true_audio) * np.linalg.norm(pred_audio)))
        else:
            metrics["max_xcorr"] = 0.0
            
        return metrics
    
    def evaluate_sample(self, sample_idx: int, plot: bool = True, save_audio: bool = False, 
                       output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a single sample with complete analysis.
        
        Args:
            sample_idx: Index of sample to evaluate
            plot: Whether to create visualization plots
            save_audio: Whether to save audio files
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary containing all evaluation results
        """
        log.info(f"Evaluating sample {sample_idx}")
        
        # Run inference
        predicted_params, true_params, input_spec = self.run_inference(sample_idx)
        
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
                    "relative_error": abs(pred_val - true_val) / abs(true_val) if true_val != 0 else float('inf')
                }
        
        results = {
            "sample_idx": sample_idx,
            "predicted_params": predicted_params,
            "true_params": true_params,
            "param_errors": param_errors,
            "audio_metrics": audio_metrics,
            "true_audio": true_audio,
            "pred_audio": pred_audio,
            "input_spectrogram": input_spec.numpy()
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
        
        # Remove channel dimension and normalize for plotting
        if len(input_spec.shape) == 3:
            if input_spec.shape[0] == 1:  # (1, H, W) format - channels first
                input_spec = input_spec[0, :, :]  # Remove channel dimension: (1, H, W) -> (H, W)
            else:  # (H, W, 1) format - channels last  
                input_spec = input_spec[:, :, 0]  # Remove channel dimension: (H, W, 1) -> (H, W)
        
        # Keep input spectrogram in floating-point format to match true spectrogram
        # Both should be displayed with the same scale for proper comparison
        # No conversion needed - input_spec is already in proper float format
        
        # Time axis
        t = np.arange(len(true_audio)) / self.sample_rate
        
        # 1. Input spectrogram (from dataset)
        plt.subplot(3, 3, 1)
        plt.imshow(input_spec, aspect='auto', origin='lower', cmap='viridis')
        plt.title(f"Input Spectrogram (Sample {results['sample_idx']})")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        
        # 2. True audio waveform
        plt.subplot(3, 3, 2)
        plt.plot(t, true_audio, 'b-', alpha=0.7, label='True')
        plt.title("True Audio Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        
        # 3. Predicted audio waveform
        plt.subplot(3, 3, 3)
        plt.plot(t, pred_audio, 'r-', alpha=0.7, label='Predicted')
        plt.title("Predicted Audio Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        
        # 4. Waveform overlay
        plt.subplot(3, 3, 4)
        plt.plot(t, true_audio, 'b-', alpha=0.7, label='True')
        plt.plot(t, pred_audio, 'r--', alpha=0.7, label='Predicted')
        plt.title("Waveform Comparison")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. True audio spectrogram (re-synthesized)
        plt.subplot(3, 3, 5)
        # Use the same parameters that were used for original dataset generation
        true_spec, _, _ = self.spec_processor.audio_to_spectrogram(results["true_params"], true_audio)
        
        # The spectrograms should now match since we're using the same parameters and processing
        
        plt.imshow(true_spec, aspect='auto', origin='lower', cmap='viridis')
        plt.title("True Synthesis Spectrogram")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        
        # 6. Predicted audio spectrogram
        plt.subplot(3, 3, 6)
        pred_spec, _, _ = self.spec_processor.audio_to_spectrogram({}, pred_audio)
        plt.imshow(pred_spec, aspect='auto', origin='lower', cmap='viridis')
        plt.title("Predicted Synthesis Spectrogram")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        
        # 7. Parameter comparison (normalized 0-1)
        plt.subplot(3, 3, 7)
        param_names = list(results["param_errors"].keys())
        # Normalize using dataset parameter mappings
        norm_true_vals = []
        norm_pred_vals = []
        for p in param_names:
            mapping = self.param_mappings.get(p, {"min": 0.0, "max": 1.0})
            pmin = mapping.get("min", 0.0)
            pmax = mapping.get("max", 1.0)
            prange = max(1e-12, pmax - pmin)
            tval = results["param_errors"][p]["true"]
            pval = results["param_errors"][p]["predicted"]
            norm_true_vals.append(np.clip((tval - pmin) / prange, 0.0, 1.0))
            norm_pred_vals.append(np.clip((pval - pmin) / prange, 0.0, 1.0))

        x = np.arange(len(param_names))
        width = 0.35
        bars_true = plt.bar(x - width/2, norm_true_vals, width, label='True', alpha=0.7)
        bars_pred = plt.bar(x + width/2, norm_pred_vals, width, label='Predicted', alpha=0.7)
        # Add natural-value labels on top of bars
        def _fmt(v: float) -> str:
            av = abs(v)
            if av >= 1000 or (av > 0 and av < 1e-3):
                return f"{v:.2e}"
            if av >= 100:
                return f"{v:.0f}"
            if av >= 10:
                return f"{v:.1f}"
            return f"{v:.3f}"
        true_vals_nat = [results["param_errors"][p]["true"] for p in param_names]
        pred_vals_nat = [results["param_errors"][p]["predicted"] for p in param_names]
        for i, (bt, bp) in enumerate(zip(bars_true, bars_pred)):
            plt.text(bt.get_x() + bt.get_width()/2, bt.get_height() + 0.02,
                     _fmt(true_vals_nat[i]), ha='center', va='bottom', fontsize=8, color='tab:blue')
            plt.text(bp.get_x() + bp.get_width()/2, bp.get_height() + 0.02,
                     _fmt(pred_vals_nat[i]), ha='center', va='bottom', fontsize=8, color='tab:orange')
        plt.ylim(0, 1.1)
        plt.xlabel('Parameters')
        plt.ylabel('Normalized Value [0,1]')
        plt.title('Parameter Comparison (Normalized)')
        plt.xticks(x, param_names, rotation=45)
        plt.legend()
        
        # 8. Audio metrics
        plt.subplot(3, 3, 8)
        metrics = results["audio_metrics"]
        metric_names = list(metrics.keys())
        metric_vals = list(metrics.values())
        
        # Filter out infinite values for plotting
        finite_metrics = [(name, val) for name, val in zip(metric_names, metric_vals) 
                         if np.isfinite(val)]
        if finite_metrics:
            names, vals = zip(*finite_metrics)
            plt.bar(names, vals)
            plt.title('Audio Quality Metrics')
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, 'No finite metrics', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Audio Quality Metrics')
        
        # 9. Error difference waveform
        plt.subplot(3, 3, 9)
        min_len = min(len(true_audio), len(pred_audio))
        error = true_audio[:min_len] - pred_audio[:min_len]
        plt.plot(t[:min_len], error, 'g-', alpha=0.7)
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
                    if 'true' in title and 'waveform' in title:
                        audio = results["true_audio"]
                        print(f"🎵 Playing true audio waveform (Sample {results['sample_idx']})")
                    elif 'predicted' in title and 'waveform' in title:
                        audio = results["pred_audio"] 
                        print(f"🎵 Playing predicted audio waveform (Sample {results['sample_idx']})")
                    elif 'true' in title and 'spectrogram' in title:
                        audio = results["true_audio"]
                        print(f"🎵 Playing true audio from spectrogram (Sample {results['sample_idx']})")
                    elif 'predicted' in title and 'spectrogram' in title:
                        audio = results["pred_audio"]
                        print(f"🎵 Playing predicted audio from spectrogram (Sample {results['sample_idx']})")
                    elif 'comparison' in title:
                        # For comparison plot, play true audio
                        audio = results["true_audio"]
                        print(f"🎵 Playing true audio from comparison (Sample {results['sample_idx']})")
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
        fig.canvas.mpl_connect('button_press_event', on_click)
        
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
            "audio_metrics": results["audio_metrics"]
        }
        
        metadata_path = os.path.join(output_dir, f"sample_{sample_idx}_metadata.json")
        with open(metadata_path, 'w') as f:
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
        
        # Create interactive widget
        self.create_widget()
    
    def create_widget(self):
        """Create the interactive matplotlib widget."""
        # Create figure and explicit layout: 2x3 plot grid on top, control strip below
        self.fig = plt.figure(figsize=(18, 10))
        outer_gs = self.fig.add_gridspec(nrows=3, ncols=3,
                                         height_ratios=[1.0, 1.0, 0.18],
                                         hspace=0.35, wspace=0.3)

        # Axes matrix for the six plots
        import numpy as _np
        self.axes = _np.empty((2, 3), dtype=object)
        for r in range(2):
            for c in range(3):
                self.axes[r, c] = self.fig.add_subplot(outer_gs[r, c])

        # Controls row spanning all columns
        controls_gs = outer_gs[2, :].subgridspec(1, 5, width_ratios=[6, 1, 1, 1, 1])

        # Slider occupies most of the bottom strip
        ax_slider = self.fig.add_subplot(controls_gs[0, 0])
        # Buttons arranged to the right
        ax_prev = self.fig.add_subplot(controls_gs[0, 1])
        ax_next = self.fig.add_subplot(controls_gs[0, 2])
        ax_play_true = self.fig.add_subplot(controls_gs[0, 3])
        ax_play_pred = self.fig.add_subplot(controls_gs[0, 4])

        # Reduce visual clutter in control axes
        for ax in [ax_slider, ax_prev, ax_next, ax_play_true, ax_play_pred]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        # Create interactive widgets
        self.sample_slider = Slider(
            ax_slider, 'Sample', 0, len(self.evaluator.test_dataset) - 1,
            valinit=0, valfmt='%d'
        )
        self.sample_slider.on_changed(self.update_sample)
        
        self.btn_prev = Button(ax_prev, 'Prev')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_play_true = Button(ax_play_true, 'Play True')
        self.btn_play_pred = Button(ax_play_pred, 'Play Pred')
        
        self.btn_prev.on_clicked(self.prev_sample)
        self.btn_next.on_clicked(self.next_sample)
        self.btn_play_true.on_clicked(self.play_true_audio)
        self.btn_play_pred.on_clicked(self.play_pred_audio)
        
        # Add keyboard navigation
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
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
        print("   • Use slider: Jump to specific sample")
        print("   • Click 'Play True'/'Play Pred': Play audio for current sample")
    
    def on_key_press(self, event):
        """Handle keyboard navigation events."""
        if event.key == 'left':
            # Left arrow key - go to previous sample
            self.prev_sample(None)
        elif event.key == 'right':
            # Right arrow key - go to next sample
            self.next_sample(None)
    
    def update_sample(self, val):
        """Update current sample from slider."""
        self.current_sample = int(self.sample_slider.val)
        self.update_display()
    
    def prev_sample(self, event):
        """Go to previous sample."""
        old_sample = self.current_sample
        self.current_sample = max(0, self.current_sample - 1)
        print(f"Prev: {old_sample} -> {self.current_sample}")
        self.sample_slider.set_val(self.current_sample)
        # Note: slider.set_val() automatically triggers update_display() via on_changed callback
    
    def next_sample(self, event):
        """Go to next sample."""
        old_sample = self.current_sample
        self.current_sample = min(len(self.evaluator.test_dataset) - 1, self.current_sample + 1)
        print(f"Next: {old_sample} -> {self.current_sample}")
        self.sample_slider.set_val(self.current_sample)
        # Note: slider.set_val() automatically triggers update_display() via on_changed callback
        # So we don't need to call update_display() again here
    
    def update_display(self):
        """Update all plots for current sample."""
        print(f"🔄 Updating display for sample {self.current_sample}")
        
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Evaluate current sample
        self.current_results = self.evaluator.evaluate_sample(
            self.current_sample, plot=False
        )
        
        # Debug: Print some identifying info about this sample
        if self.current_results:
            print(f"   Sample {self.current_sample} - True audio range: [{self.current_results['true_audio'].min():.3f}, {self.current_results['true_audio'].max():.3f}]")
            print(f"   Sample {self.current_sample} - Pred audio range: [{self.current_results['pred_audio'].min():.3f}, {self.current_results['pred_audio'].max():.3f}]")
            
            # Print parameter values to check if they're actually changing
            true_params = self.current_results['true_params']
            pred_params = self.current_results['predicted_params']
            
            # Just show a few key parameters to see if they vary
            key_params = ['note_number', 'filter_cutoff', 'filter_resonance']
            for param in key_params:
                if param in true_params and param in pred_params:
                    print(f"   {param}: true={true_params[param]:.3f}, pred={pred_params[param]:.3f}")
                    
            # Also print ALL true params for a couple samples to see the full picture
            if self.current_sample < 3:
                print(f"   All true params for sample {self.current_sample}: {true_params}")
                print(f"   All pred params for sample {self.current_sample}: {pred_params}")
        
        # Get data
        true_audio = self.current_results["true_audio"]
        pred_audio = self.current_results["pred_audio"]
        input_spec = self.current_results["input_spectrogram"]
        
        # Remove channel dimension and normalize for plotting  
        if len(input_spec.shape) == 3:
            if input_spec.shape[0] == 1:  # (1, H, W) format - channels first
                input_spec = input_spec[0, :, :]  # Remove channel dimension: (1, H, W) -> (H, W)
            else:  # (H, W, 1) format - channels last  
                input_spec = input_spec[:, :, 0]  # Remove channel dimension: (H, W, 1) -> (H, W)
        
        # Keep input spectrogram in floating-point format to match true spectrogram
        # No conversion needed - input_spec is already in proper float format
        
        t = np.arange(len(true_audio)) / self.evaluator.sample_rate
        
        # Plot input spectrogram
        self.axes[0, 0].imshow(input_spec, aspect='auto', origin='lower', cmap='viridis')
        self.axes[0, 0].set_title(f"Input Spectrogram (Sample {self.current_sample})")
        
        # Plot waveforms
        self.axes[0, 1].plot(t, true_audio, 'b-', alpha=0.7, label='True')
        self.axes[0, 1].plot(t, pred_audio, 'r--', alpha=0.7, label='Predicted')
        self.axes[0, 1].set_title("Audio Waveforms")
        self.axes[0, 1].legend()
        self.axes[0, 1].grid(True, alpha=0.3)
        
        # Plot spectrograms
        true_params = self.current_results["true_params"]
        true_spec, _, _ = self.evaluator.spec_processor.audio_to_spectrogram(true_params, true_audio)
        pred_spec, _, _ = self.evaluator.spec_processor.audio_to_spectrogram({}, pred_audio)
        
        self.axes[0, 2].imshow(true_spec, aspect='auto', origin='lower', cmap='viridis')
        self.axes[0, 2].set_title("True Audio Spectrogram")
        
        self.axes[1, 0].imshow(pred_spec, aspect='auto', origin='lower', cmap='viridis')
        self.axes[1, 0].set_title("Predicted Audio Spectrogram")
        
        # Plot parameters (normalized to 0-1) with natural-value labels
        param_names = list(self.current_results["param_errors"].keys())
        if param_names:
            # Natural values
            true_vals = [self.current_results["param_errors"][p]["true"] for p in param_names]
            pred_vals = [self.current_results["param_errors"][p]["predicted"] for p in param_names]

            # Normalize using dataset parameter ranges
            norm_true = []
            norm_pred = []
            for p, tval, pval in zip(param_names, true_vals, pred_vals):
                mapping = self.evaluator.param_mappings.get(p, {"min": 0.0, "max": 1.0})
                pmin = mapping.get("min", 0.0)
                pmax = mapping.get("max", 1.0)
                prange = max(1e-12, pmax - pmin)
                norm_true.append(np.clip((tval - pmin) / prange, 0.0, 1.0))
                norm_pred.append(np.clip((pval - pmin) / prange, 0.0, 1.0))

            # Helper to format natural values compactly for labels
            def _fmt(v: float) -> str:
                av = abs(v)
                if av >= 1000 or (av > 0 and av < 1e-3):
                    return f"{v:.2e}"
                if av >= 100:
                    return f"{v:.0f}"
                if av >= 10:
                    return f"{v:.1f}"
                return f"{v:.3f}"

            x = np.arange(len(param_names))
            width = 0.35
            bars_true = self.axes[1, 1].bar(x - width/2, norm_true, width, label='True', alpha=0.7)
            bars_pred = self.axes[1, 1].bar(x + width/2, norm_pred, width, label='Predicted', alpha=0.7)

            # Add labels with natural values on top of each bar
            for i, (bt, bp) in enumerate(zip(bars_true, bars_pred)):
                self.axes[1, 1].text(bt.get_x() + bt.get_width()/2, bt.get_height() + 0.02,
                                      _fmt(true_vals[i]), ha='center', va='bottom', fontsize=8, color='tab:blue')
                self.axes[1, 1].text(bp.get_x() + bp.get_width()/2, bp.get_height() + 0.02,
                                      _fmt(pred_vals[i]), ha='center', va='bottom', fontsize=8, color='tab:orange')

            self.axes[1, 1].set_ylim(0, 1.1)
            self.axes[1, 1].set_ylabel('Normalized Value [0,1]')
            self.axes[1, 1].set_xticks(x)
            self.axes[1, 1].set_xticklabels(param_names, rotation=45)
            self.axes[1, 1].set_title('Parameters (Normalized)')
            self.axes[1, 1].legend()
        
        # Plot metrics
        metrics = self.current_results["audio_metrics"]
        finite_metrics = {k: v for k, v in metrics.items() if np.isfinite(v)}
        if finite_metrics:
            self.axes[1, 2].bar(finite_metrics.keys(), finite_metrics.values())
            self.axes[1, 2].set_title('Audio Metrics')
            self.axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # Force figure to update in interactive backends
        plt.pause(0.01)
    
    def play_true_audio(self, event):
        """Play true audio with multiple playback options."""
        if not self.current_results:
            print("⚠️  No audio data available")
            return
            
        try:
            sample_rate = self.evaluator.sample_rate
            audio = self.current_results["true_audio"]
            sample_idx = self.current_results["sample_idx"]
            
            print(f"🎵 Playing true audio (Sample {sample_idx})")
            
            # Try multiple playback options
            success = False
            
            # Option 1: Use sounddevice for direct playback (best for Mac command line)
            try:
                import sounddevice as sd
                sd.play(audio, sample_rate)
                sd.wait()  # Wait until playback is finished
                success = True
                print(f"   ✓ Played via sounddevice at {sample_rate} Hz")
            except ImportError:
                print("   sounddevice not available")
            except Exception as e:
                print(f"   sounddevice playback failed: {e}")
            
            # Option 2: Jupyter/IPython display (only if sounddevice failed)
            if not success and IPYTHON_AVAILABLE:
                try:
                    from IPython.display import display
                    # Only use IPython if we're actually in a Jupyter environment
                    try:
                        get_ipython()  # This will raise NameError if not in IPython
                        display(ipd.Audio(audio, rate=sample_rate))
                        success = True
                        print(f"   ✓ Displayed via Jupyter")
                    except NameError:
                        # Not in IPython/Jupyter, skip this method
                        pass
                except Exception as e:
                    print(f"   Jupyter playback failed: {e}")
            
            # Option 3: Save to temporary file and use system player
            if not success:
                try:
                    import tempfile
                    import subprocess
                    import platform
                    
                    # Create temporary wav file
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                        tmp_path = tmp_file.name
                    
                    try:
                        import soundfile as sf
                        sf.write(tmp_path, audio, sample_rate)
                        
                        # Try to play with system command
                        system = platform.system()
                        if system == "Darwin":  # macOS
                            subprocess.run(["afplay", tmp_path], check=True)
                        elif system == "Linux":
                            subprocess.run(["aplay", tmp_path], check=True)
                        elif system == "Windows":
                            subprocess.run(["start", tmp_path], shell=True, check=True)
                        
                        success = True
                        print(f"   ✓ Played via system audio player")
                        
                    except Exception as e:
                        print(f"   System player failed: {e}")
                    finally:
                        # Clean up temp file after a delay
                        try:
                            import os, time
                            time.sleep(2)  # Wait for playback to start
                            os.unlink(tmp_path)
                        except:
                            pass
                            
                except Exception as e:
                    print(f"   Temp file playback failed: {e}")
            
            if not success:
                print("   ⚠️  No audio playback method available")
                print("   💡 Try installing: pip install sounddevice")
                print("   💡 Or run in Jupyter notebook for web audio")
                
        except Exception as e:
            print(f"⚠️  Error playing true audio: {e}")
    
    def play_pred_audio(self, event):
        """Play predicted audio with multiple playback options."""
        if not self.current_results:
            print("⚠️  No audio data available")
            return
            
        try:
            sample_rate = self.evaluator.sample_rate
            audio = self.current_results["pred_audio"]
            sample_idx = self.current_results["sample_idx"]
            
            print(f"🎵 Playing predicted audio (Sample {sample_idx})")
            
            # Try multiple playback options
            success = False
            
            # Option 1: Use sounddevice for direct playback (best for Mac command line)
            try:
                import sounddevice as sd
                sd.play(audio, sample_rate)
                sd.wait()  # Wait until playback is finished
                success = True
                print(f"   ✓ Played via sounddevice at {sample_rate} Hz")
            except ImportError:
                print("   sounddevice not available")
            except Exception as e:
                print(f"   sounddevice playback failed: {e}")
            
            # Option 2: Jupyter/IPython display (only if sounddevice failed)
            if not success and IPYTHON_AVAILABLE:
                try:
                    from IPython.display import display
                    # Only use IPython if we're actually in a Jupyter environment
                    try:
                        get_ipython()  # This will raise NameError if not in IPython
                        display(ipd.Audio(audio, rate=sample_rate))
                        success = True
                        print(f"   ✓ Displayed via Jupyter")
                    except NameError:
                        # Not in IPython/Jupyter, skip this method
                        pass
                except Exception as e:
                    print(f"   Jupyter playback failed: {e}")
            
            # Option 3: Save to temporary file and use system player
            if not success:
                try:
                    import tempfile
                    import subprocess
                    import platform
                    
                    # Create temporary wav file
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                        tmp_path = tmp_file.name
                    
                    try:
                        import soundfile as sf
                        sf.write(tmp_path, audio, sample_rate)
                        
                        # Try to play with system command
                        system = platform.system()
                        if system == "Darwin":  # macOS
                            subprocess.run(["afplay", tmp_path], check=True)
                        elif system == "Linux":
                            subprocess.run(["aplay", tmp_path], check=True)
                        elif system == "Windows":
                            subprocess.run(["start", tmp_path], shell=True, check=True)
                        
                        success = True
                        print(f"   ✓ Played via system audio player")
                        
                    except Exception as e:
                        print(f"   System player failed: {e}")
                    finally:
                        # Clean up temp file after a delay
                        try:
                            import os, time
                            time.sleep(2)  # Wait for playback to start
                            os.unlink(tmp_path)
                        except:
                            pass
                            
                except Exception as e:
                    print(f"   Temp file playback failed: {e}")
            
            if not success:
                print("   ⚠️  No audio playback method available")
                print("   💡 Try installing: pip install sounddevice")
                print("   💡 Or run in Jupyter notebook for web audio")
                
        except Exception as e:
            print(f"⚠️  Error playing predicted audio: {e}")


@task_wrapper
def evaluate_audio_reconstruction(cfg: DictConfig) -> Dict[str, Any]:
    """
    Main evaluation function for audio reconstruction.
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        Dictionary of evaluation results
    """
    # Auto-discover checkpoint if not provided
    ckpt_path = cfg.get("ckpt_path")
    if not ckpt_path:
        ckpt_path = find_latest_checkpoint()
        if not ckpt_path:
            raise ValueError(
                "No checkpoint path provided and no checkpoints found. "
                "Please either:\n"
                "1. Provide ckpt_path=path/to/checkpoint.ckpt, or\n" 
                "2. Ensure you have trained checkpoints in logs/train/"
            )
    
    # Verify checkpoint exists
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    # Load checkpoint to extract metadata first
    log.info(f"Loading model from checkpoint: {ckpt_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Extract dataset metadata from checkpoint to properly configure datamodule
    vimh_data = checkpoint.get('VIMHDataModule', {})
    dataset_metadata = vimh_data.get('dataset_metadata', {})
    
    # Configure datamodule to use the same dataset that was used for training
    if dataset_metadata and 'dataset_name' in dataset_metadata:
        trained_dataset_name = dataset_metadata['dataset_name']
        log.info(f"Training used dataset: {trained_dataset_name}")
        
        # Try to use the same dataset directory for evaluation
        potential_data_dir = f"data/{trained_dataset_name}"
        if os.path.exists(potential_data_dir):
            log.info(f"Using training dataset directory: {potential_data_dir}")
            cfg.data.data_dir = potential_data_dir
        else:
            log.warning(f"Training dataset directory {potential_data_dir} not found, using config default")
    
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    # Explicitly use identity transforms for spectrograms (no fallbacks)
    from torchvision.transforms import transforms
    identity_transform = transforms.Compose([])
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data,
        train_transform=identity_transform,
        val_transform=identity_transform,
        test_transform=identity_transform,
    )
    
    # If we have saved metadata, inject it into the datamodule
    if dataset_metadata and 'parameter_names' in dataset_metadata:
        log.info(f"Found saved dataset metadata with parameters: {dataset_metadata['parameter_names']}")
        datamodule._saved_dataset_metadata = dataset_metadata
        log.info("Injected saved metadata into datamodule")
    
    # Setup the datamodule to load datasets (setup train for auto-configuration)
    datamodule.setup("fit")  # This sets up train dataset which is needed for auto-configuration
    
    # Try to load model directly from checkpoint using Lightning's built-in method first
    try:
        # Try to use Lightning's load_from_checkpoint with the exact class
        from src.models.multihead_module import MultiheadLitModule
        model: LightningModule = MultiheadLitModule.load_from_checkpoint(
            ckpt_path, 
            map_location=device,
            strict=False  # Allow loading with missing/extra keys
        )
        log.info("Successfully loaded model from checkpoint")
    except Exception as e:
        log.warning(f"Failed to load from checkpoint directly: {e}")
        log.info("Trying alternative loading method with original model config...")
        
        # Extract the original model configuration from the checkpoint hyperparameters
        hyper_parameters = checkpoint.get('hyper_parameters', {})
        log.info(f"Checkpoint hyperparameters keys: {list(hyper_parameters.keys())}")
        
        # Check for stored architecture metadata first
        architecture_metadata = hyper_parameters.get('architecture_metadata', {})
        log.info(f"Architecture metadata: {architecture_metadata}")
        
        # Get checkpoint state dict for later use
        checkpoint_state_dict = checkpoint["state_dict"]
        first_weight_key = list(checkpoint_state_dict.keys())[0]
        log.info(f"First weight key: {first_weight_key}")
        
        # Get heads config from the properly configured datamodule
        if hasattr(datamodule, 'data_train') and datamodule.data_train is not None:
            heads_config = datamodule.data_train.get_heads_config()
            log.info(f"Retrieved heads config from datamodule: {list(heads_config.keys())}")
            
            # Use stored architecture metadata if available
            if architecture_metadata and architecture_metadata.get('type') == 'ViT':
                log.info("Using stored ViT architecture metadata from checkpoint")
                
                # Create ViT with stored parameters
                from src.models.components.vision_transformer import VisionTransformer
                net = VisionTransformer(
                    image_size=architecture_metadata['image_size'],
                    patch_size=architecture_metadata['patch_size'], 
                    n_channels=architecture_metadata['input_channels'],
                    embed_dim=architecture_metadata['embed_dim'],
                    n_layers=architecture_metadata['n_layers'],
                    n_attention_heads=architecture_metadata['n_attention_heads'],
                    heads_config=heads_config,
                    forward_mul=architecture_metadata.get('forward_mul', 2),
                    dropout=architecture_metadata.get('dropout', 0.1),
                    use_torch_layers=architecture_metadata.get('use_torch_layers', False)
                )
                
            elif architecture_metadata and architecture_metadata.get('type') == 'CNN':
                log.info("Using stored CNN architecture metadata from checkpoint")
                
                # Create CNN with stored parameters (provide defaults for missing params)
                from src.models.components.simple_cnn import SimpleCNN
                net = SimpleCNN(
                    input_channels=architecture_metadata.get('input_channels', 1),
                    conv1_channels=architecture_metadata.get('conv1_channels', 16),
                    conv2_channels=architecture_metadata.get('conv2_channels', 32),
                    fc_hidden=architecture_metadata.get('fc_hidden', 64),
                    heads_config=heads_config,
                    dropout=architecture_metadata.get('dropout', 0.5),
                    input_size=architecture_metadata.get('input_size', 32)
                )
                
            else:
                # Fallback: try to infer from state dict structure - FAIL FAST!
                log.warning("No architecture metadata found in checkpoint, trying to infer from weights")
                
                if 'net.embedding.pos_embedding' in checkpoint_state_dict:
                    log.info("Detected Vision Transformer architecture from checkpoint")
                    log.error("Cannot reliably infer ViT architecture from checkpoint weights alone")
                    log.error("ViT architecture must be fully specified in hyperparameters or config")
                    sys.exit(1)
                    
                elif 'net.conv_layers.0.weight' in checkpoint_state_dict or 'net.features.0.weight' in checkpoint_state_dict or 'net.conv1.weight' in checkpoint_state_dict:
                    log.info("Detected CNN architecture from checkpoint")
                
                    # FAIL FAST: Infer CNN architecture parameters from checkpoint weights - no defaults!
                    if 'net.conv_layers.0.weight' not in checkpoint_state_dict:
                        log.error("SimpleCNN checkpoint missing net.conv_layers.0.weight")
                        sys.exit(1)
                    
                    conv1_weight = checkpoint_state_dict['net.conv_layers.0.weight']
                    conv1_channels = conv1_weight.shape[0]  # Number of output channels
                    input_channels = conv1_weight.shape[1]   # Should be 1
                    
                    # Get conv2 channels - REQUIRED, no fallback
                    if 'net.conv_layers.4.weight' not in checkpoint_state_dict:
                        log.error("SimpleCNN checkpoint missing net.conv_layers.4.weight")
                        sys.exit(1)
                        
                    conv2_weight = checkpoint_state_dict['net.conv_layers.4.weight']
                    conv2_channels = conv2_weight.shape[0]
                    
                    # Get FC hidden size from shared_features layer - REQUIRED, no fallback
                    if 'net.shared_features.1.weight' in checkpoint_state_dict:
                        shared_fc_weight = checkpoint_state_dict['net.shared_features.1.weight']
                        fc_hidden = shared_fc_weight.shape[0]
                    elif 'net.shared_features.0.weight' in checkpoint_state_dict:
                        shared_fc_weight = checkpoint_state_dict['net.shared_features.0.weight']
                        fc_hidden = shared_fc_weight.shape[0]
                    else:
                        log.error("SimpleCNN checkpoint missing both net.shared_features.0.weight and net.shared_features.1.weight")
                        sys.exit(1)
                    
                    # Extract input size from dataset metadata - REQUIRED
                    if not dataset_metadata or 'height' not in dataset_metadata or 'width' not in dataset_metadata:
                        log.error("CNN requires dataset metadata with height and width")
                        sys.exit(1)
                        
                    input_size = max(dataset_metadata['height'], dataset_metadata['width'])
                    
                    log.info(f"Inferred CNN params: conv1_channels={conv1_channels}, conv2_channels={conv2_channels}, fc_hidden={fc_hidden}, input_size={input_size}")
                    
                    # Create CNN with inferred parameters
                    from src.models.components.simple_cnn import SimpleCNN
                    net = SimpleCNN(
                        input_channels=input_channels,
                        conv1_channels=conv1_channels,
                        conv2_channels=conv2_channels,
                        fc_hidden=fc_hidden,
                        heads_config=heads_config,
                        dropout=0.5,  # This should also be extracted from checkpoint, but SimpleCNN doesn't save it
                        input_size=input_size
                    )
                else:
                    log.error("Could not determine model architecture from checkpoint - no recognizable architecture found")
                    log.error("Expected either:")
                    log.error("  - ViT: net.embedding.pos_embedding")  
                    log.error("  - CNN: net.conv_layers.0.weight")
                    sys.exit(1)
                
            # Create multihead module with inferred network (common for both ViT and CNN)
            from src.models.multihead_module import MultiheadLitModule
            
            # Create default criterion for each head to satisfy the model requirements
            from torch.nn import CrossEntropyLoss
            criteria = {head_name: CrossEntropyLoss() for head_name in heads_config.keys()}
            
            model = MultiheadLitModule(
                net=net,
                optimizer=hyper_parameters.get('optimizer'),
                scheduler=None,  # Will be set later if needed
                loss_weights=hyper_parameters.get('loss_weights', {}),
                compile=hyper_parameters.get('compile', False),
                criteria=criteria,  # Provide explicit criteria
                auto_configure_from_dataset=False,  # Disable auto-config to preserve loaded model structure
                output_mode=hyper_parameters.get('output_mode', 'classification')
            )
            
            log.info(f"Model created with inferred architecture")
        else:
            log.warning("Could not get heads config from datamodule, using default model")
            model: LightningModule = hydra.utils.instantiate(cfg.model)
        
        # Try to load state dict with relaxed matching
        model_state_dict = model.state_dict()
        
        # Filter to only load weights that match in shape and name
        compatible_weights = {}
        for key, value in checkpoint_state_dict.items():
            if key in model_state_dict and model_state_dict[key].shape == value.shape:
                compatible_weights[key] = value
            else:
                log.error(f"Incompatible weight: {key} (checkpoint: {value.shape}, model: {model_state_dict.get(key, 'missing').shape if key in model_state_dict else 'missing'})")
                sys.exit(1)
        
        model.load_state_dict(compatible_weights, strict=False)
        log.info(f"Loaded {len(compatible_weights)} compatible weights from checkpoint")
    
    # Create evaluator
    evaluator = AudioReconstructionEvaluator(model, datamodule, device)
    
    # Configuration options
    num_samples = cfg.get("num_samples", 5)
    interactive = cfg.get("interactive", False)
    save_audio = cfg.get("save_audio", False)
    output_dir = cfg.get("output_dir", "audio_eval_results")
    
    if interactive:
        # Launch interactive widget
        log.info("Launching interactive evaluator...")
        interactive_eval = InteractiveAudioEvaluator(evaluator)
        plt.show()
        return {"message": "Interactive evaluation launched"}, None
    else:
        # Batch evaluation
        log.info(f"Evaluating {num_samples} samples...")
        results = []
        
        for i in range(min(num_samples, len(evaluator.test_dataset))):
            log.info(f"Evaluating sample {i+1}/{num_samples}")
            result = evaluator.evaluate_sample(
                i, plot=True, save_audio=save_audio, output_dir=output_dir
            )
            results.append(result)
        
        # Compute aggregate statistics
        all_metrics = [r["audio_metrics"] for r in results]
        aggregate_metrics = {}
        
        for metric_name in all_metrics[0].keys():
            values = [m[metric_name] for m in all_metrics if np.isfinite(m[metric_name])]
            if values:
                aggregate_metrics[f"mean_{metric_name}"] = np.mean(values)
                aggregate_metrics[f"std_{metric_name}"] = np.std(values)
        
        log.info("Aggregate metrics:")
        for metric, value in aggregate_metrics.items():
            log.info(f"  {metric}: {value:.6f}")
        
        return {
            "individual_results": results,
            "aggregate_metrics": aggregate_metrics,
            "num_samples_evaluated": len(results)
        }, None


@hydra.main(version_base="1.3", config_path="../configs", config_name="audio_eval")
def main(cfg: DictConfig) -> None:
    """Main entry point for audio reconstruction evaluation."""
    
    # Show usage info
    if cfg.get("ckpt_path") is None:
        log.info("🎵 Audio Reconstruction Evaluation")
        log.info("Auto-discovering latest checkpoint...")
        log.info("💡 Tip: Run with ckpt_path=path/to/checkpoint.ckpt to use specific checkpoint")
    
    extras(cfg)
    evaluate_audio_reconstruction(cfg)


if __name__ == "__main__":
    main()
