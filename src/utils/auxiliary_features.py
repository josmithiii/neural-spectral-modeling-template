import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch.nn.functional as F


def extract_decay_time_features(temporal_envelope: torch.Tensor,
                               sample_rate: float = 8000.0,
                               frame_hop: float = 0.01) -> torch.Tensor:
    """
    Extract decay time scalar features from temporal envelope.

    Args:
        temporal_envelope: Temporal envelope tensor [B, C, H, W]
        sample_rate: Audio sample rate in Hz
        frame_hop: Time between frames in seconds

    Returns:
        decay_features: [B, 1] tensor with log10_decay_time_measured
    """
    B, C, H, W = temporal_envelope.shape

    # Sum over frequency and channel dimensions to get overall temporal profile
    temporal_profile = temporal_envelope.sum(dim=[1, 2])  # [B, W]

    batch_features = []

    for b in range(B):
        profile = temporal_profile[b].cpu().numpy()  # [W]

        # Normalize to [0, 1] range
        profile_max = profile.max()
        if profile_max > 0:
            profile = profile / profile_max
        else:
            profile = np.ones_like(profile) * 0.001  # Avoid all-zero profiles

        # Time axis in seconds
        times = np.arange(W) * frame_hop

        # Measure exponential decay time
        log10_decay_time = _estimate_decay_time(profile, times)

        batch_features.append([log10_decay_time])

    # Convert to numpy array first to avoid performance warning
    batch_features_array = np.array(batch_features)
    return torch.tensor(batch_features_array, dtype=temporal_envelope.dtype, device=temporal_envelope.device)


def _estimate_decay_time(profile: np.ndarray, times: np.ndarray) -> float:
    """
    Estimate exponential decay time constant from temporal profile.

    Fits y = A * exp(-t/tau) to the profile and returns log10(tau).
    """
    # Find peak and use data from peak onwards
    peak_idx = np.argmax(profile)
    decay_profile = profile[peak_idx:]
    decay_times = times[peak_idx:]

    if len(decay_profile) < 3:
        return -1.0  # Default for too short profiles

    # Take log of positive values only
    positive_mask = decay_profile > 1e-6
    if np.sum(positive_mask) < 2:
        return -1.0  # Not enough positive values

    log_profile = np.log(decay_profile[positive_mask])
    valid_times = decay_times[positive_mask]

    # Linear fit to log(profile) vs time gives slope = -1/tau
    if len(valid_times) < 2:
        return -1.0

    # Robust linear fit (handles some outliers)
    try:
        coeffs = np.polyfit(valid_times, log_profile, deg=1, w=decay_profile[positive_mask])
        slope = coeffs[0]

        if slope < -1e-6:  # Decay should have negative slope
            tau = -1.0 / slope  # Decay time constant
            log10_decay_time = np.log10(tau)

            # Clamp to reasonable range [-3, 1]
            log10_decay_time = np.clip(log10_decay_time, -3.0, 1.0)
            return log10_decay_time
        else:
            return -1.0  # No decay detected
    except:
        return -1.0  # Fitting failed


def compute_temporal_envelope_from_spectrogram(spectrogram: torch.Tensor,
                                              kernel_size: int = 5) -> torch.Tensor:
    """
    Compute temporal envelope from raw spectrogram.

    Args:
        spectrogram: Input spectrogram [C, H, W] or [H, W]
        kernel_size: Size of smoothing kernel

    Returns:
        temporal_envelope: [1, C, H, W] temporal envelope (matching extract_decay_time_features input format)
    """
    # Handle different input shapes
    if spectrogram.dim() == 2:
        # [H, W] -> [1, H, W]
        spectrogram = spectrogram.unsqueeze(0)
    if spectrogram.dim() == 3:
        # [C, H, W] -> [1, C, H, W] (add batch dimension)
        spectrogram = spectrogram.unsqueeze(0)

    B, C, H, W = spectrogram.shape

    # Compute energy per time frame (sum over frequency)
    temporal_energy = spectrogram.sum(dim=2, keepdim=True)  # [B, C, 1, W]

    # Smooth the envelope to remove noise
    if kernel_size >= 3 and W >= kernel_size:
        # Apply smoothing to each channel separately
        smoothed_channels = []
        for c in range(C):
            channel_energy = temporal_energy[:, c:c+1, :, :]  # [B, 1, 1, W]
            kernel = torch.ones(1, 1, 1, kernel_size, device=spectrogram.device) / kernel_size
            smoothed = F.conv2d(channel_energy, kernel, padding=(0, kernel_size//2))
            smoothed_channels.append(smoothed)
        temporal_energy = torch.cat(smoothed_channels, dim=1)

    # Broadcast back to full spectrogram size
    temporal_envelope = temporal_energy.expand(-1, -1, H, -1)  # [B, C, H, W]

    return temporal_envelope


def extract_auxiliary_features(data: Dict[str, torch.Tensor],
                             feature_types: List[str] = ["decay_time"]) -> torch.Tensor:
    """
    Extract auxiliary scalar features from VIMH data dict.

    Args:
        data: Dictionary with 'image', 'temporal_envelope', etc.
        feature_types: List of feature types to extract

    Returns:
        auxiliary_features: [B, num_features] tensor
    """
    features_list = []

    if "decay_time" in feature_types:
        # First try to use pre-computed temporal envelope
        if "temporal_envelope" in data:
            decay_features = extract_decay_time_features(data["temporal_envelope"])
            features_list.append(decay_features)
        # If no temporal envelope available, compute it from raw spectrogram
        elif "image" in data:
            # Compute temporal envelope from raw spectrogram
            temporal_envelope = compute_temporal_envelope_from_spectrogram(data["image"])
            decay_features = extract_decay_time_features(temporal_envelope)
            features_list.append(decay_features)

    if len(features_list) == 0:
        # No features available - return empty tensor
        if "image" in data:
            batch_size = data["image"].shape[0] if data["image"].dim() > 2 else 1
        else:
            batch_size = 1
        device = data["image"].device if "image" in data else torch.device("cpu")
        dtype = data["image"].dtype if "image" in data else torch.float32
        return torch.zeros(batch_size, 0, dtype=dtype, device=device)

    # Concatenate all feature types
    return torch.cat(features_list, dim=1)