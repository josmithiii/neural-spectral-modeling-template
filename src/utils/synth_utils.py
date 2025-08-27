"""
Shared synthesis utilities for audio generation and spectrogram processing.

This module consolidates the synthesis components used by both generate_vimh.py
and the SynthesisValidationLoss to avoid code duplication.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants from generate_vimh.py
DEFAULT_NOTE_NUMBER = 69.0
DEFAULT_NOTE_VELOCITY = 100.0
DEFAULT_LOG10_DECAY_TIME = -0.301  # log10(0.5)
DEFAULT_DURATION = 1.0
MIN_SPECTROGRAM_VALUE = 1e-10
QUANTIZATION_LEVELS = 255


class STFT(nn.Module):
    """STFT module from generate_vimh.py"""

    def __init__(
        self,
        fftsize: int,
        winsize: int,
        hopsize: int,
        complex: bool = False,
        window_type: str = "rectangular",
    ):
        super().__init__()
        self.fftsize = fftsize
        self.winsize = winsize
        self.hopsize = hopsize
        self.window_type = window_type

        # Create window based on type
        if window_type == "rectangular":
            window = torch.ones(winsize)
        elif window_type == "hann":
            window = torch.hann_window(winsize, periodic=False)
        elif window_type == "hamming":
            window = torch.hamming_window(winsize, periodic=False)
        elif window_type == "blackman":
            window = torch.blackman_window(winsize, periodic=False)
        else:
            raise ValueError(
                f"Unsupported window type: {window_type}. Supported: 'rectangular', 'hann', 'hamming', 'blackman'"
            )

        self.register_buffer("window", window, persistent=False)
        self.complex = complex

    def compute_stft_kernel(self):
        # use CPU STFT of dirac impulses to derive conv1d weights if we're using that b/c torch.stft failed
        diracs = torch.eye(self.winsize)
        w = torch.stft(
            diracs,
            n_fft=self.fftsize,
            hop_length=self.hopsize,
            win_length=self.winsize,
            window=self.window.to(diracs),
            center=False,
            normalized=True,
            return_complex=True,
        )
        w = torch.view_as_real(w)
        # squash real/complex, transpose to (1, winsize+2, winsize)
        w = w.flatten(1).T[:, np.newaxis]
        return w

    def forward(self, x):
        # we want each channel to be treated separately, so we mash
        # up the channels and batch size and split them up afterwards
        batchsize, channels = x.shape[:2]
        x = x.reshape((-1,) + x.shape[2:])
        # we apply the STFT
        if not hasattr(self, "stft_kernel"):
            try:
                x = torch.stft(
                    x,
                    n_fft=self.fftsize,
                    hop_length=self.hopsize,
                    win_length=self.winsize,
                    window=self.window,
                    center=False,
                    normalized=True,
                    return_complex=True,
                )
                x = torch.view_as_real(x)
            except RuntimeError as exc:
                if len(exc.args) > 0 and (
                    ("doesn't support" in exc.args[0]) or ("only supports" in exc.args[0])
                ):
                    # half precision STFT not supported everywhere, improvise!
                    # compute equivalent conv1d weights and register as buffer
                    self.register_buffer(
                        "stft_kernel", self.compute_stft_kernel().to(x), persistent=False
                    )
                else:
                    raise
        if hasattr(self, "stft_kernel"):
            # we use the conv1d replacement if we found that stft() fails
            x = F.conv1d(x[:, None], self.stft_kernel, stride=self.hopsize)
            # split real/complex and move to the end
            x = x.reshape((batchsize, -1, 2, x.shape[-1])).transpose(-1, -2)
        # we compute magnitudes, if requested
        if not self.complex:
            x = x.norm(p=2, dim=-1)
        # restore original batchsize and channels in case we mashed them
        x = x.reshape((batchsize, channels, -1) + x.shape[2:])
        return x


def create_mel_filterbank(
    sample_rate: float,
    frame_len: int,
    num_bands: int,
    min_freq: float,
    max_freq: float,
    norm: bool = True,
    crop: bool = False,
) -> torch.Tensor:
    """
    Creates a mel filterbank of `num_bands` triangular filters, with the first
    filter starting at `min_freq` and the last one stopping at `max_freq`.
    """
    # mel-spaced peak frequencies
    min_mel = 1127 * np.log1p(min_freq / 700.0)
    max_mel = 1127 * np.log1p(max_freq / 700.0)
    peaks_mel = torch.linspace(min_mel, max_mel, num_bands + 2)
    peaks_hz = 700 * (torch.expm1(peaks_mel / 1127))
    peaks_bin = peaks_hz * frame_len / sample_rate

    # create filterbank
    input_bins = (frame_len // 2) + 1
    if crop:
        input_bins = min(input_bins, int(np.ceil(max_freq * frame_len / float(sample_rate))))
    x = torch.arange(input_bins, dtype=peaks_bin.dtype)[:, np.newaxis]
    l, c, r = peaks_bin[0:-2], peaks_bin[1:-1], peaks_bin[2:]
    # triangles are the minimum of two linear functions f(x) = a*x + b
    # left side of triangles: f(l) = 0, f(c) = 1 -> a=1/(c-l), b=-a*l
    tri_left = (x - l) / (c - l)
    # right side of triangles: f(c) = 1, f(r) = 0 -> a=1/(c-r), b=-a*r
    tri_right = (x - r) / (c - r)
    # combine by taking the minimum of the left and right sides
    tri = torch.min(tri_left, tri_right)
    # and clip to only keep positive values
    filterbank = torch.clamp(tri, min=0)

    # normalize by area
    if norm:
        filterbank /= filterbank.sum(0)

    return filterbank


class MelFilter(nn.Module):
    """
    Transform a spectrogram created with the given `sample_rate` and `winsize`
    into a mel spectrogram of `num_bands` from `min_freq` to `max_freq`.
    """

    def __init__(
        self, sample_rate: float, winsize: int, num_bands: int, min_freq: float, max_freq: float
    ):
        super().__init__()
        melbank = create_mel_filterbank(
            sample_rate, winsize, num_bands, min_freq, max_freq, crop=True
        )
        self.register_buffer("bank", melbank, persistent=False)

    def forward(self, x):
        x = x.transpose(-1, -2)  # put fft bands last
        x = x[..., : self.bank.shape[0]]  # remove unneeded fft bands
        x = x.matmul(self.bank)  # turn fft bands into mel bands
        x = x.transpose(-1, -2)  # put time last
        return x


class SpectrogramProcessor:
    """Handles efficient spectrogram generation with reusable modules."""

    def __init__(
        self,
        sample_rate: int,
        height: int,
        width: int,
        stft_config: Dict[str, Any],
        mel_config: Dict[str, Any],
    ):
        self.sample_rate = sample_rate
        self.height = height
        self.width = width
        self.stft_config = stft_config
        self.mel_config = mel_config

        # Extract spectrogram configuration
        self.spectrogram_type = stft_config["type"]

        self.n_fft = stft_config.get("n_fft", 512)
        self.n_window = stft_config.get("n_window", 128)
        self.hop_length = stft_config.get("hop_length", 64)
        self.window_type = stft_config.get("window_type", "rectangular")
        self.bins_per_harmonic = stft_config.get("bins_per_harmonic", 1.0)

        self.freq_min = mel_config.get("freq_min", 40.0)
        self.freq_max_ratio = mel_config.get("freq_max_ratio", 0.9)

        # Validate configuration
        if self.spectrogram_type not in ["mel", "stft"]:
            raise ValueError(
                f"Unknown spectrogram type: {self.spectrogram_type}. Must be 'mel' or 'stft'."
            )

        # Create reusable modules
        self.stft_module = None
        self.mel_filter_module: Optional[MelFilter] = None

        # Create default module
        self.stft_module = STFT(
            fftsize=self.n_fft,
            winsize=self.n_window,
            hopsize=self.hop_length,
            window_type=self.window_type,
        )

        if self.spectrogram_type == "mel":
            freq_max = self.freq_max_ratio * sample_rate / 2
            self.mel_filter_module = MelFilter(
                sample_rate=sample_rate,
                winsize=self.n_fft,
                num_bands=height,
                min_freq=self.freq_min,
                max_freq=freq_max,
            )

    def _prepare_audio_tensor(self, audio: np.ndarray) -> torch.Tensor:
        """Convert audio to tensor and handle padding."""
        audio_tensor = torch.from_numpy(audio).float()

        # Handle very short audio by padding
        if len(audio_tensor) < self.n_fft:
            padding = self.n_fft - len(audio_tensor)
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))

        # Add batch and channel dimensions: (batch, channels, time)
        return audio_tensor.unsqueeze(0).unsqueeze(0)

    def _normalize_spectrogram(self, spec: torch.Tensor) -> Tuple[np.ndarray, float, float]:
        """Normalize spectrogram to [0, 255] range and return scale factors.

        Returns:
            normalized_spec: uint8 array normalized to [0, 255]
            spec_min: minimum value before normalization (in dB)
            spec_max: maximum value before normalization (in dB)
        """
        spec_np = spec.numpy()
        spec_min = float(spec_np.min())
        spec_max = float(spec_np.max())

        if spec_max > spec_min:
            # In-place normalization to avoid memory copies
            spec_normalized = spec_np.copy()
            spec_normalized -= spec_min
            spec_normalized /= spec_max - spec_min
            # Apply smooth scaling with rounding to reduce stepping artifacts
            spec_normalized = np.round(spec_normalized * QUANTIZATION_LEVELS)
            spec_normalized = np.clip(spec_normalized, 0, QUANTIZATION_LEVELS)
            return spec_normalized.astype(np.uint8), spec_min, spec_max
        else:
            return np.zeros_like(spec_np, dtype=np.uint8), spec_min, spec_max

    def audio_to_spectrogram(
        self, params: Dict[str, float], audio: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """Process audio into spectrogram based on configured type.

        Returns:
            normalized_spec: uint8 array normalized to [0, 255]
            spec_min: minimum value before normalization (in dB)
            spec_max: maximum value before normalization (in dB)
        """
        try:
            audio_tensor = self._prepare_audio_tensor(audio)

            with torch.no_grad():
                # Compute STFT first (required for both 'stft' and 'mel' types)
                # STFT transforms audio from time domain to frequency domain
                spec = self.stft_module(
                    audio_tensor
                )  # (batch, channels, freq [0:n_fft/2+1], time)

                # Apply mel filtering if needed (converts linear frequency bins to mel-scale)
                if self.spectrogram_type == "mel" and self.mel_filter_module is not None:
                    spec = self.mel_filter_module(spec)

                # Convert to dB
                spec_db = torch.log10(torch.clamp(spec, min=MIN_SPECTROGRAM_VALUE)) * 20.0

            # Remove batch and channel dimensions
            spec_db = spec_db.squeeze(0).squeeze(0)

            # Slice frequency bins and time frames based on spectrogram type
            max_time_frame = min(self.width, spec_db.shape[1])

            if self.spectrogram_type == "mel":
                # For mel spectrograms, keep all frequency bins (no DC bin to skip)
                max_freq_bin = min(self.height, spec_db.shape[0])
                spec_db = spec_db[:max_freq_bin, :max_time_frame]
            else:
                # For STFT spectrograms, skip DC bin (keep bins 1:height+1)
                max_freq_bin = min(self.height + 1, spec_db.shape[0])
                spec_db = spec_db[1:max_freq_bin, :max_time_frame]

            return self._normalize_spectrogram(spec_db)

        except Exception as e:
            raise RuntimeError(f"Error processing audio: {e}")


def check_params(params: Dict[str, float], *required_params: str) -> None:
    """Check if required parameters exist in params dict and log warnings for missing ones."""
    import logging

    logger = logging.getLogger(__name__)

    for param_name in required_params:
        if param_name not in params:
            logger.warning(f"Parameter '{param_name}' not found in params")


class SimpleSynth:
    """Simple exponentially decaying sawtooth synthesizer."""

    def __init__(self, sample_rate: int = 8000):
        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")
        self.sample_rate = sample_rate

    def generate_audio(self, params: Dict[str, float]) -> np.ndarray:
        """Generate audio with given parameters."""
        check_params(params, "note_number", "note_velocity", "duration", "log10_decay_time")

        try:
            note_number = params.get("note_number", DEFAULT_NOTE_NUMBER)
            note_velocity = params.get("note_velocity", DEFAULT_NOTE_VELOCITY)
            duration = params.get("duration", DEFAULT_DURATION)

            if "log10_decay_time" in params:
                log10_decay_time = params["log10_decay_time"]
                decay_time = 10.0**log10_decay_time
            else:
                decay_time = 10.0**DEFAULT_LOG10_DECAY_TIME

            # Validate parameters
            if duration <= 0:
                raise ValueError(f"Duration must be positive, got {duration}")
            if note_velocity < 0 or note_velocity > 127:
                raise ValueError(f"Note velocity must be in [0, 127], got {note_velocity}")

            # Convert MIDI note to frequency
            freq = 440.0 * (2.0 ** ((note_number - 69.0) / 12.0))

            # Generate time array with precise sample timing
            num_samples = int(duration * self.sample_rate)
            if num_samples <= 0:
                raise ValueError(f"Duration too short for sample rate, got {num_samples} samples")

            # Use arange for exact sample timing instead of linspace
            t = np.arange(num_samples, dtype=np.float64) / self.sample_rate

            # Generate sawtooth wave with proper 1/f spectrum
            # Use fractional part of phase for cleaner sawtooth
            phase = freq * t
            phase -= np.floor(phase)
            sawtooth = 2.0 * phase - 1.0

            # Apply exponential decay
            decay_envelope = np.exp(-t / decay_time)

            # Apply velocity (amplitude scaling)
            amplitude = note_velocity / 127.0

            # Generate impulse train at fundamental frequency
            # Detect rising edges of sawtooth (new cycles)
            impulse_train = np.zeros_like(sawtooth)
            if len(sawtooth) > 1:
                # Find where sawtooth wraps around (goes from high to low value)
                phase_wraps = np.diff(sawtooth) < -1.0  # Large negative diff indicates wrap
                impulse_train[1:][phase_wraps] = 1.0
                # Also set first sample as impulse
                impulse_train[0] = 1.0
            else:
                impulse_train[0] = 1.0

            # Combine all effects
            audio = amplitude * sawtooth * decay_envelope
            # TEST IMPULSE-TRAIN HACK - flat spectrum (no decay envelope):
            # audio = amplitude * impulse_train

            return audio.astype(np.float32)

        except Exception as e:
            raise RuntimeError(f"Error generating audio: {e}")


def pre_emphasis_filter(audio: np.ndarray, coefficient: float = 0.97) -> np.ndarray:
    """Apply pre-emphasis filter to audio signal."""
    if coefficient <= 0:
        return audio
    emphasized = np.append(audio[0], audio[1:] - coefficient * audio[:-1])
    return emphasized


def prepare_channels(spectrogram: np.ndarray, channels: int) -> np.ndarray:
    """Prepare spectrogram with correct number of channels."""
    height, width = spectrogram.shape

    if channels == 1:
        return spectrogram
    else:
        # For spectrograms, replicate the single-channel data across all channels
        # This maintains compatibility with multi-channel image processing pipelines
        final_spectrogram = np.zeros((height, width, channels), dtype=spectrogram.dtype)
        for c in range(channels):
            final_spectrogram[:, :, c] = spectrogram
        return final_spectrogram
