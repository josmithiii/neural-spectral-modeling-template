"""Display utilities for audio reconstruction evaluation.

Shared functions for plotting spectrograms, parameters, and metrics
in both interactive and non-interactive display modes.
"""

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def prepare_spectrogram_for_display(
    input_spec: np.ndarray, debug: bool = False
) -> np.ndarray:
    """Prepare input spectrogram for display by extracting first channel.

    Args:
        input_spec: Input spectrogram tensor, either (H, W) or (C, H, W)
        debug: Whether to print debug information

    Returns:
        2D spectrogram array ready for display (H, W)

    Raises:
        ValueError: If input shape is not 2D or 3D
    """
    if debug:
        print(f"🔍 Input spectrogram original shape: {input_spec.shape}")

    # Remove channel dimension if present
    if len(input_spec.shape) == 3:
        # PyTorch uses channels-first format: (C, H, W)
        # Always take first channel for visualization
        if debug:
            print(f"   → Taking first channel from 3D tensor")
        input_spec = input_spec[0, :, :]  # (C, H, W) -> (H, W)
    elif len(input_spec.shape) != 2:
        raise ValueError(
            f"Unexpected input_spec shape: {input_spec.shape}. Expected 2D or 3D."
        )
    else:
        if debug:
            print(f"   → Already 2D, shape: {input_spec.shape}")

    if debug:
        print(f"   → After channel extraction: {input_spec.shape}")

    return input_spec


def format_parameter_value(value: float) -> str:
    """Format parameter value for compact display.

    Args:
        value: Parameter value to format

    Returns:
        Formatted string representation
    """
    av = abs(value)
    if av >= 1000 or (av > 0 and av < 1e-3):
        return f"{value:.2e}"
    if av >= 100:
        return f"{value:.0f}"
    if av >= 10:
        return f"{value:.1f}"
    return f"{value:.3f}"


def plot_parameter_comparison(
    ax: plt.Axes,
    param_names: List[str],
    true_vals: List[float],
    pred_vals: List[float],
    param_mappings: Dict[str, Dict[str, Any]],
    title_suffix: str = "",
) -> None:
    """Plot parameter comparison bars with normalized values.

    Args:
        ax: Matplotlib axes to plot on
        param_names: List of parameter names
        true_vals: List of true parameter values (natural scale)
        pred_vals: List of predicted parameter values (natural scale)
        param_mappings: Dictionary of parameter ranges for normalization
        title_suffix: Optional suffix for title (e.g., "[AVIX: Click to navigate]")
    """
    # Normalize values to [0, 1] using parameter mappings
    norm_true = []
    norm_pred = []
    param_ranges = []  # Store (min, max, step) for each parameter

    for p, tval, pval in zip(param_names, true_vals, pred_vals):
        mapping = param_mappings.get(p, {"min": 0.0, "max": 1.0})
        pmin = mapping.get("min", 0.0)
        pmax = mapping.get("max", 1.0)
        pstep = mapping.get("step", None)
        prange = max(1e-12, pmax - pmin)
        norm_true.append(np.clip((tval - pmin) / prange, 0.0, 1.0))
        norm_pred.append(np.clip((pval - pmin) / prange, 0.0, 1.0))
        param_ranges.append((pmin, pmax, pstep))

    # Create bar chart
    x = np.arange(len(param_names))
    width = 0.35
    bars_true = ax.bar(x - width / 2, norm_true, width, label="True", alpha=0.7)
    bars_pred = ax.bar(x + width / 2, norm_pred, width, label="Predicted", alpha=0.7)

    # Add natural value labels on top of bars
    for i, (bt, bp) in enumerate(zip(bars_true, bars_pred)):
        ax.text(
            bt.get_x() + bt.get_width() / 2,
            bt.get_height() + 0.02,
            format_parameter_value(true_vals[i]),
            ha="center",
            va="bottom",
            fontsize=8,
            color="tab:blue",
        )
        ax.text(
            bp.get_x() + bp.get_width() / 2,
            bp.get_height() + 0.02,
            format_parameter_value(pred_vals[i]),
            ha="center",
            va="bottom",
            fontsize=8,
            color="tab:orange",
        )

    # Add horizontal gridlines at normalized positions with step marks
    # Draw faint gridlines at 0.0, 0.25, 0.5, 0.75, 1.0
    for y_pos in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ax.axhline(y_pos, color='gray', linewidth=0.5, alpha=0.3, linestyle='--', zorder=0)

    # Add min/max labels for each parameter
    for i, (pname, (pmin, pmax, pstep)) in enumerate(zip(param_names, param_ranges)):
        # Position labels at the sides of each parameter's bars
        x_pos = x[i]

        # Min label at bottom (y=0)
        ax.text(
            x_pos, -0.03,
            format_parameter_value(pmin),
            ha='center', va='top',
            fontsize=7, color='gray', style='italic',
            transform=ax.get_xaxis_transform()
        )

        # Max label at top (y=1.0)
        ax.text(
            x_pos, 1.0,
            format_parameter_value(pmax),
            ha='center', va='bottom',
            fontsize=7, color='gray', style='italic'
        )

        # If we have step information, draw additional tick marks
        if pstep is not None and pstep > 0:
            prange = pmax - pmin
            num_steps = int(round(prange / pstep))
            if 2 < num_steps <= 20:  # Only draw if reasonable number of steps
                for step_idx in range(1, num_steps):
                    y_norm = step_idx / num_steps
                    # Draw short tick mark at this parameter's x position
                    ax.plot(
                        [x_pos - width*0.6, x_pos + width*0.6],
                        [y_norm, y_norm],
                        color='lightgray', linewidth=0.8, alpha=0.5, zorder=0
                    )

    # Configure axes
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Normalized Value [0,1]")
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45)
    title = "Parameters (Normalized)"
    if title_suffix:
        title += f" {title_suffix}"
    ax.set_title(title)
    ax.legend()


def get_metric_color(name: str, value: float) -> str:
    """Get color for metric based on quality thresholds.

    Args:
        name: Metric name
        value: Metric value

    Returns:
        Color string (green=good, orange=ok, red=poor, black=neutral)
    """
    if name == "mae":
        return "green" if value < 0.05 else "orange" if value < 0.15 else "red"
    elif name == "mse":
        return "green" if value < 0.01 else "orange" if value < 0.1 else "red"
    elif name == "mss_distance":
        # MSS distance: lower is better (PNP reports ~0.78, we see ~0.15-0.63)
        return "green" if value < 0.3 else "orange" if value < 0.6 else "red"
    elif name == "snr_db":
        return "green" if value > 20 else "orange" if value > 10 else "red"
    elif name in ["correlation", "max_xcorr"]:
        return "green" if value > 0.9 else "orange" if value > 0.7 else "red"
    elif name == "max_xcorr_lag_samples":
        return "green" if abs(value) <= 1 else "orange" if abs(value) <= 5 else "red"
    return "black"


def plot_audio_metrics(
    ax: plt.Axes, metrics: Dict[str, float], sample_rate: int
) -> None:
    """Plot audio quality metrics on axes.

    Args:
        ax: Matplotlib axes to plot on
        metrics: Dictionary of metric name -> value
        sample_rate: Audio sample rate for lag conversion
    """
    ax.set_title("Audio Metrics")
    ax.axis("off")  # Hide axes for text-only display

    # Display metrics in order with color coding
    y_pos = 0.9
    line_height = 0.15

    # Define metric order and formatting
    metric_order = ["snr_db", "correlation", "max_xcorr", "max_xcorr_lag_samples", "mae", "mse", "mss_distance"]

    for name in metric_order:
        if name in metrics and np.isfinite(metrics[name]):
            value = metrics[name]
            color = get_metric_color(name, value)

            if name == "snr_db":
                text = f"Waveform SNR: {value:.2f} dB"
            elif name == "correlation":
                text = f"Correlation: {value:.4f} / 1"
            elif name == "max_xcorr":
                text = f"Max XCorr: {value:.4f} / 1"
            elif name == "max_xcorr_lag_samples":
                lag_samples = int(round(value))
                lag_ms = (lag_samples / sample_rate) * 1000 if sample_rate else 0.0
                text = f"Max XCorr Lag: {lag_samples:+d} samples ({lag_ms:+.3f} ms)"
            elif name == "mae":
                text = f"MAE: {value:.8f}"
            elif name == "mse":
                text = f"MSE: {value:.8f}"
            elif name == "mss_distance":
                text = f"MSS Distance: {value:.6f}"
            else:
                text = f"{name}: {value:.4f}"

            ax.text(
                0.05,
                y_pos,
                text,
                ha="left",
                va="top",
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
                color=color,
            )
            y_pos -= line_height

    # Add definitions if there's room
    if y_pos > 0.3:
        ax.text(
            0.05,
            0.25,
            "Correlation: Linear relationship (-1 to +1)\nMax XCorr: Best time-aligned similarity",
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
            style="italic",
            color="gray",
        )
