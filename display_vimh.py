#!/usr/bin/env python3
"""
Interactive dataset viewer for VIMH format audio spectrogram datasets.
Displays synthesis parameters and mel spectrograms with support for variable parameter counts.
"""

import argparse
import json
import pickle
import struct
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D

from src.utils.synth_utils import SimpleSawSynth
from src.data.vimh_dataset import VIMHDataset


class VIMHViewer:
    def __init__(self, dataset_path: str, channel: int = 0):
        self.dataset_path = Path(dataset_path)
        self.current_idx = 0
        self.channel = channel

        # Load dataset using the existing VIMHDataset class
        try:
            self.dataset = VIMHDataset(str(dataset_path), train=True)
        except FileNotFoundError:
            # Try loading test data if train data doesn't exist
            self.dataset = VIMHDataset(str(dataset_path), train=False)

        # Get dataset info from the loaded dataset
        self.dataset_info = self.dataset.metadata_format

        # Get image shape and determine number of channels
        image_shape = self.dataset.get_image_shape()  # Returns (channels, height, width)
        self.num_channels = image_shape[0]

        # Get channel labels (VIMH v2.1+ feature)
        self.channel_labels = self.dataset_info.get("channel_labels", None)
        if self.channel_labels is None:
            # Use default labels for backward compatibility
            if self.num_channels == 1:
                self.channel_labels = ["Gray"]
            elif self.num_channels == 3:
                self.channel_labels = ["R", "G", "B"]
            elif self.num_channels == 4:
                self.channel_labels = ["R", "G", "B", "A"]
            else:
                self.channel_labels = [f"Ch{i}" for i in range(self.num_channels)]

        self.total_samples = len(self.dataset)

        # Initialize synthesizer for audio generation
        sample_rate = self.dataset_info.get("sample_rate", 8000)
        self.synth = SimpleSawSynth(sample_rate=sample_rate)

        # Setup matplotlib - create all figures first so tabs are accounted for
        self.audio_waterfall_created = True
        self.setup_plot()
        self.setup_waterfall_plot()
        self.setup_audio_waterfall_plot()

        # Force redraw of Figure 1 after all figures exist (for proper tab spacing)
        self.fig.canvas.draw()



    def get_sample_info(self, idx: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Get spectrogram data and synthesis parameters for sample idx."""
        # Get the sample from the dataset (returns torch tensor in CHW format)
        sample_data = self.dataset[idx]
        
        # Handle both 2-tuple (pickle format) and 3-tuple (binary format) cases
        if len(sample_data) == 2:
            image_tensor, labels_dict = sample_data
        else:
            image_tensor, labels_dict = sample_data[0], sample_data[1]
        
        # Convert from CHW to HWC format for display
        spectrogram = image_tensor.permute(1, 2, 0).numpy()
        
        # Get metadata for this sample
        sample_metadata = self.dataset._get_sample_metadata(idx)
        
        # Extract synthesis parameters from the dataset info
        param_info = {
            "vimh_labels": sample_metadata.get("labels", {}),
            "sample_index": idx,
            "parameters": self.dataset_info.get("parameter_mappings", {}),
            "actual_values": self.decode_actual_values(labels_dict),
        }

        return spectrogram, param_info

    def decode_actual_values(self, labels_dict: Dict[str, int]) -> Dict[str, float]:
        """Decode actual parameter values from labels dictionary."""
        params = {}
        param_mappings = self.dataset_info.get("parameter_mappings", {})
        
        # Initialize all parameters to their minimum values
        for param_name, param_info in param_mappings.items():
            params[param_name] = param_info["min"]
        
        # Decode quantized values back to actual parameter ranges
        for param_name, quantized_value in labels_dict.items():
            if param_name in param_mappings:
                param_info = param_mappings[param_name]
                min_val = param_info.get("min", 0)
                max_val = param_info.get("max", 255)
                
                # Convert quantized value (0-255) to normalized (0-1) then to actual range
                normalized_value = quantized_value / 255.0
                actual_value = min_val + normalized_value * (max_val - min_val)
                params[param_name] = actual_value
            else:
                # If parameter not in mappings, store as-is
                params[param_name] = quantized_value
        
        return params

    def setup_plot(self):
        """Setup the matplotlib figure and axes."""
        self.fig = plt.figure(figsize=(15, 8))
        dataset_name = self.dataset_info.get("dataset_name", self.dataset_path.name)
        self.fig.suptitle(f"VIMH Dataset: {dataset_name}", fontsize=14)

        # Create layout: parameters (left), spectrogram (center), colorbar (right)
        # Increased left column width to prevent text overlap
        # NOTE: Due to matplotlib layout bug with tabs, initial display may cut off title - resize window to fix
        gs = self.fig.add_gridspec(2, 3, width_ratios=[1.4, 2, 0.05], height_ratios=[4, 1], top=0.88, bottom=0.15)

        # Parameters text area
        self.ax_params = self.fig.add_subplot(gs[0, 0])
        self.ax_params.set_title("Synthesis Parameters")
        self.ax_params.axis("off")

        # Spectrogram (type determined from dataset info)
        self.ax_spec = self.fig.add_subplot(gs[0, 1])
        # Title will be set dynamically based on spectrogram type
        self.ax_spec.set_xlabel("Time (frames)")
        # Y-axis label will be set dynamically based on spectrogram type

        # Colorbar axis
        self.ax_cbar = self.fig.add_subplot(gs[0, 2])

        # Navigation buttons
        self.ax_prev = self.fig.add_subplot(gs[1, 0])
        self.ax_next = self.fig.add_subplot(gs[1, 1])

        self.btn_prev = Button(self.ax_prev, "Previous")
        self.btn_next = Button(self.ax_next, "Next")

        self.btn_prev.on_clicked(self.prev_sample)
        self.btn_next.on_clicked(self.next_sample)

        # Initial display
        self.update_display()

        # DID NOT HELP: # Force layout refresh to apply gridspec settings - microscopic resize trick
        # current_size = self.fig.get_size_inches()
        # self.fig.set_size_inches(current_size[0]+0.1, current_size[1]+0.1)
        # self.fig.canvas.draw()
        # self.fig.set_size_inches(current_size[0], current_size[1])

        self.fig.canvas.draw()

        # Keyboard shortcuts
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

    def setup_waterfall_plot(self):
        """Setup the waterfall plot in a second window."""
        self.fig_3d = plt.figure(figsize=(12, 8))

        # Get spectrogram type for dynamic labeling
        spectrogram_type = self.dataset_info.get("spectrogram_config", {}).get("type", "mel")
        spec_name = spectrogram_type.upper() if spectrogram_type else "Spectral"

        self.fig_3d.suptitle(
            f"Waterfall Plot - {spec_name} Spectral Slices Over Time", fontsize=14, y=0.95
        )

        # Create 3D axis with top margin
        self.ax_3d = self.fig_3d.add_subplot(111, projection="3d")
        self.fig_3d.subplots_adjust(top=0.85)
        self.ax_3d.set_xlabel("Time (frames)")

        # Set Y-axis label based on spectrogram type
        if spectrogram_type == "mel":
            self.ax_3d.set_ylabel("Frequency (mel bins)")
        elif spectrogram_type == "stft":
            self.ax_3d.set_ylabel("Frequency (bins)")
        else:
            self.ax_3d.set_ylabel("Frequency (bins)")

        self.ax_3d.set_zlabel("Magnitude")

        # Connect keyboard events for the 3D plot
        self.fig_3d.canvas.mpl_connect("key_press_event", self.on_key_press_3d)

        # Initial waterfall display
        self.update_waterfall_display()

    def update_waterfall_display(self):
        """Update the waterfall plot with current sample."""
        spectrogram, info = self.get_sample_info(self.current_idx)

        # Clear previous plot
        self.ax_3d.clear()

        # Handle different spectrogram shapes (grayscale vs RGB)
        if len(spectrogram.shape) == 3:
            height, width, channels = spectrogram.shape
            # Use selected channel for 3D visualization
            channel_idx = min(self.channel, channels - 1)
            spectrogram_2d = spectrogram[:, :, channel_idx]
        else:
            height, width = spectrogram.shape
            spectrogram_2d = spectrogram

        # Create coordinate arrays
        time_frames = np.arange(width)
        freq_bins = np.arange(height)

        # Create meshgrids for interpolated surface
        T, F = np.meshgrid(time_frames, freq_bins)

        # Plot both smooth surface and wireframe for best visualization
        # 1. Smooth interpolated surface (semi-transparent)
        surf = self.ax_3d.plot_surface(
            T, F, spectrogram_2d, cmap="viridis", alpha=0.6, linewidth=0, antialiased=True
        )

        # 2. Wireframe lattice to show discrete data structure
        wire = self.ax_3d.plot_wireframe(
            T, F, spectrogram_2d, color="black", linewidth=0.5, alpha=0.8
        )

        # Add grid lines to show actual data points
        # Vertical lines (time frames) - sample every few to avoid clutter
        for t in range(0, width, max(1, width // 8)):
            spectrum_slice = spectrogram_2d[:, t]
            time_line = np.full_like(freq_bins, t)
            self.ax_3d.plot(
                time_line, freq_bins, spectrum_slice, color="black", linewidth=0.8, alpha=0.6
            )

        # Horizontal lines (frequency bins) - sample every few to avoid clutter
        for f in range(0, height, max(1, height // 8)):
            time_slice = spectrogram_2d[f, :]
            freq_line = np.full_like(time_frames, f)
            self.ax_3d.plot(
                time_frames, freq_line, time_slice, color="black", linewidth=0.8, alpha=0.6
            )

        # Set labels and title
        self.ax_3d.set_xlabel("Time (frames)")

        # Set Y-axis label based on spectrogram type
        spectrogram_type = self.dataset_info.get("spectrogram_config", {}).get("type", "mel")
        if spectrogram_type == "mel":
            self.ax_3d.set_ylabel("Frequency (mel bins)")
        elif spectrogram_type == "stft":
            self.ax_3d.set_ylabel("Frequency (bins)")
        else:
            self.ax_3d.set_ylabel("Frequency (bins)")

        self.ax_3d.set_zlabel("Magnitude")

        # Update title with sample info
        labels = info["vimh_labels"]
        if isinstance(labels, dict):
            labels_str = " ".join(f"{k}={v:.3f}" for k, v in labels.items())
        else:
            labels_str = " ".join(map(str, labels))
        if len(labels_str) > 50:
            labels_str = labels_str[:50] + "..."
        # Get spectrogram type for dynamic title
        spectrogram_type = self.dataset_info.get("spectrogram_config", {}).get("type", "mel")
        spec_name = spectrogram_type.upper() if spectrogram_type else "Spectral"

        # Get channel label name
        channel_label = self.channel_labels[self.channel] if self.channel < len(self.channel_labels) else f"Ch{self.channel}"

        title = f"Sample {self.current_idx + 1}/{self.total_samples}, Channel {self.channel} ({channel_label})\nVIMH Parameter Labels: [{labels_str}]"
        self.fig_3d.suptitle(
            f"Waterfall Plot - {spec_name} Spectral Slices Over Time\n{title}", fontsize=12, y=0.95
        )

        # Set symmetric Z-axis limits
        z_max = np.abs(spectrogram_2d).max()
        self.ax_3d.set_zlim(-z_max, z_max)

        # Set viewing angle for better perspective
        self.ax_3d.view_init(elev=20, azim=45)

        # Draw the plot
        self.fig_3d.canvas.draw()

    def on_key_press_3d(self, event):
        """Handle keyboard shortcuts for 3D plot."""
        if event.key in ["left", "p"]:
            self.prev_sample()
        elif event.key in ["right", "n"]:
            self.next_sample()
        elif event.key in ["up", "u"]:
            self.prev_channel()
        elif event.key in ["down", "d"]:
            self.next_channel()
        elif event.key == "q":
            plt.close(self.fig_3d)
        elif event.key in ["home", "h"]:
            self.current_idx = 0
            self.update_display()
        elif event.key in ["end", "e"]:
            self.current_idx = self.total_samples - 1
            self.update_display()
        elif event.key == "r":
            # Reset view angle
            self.ax_3d.view_init(elev=20, azim=45)
            self.fig_3d.canvas.draw()
        elif event.key == "a":
            # Toggle audio waterfall plot
            if not self.audio_waterfall_created:
                self.setup_audio_waterfall_plot()
                self.audio_waterfall_created = True
            else:
                if hasattr(self, "fig_audio") and plt.fignum_exists(self.fig_audio.number):
                    plt.close(self.fig_audio)
                    self.audio_waterfall_created = False
        elif event.key == "cmd+1":
            # Bring Figure 1 (main spectrogram) to front
            if hasattr(self, "fig") and plt.fignum_exists(self.fig.number):
                self.fig.canvas.manager.show()
                self.fig.canvas.draw()
        elif event.key == "cmd+2":
            # Bring Figure 2 (waterfall) to front
            if hasattr(self, "fig_3d") and plt.fignum_exists(self.fig_3d.number):
                self.fig_3d.canvas.manager.show()
                self.fig_3d.canvas.draw()
        elif event.key == "cmd+3":
            # Bring Figure 3 (audio waterfall) to front
            if hasattr(self, "fig_audio") and plt.fignum_exists(self.fig_audio.number):
                self.fig_audio.canvas.manager.show()
                self.fig_audio.canvas.draw()

    def setup_audio_waterfall_plot(self):
        """Setup the audio waterfall plot in a third window."""
        self.fig_audio = plt.figure(figsize=(12, 8))
        self.fig_audio.suptitle("Audio Waveform Waterfall - Raw Audio Frames", fontsize=14, y=0.95)

        # Create 3D axis with top margin
        self.ax_audio = self.fig_audio.add_subplot(111, projection="3d")
        self.fig_audio.subplots_adjust(top=0.85)
        self.ax_audio.set_xlabel("Waveform Sample Index")
        self.ax_audio.set_ylabel("Time Frame")
        self.ax_audio.set_zlabel("Amplitude")

        # Connect keyboard events for the audio plot
        self.fig_audio.canvas.mpl_connect("key_press_event", self.on_key_press_audio)

        # Initial audio waterfall display
        self.update_audio_waterfall_display()

    def update_audio_waterfall_display(self):
        """Update the audio waterfall plot with current sample."""
        spectrogram, info = self.get_sample_info(self.current_idx)

        # Clear previous plot
        self.ax_audio.clear()

        # Generate audio from synthesis parameters
        actual_params = info["actual_values"]

        # Ensure duration is set
        if "duration" not in actual_params:
            actual_params["duration"] = self.dataset_info.get("duration", 1.0)

        audio = self.synth.generate_audio(actual_params)

        # Calculate frame parameters based on STFT config
        stft_config = self.dataset_info.get("spectrogram_config", {})
        n_fft = stft_config.get("n_fft", 512)
        hop_length = stft_config.get("hop_length", 64)

        # Split audio into frames for waterfall display
        num_frames = (len(audio) - n_fft) // hop_length + 1
        frame_indices = np.arange(n_fft)

        # Plot audio frames as waterfall
        for frame_idx in range(0, min(num_frames, 32), max(1, num_frames // 32)):
            start_sample = frame_idx * hop_length
            end_sample = start_sample + n_fft

            if end_sample <= len(audio):
                audio_frame = audio[start_sample:end_sample]
                frame_line = np.full_like(frame_indices, frame_idx)

                self.ax_audio.plot(
                    frame_indices,
                    frame_line,
                    audio_frame,
                    color=plt.cm.viridis(frame_idx / num_frames),
                    linewidth=1.0,
                    alpha=0.8,
                )

        # Set labels and title
        self.ax_audio.set_xlabel("Waveform Sample Index")
        self.ax_audio.set_ylabel("Time Frame")
        self.ax_audio.set_zlabel("Amplitude")

        # Update title with sample info
        labels = info["vimh_labels"]
        if isinstance(labels, dict):
            labels_str = " ".join(f"{k}={v:.3f}" for k, v in labels.items())
        else:
            labels_str = " ".join(map(str, labels))
        if len(labels_str) > 50:
            labels_str = labels_str[:50] + "..."
        title = f"Sample {self.current_idx + 1}/{self.total_samples}\nVIMH Parameter Labels: [{labels_str}]"
        self.fig_audio.suptitle(
            f"Audio Waveform Waterfall - Raw Audio Frames\n{title}", fontsize=12, y=0.95
        )

        # Set symmetric Z-axis limits for audio amplitude
        z_max = np.abs(audio).max()
        self.ax_audio.set_zlim(-z_max, z_max)

        # Set viewing angle for better perspective
        self.ax_audio.view_init(elev=20, azim=45)

        # Draw the plot
        self.fig_audio.canvas.draw()
        self.fig_audio.canvas.flush_events()

    def on_key_press_audio(self, event):
        """Handle keyboard shortcuts for audio plot."""
        if event.key in ["left", "p"]:
            self.prev_sample()
        elif event.key in ["right", "n"]:
            self.next_sample()
        elif event.key in ["up", "u"]:
            self.prev_channel()
        elif event.key in ["down", "d"]:
            self.next_channel()
        elif event.key == "q":
            plt.close(self.fig_audio)
        elif event.key in ["home", "h"]:
            self.current_idx = 0
            self.update_display()
        elif event.key in ["end", "e"]:
            self.current_idx = self.total_samples - 1
            self.update_display()
        elif event.key == "r":
            # Reset view angle
            self.ax_audio.view_init(elev=20, azim=45)
            self.fig_audio.canvas.draw()
        elif event.key == "a":
            # Toggle audio waterfall plot
            if not self.audio_waterfall_created:
                self.setup_audio_waterfall_plot()
                self.audio_waterfall_created = True
            else:
                if hasattr(self, "fig_audio") and plt.fignum_exists(self.fig_audio.number):
                    plt.close(self.fig_audio)
                    self.audio_waterfall_created = False
        elif event.key == "cmd+1":
            # Bring Figure 1 (main spectrogram) to front
            if hasattr(self, "fig") and plt.fignum_exists(self.fig.number):
                self.fig.canvas.manager.show()
                self.fig.canvas.draw()
        elif event.key == "cmd+2":
            # Bring Figure 2 (waterfall) to front
            if hasattr(self, "fig_3d") and plt.fignum_exists(self.fig_3d.number):
                self.fig_3d.canvas.manager.show()
                self.fig_3d.canvas.draw()
        elif event.key == "cmd+3":
            # Bring Figure 3 (audio waterfall) to front
            if hasattr(self, "fig_audio") and plt.fignum_exists(self.fig_audio.number):
                self.fig_audio.canvas.manager.show()
                self.fig_audio.canvas.draw()

    def update_display(self):
        """Update the display with current sample."""
        spectrogram, info = self.get_sample_info(self.current_idx)

        # Clear previous content
        self.ax_params.clear()
        self.ax_spec.clear()
        self.ax_cbar.clear()

        # Format VIMH labels for display
        labels_str = " ".join(map(str, info["vimh_labels"]))
        if len(labels_str) > 50:
            labels_str = labels_str[:50] + "..."

        # Update title with sample info
        title = (
            f"Sample {self.current_idx + 1}/{self.total_samples}, Channel {self.channel}\n"
            f"VIMH Labels: [{labels_str}]"
        )
        dataset_name = self.dataset_info.get("dataset_name", self.dataset_path.name)
        self.fig.suptitle(f"VIMH Dataset: {dataset_name}\n{title}", fontsize=11, y=0.98)

        # Display synthesis parameters
        self.ax_params.set_title("Synthesis Parameters", fontsize=12)
        self.ax_params.axis("off")

        param_text = []
        varying_params = []
        fixed_params = []

        # Get actual parameter values for this sample
        actual_values = info.get("actual_values", {})

        for param_name, param_info in info["parameters"].items():
            min_val = param_info["min"]
            max_val = param_info["max"]
            actual_val = actual_values.get(param_name, min_val)

            if min_val == max_val:
                fixed_params.append(f"{actual_val:6.2f}  {param_name}")
            else:
                varying_params.append(
                    f"{actual_val:6.2f}  {param_name} ({min_val:.2f}-{max_val:.2f})"
                )

        # Add VIMH format info
        labels = info["vimh_labels"]
        if labels:
            # For new dictionary format, count non-zero parameters
            N = len([v for v in labels.values() if v != 0.0]) if isinstance(labels, dict) else len(labels)
            vimh_version = self.dataset_info.get("version", "1.0")
            param_text.append(f"VIMH FORMAT v{vimh_version}:")
            param_text.append(f"N={N} varying parameters")

            # Add image dimensions and channel labels
            spectrogram, _ = self.get_sample_info(self.current_idx)
            if len(spectrogram.shape) == 3:
                height, width, channels = spectrogram.shape
                param_text.append(f"{height} x {width} x {channels}")
                # Show channel labels
                if self.channel_labels:
                    param_text.append("CHANNELS:")
                    for i, label in enumerate(self.channel_labels):
                        marker = " ►" if i == self.channel else "  "
                        param_text.append(f"{marker} {i}: {label}")
            else:
                height, width = spectrogram.shape
                param_text.append(f"{height} x {width} x 1")
                if self.channel_labels and len(self.channel_labels) > 0:
                    param_text.append(f"CHANNEL: {self.channel_labels[0]}")
            param_text.append("")

        if varying_params:
            param_text.append("VARYING PARAMETERS:")
            param_text.extend(varying_params)
            param_text.append("")

        if fixed_params:
            param_text.append("FIXED PARAMETERS:")
            param_text.extend(fixed_params)

        self.ax_params.text(
            0.05,
            0.95,
            "\n".join(param_text),
            transform=self.ax_params.transAxes,
            fontfamily="monospace",
            fontsize=9,
            verticalalignment="top",
        )

        # Display spectrogram with dynamic title based on type and channel label
        spectrogram_type = self.dataset_info.get("spectrogram_config", {}).get("type", "mel")
        spec_type_name = f"{spectrogram_type.upper()} Spectrogram" if spectrogram_type else "Spectrogram"

        # Add channel label to title
        if len(spectrogram.shape) == 3:
            channels = spectrogram.shape[2]
            channel_idx = min(self.channel, channels - 1)
            channel_label = self.channel_labels[channel_idx] if channel_idx < len(self.channel_labels) else f"Ch{channel_idx}"
            spec_title = f"{spec_type_name} - {channel_label}"
        else:
            spec_title = spec_type_name

        self.ax_spec.set_title(spec_title, fontsize=12)

        # Use the selected channel as the primary spectrogram data
        if len(spectrogram.shape) == 3:
            channels = spectrogram.shape[2]
            channel_idx = min(self.channel, channels - 1)
            spec_data = spectrogram[:, :, channel_idx]
        else:
            spec_data = spectrogram

        # Use nearest neighbor to show actual data points, with extent for proper scaling
        height, width = spec_data.shape
        im = self.ax_spec.imshow(
            spec_data,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
            extent=[-0.5, width - 0.5, -0.5, height - 0.5],
        )

        # Add grid lines to show actual data points - enhanced visibility
        # Vertical lines (time frames)
        for i in range(width + 1):
            self.ax_spec.axvline(x=i - 0.5, color="white", linewidth=0.8, alpha=0.7)
        # Horizontal lines (frequency bins)
        for i in range(height + 1):
            self.ax_spec.axhline(y=i - 0.5, color="white", linewidth=0.8, alpha=0.7)

        self.ax_spec.set_xlabel("Time (frames)")

        # Set Y-axis label based on spectrogram type
        spectrogram_type = self.dataset_info.get("spectrogram_config", {}).get("type", "mel")
        if spectrogram_type == "mel":
            self.ax_spec.set_ylabel("Mel Bin")
        elif spectrogram_type == "stft":
            self.ax_spec.set_ylabel("Frequency Bin")
        else:
            self.ax_spec.set_ylabel("Frequency Bin")

        # Add colorbar
        cbar = plt.colorbar(im, cax=self.ax_cbar)
        cbar.set_label("Magnitude", rotation=270, labelpad=20)

        # Update navigation info with channel label
        current_channel_label = self.channel_labels[self.channel] if self.channel < len(self.channel_labels) else f"Ch{self.channel}"
        nav_text = f"Left/Right: samples, Up/Down: channels, Q: quit, A: audio waterfall, Cmd+1/2/3: figures\nSample {self.current_idx + 1}/{self.total_samples}, Channel {self.channel + 1}/{self.num_channels} ({current_channel_label})"
        self.fig.text(0.5, 0.02, nav_text, ha="center", fontsize=10)

        # Force complete redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Update waterfall plot if it exists
        if hasattr(self, "fig_3d") and plt.fignum_exists(self.fig_3d.number):
            self.update_waterfall_display()

        # Update audio waterfall plot if it exists
        if hasattr(self, "fig_audio") and plt.fignum_exists(self.fig_audio.number):
            self.update_audio_waterfall_display()

    def prev_sample(self, event=None):
        """Go to previous sample."""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_display()

    def next_sample(self, event=None):
        """Go to next sample."""
        if self.current_idx < self.total_samples - 1:
            self.current_idx += 1
            self.update_display()

    def prev_channel(self, event=None):
        """Go to previous channel."""
        if self.channel > 0:
            self.channel -= 1
            self.update_display()

    def next_channel(self, event=None):
        """Go to next channel."""
        if self.channel < self.num_channels - 1:
            self.channel += 1
            self.update_display()

    def on_key_press(self, event):
        """Handle keyboard shortcuts."""
        if event.key in ["left", "p"]:
            self.prev_sample()
        elif event.key in ["right", "n"]:
            self.next_sample()
        elif event.key in ["up", "u"]:
            self.prev_channel()
        elif event.key in ["down", "d"]:
            self.next_channel()
        elif event.key == "q":
            plt.close(self.fig)
        elif event.key in ["home", "h"]:
            self.current_idx = 0
            self.update_display()
        elif event.key in ["end", "e"]:
            self.current_idx = self.total_samples - 1
            self.update_display()
        elif event.key == "a":
            # Toggle audio waterfall plot
            if not self.audio_waterfall_created:
                self.setup_audio_waterfall_plot()
                self.audio_waterfall_created = True
            else:
                if hasattr(self, "fig_audio") and plt.fignum_exists(self.fig_audio.number):
                    plt.close(self.fig_audio)
                    self.audio_waterfall_created = False
        elif event.key == "cmd+1":
            # Bring Figure 1 (main spectrogram) to front
            if hasattr(self, "fig") and plt.fignum_exists(self.fig.number):
                self.fig.canvas.manager.show()
                self.fig.canvas.draw()
        elif event.key == "cmd+2":
            # Bring Figure 2 (waterfall) to front
            if hasattr(self, "fig_3d") and plt.fignum_exists(self.fig_3d.number):
                self.fig_3d.canvas.manager.show()
                self.fig_3d.canvas.draw()
        elif event.key == "cmd+3":
            # Bring Figure 3 (audio waterfall) to front
            if hasattr(self, "fig_audio") and plt.fignum_exists(self.fig_audio.number):
                self.fig_audio.canvas.manager.show()
                self.fig_audio.canvas.draw()

    def show(self):
        """Display the interactive viewer."""
        plt.show()


def find_most_recent_dataset() -> str:
    """Find the most recent dataset in the data directory."""
    data_dir = Path("./data")
    if not data_dir.exists():
        raise FileNotFoundError("data directory not found")

    # Find all subdirectories that contain vimh_dataset_info.json
    dataset_dirs = []
    for subdir in data_dir.iterdir():
        if subdir.is_dir() and (subdir / "vimh_dataset_info.json").exists():
            dataset_dirs.append(subdir)

    if not dataset_dirs:
        raise FileNotFoundError("No valid VIMH datasets found in data/")

    # Sort by modification time (most recent first)
    dataset_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    return str(dataset_dirs[0])


def main():
    parser = argparse.ArgumentParser(
        description="Interactive VIMH dataset viewer for audio spectrograms"
    )
    parser.add_argument(
        "dataset_path",
        nargs="?",
        help="Path to VIMH dataset directory (defaults to most recent in data/)",
    )
    parser.add_argument(
        "--channel", "-c", type=int, default=0, help="Channel to display (default: 0)"
    )

    args = parser.parse_args()

    # Use provided path or find the most recent dataset
    if args.dataset_path:
        dataset_path = args.dataset_path
    else:
        try:
            dataset_path = find_most_recent_dataset()
            print(f"Using most recent dataset: {dataset_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1

    try:
        viewer = VIMHViewer(dataset_path, channel=args.channel)
        print(f"Loaded VIMH dataset with {viewer.total_samples} samples")
        print(f"Channels: {viewer.num_channels}")
        print(f"Varying parameters: {viewer.dataset_info.get('varying_parameters', 0)}")
        print("Navigation:")
        print("  Left/Right arrows or P/N: Previous/Next sample")
        print("  Up/Down arrows or U/D: Previous/Next channel")
        print("  Home/H: First sample")
        print("  End/E: Last sample")
        print("  A: Toggle audio waterfall plot")
        print("  Cmd+1/2/3: Bring Figure 1/2/3 to front")
        print("  Q: Quit")
        print("")
        print("NOTE: If Figure 1 title is cut off initially, resize the window to fix layout.")
        viewer.show()
    except Exception as e:
        print(f"Error loading VIMH dataset: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
