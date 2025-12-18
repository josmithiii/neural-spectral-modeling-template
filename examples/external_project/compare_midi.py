#!/usr/bin/env python3
"""MIDI Comparison Viewer - Compare NSMT vs Basic Pitch onset detections.

Visual tool to compare MIDI onset detections overlaid on audio waveform.
Navigate through segments with Next/Previous buttons or arrow keys.

Usage:
    python compare_midi.py bass-mono.wav bass-nsmt.mid bass.mid
    python compare_midi.py bass-mono.wav bass-nsmt.mid bass.mid --window 5.0
    python compare_midi.py bass-mono.wav bass-nsmt.mid bass.mid --start 120.0
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import mido
import numpy as np
import soundfile as sf
from matplotlib.widgets import Button, TextBox


# MIDI note names for display
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def midi_note_to_name(note: int) -> str:
    """Convert MIDI note number to note name (e.g., 60 -> 'C4')."""
    octave = (note // 12) - 1
    name = NOTE_NAMES[note % 12]
    return f"{name}{octave}"


@dataclass
class OnsetEvent:
    """A single onset from either source."""
    time: float           # Onset time in seconds
    midi_note: int        # MIDI note number (0-127)
    velocity: int         # MIDI velocity (0-127)
    source: str           # "nsmt" or "basic_pitch"


@dataclass
class ViewerState:
    """Current state of the viewer."""
    audio: np.ndarray     # Full audio waveform
    sample_rate: int      # Audio sample rate
    nsmt_onsets: List[OnsetEvent]
    bp_onsets: List[OnsetEvent]
    current_idx: int      # Current segment index
    window_size: float    # Time window in seconds
    segment_times: List[float]  # Start time of each segment


def load_audio(wav_path: str) -> Tuple[np.ndarray, int]:
    """Load audio file, return (samples, sample_rate)."""
    audio, sr = sf.read(wav_path)
    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def load_midi_onsets(midi_path: str, source: str) -> List[OnsetEvent]:
    """Parse MIDI file, extract note-on events as OnsetEvent list."""
    midi_file = mido.MidiFile(midi_path)
    onsets = []

    # Track cumulative time in seconds
    current_time = 0.0

    for track in midi_file.tracks:
        current_time = 0.0
        for msg in track:
            # Convert delta time to seconds
            current_time += mido.tick2second(msg.time, midi_file.ticks_per_beat,
                                             get_tempo(midi_file, current_time))

            # Capture note_on events with velocity > 0
            if msg.type == 'note_on' and msg.velocity > 0:
                onsets.append(OnsetEvent(
                    time=current_time,
                    midi_note=msg.note,
                    velocity=msg.velocity,
                    source=source
                ))

    # Sort by time
    onsets.sort(key=lambda x: x.time)
    return onsets


def get_tempo(midi_file: mido.MidiFile, current_time: float) -> int:
    """Get tempo at current time (simplified - uses first tempo found or default)."""
    for track in midi_file.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                return msg.tempo
    return 500000  # Default: 120 BPM


def compute_segments(audio_duration: float, window_size: float,
                     overlap: float = 0.5) -> List[float]:
    """Compute segment start times with overlap."""
    step = window_size * (1 - overlap)
    segments = []
    t = 0.0
    while t < audio_duration - window_size * 0.5:
        segments.append(t)
        t += step
    if not segments:
        segments = [0.0]
    return segments


def format_note_info(onsets: List[OnsetEvent], t_start: float,
                     t_end: float, source_name: str, color: str) -> List[str]:
    """Format onset info text for display panel."""
    in_window = [o for o in onsets if t_start <= o.time < t_end]
    if not in_window:
        return [f"{source_name} ({color}): (none)"]

    lines = [f"{source_name} ({color}):"]
    for o in in_window:
        note_name = midi_note_to_name(o.midi_note)
        lines.append(f"  {note_name} ({o.midi_note}) v={o.velocity} @ {o.time:.2f}s")
    return lines


class MidiComparisonViewer:
    """Main viewer class with matplotlib figure and widgets."""

    def __init__(self, wav_path: str, nsmt_midi: str, bp_midi: str,
                 window_size: float = 3.0, start_time: float = 0.0):
        # Load data
        print(f"Loading audio: {wav_path}")
        audio, sr = load_audio(wav_path)

        print(f"Loading NSMT MIDI: {nsmt_midi}")
        nsmt_onsets = load_midi_onsets(nsmt_midi, "nsmt")

        print(f"Loading Basic Pitch MIDI: {bp_midi}")
        bp_onsets = load_midi_onsets(bp_midi, "basic_pitch")

        # Compute segments
        audio_duration = len(audio) / sr
        segment_times = compute_segments(audio_duration, window_size)

        # Find starting segment
        start_idx = 0
        if start_time > 0:
            for i, t in enumerate(segment_times):
                if t >= start_time:
                    start_idx = max(0, i - 1)
                    break

        # Initialize state
        self.state = ViewerState(
            audio=audio,
            sample_rate=sr,
            nsmt_onsets=nsmt_onsets,
            bp_onsets=bp_onsets,
            current_idx=start_idx,
            window_size=window_size,
            segment_times=segment_times
        )

        self.wav_path = Path(wav_path).name

        # Print summary
        print(f"\nSummary:")
        print(f"  Audio duration: {audio_duration:.1f}s")
        print(f"  NSMT onsets: {len(nsmt_onsets)}")
        print(f"  Basic Pitch onsets: {len(bp_onsets)}")
        print(f"  Segments: {len(segment_times)}")
        print(f"  Window size: {window_size}s")

        # Setup figure
        self._setup_figure()
        self._setup_widgets()
        self.update_plot()

    def _setup_figure(self):
        """Create matplotlib figure with axes."""
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title(f'MIDI Comparison: {self.wav_path}')

        # Main waveform axes (top portion)
        self.ax_wave = self.fig.add_axes([0.08, 0.35, 0.84, 0.55])

        # Info text axes (middle portion) - invisible axes for text
        self.ax_info = self.fig.add_axes([0.08, 0.12, 0.84, 0.20])
        self.ax_info.axis('off')

        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def _setup_widgets(self):
        """Setup navigation buttons and controls."""
        # Previous button
        ax_prev = self.fig.add_axes([0.08, 0.02, 0.12, 0.05])
        self.btn_prev = Button(ax_prev, '<< Prev')
        self.btn_prev.on_clicked(self.on_prev)

        # Next button
        ax_next = self.fig.add_axes([0.22, 0.02, 0.12, 0.05])
        self.btn_next = Button(ax_next, 'Next >>')
        self.btn_next.on_clicked(self.on_next)

        # Window size buttons
        ax_w2 = self.fig.add_axes([0.45, 0.02, 0.08, 0.05])
        self.btn_w2 = Button(ax_w2, '2s')
        self.btn_w2.on_clicked(lambda e: self.set_window(2.0))

        ax_w3 = self.fig.add_axes([0.54, 0.02, 0.08, 0.05])
        self.btn_w3 = Button(ax_w3, '3s')
        self.btn_w3.on_clicked(lambda e: self.set_window(3.0))

        ax_w5 = self.fig.add_axes([0.63, 0.02, 0.08, 0.05])
        self.btn_w5 = Button(ax_w5, '5s')
        self.btn_w5.on_clicked(lambda e: self.set_window(5.0))

        ax_w10 = self.fig.add_axes([0.72, 0.02, 0.08, 0.05])
        self.btn_w10 = Button(ax_w10, '10s')
        self.btn_w10.on_clicked(lambda e: self.set_window(10.0))

        # Jump to time text box
        ax_jump = self.fig.add_axes([0.86, 0.02, 0.08, 0.05])
        self.text_jump = TextBox(ax_jump, 'Go to:', initial='')
        self.text_jump.on_submit(self.on_jump)

    def set_window(self, window_size: float):
        """Change window size and recompute segments."""
        audio_duration = len(self.state.audio) / self.state.sample_rate
        current_time = self.state.segment_times[self.state.current_idx]

        self.state.window_size = window_size
        self.state.segment_times = compute_segments(audio_duration, window_size)

        # Find closest segment to current time
        self.state.current_idx = 0
        for i, t in enumerate(self.state.segment_times):
            if t > current_time:
                self.state.current_idx = max(0, i - 1)
                break
            self.state.current_idx = i

        self.update_plot()

    def on_next(self, event):
        """Handle Next button click."""
        if self.state.current_idx < len(self.state.segment_times) - 1:
            self.state.current_idx += 1
            self.update_plot()

    def on_prev(self, event):
        """Handle Previous button click."""
        if self.state.current_idx > 0:
            self.state.current_idx -= 1
            self.update_plot()

    def on_key(self, event):
        """Handle keyboard navigation."""
        if event.key == 'right':
            self.on_next(event)
        elif event.key == 'left':
            self.on_prev(event)
        elif event.key == 'home':
            self.state.current_idx = 0
            self.update_plot()
        elif event.key == 'end':
            self.state.current_idx = len(self.state.segment_times) - 1
            self.update_plot()

    def on_jump(self, text):
        """Handle jump to time."""
        try:
            target_time = float(text)
            for i, t in enumerate(self.state.segment_times):
                if t > target_time:
                    self.state.current_idx = max(0, i - 1)
                    break
                self.state.current_idx = i
            self.update_plot()
        except ValueError:
            pass  # Ignore invalid input

    def update_plot(self):
        """Redraw current segment."""
        state = self.state
        t_start = state.segment_times[state.current_idx]
        t_end = t_start + state.window_size

        # Clamp to audio duration
        audio_duration = len(state.audio) / state.sample_rate
        t_end = min(t_end, audio_duration)

        # Get audio samples for window
        start_sample = int(t_start * state.sample_rate)
        end_sample = int(t_end * state.sample_rate)
        audio_window = state.audio[start_sample:end_sample]
        time_axis = np.linspace(t_start, t_end, len(audio_window))

        # Clear and redraw waveform
        self.ax_wave.clear()
        self.ax_wave.plot(time_axis, audio_window, 'k-', linewidth=0.5, alpha=0.7)

        # Get Y limits for onset markers
        y_min, y_max = self.ax_wave.get_ylim()

        # Draw NSMT onsets (blue, solid)
        nsmt_in_window = [o for o in state.nsmt_onsets if t_start <= o.time < t_end]
        for onset in nsmt_in_window:
            self.ax_wave.axvline(x=onset.time, color='blue', linestyle='-',
                                 linewidth=1.5, alpha=0.8, label='NSMT' if onset == nsmt_in_window[0] else '')

        # Draw Basic Pitch onsets (orange, dashed)
        bp_in_window = [o for o in state.bp_onsets if t_start <= o.time < t_end]
        for onset in bp_in_window:
            self.ax_wave.axvline(x=onset.time, color='orange', linestyle='--',
                                 linewidth=1.5, alpha=0.8, label='Basic Pitch' if onset == bp_in_window[0] else '')

        # Title and labels
        segment_str = f"Segment {state.current_idx + 1} of {len(state.segment_times)}"
        self.ax_wave.set_title(f'MIDI Comparison: {self.wav_path}  |  {segment_str}', fontsize=12)
        self.ax_wave.set_xlabel(f'Time (s)  |  Window: {t_start:.2f}s - {t_end:.2f}s  |  Size: {state.window_size}s')
        self.ax_wave.set_ylabel('Amplitude')
        self.ax_wave.set_xlim(t_start, t_end)

        # Legend (only if onsets present)
        if nsmt_in_window or bp_in_window:
            self.ax_wave.legend(loc='upper right')

        # Update info panel
        self.ax_info.clear()
        self.ax_info.axis('off')

        nsmt_lines = format_note_info(state.nsmt_onsets, t_start, t_end, "NSMT", "blue")
        bp_lines = format_note_info(state.bp_onsets, t_start, t_end, "Basic Pitch", "orange")

        info_text = '\n'.join(nsmt_lines + [''] + bp_lines)
        self.ax_info.text(0.0, 1.0, info_text, transform=self.ax_info.transAxes,
                         fontsize=10, verticalalignment='top', fontfamily='monospace')

        self.fig.canvas.draw_idle()

    def run(self):
        """Start the viewer (plt.show())."""
        print("\nControls:")
        print("  Left/Right arrows or buttons: Navigate segments")
        print("  Home/End: Jump to first/last segment")
        print("  Window buttons: Change time window size")
        print("  'Go to:' box: Jump to specific time (seconds)")
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Compare MIDI onset detections from NSMT vs Basic Pitch')
    parser.add_argument('wav_path', help='Path to audio WAV file')
    parser.add_argument('nsmt_midi', help='Path to NSMT MIDI output')
    parser.add_argument('bp_midi', help='Path to Basic Pitch MIDI output')
    parser.add_argument('--window', type=float, default=3.0,
                        help='Time window size in seconds (default: 3.0)')
    parser.add_argument('--start', type=float, default=0.0,
                        help='Start time in seconds (default: 0.0)')

    args = parser.parse_args()

    # Validate files exist
    for path, name in [(args.wav_path, 'WAV'),
                       (args.nsmt_midi, 'NSMT MIDI'),
                       (args.bp_midi, 'Basic Pitch MIDI')]:
        if not Path(path).exists():
            print(f"Error: {name} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    viewer = MidiComparisonViewer(
        args.wav_path, args.nsmt_midi, args.bp_midi,
        window_size=args.window, start_time=args.start
    )
    viewer.run()


if __name__ == '__main__':
    main()
