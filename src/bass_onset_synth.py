# src/bass_onset_synth.py

from __future__ import annotations
from typing import Dict, Any
import numpy as np


class BassOnsetSynth:
    """
    Synthetic bass-note generator for onset-delay training.

    Expects params dict to contain:
      - onset_delay_ms
      - midi_pitch
      - note_duration_s
      - duration  (overall frame length, from cfg.dataset.duration)
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = int(sample_rate)

    def generate_audio(self, params: Dict[str, Any]) -> np.ndarray:
        sr = self.sr

        onset_delay_ms = float(params["onset_delay_ms"])
        midi_pitch = int(params["midi_pitch"])
        note_duration_s = float(params["note_duration_s"])
        frame_seconds = float(params.get("duration", 1.0))

        n_frame = int(round(frame_seconds * sr))

        onset_delay_ms = max(0.0, onset_delay_ms)
        note_duration_s = max(0.05, note_duration_s)

        onset_delay_s = onset_delay_ms / 1000.0
        n_delay = int(round(onset_delay_s * sr))
        n_note = int(round(note_duration_s * sr))

        t = np.linspace(0.0, note_duration_s, num=n_note, endpoint=False)
        f0 = 440.0 * (2.0 ** ((midi_pitch - 69.0) / 12.0))

        fundamental = np.sin(2.0 * np.pi * f0 * t)
        harmonic2 = 0.4 * np.sin(2.0 * np.pi * 2.0 * f0 * t)

        env = np.exp(-3.0 * t / max(note_duration_s, 1e-3))
        note = (0.7 * fundamental + harmonic2) * env

        audio = np.zeros(n_frame, dtype=np.float32)
        start = min(n_frame, n_delay)
        end = min(n_frame, start + n_note)
        if end > start:
            audio[start:end] += note[: end - start].astype(np.float32)

        audio += 1e-4 * np.random.randn(n_frame).astype(np.float32)

        peak = np.max(np.abs(audio))
        if peak > 0.99:
            audio = 0.99 * audio / peak

        return audio
