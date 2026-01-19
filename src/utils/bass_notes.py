# src/utils/bass_notes.py
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


def midi_to_hz(midi_note: float) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))


def one_pole_lowpass(x: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    cutoff_hz = float(np.clip(cutoff_hz, 10.0, 0.45 * sr))
    a = math.exp(-2.0 * math.pi * cutoff_hz / sr)
    y = np.empty_like(x)
    y0 = 0.0
    for i in range(len(x)):
        y0 = a * y0 + (1.0 - a) * x[i]
        y[i] = y0
    return y


def softclip_tanh(x: np.ndarray, drive: float) -> np.ndarray:
    drive = float(max(0.0, drive))
    if drive == 0.0:
        return x
    return np.tanh(drive * x) / np.tanh(drive)


@dataclass
class BassNotesSynthConfig:
    osc: str = "saw_lp"              # "saw_lp" or "sine_harm"
    add_attack_click: bool = True
    click_level: float = 0.08
    click_ms: float = 6.0
    lp_cutoff_hz: float = 900.0
    drive: float = 0.5


class BassNotesSynth:
    """
    Renders a synthetic monophonic bass note clip.

        Expected params keys:
      - duration          (seconds)
      - note_number       (MIDI pitch, float ok)
      - note_velocity     (0..127)
      - log10_decay_time  (log10 seconds)
      - onset_delay_ms    (ms)

    """

    def __init__(self, sample_rate: int = 16000, cfg: dict | None = None):
        cfg = cfg or {}
        self.sample_rate = int(sample_rate)
        self.cfg = BassNotesSynthConfig(
            osc=str(cfg.get("osc", "saw_lp")),
            add_attack_click=bool(cfg.get("add_attack_click", True)),
            click_level=float(cfg.get("click_level", 0.08)),
            click_ms=float(cfg.get("click_ms", 6.0)),
            lp_cutoff_hz=float(cfg.get("lp_cutoff_hz", 900.0)),
            drive=float(cfg.get("drive", 0.5)),
        )

    def generate_audio(self, params: dict) -> np.ndarray:
        duration_s = float(params.get("duration", 1.0))
        return self.render(params=params, sr=self.sample_rate, duration_s=duration_s)

    def render(self, params: dict, sr: int, duration_s: float) -> np.ndarray:
        n = int(round(sr * float(duration_s)))
        if n <= 0:
            return np.zeros((0,), dtype=np.float32)


        note_number = float(params.get("note_number", 40.0))
        note_velocity = float(params.get("note_velocity", 90.0))
        log10_decay_time = float(params.get("log10_decay_time", -0.3))
        onset_delay_ms = float(params.get("onset_delay_ms", 0.0))

        f0 = midi_to_hz(note_number)

        vel = float(np.clip(note_velocity, 0.0, 127.0))
        amp = (vel / 127.0) ** 1.2

        # Center the onset, then shift by onset_delay_ms
        center_t = 0.5 * float(duration_s)
        onset_t = center_t + (onset_delay_ms / 1000.0)
        onset_t = float(np.clip(onset_t, 0.01, float(duration_s) - 0.01))
        onset_i = int(round(onset_t * sr))

        tau = 10.0 ** log10_decay_time
        tau = float(np.clip(tau, 0.01, 4.0))
        attack_s = 0.005
        attack_n = max(1, int(round(attack_s * sr)))

        env = np.zeros((n,), dtype=np.float32)

        a0 = onset_i
        a1 = min(n, onset_i + attack_n)
        if a0 < n:
            if a1 > a0:
                env[a0:a1] = np.linspace(0.0, 1.0, a1 - a0, endpoint=False, dtype=np.float32)
            d0 = a1
            if d0 < n:
                t = (np.arange(d0, n, dtype=np.float32) - d0) / float(sr)
                env[d0:n] = np.exp(-t / tau).astype(np.float32)

        t = (np.arange(n, dtype=np.float32) / float(sr))
        phase0 = np.random.rand() * 2.0 * math.pi

        if self.cfg.osc == "sine_harm":
            y0 = np.sin(2.0 * math.pi * f0 * t + phase0)
            y1 = 0.25 * np.sin(2.0 * math.pi * (2.0 * f0) * t + (0.7 * phase0))
            sig = (y0 + y1).astype(np.float32)
        else:
            frac = np.modf((f0 * t) + (phase0 / (2.0 * math.pi)))[0]
            sig = (2.0 * frac - 1.0).astype(np.float32)
            sig = one_pole_lowpass(sig, sr=sr, cutoff_hz=self.cfg.lp_cutoff_hz)

        sig *= env * amp
        sig[:onset_i] = 0.0

        if self.cfg.add_attack_click and onset_i < n:
            click_n = max(1, int(round((self.cfg.click_ms / 1000.0) * sr)))
            c0 = onset_i
            c1 = min(n, onset_i + click_n)
            if c1 > c0:
                noise = np.random.randn(c1 - c0).astype(np.float32)
                noise = noise - np.concatenate(([0.0], noise[:-1]))
                win = np.hanning(c1 - c0).astype(np.float32)
                sig[c0:c1] += self.cfg.click_level * noise * win

        sig = softclip_tanh(sig, drive=self.cfg.drive)

        peak = float(np.max(np.abs(sig)) + 1e-8)
        if peak > 1.0:
            sig = (sig / peak).astype(np.float32)

        return sig.astype(np.float32)
