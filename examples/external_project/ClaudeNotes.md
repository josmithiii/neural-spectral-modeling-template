2026-01-01 18:04 - CCO

  Session Summary

  Commit: 93c40f0 - examples: Major improvements to onset detection and parameter prediction

  Key Fixes

  1. Spectrogram normalization bug - Model expected [0,255] but
     inference used [0,1]. This was causing all predictions to be
     stuck at minimum values.

  2. Replaced onset detection - Switched from custom RMS-based to librosa with:
    - Pre-emphasis filtering for soft attacks
    - Spectral flux with adaptive thresholding
    - 500ms gap filter to prevent false positives

  3. Added onset comparison table - Console output now shows NSMT vs Truth timing errors

  4. Updated docs - transcription_pipeline.md reflects current architecture

  Results

  | Metric          | Before                 | After                      |
  |-----------------|------------------------|----------------------------|
  | Detection rate  | ~0% (all stuck at min) | 96%                        |
  | Timing error    | N/A                    | < 50ms typical             |
  | False positives | Many                   | Eliminated by 500ms filter |

  Remaining Work

  - 4 soft onsets still missed (heavily wah-filtered)
  - Future: Viterbi alignment for robust matching
