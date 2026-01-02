# Makefile for external_project examples
#
# Demonstrates using NSMT tools for audio analysis and MIDI comparison
#
# Prerequisites:
#   - Activate venv: source ../../.venv/bin/activate
#   - Reference checkpoint must exist (run from nsm-synth-match root if needed)

PYTHON = python
SHELL = /bin/bash

# Default number of samples for test generation
N_SAMPLES ?= 100

# Output files
TEST_WAV = test_$(N_SAMPLES).wav
TEST_TRUTH = test_$(N_SAMPLES)_truth.txt
TEST_NSMT_MID = test_$(N_SAMPLES)_nsmt.mid

.PHONY: help gen pred pred-multi compare compare-bass clean all

help:
	@echo "NSMT External Project Examples"
	@echo "=============================="
	@echo ""
	@echo "Quick start:"
	@echo "  make all          - Generate test audio, predict, and compare"
	@echo ""
	@echo "Test audio generation:"
	@echo "  make gen          - Generate $(N_SAMPLES) test samples (N_SAMPLES=$(N_SAMPLES))"
	@echo "  make gen N_SAMPLES=50  - Generate 50 test samples"
	@echo ""
	@echo "Parameter prediction:"
	@echo "  make pred         - Predict params for single note (test.wav)"
	@echo "  make pred-multi   - Multi-note prediction on generated test audio"
	@echo "  make pred-bass    - Multi-note prediction on bass-mono.wav"
	@echo ""
	@echo "MIDI comparison:"
	@echo "  make compare      - Compare NSMT vs ground truth on test audio"
	@echo "  make compare-bass - Compare NSMT vs Basic Pitch on bass-mono.wav"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean        - Remove generated files"
	@echo "  make list         - List available audio/MIDI files"
	@echo ""
	@echo "Individual tools:"
	@echo "  $(PYTHON) generate_test_audio.py --help"
	@echo "  $(PYTHON) predict_params.py --help"
	@echo "  $(PYTHON) compare_midi.py --help"

# Generate test audio from dataset ground truth
gen:
	$(PYTHON) generate_test_audio.py -n $(N_SAMPLES) -o $(TEST_WAV)

# Single-note prediction (on test.wav if it exists)
pred:
	@if [ -f test.wav ]; then \
		$(PYTHON) predict_params.py test.wav; \
	else \
		echo "No test.wav found. Create one or use 'make pred-multi'"; \
	fi

# Multi-note prediction on generated test audio
pred-multi: $(TEST_WAV)
	$(PYTHON) predict_params.py --multi --midi-out $(TEST_NSMT_MID) $(TEST_WAV)

# Multi-note prediction on bass recording
pred-bass:
	$(PYTHON) predict_params.py --multi --midi-out bass-nsmt.mid bass-mono.wav

# Compare NSMT predictions vs ground truth on test audio
compare: $(TEST_WAV) $(TEST_NSMT_MID)
	@if [ -f test_$(N_SAMPLES)_truth.mid ]; then \
		TRUTH_ARG=$$(if [ -f test_$(N_SAMPLES)_truth.txt ]; then echo "--truth test_$(N_SAMPLES)_truth.txt"; fi); \
		$(PYTHON) compare_midi.py $(TEST_WAV) $(TEST_NSMT_MID) test_$(N_SAMPLES)_truth.mid $$TRUTH_ARG; \
	else \
		echo "Ground truth MIDI not available (no pitch info in dataset)"; \
		echo "View predictions only:"; \
		$(PYTHON) compare_midi.py $(TEST_WAV) $(TEST_NSMT_MID) $(TEST_NSMT_MID); \
	fi

# Compare NSMT vs Basic Pitch on bass recording
compare-bass: bass-mono.wav bass-nsmt.mid bass.mid
	$(PYTHON) compare_midi.py bass-mono.wav bass-nsmt.mid bass.mid

# Full workflow: generate, predict, compare
all: gen pred-multi compare

# Ensure test audio exists
$(TEST_WAV):
	$(MAKE) gen

# Ensure NSMT predictions exist
$(TEST_NSMT_MID): $(TEST_WAV)
	$(MAKE) pred-multi

# List available files
list:
	@echo "Audio files:"
	@ls -la *.wav 2>/dev/null || echo "  (none)"
	@echo ""
	@echo "MIDI files:"
	@ls -la *.mid 2>/dev/null || echo "  (none)"
	@echo ""
	@echo "Ground truth files:"
	@ls -la *_truth.txt 2>/dev/null || echo "  (none)"

# Clean generated files (preserves bass-mono.wav and bass.mid)
clean:
	rm -f test_*.wav test_*.txt test_*.mid
	rm -f test.wav test-truth.txt test-truth.jpg
	@echo "Cleaned generated files (kept bass-mono.wav, bass.mid)"
