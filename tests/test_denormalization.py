import tempfile
from pathlib import Path

import torch

from src.audio_reconstruction_eval import AudioReconstructionEvaluator, CheckpointArchitectureReconstructor


class _FakeDataset:
    def __init__(self, param_mappings, param_names, height=32, width=32):
        self._param_mappings = param_mappings
        self._param_names = param_names
        self._height = height
        self._width = width

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # Minimal sample: image tensor and dummy labels dict
        image = torch.zeros(1, self._height, self._width)
        labels = {name: 0 for name in self._param_names}
        return image, labels

    # The evaluator calls this on the underlying dataset of the datamodule for metadata
    def get_dataset_info(self):
        return {
            "metadata_format": {
                "parameter_mappings": self._param_mappings,
                "parameter_names": self._param_names,
                "height": self._height,
                "width": self._width,
                "channels": 1,
                "sample_rate": 8000,
                "duration": 1.0,
                "spectrogram_config": {"type": "stft", "n_fft": 80, "n_window": 80, "hop_length": 40},
                "mel_config": {"freq_min": 40.0, "freq_max_ratio": 0.9},
                "synth_type": "SimpleSawSynth",
            }
        }

    # AudioReconstructionEvaluator uses this on its datamodule wrapper, but we simulate via datamodule


class _FakeDataModule:
    def __init__(self, param_mappings, param_names):
        self._info = {
            "parameter_mappings": param_mappings,
            "parameter_names": param_names,
            "height": 32,
            "width": 32,
            "channels": 1,
            "sample_rate": 8000,
            "duration": 1.0,
            "spectrogram_config": {"type": "stft", "n_fft": 80, "n_window": 80, "hop_length": 40},
            "mel_config": {"freq_min": 40.0, "freq_max_ratio": 0.9},
            "synth_type": "SimpleSawSynth",
        }
        self.data_train = _FakeDataset(param_mappings, param_names)
        self.data_test = _FakeDataset(param_mappings, param_names)

    def setup(self, stage=None):
        return None

    def get_dataset_info(self):
        # Mimic VIMHDataModule.get_dataset_info() shape
        base = {"heads_config": {name: 1 for name in self._info["parameter_names"]}}
        base.update(self._info)
        return base


class _FakeModel(torch.nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.output_mode = "regression"
        self.net = type("N", (), {"heads_config": {h: 1 for h in heads}})()
        self.heads = heads

    def forward(self, x):
        # Not used in these tests; evaluator.denormalize_parameters is tested in isolation
        return {h: torch.tensor([[0.5]]) for h in self.heads}


def test_denormalize_parameters_regression_and_classification():
    # Parameter mappings as used by the wah dataset
    param_mappings = {
        "log10_decay_time": {"min": -1.0, "max": 0.3, "step": 0.1, "num_classes": 14},
        "wah_position": {"min": 0.0, "max": 0.9, "step": 0.05, "num_classes": 19},
    }
    param_names = ["log10_decay_time", "wah_position"]

    dm = _FakeDataModule(param_mappings, param_names)
    model = _FakeModel(param_names)

    evaluator = AudioReconstructionEvaluator(model, dm, device="cpu")

    # 1) Regression: inputs are [0,1] -> map to natural space
    predicted = {
        "wah_position": torch.tensor([0.5]),  # normalized 0.5 => 0.45 in natural space
        "log10_decay_time": torch.tensor([0.0]),  # normalized 0.0 => -1.0
    }
    denorm = evaluator.denormalize_parameters(predicted)
    assert abs(denorm["wah_position"] - 0.45) < 1e-6
    assert abs(denorm["log10_decay_time"] - (-1.0)) < 1e-6

    # 2) Classification: inputs are class indices -> normalized -> natural space
    predicted_cls = {
        "wah_position": torch.tensor([9]),  # midpoint class for 19 levels -> ~0.45
        "log10_decay_time": torch.tensor([5]),  # 5/13 of range 1.3 -> -1 + 0.5 ≈ -0.5
    }
    denorm_cls = evaluator.denormalize_parameters(predicted_cls)
    assert abs(denorm_cls["wah_position"] - 0.45) < 1e-6
    assert abs(denorm_cls["log10_decay_time"] - (-0.5)) < 1e-6


def test_checkpoint_reconstruction_sets_regression_heads(tmp_path: Path):
    # Minimal fake checkpoint with regression output_mode and CNN metadata
    ckpt = {
        "hyper_parameters": {
            "output_mode": "regression",
            "architecture_metadata": {
                "type": "CNN",
                "input_channels": 1,
                "conv1_channels": 16,
                "conv2_channels": 32,
                "fc_hidden": 64,
                "dropout": 0.3,
                "input_size": 32,
            },
        },
        "state_dict": {},
    }
    ckpt_path = tmp_path / "fake.ckpt"
    torch.save(ckpt, ckpt_path)

    heads_config = {"log10_decay_time": 1, "wah_position": 1}
    dataset_metadata = {"parameter_names": ["log10_decay_time", "wah_position"]}

    recon = CheckpointArchitectureReconstructor()
    net, cfg = recon.reconstruct_from_checkpoint(str(ckpt_path), heads_config, dataset_metadata)

    assert cfg.architecture_type == "CNN"
    # Ensure regression heads were built (1 output with Sigmoid per head)
    assert hasattr(net, "heads"), "Network should expose heads in regression mode"
    assert set(net.heads.keys()) == set(heads_config.keys())
    # Each head should output a single value in [0,1]
    x = torch.zeros(1, 1, 32, 32)
    out = net(x)
    assert set(out.keys()) == set(heads_config.keys())
    for t in out.values():
        assert t.shape[-1] == 1
