import math

import pytest
import torch

from src.models.components.simple_cnn import SimpleCNN
from src.utils.auxiliary_features import extract_auxiliary_features


def _decaying_spectrogram(tau: float, frame_hop: float, width: int = 32, height: int = 8):
    """[1, 1, H, W] spectrogram whose energy decays as exp(-t / tau)."""
    t = torch.arange(width, dtype=torch.float32) * frame_hop
    return torch.exp(-t / tau).expand(height, width).reshape(1, 1, height, width).clone()


@pytest.mark.parametrize("frame_hop", [0.01, 0.004])
def test_decay_time_uses_dataset_frame_hop(frame_hop: float):
    """The measured decay must be in seconds for the dataset's actual frame spacing.

    Regression test: frame_hop was hardcoded to 0.01 s, so datasets with a 4 ms hop
    (e.g. wah_envelope_9p, hop 32 @ 8 kHz) got log10 decay times off by ~0.4.
    """
    tau = 0.05
    spec = _decaying_spectrogram(tau, frame_hop)
    feats = extract_auxiliary_features({"image": spec}, ["log10_decay_time"], frame_hop=frame_hop)
    assert feats.shape == (1, 1)
    # Light smoothing biases the fit a little; units are what matter here
    assert float(feats[0, 0]) == pytest.approx(math.log10(tau), abs=0.1)


def test_unknown_auxiliary_feature_raises():
    """Unknown feature names used to silently produce a size-0 feature vector."""
    spec = _decaying_spectrogram(0.05, 0.01)
    with pytest.raises(ValueError, match="Unsupported auxiliary features"):
        extract_auxiliary_features({"image": spec}, ["decay_time"], frame_hop=0.01)


def test_simple_cnn_auxiliary_input_must_match_construction():
    """The auxiliary input was silently ignored when the net was built without aux support."""
    x = torch.rand(2, 1, 32, 32)
    aux = torch.rand(2, 1)
    plain = SimpleCNN(input_size=32, heads_config={"a": 3, "b": 4})
    with pytest.raises(ValueError, match="auxiliary"):
        plain(x, aux)

    with_aux = SimpleCNN(input_size=32, heads_config={"a": 3, "b": 4}, auxiliary_input_size=1)
    with pytest.raises(ValueError, match="auxiliary"):
        with_aux(x)
    out = with_aux(x, aux)
    assert out["a"].shape == (2, 3) and out["b"].shape == (2, 4)


def test_simple_cnn_single_head_rebuild():
    """_build_heads used to leave single-head forward() on a stale `classifier`."""
    net = SimpleCNN(input_size=32, heads_config={"a": 3, "b": 4})
    net._build_heads({"only": 14})
    assert net(torch.rand(2, 1, 32, 32)).shape == (2, 14)
