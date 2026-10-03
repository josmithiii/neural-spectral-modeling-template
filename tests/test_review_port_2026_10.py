"""Regression tests for fixes ported from nsm-synth-match's Oct-2026 code review."""

import pytest
import torch

from src.data.vimh_datamodule import VIMHDataModule
from src.models.components.simple_dense_net import SimpleDenseNet
from src.models.components.simple_mlp import SimpleMLP
from src.models.jnd_accuracy import ExactAccuracy, JNDToleranceAccuracy
from src.models.losses import OrdinalRegressionLoss


@pytest.mark.parametrize("tol", [1, 3, 5])
def test_jnd_tolerance_gives_partial_credit_out_to_tolerance(tol: int) -> None:
    """Score is positive for |d| <= tol and zero for |d| = tol + 1 (support is 2*tol+1 steps)."""
    metric = JNDToleranceAccuracy(tolerance_jnds=tol)
    target = torch.zeros(1, dtype=torch.long)
    for d in range(tol + 2):
        metric.reset()
        metric.update(torch.tensor([float(d)]), target)
        assert metric.compute().item() == pytest.approx(max(0.0, 1.0 - d / (tol + 1)))


def test_jnd1_differs_from_exact_accuracy() -> None:
    """Previously jnd1 == exact accuracy because a 1-step miss scored 1 - 1/1 = 0."""
    preds = torch.tensor([0.0, 1.0])
    target = torch.tensor([0, 0])
    exact, jnd1 = ExactAccuracy(), JNDToleranceAccuracy(tolerance_jnds=1)
    exact.update(preds, target)
    jnd1.update(preds, target)
    assert exact.compute().item() == pytest.approx(0.5)
    assert jnd1.compute().item() == pytest.approx(0.75)


def test_exact_accuracy_empty_is_not_nan() -> None:
    assert ExactAccuracy().compute().item() == 0.0


def test_ordinal_l2_scales_by_step_squared() -> None:
    """l2 is in squared units: one step of error must equal step**2, not step."""
    step = 0.5
    loss_fn = OrdinalRegressionLoss(
        num_classes=5, param_range=step * 4, regression_loss="l2", alpha=0.0
    )
    logits = torch.full((1, 5), -1e4)
    logits[0, 3] = 1e4  # all mass on class 3, target 2
    assert loss_fn(logits, torch.tensor([2])).item() == pytest.approx(step**2, rel=1e-4)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("vimh-32x32x3_8000Hz_1p0s_256dss_simple_2p", (32, 32, 3)),
        ("vimh-avix-32x64x1_8000Hz_1p0s_72dss_simple_2p", (32, 64, 1)),
        ("some_other_dataset", None),
    ],
)
def test_parse_image_dims_from_path_handles_avix_names(name: str, expected) -> None:
    dm = VIMHDataModule(data_dir="unused")
    assert dm._parse_image_dims_from_path(f"/tmp/{name}") == expected


@pytest.mark.parametrize("cls", [SimpleMLP, SimpleDenseNet])
def test_single_named_head_returns_dict(cls) -> None:
    """A single head not named 'digit' used to raise KeyError."""
    net = cls(input_size=16, heads_config={"log10_decay_time": 7})
    net.eval()
    out = net(torch.randn(2, 1, 4, 4))
    assert set(out) == {"log10_decay_time"}
    assert out["log10_decay_time"].shape == (2, 7)
