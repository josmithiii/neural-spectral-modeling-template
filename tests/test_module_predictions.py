import types

import pytest
import torch

from src.models.components.simple_cnn import SimpleCNN
from src.models.vimh_lit_module import VIMHLitModule


class DummyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads_config = {}

    def forward(self, x):
        return x

    def _build_heads(self, heads_config):
        self.heads_config = heads_config


def _regression_module(heads, bounds, criteria=None):
    module = VIMHLitModule(
        net=DummyNet(),
        optimizer=lambda **kw: None,  # not used
        scheduler=None,
        criteria=criteria or {h: torch.nn.MSELoss() for h in heads},
        loss_weights={h: 1.0 for h in heads},
        compile=False,
        auto_configure_from_dataset=False,
        loss_type="normalized_regression",
    )
    module.heads_config = dict(heads)
    module.param_bounds = dict(bounds)
    return module


def _fake_dataset(heads, bounds, steps, auxiliary=None, channels=1):
    return types.SimpleNamespace(
        get_heads_config=lambda: dict(heads),
        get_image_shape=lambda: (channels, 32, 32),
        auxiliary_features=auxiliary or [],
        metadata_format={
            "parameter_mappings": {
                h: {"min": bounds[h][0], "max": bounds[h][1], "step": steps[h]} for h in heads
            }
        },
    )


def test_vimhlitmodule_compute_predictions_regression_denorm():
    module = _regression_module({"a": 10, "b": 14}, {"a": (0.0, 0.9), "b": (-1.0, 0.3)})

    # a: 0.5 of (0..0.9) → 0.45;  b: 0.0 of (-1..0.3) → -1.0
    pa = module._compute_predictions(torch.tensor([[0.5]]), module.criteria["a"], "a")
    pb = module._compute_predictions(torch.tensor([[0.0]]), module.criteria["b"], "b")

    assert torch.allclose(pa.squeeze(), torch.tensor(0.45), atol=1e-6)
    assert torch.allclose(pb.squeeze(), torch.tensor(-1.0), atol=1e-6)


def test_regression_predictions_without_bounds_fail_loudly():
    """Missing bounds used to silently leave predictions normalized (wrong units)."""
    module = _regression_module({"a": 10}, {})
    with pytest.raises(RuntimeError, match="No parameter bounds"):
        module._compute_predictions(torch.tensor([[0.5]]), module.criteria["a"], "a")


def _regression_module_with_bounds():
    """Regression module with one head 'a' over [0, 0.9] with 10 classes (step 0.1)."""
    return _regression_module({"a": 10}, {"a": (0.0, 0.9)})


def test_to_jnd_index_space_maps_physical_to_steps():
    """Physical parameter values must map to class-index (JND step) units."""
    module = _regression_module_with_bounds()
    # step = 0.1: 0.0 -> 0, 0.45 -> 4.5, 0.9 -> 9.0
    values = torch.tensor([0.0, 0.45, 0.9])
    idx = module._to_jnd_index_space(values, "a")
    assert torch.allclose(idx, torch.tensor([0.0, 4.5, 9.0]), atol=1e-6)


def test_regression_jnd_metric_is_meaningful_after_conversion():
    """In index space the JND metric distinguishes close vs far predictions.

    Regression test for the bug where physical-unit preds/targets were fed
    directly to JNDToleranceAccuracy (tolerance in steps), making ~5-JND errors
    score as near-perfect.
    """
    from src.models.jnd_accuracy import JNDToleranceAccuracy

    module = _regression_module_with_bounds()
    # step = 0.1, so these physical values map to indices [0, 4, 8].
    targets = torch.tensor([0.00, 0.40, 0.80])  # physical units
    near = torch.tensor([0.00, 0.40, 0.80])  # exact -> 0 steps off
    far = torch.tensor([0.40, 0.00, 0.40])  # 4 steps off each

    t_idx = module._to_jnd_index_space(targets, "a")
    near_acc = JNDToleranceAccuracy(tolerance_jnds=1)
    near_acc.update(module._to_jnd_index_space(near, "a"), t_idx)
    far_acc = JNDToleranceAccuracy(tolerance_jnds=1)
    far_acc.update(module._to_jnd_index_space(far, "a"), t_idx)

    assert float(near_acc.compute()) == 1.0  # exact match
    assert float(far_acc.compute()) == 0.0  # 4 JNDs off at tolerance 1

    # Without the conversion (physical units), the same 4-step error looks
    # "accurate" because the small physical values collapse to 0/1 when rounded —
    # demonstrating the original bug (true index distance is 4 everywhere).
    buggy = JNDToleranceAccuracy(tolerance_jnds=1)
    buggy.update(far, targets)
    assert float(buggy.compute()) > 0.5


def _auto_module(loss_type="normalized_regression", net=None, **kwargs):
    return VIMHLitModule(
        net=net or DummyNet(),
        optimizer=lambda **kw: None,
        scheduler=None,
        criteria=None,  # force the auto-generated criteria path
        compile=False,
        auto_configure_from_dataset=True,
        loss_type=loss_type,
        **kwargs,
    )


def test_auto_configured_normalized_regression_gets_dataset_bounds():
    """Auto-generated normalized_regression criteria must receive real param bounds.

    Regression test: the auto-config path once built each NormalizedRegressionLoss
    with placeholder bounds (0, 1). Targets outside [0, 1] (e.g. log10_decay_time
    in [-1, 0.3]) were then mis-normalized.
    """
    from src.models.losses import NormalizedRegressionLoss

    module = _auto_module()
    assert module.criteria == {}  # nothing configured until auto-config runs

    bounds = {"wah_position": (0.0, 0.9), "log10_decay_time": (-1.0, 0.3)}
    dataset = _fake_dataset(
        {"wah_position": 19, "log10_decay_time": 14}, bounds, {"wah_position": 0.05, "log10_decay_time": 0.1}
    )
    module._auto_configure_from_dataset(dataset)

    for head, (pmin, pmax) in bounds.items():
        crit = module.criteria[head]
        assert isinstance(crit, NormalizedRegressionLoss)
        assert crit.param_min == pmin
        assert crit.param_max == pmax
        assert crit.param_range == pmax - pmin
    assert module.param_bounds == bounds
    # Each criterion knows its JND step count (dataset class counts)...
    assert module.criteria["wah_position"].num_classes == 19
    assert module.criteria["log10_decay_time"].num_classes == 14
    # ...so regression losses are in JND steps and head weights are uniform
    assert module.loss_weights == {"wah_position": 1.0, "log10_decay_time": 1.0}


def test_auto_configuration_runs_once_and_keeps_trained_heads():
    """Lightning calls setup() again for validate/test; heads must not be re-randomized."""
    heads = {"wah_position": 19, "log10_decay_time": 14}
    bounds = {"wah_position": (0.0, 0.9), "log10_decay_time": (-1.0, 0.3)}
    steps = {"wah_position": 0.05, "log10_decay_time": 0.1}
    net = SimpleCNN(input_channels=1, conv1_channels=4, conv2_channels=4, fc_hidden=8, input_size=32)
    module = _auto_module(loss_type="cross_entropy", net=net)
    dataset = _fake_dataset(heads, bounds, steps)

    module._auto_configure_from_dataset(dataset)
    assert net.heads_config == heads
    trained = {k: v.clone() for k, v in net.heads.state_dict().items()}

    module._auto_configure_from_dataset(dataset)  # e.g. setup("test")
    for k, v in net.heads.state_dict().items():
        assert torch.equal(v, trained[k])

    other = _fake_dataset({"wah_position": 21, "log10_decay_time": 14}, bounds, steps)
    with pytest.raises(ValueError, match="differ"):
        module._auto_configure_from_dataset(other)


def test_preconfigured_net_heads_are_not_rebuilt():
    """train.py pre-configures the net's heads; auto-config must keep those weights."""
    heads = {"wah_position": 19, "log10_decay_time": 14}
    net = SimpleCNN(input_channels=1, conv1_channels=4, conv2_channels=4, fc_hidden=8,
                    input_size=32, heads_config=dict(heads))
    before = {k: v.clone() for k, v in net.heads.state_dict().items()}
    module = _auto_module(loss_type="cross_entropy", net=net)
    module._auto_configure_from_dataset(
        _fake_dataset(heads, {"wah_position": (0.0, 0.9), "log10_decay_time": (-1.0, 0.3)},
                      {"wah_position": 0.05, "log10_decay_time": 0.1})
    )
    for k, v in net.heads.state_dict().items():
        assert torch.equal(v, before[k])


def test_auto_configuration_excludes_auxiliary_features_and_checks_channels():
    heads = {"wah_position": 19, "log10_decay_time": 14}
    bounds = {"wah_position": (0.0, 0.9), "log10_decay_time": (-1.0, 0.3)}
    steps = {"wah_position": 0.05, "log10_decay_time": 0.1}
    module = _auto_module(loss_type="cross_entropy", net=SimpleCNN(input_channels=1, input_size=32))
    module._auto_configure_from_dataset(_fake_dataset(heads, bounds, steps, auxiliary=["log10_decay_time"]))
    assert module.heads_config == {"wah_position": 19}
    assert list(module.criteria) == ["wah_position"]

    module3 = _auto_module(loss_type="cross_entropy", net=SimpleCNN(input_channels=3, input_size=32))
    with pytest.raises(ValueError, match="input channel"):
        module3._auto_configure_from_dataset(_fake_dataset(heads, bounds, steps))


def test_explicit_loss_weights_must_match_heads():
    heads = {"wah_position": 19, "log10_decay_time": 14}
    bounds = {"wah_position": (0.0, 0.9), "log10_decay_time": (-1.0, 0.3)}
    steps = {"wah_position": 0.05, "log10_decay_time": 0.1}
    module = _auto_module(loss_type="cross_entropy", loss_weights={"wah_position": 1.0})
    with pytest.raises(ValueError, match="loss_weights"):
        module._auto_configure_from_dataset(_fake_dataset(heads, bounds, steps))

    equal = _auto_module(loss_type="cross_entropy", loss_weights={"wah_position": 1.0, "log10_decay_time": 1.0})
    equal._auto_configure_from_dataset(_fake_dataset(heads, bounds, steps))
    assert equal.loss_weights == {"wah_position": 1.0, "log10_decay_time": 1.0}


def test_model_step_rejects_wrong_label_space_and_missing_heads():
    """Class-index targets with a regression loss used to be clamped into garbage."""
    net = SimpleCNN(input_channels=1, conv1_channels=4, conv2_channels=4, fc_hidden=8,
                    input_size=32, output_mode="regression", parameter_names=["a", "b"])
    from src.models.losses import NormalizedRegressionLoss

    module = _regression_module(
        {"a": 10, "b": 10},
        {"a": (0.0, 0.9), "b": (0.0, 0.9)},
        criteria={h: NormalizedRegressionLoss(param_range=(0.0, 0.9), num_classes=10) for h in "ab"},
    )
    module.net = net
    x = torch.rand(4, 1, 32, 32)

    with pytest.raises(TypeError, match="label_mode=regression"):
        module.model_step((x, {"a": torch.tensor([1, 2, 3, 4]), "b": torch.tensor([1, 2, 3, 4])}))

    with pytest.raises(RuntimeError, match="missing"):
        module.model_step((x, {"a": torch.rand(4)}))

    loss, preds, _ = module.model_step((x, {"a": torch.rand(4) * 0.9, "b": torch.rand(4) * 0.9}))
    assert torch.isfinite(loss)
    assert set(preds) == {"a", "b"}
