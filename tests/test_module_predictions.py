import types

import torch

from src.models.vimh_lit_module import VIMHLitModule


class DummyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def test_vimhlitmodule_compute_predictions_regression_denorm():
    # Create module with two heads in regression mode
    heads = {"a": 1, "b": 1}
    module = VIMHLitModule(
        net=DummyNet(),
        optimizer=lambda **kw: None,  # not used
        scheduler=None,
        criteria={"a": torch.nn.MSELoss(), "b": torch.nn.MSELoss()},
        loss_weights={"a": 1.0, "b": 1.0},
        compile=False,
        auto_configure_from_dataset=False,
        output_mode="regression",
    )

    # Attach a fake trainer/datamodule exposing param_bounds
    fake_dm = types.SimpleNamespace(param_bounds={"a": (0.0, 0.9), "b": (-1.0, 0.3)})
    module.trainer = types.SimpleNamespace(datamodule=fake_dm)

    # Normalized prediction → denormalized via _compute_predictions
    # a: 0.5 of (0..0.9) → 0.45
    # b: 0.0 of (-1..0.3) → -1.0
    logits_a = torch.tensor([[0.5]])
    logits_b = torch.tensor([[0.0]])
    pa = module._compute_predictions(logits_a, module.criteria["a"], "a")
    pb = module._compute_predictions(logits_b, module.criteria["b"], "b")

    assert torch.allclose(pa.squeeze(), torch.tensor(0.45), atol=1e-6)
    assert torch.allclose(pb.squeeze(), torch.tensor(-1.0), atol=1e-6)


def _regression_module_with_bounds():
    """Build a regression module with one head 'a' over [0, 0.9] with 10 classes."""
    module = VIMHLitModule(
        net=DummyNet(),
        optimizer=lambda **kw: None,
        scheduler=None,
        criteria={"a": torch.nn.MSELoss()},
        loss_weights={"a": 1.0},
        compile=False,
        auto_configure_from_dataset=False,
        output_mode="regression",
    )
    module.net.heads_config = {"a": 10}  # step = 0.9 / 9 = 0.1
    fake_dm = types.SimpleNamespace(param_bounds={"a": (0.0, 0.9)})
    module.trainer = types.SimpleNamespace(datamodule=fake_dm)
    return module


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


def test_auto_configured_normalized_regression_gets_dataset_bounds():
    """Auto-generated normalized_regression criteria must receive real param bounds.

    Regression test: the auto-config (no explicit `criteria:`) path built each
    NormalizedRegressionLoss via _create_loss_function, which hardcodes
    param_range=(0, 1), and never applied the dataset's actual bounds. Targets
    outside [0, 1] (e.g. log10_decay_time in [-1, 0.3]) were then mis-normalized.
    """
    from src.models.losses import NormalizedRegressionLoss

    module = VIMHLitModule(
        net=DummyNet(),
        optimizer=lambda **kw: None,
        scheduler=None,
        criteria=None,  # force the auto-generated criteria path
        compile=False,
        auto_configure_from_dataset=True,
        loss_type="normalized_regression",
    )
    assert module.criteria == {}  # nothing configured until auto-config runs

    bounds = {"wah_position": (0.0, 0.9), "log10_decay_time": (-1.0, 0.3)}
    fake_dm = types.SimpleNamespace(
        param_bounds=bounds,
        param_ranges={k: (lo, hi) for k, (lo, hi) in bounds.items()},
    )
    module.trainer = types.SimpleNamespace(datamodule=fake_dm)

    fake_dataset = types.SimpleNamespace(
        get_heads_config=lambda: {"wah_position": 19, "log10_decay_time": 14},
        metadata_format={},  # falsy -> skip JND weight auto-config
    )

    module._auto_configure_from_dataset(fake_dataset)

    for head, (pmin, pmax) in bounds.items():
        crit = module.criteria[head]
        assert isinstance(crit, NormalizedRegressionLoss)
        assert crit.param_min == pmin
        assert crit.param_max == pmax
        assert crit.param_range == pmax - pmin
    # The negative-min head must not retain the placeholder (0, 1) bounds.
    assert module.criteria["log10_decay_time"].param_min == -1.0

