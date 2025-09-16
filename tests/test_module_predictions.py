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

