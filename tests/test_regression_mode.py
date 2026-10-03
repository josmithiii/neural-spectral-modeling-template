"""Tests for regression mode functionality in multihead models."""

import pytest
import torch
import torch.nn as nn

from src.models.components.simple_cnn import SimpleCNN
from src.models.losses import NormalizedRegressionLoss
from src.models.vimh_lit_module import VIMHLitModule


class TestRegressionNetworkArchitecture:
    """Test regression network architecture components."""

    def test_regression_network_initialization(self):
        """Test that regression networks initialize correctly."""
        net = SimpleCNN(
            input_channels=3,
            output_mode="regression",
            parameter_names=["note_number", "note_velocity"],
            parameter_ranges={"note_number": (50.0, 52.0), "note_velocity": (80.0, 82.0)},
            input_size=32,
        )

        assert net.output_mode == "regression"
        assert net.parameter_names == ["note_number", "note_velocity"]
        assert net.parameter_ranges == {"note_number": (50.0, 52.0), "note_velocity": (80.0, 82.0)}
        assert net.is_multihead
        assert len(net.heads_config) == 2
        assert net.heads_config["note_number"] == 1
        assert net.heads_config["note_velocity"] == 1

    def test_regression_network_forward_pass(self):
        """Test regression network forward pass produces correct output."""
        net = SimpleCNN(
            input_channels=3,
            output_mode="regression",
            parameter_names=["note_number", "note_velocity"],
            parameter_ranges={"note_number": (50.0, 52.0), "note_velocity": (80.0, 82.0)},
            input_size=32,
        )

        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32)

        with torch.no_grad():
            output = net(x)

        # Check output structure
        assert isinstance(output, dict)
        assert "note_number" in output
        assert "note_velocity" in output

        # Check output shapes and ranges (should be sigmoid-activated [0,1])
        for param_name, param_output in output.items():
            assert param_output.shape == (batch_size, 1)
            assert torch.all(param_output >= 0.0), f"Output should be >= 0 for {param_name}"
            assert torch.all(param_output <= 1.0), f"Output should be <= 1 for {param_name}"

    def test_regression_network_empty_parameter_names_allowed(self):
        """Test that empty parameter_names is allowed for regression mode (auto-configuration)."""
        # This should not raise an error - empty parameter_names allows auto-configuration
        net = SimpleCNN(input_channels=3, output_mode="regression", input_size=32)

        assert net.output_mode == "regression"
        assert net.parameter_names == []
        assert net.heads_config == {}  # Empty until auto-configured

    def test_regression_network_backward_compatibility(self):
        """Test that classification mode still works (backward compatibility)."""
        net = SimpleCNN(
            input_channels=3,
            output_mode="classification",
            heads_config={"note_number": 256, "note_velocity": 256},
            input_size=32,
        )

        assert net.output_mode == "classification"
        assert net.is_multihead
        assert net.heads_config["note_number"] == 256
        assert net.heads_config["note_velocity"] == 256

        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32)

        with torch.no_grad():
            output = net(x)

        # Check output structure for classification
        assert isinstance(output, dict)
        assert output["note_number"].shape == (batch_size, 256)
        assert output["note_velocity"].shape == (batch_size, 256)


class TestNormalizedRegressionLoss:
    """Test the NormalizedRegressionLoss function (loss in JND-step units)."""

    def test_normalized_regression_loss_initialization(self):
        """Test that NormalizedRegressionLoss initializes correctly."""
        loss_fn = NormalizedRegressionLoss(param_range=(50.0, 52.0), num_classes=21, loss_type="l1")

        assert loss_fn.param_min == 50.0
        assert loss_fn.param_max == 52.0
        assert loss_fn.param_range == 2.0
        assert loss_fn.loss_type == "l1"
        assert loss_fn.num_classes == 21

    def test_normalized_regression_loss_invalid_range(self):
        """Test that invalid parameter ranges raise errors."""
        with pytest.raises(ValueError, match="Parameter range must be positive"):
            NormalizedRegressionLoss(param_range=(52.0, 50.0))  # Invalid range

    @pytest.mark.parametrize("loss_type", ["l1", "mse", "huber"])
    def test_normalized_regression_loss_types(self, loss_type):
        """Test different loss types work correctly."""
        loss_fn = NormalizedRegressionLoss(
            param_range=(50.0, 52.0), num_classes=21, loss_type=loss_type
        )

        preds = torch.tensor([[0.5], [0.3], [0.7], [0.9]])
        targets = torch.tensor([51.0, 50.6, 51.4, 51.8])

        loss = loss_fn(preds, targets)

        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_normalized_regression_loss_unknown_type(self):
        """Unknown loss types are rejected at construction."""
        with pytest.raises(ValueError, match="Unknown loss type"):
            NormalizedRegressionLoss(param_range=(50.0, 52.0), loss_type="unknown")

    @pytest.mark.parametrize("loss_type, expected", [("l1", 1.0), ("mse", 1.0), ("huber", 0.5)])
    def test_one_step_miss_costs_one_step_on_any_head(self, loss_type, expected):
        """A one-JND-step error costs the same whatever the parameter's units or range."""
        for param_range, num_classes in [((50.0, 52.0), 21), ((-2.0, 0.3), 24), ((0.0, 80.0), 9)]:
            loss_fn = NormalizedRegressionLoss(
                param_range=param_range, num_classes=num_classes, loss_type=loss_type
            )
            step = (param_range[1] - param_range[0]) / (num_classes - 1)
            target = torch.tensor([param_range[0] + 3 * step])
            pred = torch.tensor([[4.0 / (num_classes - 1)]])  # one step above the target
            assert loss_fn(pred, target).item() == pytest.approx(expected, rel=1e-4)

    def test_mse_is_squared_steps(self):
        loss_fn = NormalizedRegressionLoss(param_range=(0.0, 1.0), num_classes=11, loss_type="mse")
        # 3-step error -> 9 squared steps
        assert loss_fn(torch.tensor([[0.5]]), torch.tensor([0.2])).item() == pytest.approx(9.0)

    def test_num_classes_required_before_forward(self):
        loss_fn = NormalizedRegressionLoss(param_range=(50.0, 52.0))
        with pytest.raises(RuntimeError, match="num_classes is unset"):
            loss_fn(torch.tensor([[0.5]]), torch.tensor([51.0]))

    def test_normalized_regression_loss_rejects_out_of_range_targets(self):
        """Targets outside [min, max] are in the wrong units: fail instead of clamping.

        Regression test: class-index targets fed to a regression loss were silently
        clamped to the parameter range, so training "succeeded" on garbage.
        """
        loss_fn = NormalizedRegressionLoss(param_range=(50.0, 52.0), num_classes=21, loss_type="l1")
        preds = torch.tensor([[0.5]])
        with pytest.raises(ValueError, match="outside"):
            loss_fn(preds, torch.tensor([55.0]))  # Outside [50, 52]
        with pytest.raises(TypeError, match="label_mode=regression"):
            loss_fn(preds, torch.tensor([51]))  # integer class index


class TestMultiheadRegressionModule:
    """Test the VIMHLitModule with regression mode."""

    def test_multihead_regression_initialization(self):
        """Test that VIMHLitModule initializes correctly in regression mode."""
        net = SimpleCNN(
            input_channels=3,
            output_mode="regression",
            parameter_names=["note_number", "note_velocity"],
            parameter_ranges={"note_number": (50.0, 52.0), "note_velocity": (80.0, 82.0)},
            input_size=32,
        )

        criteria = {
            "note_number": NormalizedRegressionLoss(param_range=(50.0, 52.0), num_classes=21, loss_type="l1"),
            "note_velocity": NormalizedRegressionLoss(param_range=(80.0, 82.0), num_classes=21, loss_type="l1"),
        }

        module = VIMHLitModule(
            net=net,
            optimizer=torch.optim.Adam,
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
            criteria=criteria,
            loss_type="normalized_regression",
            auto_configure_from_dataset=False,
        )

        assert module.output_mode == "regression"
        assert module.is_multihead
        assert len(module.criteria) == 2

    def test_multihead_regression_forward_pass(self):
        """Test that VIMHLitModule forward pass works in regression mode."""
        net = SimpleCNN(
            input_channels=3,
            output_mode="regression",
            parameter_names=["note_number", "note_velocity"],
            parameter_ranges={"note_number": (50.0, 52.0), "note_velocity": (80.0, 82.0)},
            input_size=32,
        )

        criteria = {
            "note_number": NormalizedRegressionLoss(param_range=(50.0, 52.0), num_classes=21, loss_type="l1"),
            "note_velocity": NormalizedRegressionLoss(param_range=(80.0, 82.0), num_classes=21, loss_type="l1"),
        }

        module = VIMHLitModule(
            net=net,
            optimizer=torch.optim.Adam,
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
            criteria=criteria,
            loss_type="normalized_regression",
            auto_configure_from_dataset=False,
        )

        # Bounds are normally set by dataset auto-configuration
        bounds = {"note_number": (50.0, 52.0), "note_velocity": (80.0, 82.0)}
        module.param_bounds = bounds

        # Test model step
        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32)
        y = {
            "note_number": torch.tensor([51.0, 50.6, 51.4, 51.8]),
            "note_velocity": torch.tensor([81.0, 80.3, 81.7, 80.9]),
        }
        batch = (x, y)

        with torch.no_grad():
            loss, preds, targets = module.model_step(batch)

        # Check outputs
        assert torch.isfinite(loss)
        assert loss.item() > 0
        assert isinstance(preds, dict)
        assert isinstance(targets, dict)

        # Predictions are denormalized to physical parameter units
        for param_name, pred in preds.items():
            assert pred.shape == (batch_size,)
            pmin, pmax = bounds[param_name]
            assert torch.all(pred >= pmin)
            assert torch.all(pred <= pmax)

    def test_multihead_regression_metrics_setup(self):
        """Test that regression metrics are set up correctly."""
        net = SimpleCNN(
            input_channels=3,
            output_mode="regression",
            parameter_names=["note_number", "note_velocity"],
            parameter_ranges={"note_number": (50.0, 52.0), "note_velocity": (80.0, 82.0)},
            input_size=32,
        )

        criteria = {
            "note_number": NormalizedRegressionLoss(param_range=(50.0, 52.0), num_classes=21, loss_type="l1"),
            "note_velocity": NormalizedRegressionLoss(param_range=(80.0, 82.0), num_classes=21, loss_type="l1"),
        }

        module = VIMHLitModule(
            net=net,
            optimizer=torch.optim.Adam,
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
            criteria=criteria,
            loss_type="normalized_regression",
            auto_configure_from_dataset=False,
        )

        # Setup metrics
        module._setup_metrics()

        # Check that MAE metrics are created instead of accuracy
        assert "note_number_mae" in module.train_metrics
        assert "note_velocity_mae" in module.train_metrics
        assert "note_number_mae" in module.val_metrics
        assert "note_velocity_mae" in module.val_metrics
        assert "note_number_mae" in module.test_metrics
        assert "note_velocity_mae" in module.test_metrics

        # Check that no accuracy metrics are created
        assert "note_number_acc" not in module.train_metrics
        assert "note_velocity_acc" not in module.train_metrics


class TestRegressionModeIntegration:
    """Integration tests for regression mode across multiple components."""

    def test_regression_mode_end_to_end(self):
        """Test complete end-to-end regression mode functionality."""
        # Create network
        net = SimpleCNN(
            input_channels=3,
            output_mode="regression",
            parameter_names=["note_number", "note_velocity"],
            parameter_ranges={"note_number": (50.0, 52.0), "note_velocity": (80.0, 82.0)},
            input_size=32,
        )

        # Create loss functions
        criteria = {
            "note_number": NormalizedRegressionLoss(param_range=(50.0, 52.0), num_classes=21, loss_type="l1"),
            "note_velocity": NormalizedRegressionLoss(param_range=(80.0, 82.0), num_classes=21, loss_type="l1"),
        }

        # Create module
        module = VIMHLitModule(
            net=net,
            optimizer=torch.optim.Adam,
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
            criteria=criteria,
            loss_type="normalized_regression",
            auto_configure_from_dataset=False,
        )

        # Test training step
        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32)
        y = {
            "note_number": torch.tensor([51.0, 50.6, 51.4, 51.8]),
            "note_velocity": torch.tensor([81.0, 80.3, 81.7, 80.9]),
        }
        batch = (x, y)

        # Setup metrics; bounds are normally set by dataset auto-configuration
        module._setup_metrics()
        module.param_bounds = {"note_number": (50.0, 52.0), "note_velocity": (80.0, 82.0)}

        # Test model step
        loss, preds, targets = module.model_step(batch)

        # Verify everything works
        assert torch.isfinite(loss)
        assert loss.item() > 0
        assert isinstance(preds, dict)
        assert isinstance(targets, dict)
        assert len(preds) == 2
        assert len(targets) == 2

        # Test metrics update (would normally happen in training_step)
        for param_name in preds.keys():
            mae_metric = module.train_metrics[f"{param_name}_mae"]
            mae_metric.update(preds[param_name], targets[param_name])
            mae_value = mae_metric.compute()
            assert torch.isfinite(mae_value)
            assert mae_value.item() >= 0

    def test_regression_classification_mode_switching(self):
        """Test that the same network can switch between regression and classification."""
        # Test classification mode
        net_classification = SimpleCNN(
            input_channels=3,
            output_mode="classification",
            heads_config={"note_number": 256, "note_velocity": 256},
            input_size=32,
        )

        # Test regression mode
        net_regression = SimpleCNN(
            input_channels=3,
            output_mode="regression",
            parameter_names=["note_number", "note_velocity"],
            parameter_ranges={"note_number": (50.0, 52.0), "note_velocity": (80.0, 82.0)},
            input_size=32,
        )

        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32)

        with torch.no_grad():
            output_classification = net_classification(x)
            output_regression = net_regression(x)

        # Classification outputs should be logits
        assert output_classification["note_number"].shape == (batch_size, 256)
        assert output_classification["note_velocity"].shape == (batch_size, 256)

        # Regression outputs should be sigmoid-activated [0,1]
        assert output_regression["note_number"].shape == (batch_size, 1)
        assert output_regression["note_velocity"].shape == (batch_size, 1)
        assert torch.all(output_regression["note_number"] >= 0.0)
        assert torch.all(output_regression["note_number"] <= 1.0)
        assert torch.all(output_regression["note_velocity"] >= 0.0)
        assert torch.all(output_regression["note_velocity"] <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__])
