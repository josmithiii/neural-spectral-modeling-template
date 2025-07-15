"""Tests for regression mode functionality in multihead models."""

import pytest
import torch
import torch.nn as nn
from src.models.components.simple_cnn import SimpleCNN
from src.models.losses import NormalizedRegressionLoss
from src.models.multihead_module import MultiheadLitModule


class TestRegressionNetworkArchitecture:
    """Test regression network architecture components."""

    def test_regression_network_initialization(self):
        """Test that regression networks initialize correctly."""
        net = SimpleCNN(
            input_channels=3,
            output_mode="regression",
            parameter_names=["note_number", "note_velocity"],
            parameter_ranges={
                "note_number": (50.0, 52.0),
                "note_velocity": (80.0, 82.0)
            },
            input_size=32
        )

        assert net.output_mode == "regression"
        assert net.parameter_names == ["note_number", "note_velocity"]
        assert net.parameter_ranges == {
            "note_number": (50.0, 52.0),
            "note_velocity": (80.0, 82.0)
        }
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
            parameter_ranges={
                "note_number": (50.0, 52.0),
                "note_velocity": (80.0, 82.0)
            },
            input_size=32
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

    def test_regression_network_parameter_names_required(self):
        """Test that parameter_names is required for regression mode."""
        with pytest.raises(ValueError, match="parameter_names must be provided"):
            SimpleCNN(
                input_channels=3,
                output_mode="regression",
                input_size=32
            )

    def test_regression_network_backward_compatibility(self):
        """Test that classification mode still works (backward compatibility)."""
        net = SimpleCNN(
            input_channels=3,
            output_mode="classification",
            heads_config={"note_number": 256, "note_velocity": 256},
            input_size=32
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
    """Test the NormalizedRegressionLoss function."""

    def test_normalized_regression_loss_initialization(self):
        """Test that NormalizedRegressionLoss initializes correctly."""
        loss_fn = NormalizedRegressionLoss(
            param_range=(50.0, 52.0),
            loss_type="l1",
            return_perceptual_units=True
        )

        assert loss_fn.param_min == 50.0
        assert loss_fn.param_max == 52.0
        assert loss_fn.param_range == 2.0
        assert loss_fn.loss_type == "l1"
        assert loss_fn.return_perceptual_units is True

    def test_normalized_regression_loss_invalid_range(self):
        """Test that invalid parameter ranges raise errors."""
        with pytest.raises(ValueError, match="Parameter range must be positive"):
            NormalizedRegressionLoss(param_range=(52.0, 50.0))  # Invalid range

    @pytest.mark.parametrize("loss_type", ["l1", "mse", "huber"])
    def test_normalized_regression_loss_types(self, loss_type):
        """Test different loss types work correctly."""
        loss_fn = NormalizedRegressionLoss(
            param_range=(50.0, 52.0),
            loss_type=loss_type,
            return_perceptual_units=True
        )

        # Test data
        batch_size = 4
        preds = torch.tensor([[0.5], [0.3], [0.7], [0.9]])
        targets = torch.tensor([51.0, 50.6, 51.4, 51.8])

        loss = loss_fn(preds, targets)

        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_normalized_regression_loss_unknown_type(self):
        """Test that unknown loss types raise errors."""
        loss_fn = NormalizedRegressionLoss(
            param_range=(50.0, 52.0),
            loss_type="unknown"
        )

        preds = torch.tensor([[0.5]])
        targets = torch.tensor([51.0])

        with pytest.raises(ValueError, match="Unknown loss type"):
            loss_fn(preds, targets)

    def test_normalized_regression_loss_perceptual_units(self):
        """Test that perceptual units scaling works correctly."""
        loss_fn_perceptual = NormalizedRegressionLoss(
            param_range=(50.0, 52.0),
            loss_type="l1",
            return_perceptual_units=True
        )

        loss_fn_normalized = NormalizedRegressionLoss(
            param_range=(50.0, 52.0),
            loss_type="mse",
            return_perceptual_units=False
        )

        # Test data
        preds = torch.tensor([[0.5]])
        targets = torch.tensor([51.0])

        loss_perceptual = loss_fn_perceptual(preds, targets)
        loss_normalized = loss_fn_normalized(preds, targets)

        # Perceptual loss should be scaled by parameter range
        assert torch.allclose(loss_perceptual, loss_normalized * 2.0)

    def test_normalized_regression_loss_target_clamping(self):
        """Test that targets are properly clamped to [0,1] range."""
        loss_fn = NormalizedRegressionLoss(
            param_range=(50.0, 52.0),
            loss_type="l1",
            return_perceptual_units=False
        )

        # Test with target outside parameter range
        preds = torch.tensor([[0.5]])
        targets_out_of_range = torch.tensor([55.0])  # Outside [50, 52]

        # Should not raise error due to clamping
        loss = loss_fn(preds, targets_out_of_range)
        assert torch.isfinite(loss)


class TestMultiheadRegressionModule:
    """Test the MultiheadLitModule with regression mode."""

    def test_multihead_regression_initialization(self):
        """Test that MultiheadLitModule initializes correctly in regression mode."""
        net = SimpleCNN(
            input_channels=3,
            output_mode="regression",
            parameter_names=["note_number", "note_velocity"],
            parameter_ranges={
                "note_number": (50.0, 52.0),
                "note_velocity": (80.0, 82.0)
            },
            input_size=32
        )

        criteria = {
            "note_number": NormalizedRegressionLoss(
                param_range=(50.0, 52.0),
                loss_type="l1"
            ),
            "note_velocity": NormalizedRegressionLoss(
                param_range=(80.0, 82.0),
                loss_type="l1"
            )
        }

        module = MultiheadLitModule(
            net=net,
            optimizer=torch.optim.Adam,
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
            criteria=criteria,
            output_mode="regression",
            auto_configure_from_dataset=False
        )

        assert module.output_mode == "regression"
        assert module.is_multihead
        assert len(module.criteria) == 2

    def test_multihead_regression_forward_pass(self):
        """Test that MultiheadLitModule forward pass works in regression mode."""
        net = SimpleCNN(
            input_channels=3,
            output_mode="regression",
            parameter_names=["note_number", "note_velocity"],
            parameter_ranges={
                "note_number": (50.0, 52.0),
                "note_velocity": (80.0, 82.0)
            },
            input_size=32
        )

        criteria = {
            "note_number": NormalizedRegressionLoss(
                param_range=(50.0, 52.0),
                loss_type="l1"
            ),
            "note_velocity": NormalizedRegressionLoss(
                param_range=(80.0, 82.0),
                loss_type="l1"
            )
        }

        module = MultiheadLitModule(
            net=net,
            optimizer=torch.optim.Adam,
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
            criteria=criteria,
            output_mode="regression",
            auto_configure_from_dataset=False
        )

        # Test model step
        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32)
        y = {
            "note_number": torch.tensor([51.0, 50.6, 51.4, 51.8]),
            "note_velocity": torch.tensor([81.0, 80.3, 81.7, 80.9])
        }
        batch = (x, y)

        with torch.no_grad():
            loss, preds, targets = module.model_step(batch)

        # Check outputs
        assert torch.isfinite(loss)
        assert loss.item() > 0
        assert isinstance(preds, dict)
        assert isinstance(targets, dict)

        # Check prediction shapes and ranges
        for param_name, pred in preds.items():
            assert pred.shape == (batch_size,)
            # Since we don't have a trainer/datamodule, predictions will be in [0,1] range
            assert torch.all(pred >= 0.0)
            assert torch.all(pred <= 1.0)

    def test_multihead_regression_metrics_setup(self):
        """Test that regression metrics are set up correctly."""
        net = SimpleCNN(
            input_channels=3,
            output_mode="regression",
            parameter_names=["note_number", "note_velocity"],
            parameter_ranges={
                "note_number": (50.0, 52.0),
                "note_velocity": (80.0, 82.0)
            },
            input_size=32
        )

        criteria = {
            "note_number": NormalizedRegressionLoss(
                param_range=(50.0, 52.0),
                loss_type="l1"
            ),
            "note_velocity": NormalizedRegressionLoss(
                param_range=(80.0, 82.0),
                loss_type="l1"
            )
        }

        module = MultiheadLitModule(
            net=net,
            optimizer=torch.optim.Adam,
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
            criteria=criteria,
            output_mode="regression",
            auto_configure_from_dataset=False
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

    def test_multihead_regression_is_regression_loss(self):
        """Test the _is_regression_loss method correctly identifies regression losses."""
        net = SimpleCNN(
            input_channels=3,
            output_mode="regression",
            parameter_names=["note_number"],
            parameter_ranges={"note_number": (50.0, 52.0)},
            input_size=32
        )

        module = MultiheadLitModule(
            net=net,
            optimizer=torch.optim.Adam,
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
            criteria={"note_number": nn.CrossEntropyLoss()},
            output_mode="regression",
            auto_configure_from_dataset=False
        )

        # Test different loss types
        regression_loss = NormalizedRegressionLoss(param_range=(50.0, 52.0))
        classification_loss = nn.CrossEntropyLoss()

        assert module._is_regression_loss(regression_loss)
        assert not module._is_regression_loss(classification_loss)


class TestRegressionModeIntegration:
    """Integration tests for regression mode across multiple components."""

    def test_regression_mode_end_to_end(self):
        """Test complete end-to-end regression mode functionality."""
        # Create network
        net = SimpleCNN(
            input_channels=3,
            output_mode="regression",
            parameter_names=["note_number", "note_velocity"],
            parameter_ranges={
                "note_number": (50.0, 52.0),
                "note_velocity": (80.0, 82.0)
            },
            input_size=32
        )

        # Create loss functions
        criteria = {
            "note_number": NormalizedRegressionLoss(
                param_range=(50.0, 52.0),
                loss_type="l1"
            ),
            "note_velocity": NormalizedRegressionLoss(
                param_range=(80.0, 82.0),
                loss_type="l1"
            )
        }

        # Create module
        module = MultiheadLitModule(
            net=net,
            optimizer=torch.optim.Adam,
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
            criteria=criteria,
            output_mode="regression",
            auto_configure_from_dataset=False
        )

        # Test training step
        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32)
        y = {
            "note_number": torch.tensor([51.0, 50.6, 51.4, 51.8]),
            "note_velocity": torch.tensor([81.0, 80.3, 81.7, 80.9])
        }
        batch = (x, y)

        # Setup metrics
        module._setup_metrics()

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
            input_size=32
        )

        # Test regression mode
        net_regression = SimpleCNN(
            input_channels=3,
            output_mode="regression",
            parameter_names=["note_number", "note_velocity"],
            parameter_ranges={
                "note_number": (50.0, 52.0),
                "note_velocity": (80.0, 82.0)
            },
            input_size=32
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
