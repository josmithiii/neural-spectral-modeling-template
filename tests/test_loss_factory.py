"""Test loss function factory and general-purpose loss functions."""

import pytest
import torch
import torch.nn as nn

from src.models.losses import (
    create_loss_function,
    MultiScaleSpectralLoss,
    DistanceLoss,
    OrdinalRegressionLoss,
    QuantizedRegressionLoss,
    WeightedCrossEntropyLoss,
    NormalizedRegressionLoss
)


class TestCreateLossFunction:
    """Test loss function factory."""

    def test_create_standard_pytorch_losses(self):
        """Test creation of standard PyTorch losses."""
        # CrossEntropyLoss
        loss_config = {'_target_': 'torch.nn.CrossEntropyLoss'}
        loss = create_loss_function(loss_config)
        assert isinstance(loss, nn.CrossEntropyLoss)

        # MSELoss
        loss_config = {'_target_': 'torch.nn.MSELoss'}
        loss = create_loss_function(loss_config)
        assert isinstance(loss, nn.MSELoss)

        # L1Loss
        loss_config = {'_target_': 'torch.nn.L1Loss'}
        loss = create_loss_function(loss_config)
        assert isinstance(loss, nn.L1Loss)

        # HuberLoss with parameters
        loss_config = {
            '_target_': 'torch.nn.HuberLoss',
            'delta': 1.5
        }
        loss = create_loss_function(loss_config)
        assert isinstance(loss, nn.HuberLoss)
        assert loss.delta == 1.5

    def test_create_custom_losses(self):
        """Test creation of custom loss functions."""
        # OrdinalRegressionLoss
        loss_config = {
            '_target_': 'src.models.losses.OrdinalRegressionLoss',
            'num_classes': 256,
            'param_range': 2.0,
            'regression_loss': 'l1',
            'alpha': 0.1
        }
        loss = create_loss_function(loss_config)
        assert isinstance(loss, OrdinalRegressionLoss)
        assert loss.num_classes == 256
        assert loss.param_range == 2.0

        # MultiScaleSpectralLoss
        loss_config = {
            '_target_': 'src.models.losses.MultiScaleSpectralLoss',
            'max_n_fft': 1024,
            'num_scales': 4,
            'p': 1.0
        }
        loss = create_loss_function(loss_config)
        assert isinstance(loss, MultiScaleSpectralLoss)
        assert loss.max_n_fft == 1024
        assert len(loss.n_ffts) == 4

    def test_invalid_target_raises_error(self):
        """Test that invalid target raises ValueError."""
        loss_config = {'_target_': 'nonexistent.loss.Function'}

        with pytest.raises(ValueError, match="Unknown loss function target"):
            create_loss_function(loss_config)

    def test_missing_target_raises_error(self):
        """Test that missing '_target_' key raises ValueError."""
        loss_config = {'some_param': 'value'}

        with pytest.raises(ValueError, match="Loss configuration must contain '_target_' key"):
            create_loss_function(loss_config)


class TestMultiScaleSpectralLoss:
    """Test MultiScaleSpectralLoss functionality."""

    def test_multiscale_spectral_loss_creation(self):
        """Test creation of MultiScaleSpectralLoss."""
        loss = MultiScaleSpectralLoss(max_n_fft=1024, num_scales=4, p=1.0)

        assert len(loss.n_ffts) == 4
        assert loss.n_ffts == [1024, 512, 256, 128]
        assert len(loss.hop_lengths) == 4
        assert loss.hop_lengths == [256, 128, 64, 32]
        assert len(loss.ops) == 4
        assert loss.p == 1.0

    def test_multiscale_spectral_loss_forward(self):
        """Test forward pass of MultiScaleSpectralLoss with dummy audio data."""
        loss = MultiScaleSpectralLoss(max_n_fft=512, num_scales=3, p=2.0)

        # Create dummy audio tensors (batch_size=2, sequence_length=4096)
        x = torch.randn(2, 4096)
        y = torch.randn(2, 4096)

        output = loss(x, y)

        assert isinstance(output, torch.Tensor)
        assert output.numel() == 1  # Scalar loss
        assert output >= 0  # Loss should be non-negative

    def test_multiscale_spectral_loss_invalid_config(self):
        """Test that invalid configuration raises error."""
        with pytest.raises(AssertionError, match="max_n_fft.*too small"):
            # max_n_fft too small for num_scales
            MultiScaleSpectralLoss(max_n_fft=8, num_scales=6)


class TestDistanceLoss:
    """Test DistanceLoss base class."""

    def test_distance_loss_l1_norm(self):
        """Test distance computation with L1 norm."""
        loss = DistanceLoss(p=1.0)

        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([1.5, 2.5, 3.5])

        distance = loss.dist(x, y)
        expected = torch.abs(x - y).mean()  # L1 norm

        assert torch.allclose(distance, expected)

    def test_distance_loss_l2_norm(self):
        """Test distance computation with L2 norm."""
        loss = DistanceLoss(p=2.0)

        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([1.5, 2.5, 3.5])

        distance = loss.dist(x, y)
        expected = torch.norm(x - y, p=2.0)  # L2 norm

        assert torch.allclose(distance, expected)

    def test_distance_loss_forward_no_ops(self):
        """Test forward pass with no operations defined."""
        loss = DistanceLoss(p=2.0)

        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([1.5, 2.5, 3.5])

        # Should return 0.0 when no operations are defined
        output = loss(x, y)
        assert output == 0.0


class TestParameterValidation:
    """Test parameter validation in custom losses."""

    def test_ordinal_regression_loss_parameters(self):
        """Test OrdinalRegressionLoss parameter validation."""
        loss = OrdinalRegressionLoss(
            num_classes=10,
            param_range=1.0,
            regression_loss='huber',
            huber_delta=0.5
        )

        assert loss.num_classes == 10
        assert loss.param_range == 1.0
        assert loss.regression_loss == 'huber'
        assert loss.huber_delta == 0.5
        assert loss.quantization_step == 1.0 / 9  # (max - min) / (num_classes - 1)

    def test_normalized_regression_loss_validation(self):
        """Test NormalizedRegressionLoss parameter validation."""
        # Valid range
        loss = NormalizedRegressionLoss(param_range=(0.0, 1.0))
        assert loss.param_min == 0.0
        assert loss.param_max == 1.0
        assert loss.param_range == 1.0

        # Invalid range should raise error
        with pytest.raises(ValueError, match="Parameter range must be positive"):
            NormalizedRegressionLoss(param_range=(1.0, 1.0))


class TestLossFunctionIntegration:
    """Test integration between different loss functions."""

    def test_loss_factory_with_multihead_config(self):
        """Test loss factory with multihead configuration."""
        loss_configs = {
            'head_classification': {
                '_target_': 'torch.nn.CrossEntropyLoss'
            },
            'head_regression': {
                '_target_': 'src.models.losses.OrdinalRegressionLoss',
                'num_classes': 256,
                'param_range': 2.0
            },
            'head_spectral': {
                '_target_': 'src.models.losses.MultiScaleSpectralLoss',
                'max_n_fft': 512,
                'num_scales': 3
            }
        }

        # Create all losses
        created_losses = {}
        for head_name, loss_config in loss_configs.items():
            created_losses[head_name] = create_loss_function(loss_config)

        # Verify correct types
        assert isinstance(created_losses['head_classification'], nn.CrossEntropyLoss)
        assert isinstance(created_losses['head_regression'], OrdinalRegressionLoss)
        assert isinstance(created_losses['head_spectral'], MultiScaleSpectralLoss)

        # Verify parameters
        assert created_losses['head_regression'].num_classes == 256
        assert created_losses['head_spectral'].max_n_fft == 512

    def test_soft_target_loss_creation(self):
        """Test creation of SoftTargetLoss through factory."""
        loss_config = {
            '_target_': 'src.models.soft_target_loss.SoftTargetLoss',
            'num_classes': 10,
            'mode': 'gaussian',
            'sigma': 1.5
        }
        loss = create_loss_function(loss_config)

        # Import here to avoid circular imports
        from src.models.soft_target_loss import SoftTargetLoss
        assert isinstance(loss, SoftTargetLoss)
        assert loss.num_classes == 10
        assert loss.mode == 'gaussian'
        assert loss.sigma == 1.5

    def test_soft_target_loss_forward(self):
        """Test SoftTargetLoss forward pass."""
        from src.models.soft_target_loss import SoftTargetLoss

        loss = SoftTargetLoss(num_classes=5, mode='triangular', width=1)
        logits = torch.randn(3, 5)  # batch_size=3, num_classes=5
        targets = torch.tensor([1, 2, 4])  # target classes

        output = loss(logits, targets)

        assert isinstance(output, torch.Tensor)
        assert output.numel() == 1  # Scalar loss
        assert output >= 0  # Loss should be non-negative
