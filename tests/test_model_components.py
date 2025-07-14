import pytest
import torch
import subprocess
import sys
from pathlib import Path

from src.models.components.simple_dense_net import SimpleDenseNet
from src.models.components.simple_efficientnet import SimpleEfficientNet
from src.models.components.simple_cnn import SimpleCNN


@pytest.mark.parametrize("batch_size", [2, 4])  # Skip batch_size=1 due to BatchNorm
def test_simple_dense_net_forward_pass(batch_size: int) -> None:
    """Test SimpleDenseNet forward pass with different batch sizes."""
    # Test backward compatibility (single head)
    model_single = SimpleDenseNet(output_size=10)
    model_single.eval()  # Set to eval mode to avoid BatchNorm issues
    x = torch.randn(batch_size, 1, 28, 28)
    output = model_single(x)

    assert output.shape == (batch_size, 10)
    assert output.dtype == torch.float32
    assert not torch.isnan(output).any()

    # Test multihead mode
    model_multi = SimpleDenseNet(heads_config={'digit': 10, 'thickness': 5})
    model_multi.eval()  # Set to eval mode to avoid BatchNorm issues
    output_multi = model_multi(x)

    assert isinstance(output_multi, dict)
    assert 'digit' in output_multi
    assert 'thickness' in output_multi
    assert output_multi['digit'].shape == (batch_size, 10)
    assert output_multi['thickness'].shape == (batch_size, 5)


@pytest.mark.parametrize("batch_size", [2, 4])  # Skip batch_size=1 due to BatchNorm
def test_simple_efficientnet_forward_pass(batch_size: int) -> None:
    """Test SimpleEfficientNet forward pass with different batch sizes."""
    # Test backward compatibility (single head)
    model_single = SimpleEfficientNet(num_classes=10)
    model_single.eval()  # Set to eval mode to avoid BatchNorm issues
    x = torch.randn(batch_size, 1, 28, 28)
    output = model_single(x)

    assert output.shape == (batch_size, 10)
    assert output.dtype == torch.float32
    assert not torch.isnan(output).any()

    # Test multihead mode
    model_multi = SimpleEfficientNet(heads_config={'digit': 10, 'thickness': 5})
    model_multi.eval()  # Set to eval mode to avoid BatchNorm issues
    output_multi = model_multi(x)

    assert isinstance(output_multi, dict)
    assert 'digit' in output_multi
    assert 'thickness' in output_multi
    assert output_multi['digit'].shape == (batch_size, 10)
    assert output_multi['thickness'].shape == (batch_size, 5)


def test_simple_dense_net_parameter_counts() -> None:
    """Test parameter counts for SimpleDenseNet models."""
    # Single head model
    model_single = SimpleDenseNet(
        input_size=784,
        lin1_size=64,
        lin2_size=128,
        lin3_size=64,
        output_size=10
    )
    single_params = sum(p.numel() for p in model_single.parameters())

    # Multihead model with same architecture
    model_multi = SimpleDenseNet(
        input_size=784,
        lin1_size=64,
        lin2_size=128,
        lin3_size=64,
        heads_config={'digit': 10, 'thickness': 5}
    )
    multi_params = sum(p.numel() for p in model_multi.parameters())

    # Multihead should have more parameters due to additional head
    assert multi_params > single_params

    # Reasonable parameter range for this architecture
    assert 50_000 < single_params < 200_000
    assert 50_000 < multi_params < 200_000


def test_simple_efficientnet_parameter_counts() -> None:
    """Test parameter counts for SimpleEfficientNet models."""
    # Single head model
    model_single = SimpleEfficientNet(num_classes=10)
    single_params = sum(p.numel() for p in model_single.parameters())

    # Multihead model
    model_multi = SimpleEfficientNet(heads_config={'digit': 10, 'thickness': 5})
    multi_params = sum(p.numel() for p in model_multi.parameters())

    # Multihead should have more parameters due to additional head
    assert multi_params > single_params

    # EfficientNet has more parameters than SimpleDenseNet
    assert 100_000 < single_params < 10_000_000
    assert 100_000 < multi_params < 10_000_000


def test_backward_compatibility_simple_dense_net() -> None:
    """Test backward compatibility for SimpleDenseNet."""
    # Test old-style initialization
    model_old = SimpleDenseNet(output_size=10)
    assert not model_old.is_multihead
    assert 'digit' in model_old.heads_config
    assert model_old.heads_config['digit'] == 10

    # Test new-style single head
    model_new = SimpleDenseNet(heads_config={'digit': 10})
    assert not model_new.is_multihead

    # Both should produce same output shape
    x = torch.randn(2, 1, 28, 28)
    output_old = model_old(x)
    output_new = model_new(x)

    assert output_old.shape == output_new.shape == (2, 10)
    assert isinstance(output_old, torch.Tensor)
    assert isinstance(output_new, torch.Tensor)


def test_backward_compatibility_simple_efficientnet() -> None:
    """Test backward compatibility for SimpleEfficientNet."""
    # Test old-style initialization
    model_old = SimpleEfficientNet(num_classes=10)
    assert not model_old.is_multihead
    assert 'digit' in model_old.heads_config
    assert model_old.heads_config['digit'] == 10

    # Test new-style single head
    model_new = SimpleEfficientNet(heads_config={'digit': 10})
    assert not model_new.is_multihead

    # Both should produce same output shape
    x = torch.randn(2, 1, 28, 28)
    output_old = model_old(x)
    output_new = model_new(x)

    assert output_old.shape == output_new.shape == (2, 10)
    assert isinstance(output_old, torch.Tensor)
    assert isinstance(output_new, torch.Tensor)


def test_gradient_flow_simple_dense_net() -> None:
    """Test gradient flow through SimpleDenseNet."""
    model = SimpleDenseNet(output_size=10)
    x = torch.randn(2, 1, 28, 28, requires_grad=True)

    output = model(x)
    loss = output.sum()
    loss.backward()

    # Check that gradients exist for all parameters
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for parameter {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for parameter {name}"


def test_gradient_flow_simple_efficientnet() -> None:
    """Test gradient flow through SimpleEfficientNet."""
    model = SimpleEfficientNet(num_classes=10)
    x = torch.randn(2, 1, 28, 28, requires_grad=True)

    output = model(x)
    loss = output.sum()
    loss.backward()

    # Check that gradients exist for all parameters
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for parameter {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for parameter {name}"


def test_multihead_consistency() -> None:
    """Test that multihead models are consistent with single-head models."""
    # Create multihead models with same architecture as single head
    dense_multi = SimpleDenseNet(heads_config={'digit': 10})
    efficientnet_multi = SimpleEfficientNet(heads_config={'digit': 10})

    dense_single = SimpleDenseNet(output_size=10)
    efficientnet_single = SimpleEfficientNet(num_classes=10)

    x = torch.randn(2, 1, 28, 28)

    # Single head models return tensors
    dense_single_out = dense_single(x)
    efficientnet_single_out = efficientnet_single(x)

    assert isinstance(dense_single_out, torch.Tensor)
    assert isinstance(efficientnet_single_out, torch.Tensor)

    # Multihead models with single head return tensors too
    dense_multi_out = dense_multi(x)
    efficientnet_multi_out = efficientnet_multi(x)

    assert isinstance(dense_multi_out, torch.Tensor)
    assert isinstance(efficientnet_multi_out, torch.Tensor)

    # Shapes should match
    assert dense_single_out.shape == dense_multi_out.shape
    assert efficientnet_single_out.shape == efficientnet_multi_out.shape


def test_simple_dense_net_main_execution() -> None:
    """Test that SimpleDenseNet __main__ block executes without errors."""
    # Test the actual __main__ execution by running the file directly
    result = subprocess.run([
        sys.executable,
        "src/models/components/simple_dense_net.py"
    ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

    assert result.returncode == 0, f"SimpleDenseNet main execution failed: {result.stderr}"


def test_simple_efficientnet_main_execution() -> None:
    """Test that SimpleEfficientNet __main__ block executes without errors."""
    # Test the actual __main__ execution by running the file directly
    result = subprocess.run([
        sys.executable,
        "src/models/components/simple_efficientnet.py"
    ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

    assert result.returncode == 0, f"SimpleEfficientNet main execution failed: {result.stderr}"
    assert "Testing single head mode" in result.stdout
    assert "Testing multihead mode" in result.stdout


@pytest.mark.parametrize("model_class", [SimpleDenseNet, SimpleEfficientNet])
def test_model_training_mode(model_class) -> None:
    """Test that models work correctly in training and eval modes."""
    if model_class == SimpleDenseNet:
        model = model_class(output_size=10)
    else:
        model = model_class(num_classes=10)

    x = torch.randn(2, 1, 28, 28)

    # Training mode
    model.train()
    output_train = model(x)

    # Eval mode
    model.eval()
    output_eval = model(x)

    assert output_train.shape == output_eval.shape
    assert not torch.isnan(output_train).any()
    assert not torch.isnan(output_eval).any()


if __name__ == "__main__":
    pytest.main([__file__])
