import pytest
import torch

from src.models.components.convnext_v2 import (
    ConvNeXtV2,
    Block,
    LayerNorm,
    GRN,
    convnext_v2_mnist_tiny,
    convnext_v2_mnist_small,
    convnext_v2_mnist_base,
    convnext_v2_mnist_large,
)


@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("input_size", [28, 224])
def test_convnext_forward_pass(batch_size: int, input_size: int) -> None:
    """Test ConvNeXt-V2 forward pass with different batch sizes and input sizes.

    :param batch_size: Batch size for the input tensor.
    :param input_size: Input image size.
    """
    in_chans = 1 if input_size == 28 else 3
    output_size = 10 if input_size == 28 else 1000

    model = ConvNeXtV2(
        input_size=input_size,
        in_chans=in_chans,
        output_size=output_size,
        depths=(2, 2, 4, 2),
        dims=(24, 48, 96, 192),
        drop_path_rate=0.0,
    )

    x = torch.randn(batch_size, in_chans, input_size, input_size)
    output = model(x)

    assert output.shape == (batch_size, output_size)
    assert output.dtype == torch.float32


def test_convnext_parameter_counts() -> None:
    """Verify parameter counts for different ConvNeXt-V2 variants match expected values."""
    models = {
        "tiny": convnext_v2_mnist_tiny(),
        "small": convnext_v2_mnist_small(),
        "base": convnext_v2_mnist_base(),
        "large": convnext_v2_mnist_large(),
    }

    expected_ranges = {
        "tiny": (15_000, 25_000),    # ~18K params (ConvNeXt is inherently parameter-heavy)
        "small": (65_000, 85_000),   # ~73K params
        "base": (250_000, 350_000),  # ~288K params
        "large": (650_000, 800_000), # ~725K params
    }

    for name, model in models.items():
        param_count = sum(p.numel() for p in model.parameters())
        min_params, max_params = expected_ranges[name]
        assert min_params <= param_count <= max_params, \
            f"{name} model has {param_count} parameters, expected {min_params}-{max_params}"


@pytest.mark.parametrize("model_fn", [
    convnext_v2_mnist_tiny,
    convnext_v2_mnist_small,
    convnext_v2_mnist_base,
    convnext_v2_mnist_large,
])
def test_convnext_different_sizes(model_fn) -> None:
    """Test various ConvNeXt-V2 configurations and their outputs.

    :param model_fn: Factory function for creating the model.
    """
    model = model_fn()

    # Test MNIST input
    x = torch.randn(2, 1, 28, 28)
    output = model(x)

    assert output.shape == (2, 10)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_convnext_dropout() -> None:
    """Test dropout functionality and training vs eval modes."""
    model = ConvNeXtV2(
        input_size=28,
        in_chans=1,
        output_size=10,
        depths=(2, 2, 4, 2),
        dims=(24, 48, 96, 192),
        drop_path_rate=0.2,
    )

    x = torch.randn(4, 1, 28, 28)

    # Test training mode
    model.train()
    output_train = model(x)

    # Test eval mode
    model.eval()
    output_eval = model(x)

    assert output_train.shape == output_eval.shape == (4, 10)
    assert not torch.isnan(output_train).any()
    assert not torch.isnan(output_eval).any()


def test_convnext_gradient_flow() -> None:
    """Verify gradients flow properly through all layers."""
    model = convnext_v2_mnist_small()
    x = torch.randn(2, 1, 28, 28, requires_grad=True)

    output = model(x)
    loss = output.sum()
    loss.backward()

    # Check that gradients exist for all parameters
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for parameter {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for parameter {name}"


def test_convnext_integration_training() -> None:
    """End-to-end training test with sample data."""
    model = convnext_v2_mnist_tiny()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Create sample data
    x = torch.randn(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))

    # Training step
    model.train()
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    assert loss.item() >= 0
    assert not torch.isnan(loss)


def test_convnext_grn_functionality() -> None:
    """Test Global Response Normalization layer specifically."""
    grn = GRN(dim=64)
    x = torch.randn(2, 8, 8, 64)  # (N, H, W, C) format

    output = grn(x)

    assert output.shape == x.shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_convnext_layernorm_functionality() -> None:
    """Test LayerNorm with both data formats."""
    # Test channels_last format
    ln_last = LayerNorm(64, data_format="channels_last")
    x_last = torch.randn(2, 8, 8, 64)
    output_last = ln_last(x_last)

    assert output_last.shape == x_last.shape
    assert not torch.isnan(output_last).any()

    # Test channels_first format
    ln_first = LayerNorm(64, data_format="channels_first")
    x_first = torch.randn(2, 64, 8, 8)
    output_first = ln_first(x_first)

    assert output_first.shape == x_first.shape
    assert not torch.isnan(output_first).any()


def test_convnext_block_functionality() -> None:
    """Test ConvNeXt-V2 block functionality."""
    block = Block(dim=64, drop_path=0.1)
    x = torch.randn(2, 64, 14, 14)

    # Test forward pass
    output = block(x)

    assert output.shape == x.shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

    # Test residual connection
    # The output should not be identical to input due to the block computation
    assert not torch.equal(output, x)


@pytest.mark.parametrize("input_size", [28, 224])
def test_convnext_stem_adaptation(input_size: int) -> None:
    """Test stem layer adaptation for different input sizes."""
    in_chans = 1 if input_size == 28 else 3
    output_size = 10 if input_size == 28 else 1000

    model = ConvNeXtV2(
        input_size=input_size,
        in_chans=in_chans,
        output_size=output_size,
        depths=(2, 2, 4, 2),
        dims=(24, 48, 96, 192),
    )

    x = torch.randn(2, in_chans, input_size, input_size)

    # Test that the first downsampling layer works correctly
    features = model.downsample_layers[0](x)

    # For MNIST (28x28), stem should downsample by 2 (28->14)
    # For ImageNet (224x224), stem should downsample by 4 (224->56)
    expected_size = input_size // 2 if input_size == 28 else input_size // 4

    assert features.shape[2] == expected_size
    assert features.shape[3] == expected_size
    assert features.shape[1] == 24  # First dim size


def test_convnext_deterministic_output() -> None:
    """Test that model produces deterministic outputs with same input."""
    torch.manual_seed(42)
    model = convnext_v2_mnist_small()
    model.eval()

    x = torch.randn(2, 1, 28, 28)

    # Get output twice
    output1 = model(x)
    output2 = model(x)

    # Should be identical in eval mode
    assert torch.allclose(output1, output2, rtol=1e-5)


@pytest.mark.slow
def test_convnext_memory_efficiency() -> None:
    """Test memory efficiency for large batch sizes."""
    model = convnext_v2_mnist_tiny()
    large_batch = torch.randn(32, 1, 28, 28)

    # This should not cause memory issues
    output = model(large_batch)
    assert output.shape == (32, 10)
    assert not torch.isnan(output).any()
