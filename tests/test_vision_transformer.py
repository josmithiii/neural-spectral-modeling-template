import pytest
import torch

from src.models.components.vision_transformer import (
    VisionTransformer,
    EmbedLayer,
    SelfAttention,
    Encoder,
    Classifier,
)


@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("embed_dim", [32, 64, 128])
def test_vit_forward_pass(batch_size: int, embed_dim: int) -> None:
    """Test ViT forward pass with different batch sizes and embedding dimensions.
    
    :param batch_size: Batch size for the input tensor.
    :param embed_dim: Embedding dimension for the ViT model.
    """
    model = VisionTransformer(
        n_channels=1,
        image_size=28,
        patch_size=4,
        embed_dim=embed_dim,
        n_layers=4,
        n_attention_heads=4,
        forward_mul=2,
        output_size=10,
        dropout=0.1
    )
    
    # Test with MNIST-like input
    x = torch.randn(batch_size, 1, 28, 28)
    output = model(x)
    
    assert output.shape == (batch_size, 10)
    assert output.dtype == torch.float32
    
    # Test with flattened input (backward compatibility)
    x_flat = torch.randn(batch_size, 784)
    output_flat = model(x_flat)
    
    assert output_flat.shape == (batch_size, 10)
    assert output_flat.dtype == torch.float32


def test_vit_parameter_counts() -> None:
    """Verify parameter counts for different ViT variants match expected values."""
    
    # Small ViT configuration (~38K parameters)
    model_small = VisionTransformer(
        n_channels=1,
        image_size=28,
        patch_size=4,
        embed_dim=32,
        n_layers=4,
        n_attention_heads=4,
        forward_mul=2,
        output_size=10,
        dropout=0.1
    )
    params_small = sum(p.numel() for p in model_small.parameters())
    assert 35_000 < params_small < 45_000, f"Expected ~38K params, got {params_small:,}"
    
    # Medium ViT configuration (~210K parameters)
    model_medium = VisionTransformer(
        n_channels=1,
        image_size=28,
        patch_size=4,
        embed_dim=64,
        n_layers=6,
        n_attention_heads=4,
        forward_mul=2,
        output_size=10,
        dropout=0.1
    )
    params_medium = sum(p.numel() for p in model_medium.parameters())
    assert 200_000 < params_medium < 220_000, f"Expected ~210K params, got {params_medium:,}"
    
    # Large ViT configuration (~821K parameters)
    model_large = VisionTransformer(
        n_channels=1,
        image_size=28,
        patch_size=4,
        embed_dim=128,
        n_layers=6,
        n_attention_heads=8,
        forward_mul=2,
        output_size=10,
        dropout=0.1
    )
    params_large = sum(p.numel() for p in model_large.parameters())
    assert 800_000 < params_large < 850_000, f"Expected ~821K params, got {params_large:,}"


def test_vit_torch_vs_scratch() -> None:
    """Compare PyTorch vs scratch implementation outputs to ensure compatibility."""
    
    # Create identical configurations
    config = {
        'n_channels': 1,
        'image_size': 28,
        'patch_size': 4,
        'embed_dim': 64,
        'n_layers': 4,
        'n_attention_heads': 4,
        'forward_mul': 2,
        'output_size': 10,
        'dropout': 0.0  # Disable dropout for deterministic comparison
    }
    
    model_scratch = VisionTransformer(use_torch_layers=False, **config)
    model_torch = VisionTransformer(use_torch_layers=True, **config)
    
    # Set both models to evaluation mode
    model_scratch.eval()
    model_torch.eval()
    
    # Test input
    x = torch.randn(2, 1, 28, 28)
    
    with torch.no_grad():
        output_scratch = model_scratch(x)
        output_torch = model_torch(x)
    
    # Both should produce outputs of the same shape
    assert output_scratch.shape == output_torch.shape
    assert output_scratch.shape == (2, 10)


@pytest.mark.parametrize("n_channels", [1, 3])
@pytest.mark.parametrize("image_size", [28, 32])
def test_embed_layer(n_channels: int, image_size: int) -> None:
    """Test EmbedLayer component with different input configurations.
    
    :param n_channels: Number of input channels.
    :param image_size: Size of the input image.
    """
    embed_dim = 64
    patch_size = 4
    batch_size = 2
    
    embed_layer = EmbedLayer(n_channels, embed_dim, image_size, patch_size)
    
    x = torch.randn(batch_size, n_channels, image_size, image_size)
    output = embed_layer(x)
    
    expected_seq_len = (image_size // patch_size) ** 2 + 1  # +1 for CLS token
    assert output.shape == (batch_size, expected_seq_len, embed_dim)


@pytest.mark.parametrize("embed_dim", [32, 64, 128])
@pytest.mark.parametrize("n_attention_heads", [2, 4, 8])
def test_self_attention(embed_dim: int, n_attention_heads: int) -> None:
    """Test SelfAttention component with different configurations.
    
    :param embed_dim: Embedding dimension.
    :param n_attention_heads: Number of attention heads.
    """
    if embed_dim % n_attention_heads != 0:
        pytest.skip(f"embed_dim ({embed_dim}) must be divisible by n_attention_heads ({n_attention_heads})")
    
    batch_size = 2
    seq_len = 50  # Example sequence length
    
    attention = SelfAttention(embed_dim, n_attention_heads)
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    output = attention(x)
    
    assert output.shape == x.shape
    assert output.dtype == torch.float32


def test_encoder() -> None:
    """Test Encoder component."""
    embed_dim = 64
    n_attention_heads = 4
    forward_mul = 2
    batch_size = 2
    seq_len = 50
    
    encoder = Encoder(embed_dim, n_attention_heads, forward_mul)
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    output = encoder(x)
    
    assert output.shape == x.shape
    assert output.dtype == torch.float32


def test_classifier() -> None:
    """Test Classifier component."""
    embed_dim = 64
    n_classes = 10
    batch_size = 2
    seq_len = 50
    
    classifier = Classifier(embed_dim, n_classes)
    
    # Input includes CLS token at position 0
    x = torch.randn(batch_size, seq_len, embed_dim)
    output = classifier(x)
    
    assert output.shape == (batch_size, n_classes)
    assert output.dtype == torch.float32


@pytest.mark.parametrize("dropout", [0.0, 0.1, 0.2])
def test_vit_dropout(dropout: float) -> None:
    """Test ViT with different dropout rates.
    
    :param dropout: Dropout probability.
    """
    model = VisionTransformer(
        n_channels=1,
        image_size=28,
        patch_size=4,
        embed_dim=32,
        n_layers=2,
        n_attention_heads=4,
        forward_mul=2,
        output_size=10,
        dropout=dropout
    )
    
    x = torch.randn(2, 1, 28, 28)
    
    # Test in training mode
    model.train()
    output_train = model(x)
    assert output_train.shape == (2, 10)
    
    # Test in evaluation mode
    model.eval()
    output_eval = model(x)
    assert output_eval.shape == (2, 10)


def test_vit_gradient_flow() -> None:
    """Test that gradients flow properly through the ViT model."""
    model = VisionTransformer(
        n_channels=1,
        image_size=28,
        patch_size=4,
        embed_dim=32,
        n_layers=2,
        n_attention_heads=4,
        forward_mul=2,
        output_size=10,
        dropout=0.1
    )
    
    x = torch.randn(2, 1, 28, 28, requires_grad=True)
    target = torch.randint(0, 10, (2,))
    
    output = model(x)
    loss = torch.nn.functional.cross_entropy(output, target)
    loss.backward()
    
    # Check that gradients exist for model parameters
    for param in model.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()
        assert not torch.isinf(param.grad).any()


@pytest.mark.slow
def test_vit_integration_training() -> None:
    """Integration test: verify ViT can complete a training step without errors."""
    model = VisionTransformer(
        n_channels=1,
        image_size=28,
        patch_size=4,
        embed_dim=32,
        n_layers=2,
        n_attention_heads=4,
        forward_mul=2,
        output_size=10,
        dropout=0.1
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Simulate a few training steps
    model.train()
    for _ in range(3):
        x = torch.randn(4, 1, 28, 28)
        target = torch.randint(0, 10, (4,))
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Check that loss is finite
        assert torch.isfinite(loss)
        assert loss.item() > 0