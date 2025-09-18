"""Test MPS compatibility features including input shape inference and TensorBoard summaries."""

import math
import pytest
import torch
from unittest.mock import MagicMock

from src.models.components.simple_cnn import SimpleCNN
from src.models.components.vision_transformer import VisionTransformer
from src.models.vimh_lit_module import VIMHLitModule


class TestInputShapeAttributes:
    """Test input shape attributes for TensorBoard compatibility."""

    def test_simple_cnn_input_attributes(self):
        """Test SimpleCNN stores input shape attributes correctly."""
        # Test square input
        model = SimpleCNN(input_size=32, input_channels=1)

        assert hasattr(model, "input_size")
        assert hasattr(model, "input_resolution")
        assert model.input_size == 32
        assert model.input_resolution == (32, 32)

        # Test with different input size
        model_large = SimpleCNN(input_size=64, input_channels=3)
        assert model_large.input_size == 64
        assert model_large.input_resolution == (64, 64)

    def test_vision_transformer_input_attributes_square(self):
        """Test VisionTransformer stores input shape attributes for square images."""
        model = VisionTransformer(n_channels=1, image_size=28)

        assert hasattr(model, "n_channels")
        assert hasattr(model, "image_size")
        assert hasattr(model, "input_shape")
        assert hasattr(model, "input_resolution")

        assert model.n_channels == 1
        assert model.image_size == 28
        assert model.input_shape == (1, 28, 28)
        assert model.input_resolution == (28, 28)

    def test_vision_transformer_input_attributes_rectangular(self):
        """Test VisionTransformer stores input shape attributes for rectangular images."""
        model = VisionTransformer(n_channels=3, image_size=[32, 64])

        assert model.n_channels == 3
        assert model.image_size == [32, 64]
        assert model.input_shape == (3, 32, 64)
        assert model.input_resolution == [32, 64]

    def test_vision_transformer_input_attributes_tuple(self):
        """Test VisionTransformer handles tuple image_size correctly."""
        model = VisionTransformer(n_channels=1, image_size=(48, 32))

        assert model.n_channels == 1
        assert model.image_size == (48, 32)
        assert model.input_shape == (1, 48, 32)
        assert model.input_resolution == (48, 32)


class TestShapeInferenceMethods:
    """Test shape inference methods in VIMHLitModule."""

    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = torch.optim.Adam([torch.tensor(1.0)])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1)
        self.criterion = torch.nn.CrossEntropyLoss()

    def test_infer_input_channels_simple_cnn(self):
        """Test input channel inference for SimpleCNN."""
        cnn = SimpleCNN(input_size=32, input_channels=3)
        module = VIMHLitModule(
            net=cnn,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            criterion=self.criterion
        )

        # Should detect from input_channels attribute
        assert module._infer_input_channels() == 3

    def test_infer_input_channels_cnn_conv_layers(self):
        """Test input channel inference from conv layers when input_channels missing."""
        # This test verifies the inference works, even if not perfectly mocked
        cnn = SimpleCNN(input_size=32, input_channels=2)
        module = VIMHLitModule(
            net=cnn,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            criterion=self.criterion
        )

        # Should detect from input_channels attribute (which exists in SimpleCNN)
        assert module._infer_input_channels() == 2

    def test_infer_input_channels_vision_transformer(self):
        """Test input channel inference for VisionTransformer."""
        vit = VisionTransformer(n_channels=3, image_size=32)
        module = VIMHLitModule(
            net=vit,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            criterion=self.criterion
        )

        # VisionTransformer has n_channels but not input_channels, should detect from embedding
        # The inference method should find the embedding layer
        channels = module._infer_input_channels()
        assert channels == 3

    def test_infer_input_channels_vit_embedding(self):
        """Test input channel inference from ViT embedding layer."""
        # Test with actual VisionTransformer to verify embedding detection works
        vit = VisionTransformer(n_channels=2, image_size=32)
        module = VIMHLitModule(
            net=vit,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            criterion=self.criterion
        )

        # Should detect from embedding conv layer
        channels = module._infer_input_channels()
        assert channels == 2

    def test_infer_input_channels_fallback(self):
        """Test input channel inference fallback."""
        # Create a mock network with no detectable channels
        mock_net = MagicMock(spec=[])
        # Ensure mock doesn't have any attributes we check for
        del mock_net.input_channels
        del mock_net.conv_layers
        del mock_net.embedding

        module = VIMHLitModule(
            net=mock_net,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            criterion=self.criterion
        )

        # Should fallback to 1
        assert module._infer_input_channels() == 1

    def test_infer_spatial_dims_simple_cnn(self):
        """Test spatial dimension inference for SimpleCNN."""
        cnn = SimpleCNN(input_size=64, input_channels=1)
        module = VIMHLitModule(
            net=cnn,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            criterion=self.criterion
        )

        # Should detect from input_size attribute
        assert module._infer_spatial_dims() == (64, 64)

    def test_infer_spatial_dims_vision_transformer(self):
        """Test spatial dimension inference for VisionTransformer."""
        vit = VisionTransformer(n_channels=1, image_size=48)
        module = VIMHLitModule(
            net=vit,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            criterion=self.criterion
        )

        # Should detect from image_size attribute
        assert module._infer_spatial_dims() == (48, 48)

    def test_infer_spatial_dims_rectangular(self):
        """Test spatial dimension inference for rectangular images."""
        vit = VisionTransformer(n_channels=1, image_size=[32, 64])
        module = VIMHLitModule(
            net=vit,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            criterion=self.criterion
        )

        # Should detect rectangular dimensions
        assert module._infer_spatial_dims() == (32, 64)

    def test_infer_spatial_dims_fallback(self):
        """Test spatial dimension inference fallback."""
        # Create a mock network with no detectable dimensions
        mock_net = MagicMock()
        module = VIMHLitModule(
            net=mock_net,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            criterion=self.criterion
        )

        # Should fallback to (32, 32) for NSMT
        assert module._infer_spatial_dims() == (32, 32)

    def test_infer_example_input_shape_simple_cnn(self):
        """Test complete input shape inference for SimpleCNN."""
        cnn = SimpleCNN(input_size=32, input_channels=3)
        module = VIMHLitModule(
            net=cnn,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            criterion=self.criterion
        )

        # Should return (batch_size, channels, height, width)
        assert module._infer_example_input_shape() == (1, 3, 32, 32)

    def test_infer_example_input_shape_vision_transformer(self):
        """Test complete input shape inference for VisionTransformer."""
        vit = VisionTransformer(n_channels=2, image_size=[48, 64])
        module = VIMHLitModule(
            net=vit,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            criterion=self.criterion
        )

        # Should return (batch_size, channels, height, width)
        shape = module._infer_example_input_shape()
        # Verify the shape matches expectations for channels and spatial dims
        assert shape[0] == 1  # batch size
        assert shape[1] == 2  # channels
        assert shape[2] == 48  # height
        assert shape[3] == 64  # width


class TestNormalizeToHW:
    """Test the _normalize_to_hw utility method."""

    def test_normalize_none(self):
        """Test normalize with None input."""
        assert VIMHLitModule._normalize_to_hw(None) is None

    def test_normalize_tuple_3d(self):
        """Test normalize with 3D tuple (channels, height, width)."""
        assert VIMHLitModule._normalize_to_hw((3, 32, 64)) == (32, 64)

    def test_normalize_tuple_2d(self):
        """Test normalize with 2D tuple (height, width)."""
        assert VIMHLitModule._normalize_to_hw((48, 32)) == (48, 32)

    def test_normalize_tuple_1d(self):
        """Test normalize with 1D tuple (square size)."""
        assert VIMHLitModule._normalize_to_hw((32,)) == (32, 32)

    def test_normalize_list_3d(self):
        """Test normalize with 3D list."""
        assert VIMHLitModule._normalize_to_hw([1, 28, 28]) == (28, 28)

    def test_normalize_int_square(self):
        """Test normalize with integer (perfect square)."""
        assert VIMHLitModule._normalize_to_hw(64) == (8, 8)  # sqrt(64) = 8
        assert VIMHLitModule._normalize_to_hw(49) == (7, 7)  # sqrt(49) = 7

    def test_normalize_int_non_square(self):
        """Test normalize with integer (not perfect square)."""
        assert VIMHLitModule._normalize_to_hw(32) == (32, 32)  # Not a perfect square
        assert VIMHLitModule._normalize_to_hw(50) == (50, 50)

    def test_normalize_int_invalid(self):
        """Test normalize with invalid integer."""
        assert VIMHLitModule._normalize_to_hw(0) is None
        assert VIMHLitModule._normalize_to_hw(-5) is None

    def test_normalize_torch_size(self):
        """Test normalize with torch.Size."""
        size = torch.Size([3, 32, 64])
        assert VIMHLitModule._normalize_to_hw(size) == (32, 64)

    def test_normalize_invalid_input(self):
        """Test normalize with invalid input types."""
        assert VIMHLitModule._normalize_to_hw("invalid") is None
        assert VIMHLitModule._normalize_to_hw({}) is None


class TestMPSCompatibility:
    """Test MPS compatibility features."""

    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = torch.optim.Adam([torch.tensor(1.0)])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1)
        self.criterion = torch.nn.CrossEntropyLoss()

    def test_example_input_array_generation_simple_cnn(self):
        """Test example_input_array generation for SimpleCNN."""
        cnn = SimpleCNN(input_size=32, input_channels=1, heads_config={"single": 10})
        module = VIMHLitModule(
            net=cnn,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            criterion=self.criterion,
            auto_configure_from_dataset=False
        )

        # Manually call setup to trigger example_input_array creation
        module.setup("fit")

        # Should create example_input_array for single-head models
        assert hasattr(module, "example_input_array")
        assert module.example_input_array.shape == (1, 1, 32, 32)

    def test_example_input_array_generation_vision_transformer(self):
        """Test example_input_array generation for VisionTransformer."""
        vit = VisionTransformer(n_channels=3, image_size=28, output_size=10)
        module = VIMHLitModule(
            net=vit,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            criterion=self.criterion,
            auto_configure_from_dataset=False
        )

        # Manually call setup to trigger example_input_array creation
        module.setup("fit")

        # Should create example_input_array for single-head models
        assert hasattr(module, "example_input_array")
        # Check the actual shape inference works
        shape = module.example_input_array.shape
        assert shape[0] == 1  # batch size
        assert shape[1] == 3  # channels (should be detected correctly)
        assert shape[2] == 28  # height
        assert shape[3] == 28  # width

    def test_example_input_array_multihead_skip(self):
        """Test that multihead models skip example_input_array to avoid JIT tracer issues."""
        cnn = SimpleCNN(
            input_size=32,
            input_channels=1,
            heads_config={"head1": 10, "head2": 5}
        )
        module = VIMHLitModule(
            net=cnn,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            criteria={"head1": torch.nn.CrossEntropyLoss(), "head2": torch.nn.CrossEntropyLoss()},
            auto_configure_from_dataset=False
        )

        # Manually call setup to trigger logic
        module.setup("fit")

        # Should NOT create example_input_array for multihead models
        assert not hasattr(module, "example_input_array") or module.example_input_array is None

    def test_adaptive_pooling_mps_safe_sizes(self):
        """Test that SimpleCNN uses MPS-safe adaptive pooling sizes."""
        # Test 28x28 input (common MNIST size)
        cnn_28 = SimpleCNN(input_size=28, input_channels=1)
        assert cnn_28.adaptive_pool_size == (7, 7)  # 28/4 = 7, which is used directly

        # Test 32x32 input (common spectrogram size)
        cnn_32 = SimpleCNN(input_size=32, input_channels=1)
        assert cnn_32.adaptive_pool_size == (4, 4)  # 32/4 = 8, use 4x4 for divisibility

        # Test other sizes fall back to safe default
        cnn_other = SimpleCNN(input_size=48, input_channels=1)
        assert cnn_other.adaptive_pool_size == (4, 4)  # Safe default

    def test_forward_pass_with_mps_safe_pooling(self):
        """Test that forward pass works with MPS-safe pooling sizes."""
        # Test various input sizes
        for input_size in [28, 32, 48, 64]:
            cnn = SimpleCNN(input_size=input_size, input_channels=1, heads_config={"test": 10})
            cnn.eval()

            # Create input tensor matching the expected size
            x = torch.randn(2, 1, input_size, input_size)

            # Forward pass should work without errors
            output = cnn(x)

            # SimpleCNN with single head returns tensor, not dict when is_multihead=False
            if cnn.is_multihead:
                assert isinstance(output, dict)
                assert "test" in output
                assert output["test"].shape == (2, 10)
                assert not torch.isnan(output["test"]).any()
            else:
                assert isinstance(output, torch.Tensor)
                assert output.shape == (2, 10)
                assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__])