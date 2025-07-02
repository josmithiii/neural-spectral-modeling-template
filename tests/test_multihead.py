import pytest
import torch
from src.models.components.simple_cnn import SimpleCNN
from src.data.multihead_dataset import MultiheadMNISTDataset
from src.models.mnist_module import MNISTLitModule
from torch.nn import CrossEntropyLoss


def test_multihead_cnn_forward():
    """Test that multihead CNN produces correct output shapes."""
    model = SimpleCNN(
        input_channels=1,
        heads_config={'digit': 10, 'thickness': 5, 'smoothness': 3}
    )
    
    x = torch.randn(4, 1, 28, 28)  # Batch of 4 MNIST images
    outputs = model(x)
    
    assert isinstance(outputs, dict)
    assert 'digit' in outputs
    assert 'thickness' in outputs
    assert 'smoothness' in outputs
    
    assert outputs['digit'].shape == (4, 10)
    assert outputs['thickness'].shape == (4, 5)
    assert outputs['smoothness'].shape == (4, 3)


def test_multihead_dataset():
    """Test multihead dataset wrapper and digit-dependent mappings."""
    # Create dummy MNIST-like dataset
    class DummyMNIST:
        def __len__(self):
            return 10
        def __getitem__(self, idx):
            return torch.randn(1, 28, 28), idx % 10
    
    base_dataset = DummyMNIST()
    multihead_dataset = MultiheadMNISTDataset(base_dataset)
    
    # Test various digits to verify mappings
    test_cases = [
        # (digit, expected_thickness, expected_smoothness)
        (0, 0, 2),  # Even → thickness=0, curved → smoothness=2
        (1, 2, 0),  # Odd → thickness=2, angular → smoothness=0  
        (2, 1, 1),  # Even → thickness=1, mixed → smoothness=1
        (3, 3, 2),  # Odd → thickness=3, curved → smoothness=2
        (4, 2, 0),  # Even → thickness=2, angular → smoothness=0
        (5, 4, 1),  # Odd → thickness=4, mixed → smoothness=1
        (7, 4, 0),  # Odd → thickness=4, angular → smoothness=0
        (8, 4, 2),  # Even → thickness=4, curved → smoothness=2
        (9, 4, 2),  # Odd → thickness=4, curved → smoothness=2
    ]
    
    for digit, expected_thickness, expected_smoothness in test_cases:
        image, labels = multihead_dataset[digit]
        
        assert isinstance(labels, dict)
        assert labels['digit'] == digit
        assert labels['thickness'] == expected_thickness, f"Digit {digit}: thickness {labels['thickness']} != {expected_thickness}"
        assert labels['smoothness'] == expected_smoothness, f"Digit {digit}: smoothness {labels['smoothness']} != {expected_smoothness}"


def test_backward_compatibility():
    """Test that SimpleCNN works in single-head mode for backward compatibility."""
    # Test with old-style output_size parameter
    model_old_style = SimpleCNN(output_size=10)
    assert not model_old_style.is_multihead
    assert hasattr(model_old_style, 'classifier')
    assert not hasattr(model_old_style, 'heads')
    
    # Test with new-style single head
    model_new_style = SimpleCNN(heads_config={'digit': 10})
    assert not model_new_style.is_multihead
    assert hasattr(model_new_style, 'classifier')
    
    # Test forward pass compatibility
    x = torch.randn(2, 1, 28, 28)
    output_old = model_old_style(x)
    output_new = model_new_style(x)
    
    assert isinstance(output_old, torch.Tensor)  # Single head returns tensor
    assert isinstance(output_new, torch.Tensor)  # Single head returns tensor
    assert output_old.shape == (2, 10)
    assert output_new.shape == (2, 10)


def test_lightning_module_backward_compatibility():
    """Test that MNISTLitModule works with single criterion (backward compatibility)."""
    net = SimpleCNN(output_size=10)
    criterion = CrossEntropyLoss()
    
    # Test old-style initialization
    module = MNISTLitModule(
        net=net,
        optimizer=torch.optim.Adam,
        scheduler=None,
        criterion=criterion
    )
    
    assert not module.is_multihead
    assert 'digit' in module.criteria
    assert len(module.criteria) == 1
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)
    y = torch.randint(0, 10, (batch_size,))
    batch = (x, y)
    
    loss, preds_dict, targets_dict = module.model_step(batch)
    
    assert isinstance(loss, torch.Tensor)
    assert isinstance(preds_dict, dict)
    assert isinstance(targets_dict, dict)
    assert 'digit' in preds_dict
    assert 'digit' in targets_dict
    assert preds_dict['digit'].shape == (batch_size,)
    assert targets_dict['digit'].shape == (batch_size,)


def test_lightning_module_multihead():
    """Test that MNISTLitModule works with multiple criteria (multihead mode)."""
    net = SimpleCNN(heads_config={'digit': 10, 'thickness': 5, 'smoothness': 3})
    criteria = {
        'digit': CrossEntropyLoss(),
        'thickness': CrossEntropyLoss(),
        'smoothness': CrossEntropyLoss()
    }
    loss_weights = {'digit': 1.0, 'thickness': 0.5, 'smoothness': 0.5}
    
    module = MNISTLitModule(
        net=net,
        optimizer=torch.optim.Adam,
        scheduler=None,
        criteria=criteria,
        loss_weights=loss_weights
    )
    
    assert module.is_multihead
    assert len(module.criteria) == 3
    
    # Test forward pass with multihead labels
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)
    y = {
        'digit': torch.randint(0, 10, (batch_size,)),
        'thickness': torch.randint(0, 5, (batch_size,)),
        'smoothness': torch.randint(0, 3, (batch_size,))
    }
    batch = (x, y)
    
    loss, preds_dict, targets_dict = module.model_step(batch)
    
    assert isinstance(loss, torch.Tensor)
    assert isinstance(preds_dict, dict)
    assert isinstance(targets_dict, dict)
    
    for head_name in ['digit', 'thickness', 'smoothness']:
        assert head_name in preds_dict
        assert head_name in targets_dict
        assert preds_dict[head_name].shape == (batch_size,)
        assert targets_dict[head_name].shape == (batch_size,)


def test_multihead_collate_function():
    """Test the multihead collate function."""
    from src.data.mnist_datamodule import MNISTDataModule
    
    # Create sample batch data
    batch = [
        (torch.randn(1, 28, 28), {'digit': 5, 'thickness': 2, 'smoothness': 1}),
        (torch.randn(1, 28, 28), {'digit': 3, 'thickness': 3, 'smoothness': 2}),
        (torch.randn(1, 28, 28), {'digit': 8, 'thickness': 4, 'smoothness': 2}),
    ]
    
    images, labels_dict = MNISTDataModule._multihead_collate_fn(batch)
    
    assert images.shape == (3, 1, 28, 28)
    assert isinstance(labels_dict, dict)
    assert 'digit' in labels_dict
    assert 'thickness' in labels_dict
    assert 'smoothness' in labels_dict
    
    assert torch.equal(labels_dict['digit'], torch.tensor([5, 3, 8]))
    assert torch.equal(labels_dict['thickness'], torch.tensor([2, 3, 4]))
    assert torch.equal(labels_dict['smoothness'], torch.tensor([1, 2, 2]))


def test_digit_mappings_exhaustive():
    """Test all digit mappings comprehensively."""
    dataset = MultiheadMNISTDataset(None)  # We only need the methods
    
    # Expected mappings based on the plan
    expected_mappings = {
        0: {'thickness': 0, 'smoothness': 2},  # Even, curved
        1: {'thickness': 2, 'smoothness': 0},  # Odd, angular
        2: {'thickness': 1, 'smoothness': 1},  # Even, mixed
        3: {'thickness': 3, 'smoothness': 2},  # Odd, curved
        4: {'thickness': 2, 'smoothness': 0},  # Even, angular
        5: {'thickness': 4, 'smoothness': 1},  # Odd, mixed
        6: {'thickness': 3, 'smoothness': 2},  # Even, curved
        7: {'thickness': 4, 'smoothness': 0},  # Odd, angular
        8: {'thickness': 4, 'smoothness': 2},  # Even, curved
        9: {'thickness': 4, 'smoothness': 2},  # Odd, curved
    }
    
    for digit, expected in expected_mappings.items():
        thickness = dataset._get_thickness(digit)
        smoothness = dataset._get_smoothness(digit)
        
        assert thickness == expected['thickness'], f"Digit {digit}: thickness {thickness} != {expected['thickness']}"
        assert smoothness == expected['smoothness'], f"Digit {digit}: smoothness {smoothness} != {expected['smoothness']}"
        
        # Verify ranges
        assert 0 <= thickness <= 4, f"Thickness {thickness} out of range [0, 4]"
        assert 0 <= smoothness <= 2, f"Smoothness {smoothness} out of range [0, 2]"


if __name__ == "__main__":
    pytest.main([__file__])