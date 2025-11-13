"""
Tests for PyTorch model implementations.

Tests verify:
1. Each of the 4 PyTorch implementations
2. Equivalence with NumPy implementation
3. Training functionality
4. Parameter conversion between NumPy and PyTorch
"""

import numpy as np
import torch
import pytest
from src.models import OneHiddenLayerMLP
from src.pytorch_models import (
    ManualTensorMLP,
    AutogradMLP,
    ModuleMLP,
    SequentialMLP,
    train_pytorch_model,
    convert_numpy_to_pytorch
)
from src.data import generate_xor_data


class TestManualTensorMLP:
    """Test manual tensor implementation."""

    def test_initialization(self):
        """Model should initialize with correct dimensions."""
        model = ManualTensorMLP(input_dim=3, hidden_dim=5)
        assert model.input_dim == 3
        assert model.hidden_dim == 5
        assert model.W1.shape == (5, 3)
        assert model.b1.shape == (5,)
        assert model.W2.shape == (1, 5)
        assert model.b2.shape == (1,)

    def test_parameters_require_grad(self):
        """Parameters should have gradients enabled."""
        model = ManualTensorMLP(input_dim=2, hidden_dim=3)
        assert model.W1.requires_grad
        assert model.b1.requires_grad
        assert model.W2.requires_grad
        assert model.b2.requires_grad

    def test_forward_single_sample(self):
        """Forward pass should work for single samples."""
        model = ManualTensorMLP(input_dim=2, hidden_dim=3)
        x = torch.tensor([1.0, -1.0])
        f = model.forward(x)
        assert isinstance(f, torch.Tensor)
        assert f.shape == torch.Size([])  # Scalar

    def test_forward_batch(self):
        """Forward pass should work for batches."""
        model = ManualTensorMLP(input_dim=2, hidden_dim=3)
        X = torch.randn(10, 2)
        predictions = model.forward(X)
        assert predictions.shape == (10,)

    def test_parameters_method(self):
        """parameters() should return list of parameters."""
        model = ManualTensorMLP(input_dim=2, hidden_dim=3)
        params = model.parameters()
        assert len(params) == 4
        assert all(isinstance(p, torch.Tensor) for p in params)


class TestAutogradMLP:
    """Test autograd implementation."""

    def test_initialization(self):
        """Model should initialize with nn.Parameter."""
        model = AutogradMLP(input_dim=3, hidden_dim=5)
        assert model.input_dim == 3
        assert model.hidden_dim == 5
        assert isinstance(model.W1, torch.nn.Parameter)
        assert isinstance(model.b1, torch.nn.Parameter)

    def test_forward_single_sample(self):
        """Forward pass should work for single samples."""
        model = AutogradMLP(input_dim=2, hidden_dim=3)
        x = torch.tensor([1.0, -1.0])
        f = model.forward(x)
        assert isinstance(f, torch.Tensor)

    def test_forward_batch(self):
        """Forward pass should work for batches."""
        model = AutogradMLP(input_dim=2, hidden_dim=3)
        X = torch.randn(10, 2)
        predictions = model.forward(X)
        assert predictions.shape == (10,)

    def test_backward_with_autograd(self):
        """Autograd should compute gradients automatically."""
        model = AutogradMLP(input_dim=2, hidden_dim=3)
        x = torch.tensor([1.0, -1.0])
        y = torch.tensor(1.0)

        # Forward pass
        f = model.forward(x)
        loss = 0.5 * (f - y) ** 2

        # Backward pass
        loss.backward()

        # Check gradients exist
        assert model.W1.grad is not None
        assert model.b1.grad is not None
        assert model.W2.grad is not None
        assert model.b2.grad is not None


class TestModuleMLP:
    """Test nn.Module implementation."""

    def test_is_module(self):
        """Model should be a proper nn.Module."""
        model = ModuleMLP(input_dim=2, hidden_dim=3)
        assert isinstance(model, torch.nn.Module)

    def test_has_layers(self):
        """Model should have nn.Linear layers."""
        model = ModuleMLP(input_dim=2, hidden_dim=3)
        assert hasattr(model, 'layer1')
        assert hasattr(model, 'layer2')
        assert isinstance(model.layer1, torch.nn.Linear)
        assert isinstance(model.layer2, torch.nn.Linear)

    def test_layer_dimensions(self):
        """Layers should have correct dimensions."""
        model = ModuleMLP(input_dim=3, hidden_dim=5)
        assert model.layer1.in_features == 3
        assert model.layer1.out_features == 5
        assert model.layer2.in_features == 5
        assert model.layer2.out_features == 1

    def test_forward_single_sample(self):
        """Forward pass should work for single samples."""
        model = ModuleMLP(input_dim=2, hidden_dim=3)
        x = torch.tensor([1.0, -1.0])
        f = model.forward(x)
        assert isinstance(f, torch.Tensor)

    def test_forward_batch(self):
        """Forward pass should work for batches."""
        model = ModuleMLP(input_dim=2, hidden_dim=3)
        X = torch.randn(10, 2)
        predictions = model.forward(X)
        assert predictions.shape == (10,)

    def test_parameters_accessible(self):
        """Should be able to access parameters via nn.Module interface."""
        model = ModuleMLP(input_dim=2, hidden_dim=3)
        params = list(model.parameters())
        assert len(params) == 4  # W1, b1, W2, b2


class TestSequentialMLP:
    """Test nn.Sequential implementation."""

    def test_is_module(self):
        """Model should be a proper nn.Module."""
        model = SequentialMLP(input_dim=2, hidden_dim=3)
        assert isinstance(model, torch.nn.Module)

    def test_has_sequential(self):
        """Model should have nn.Sequential container."""
        model = SequentialMLP(input_dim=2, hidden_dim=3)
        assert hasattr(model, 'model')
        assert isinstance(model.model, torch.nn.Sequential)

    def test_sequential_structure(self):
        """Sequential should have correct layer structure."""
        model = SequentialMLP(input_dim=2, hidden_dim=3, activation='relu')
        assert len(model.model) == 4  # Linear, ReLU, Linear, Flatten
        assert isinstance(model.model[0], torch.nn.Linear)
        assert isinstance(model.model[1], torch.nn.ReLU)
        assert isinstance(model.model[2], torch.nn.Linear)

    def test_forward_single_sample(self):
        """Forward pass should work for single samples."""
        model = SequentialMLP(input_dim=2, hidden_dim=3)
        x = torch.tensor([1.0, -1.0])
        f = model.forward(x)
        assert isinstance(f, torch.Tensor)

    def test_forward_batch(self):
        """Forward pass should work for batches."""
        model = SequentialMLP(input_dim=2, hidden_dim=3)
        X = torch.randn(10, 2)
        predictions = model.forward(X)
        assert predictions.shape == (10,)


class TestEquivalenceWithNumPy:
    """Test equivalence between PyTorch and NumPy implementations."""

    def test_manual_tensor_matches_numpy(self):
        """ManualTensorMLP should give same results as NumPy version."""
        # Create NumPy model
        np_model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3, activation='relu')

        # Create PyTorch model
        pt_model = ManualTensorMLP(input_dim=2, hidden_dim=3, activation='relu')

        # Copy parameters
        convert_numpy_to_pytorch(np_model, pt_model)

        # Test same input
        x_np = np.array([1.0, -1.0])
        x_pt = torch.from_numpy(x_np.astype(np.float32))

        # Forward pass
        f_np = np_model.forward(x_np)
        f_pt = pt_model.forward(x_pt).item()

        # Should match
        assert np.isclose(f_np, f_pt, rtol=1e-5)

    def test_module_matches_numpy(self):
        """ModuleMLP should give same results as NumPy version."""
        # Create NumPy model
        np_model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3, activation='relu')

        # Create PyTorch model
        pt_model = ModuleMLP(input_dim=2, hidden_dim=3, activation='relu')

        # Copy parameters
        convert_numpy_to_pytorch(np_model, pt_model)

        # Test same input
        x_np = np.array([1.0, -1.0])
        x_pt = torch.from_numpy(x_np.astype(np.float32))

        # Forward pass
        f_np = np_model.forward(x_np)
        f_pt = pt_model.forward(x_pt).item()

        # Should match
        assert np.isclose(f_np, f_pt, rtol=1e-5)

    def test_sequential_matches_numpy(self):
        """SequentialMLP should give same results as NumPy version."""
        # Create NumPy model
        np_model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3, activation='relu')

        # Create PyTorch model
        pt_model = SequentialMLP(input_dim=2, hidden_dim=3, activation='relu')

        # Copy parameters
        convert_numpy_to_pytorch(np_model, pt_model)

        # Test same input
        x_np = np.array([1.0, -1.0])
        x_pt = torch.from_numpy(x_np.astype(np.float32))

        # Forward pass
        f_np = np_model.forward(x_np)
        f_pt = pt_model.forward(x_pt).item()

        # Should match
        assert np.isclose(f_np, f_pt, rtol=1e-5)

    def test_batch_equivalence(self):
        """Batch predictions should match between NumPy and PyTorch."""
        # Create models
        np_model = OneHiddenLayerMLP(input_dim=2, hidden_dim=4, activation='relu')
        pt_model = ModuleMLP(input_dim=2, hidden_dim=4, activation='relu')

        # Copy parameters
        convert_numpy_to_pytorch(np_model, pt_model)

        # Test batch
        X_np = np.random.randn(10, 2)
        X_pt = torch.from_numpy(X_np.astype(np.float32))

        # Forward pass
        preds_np = np_model.forward_batch(X_np)
        preds_pt = pt_model.forward(X_pt).detach().numpy()

        # Should match
        np.testing.assert_allclose(preds_np, preds_pt, rtol=1e-5)


class TestTrainPyTorchModel:
    """Test PyTorch training utility."""

    def test_train_returns_history(self):
        """train_pytorch_model should return training history."""
        model = ModuleMLP(input_dim=2, hidden_dim=4)
        X = torch.randn(50, 2)
        y = torch.randn(50)

        history = train_pytorch_model(
            model, X, y, n_epochs=10, learning_rate=0.01, verbose=False
        )

        assert 'train_loss' in history
        assert len(history['train_loss']) == 10

    def test_train_reduces_loss(self):
        """Training should reduce loss over epochs."""
        model = ModuleMLP(input_dim=2, hidden_dim=4)
        X = torch.randn(100, 2)
        y = torch.randn(100)

        history = train_pytorch_model(
            model, X, y, n_epochs=50, learning_rate=0.1, verbose=False
        )

        # Loss should decrease
        assert history['train_loss'][-1] < history['train_loss'][0]

    def test_train_with_different_models(self):
        """Training should work with all model types."""
        X = torch.randn(50, 2)
        y = torch.randn(50)

        models = [
            ManualTensorMLP(input_dim=2, hidden_dim=3),
            AutogradMLP(input_dim=2, hidden_dim=3),
            ModuleMLP(input_dim=2, hidden_dim=3),
            SequentialMLP(input_dim=2, hidden_dim=3)
        ]

        for model in models:
            history = train_pytorch_model(
                model, X, y, n_epochs=5, verbose=False
            )
            assert len(history['train_loss']) == 5

    def test_train_learns_xor(self):
        """PyTorch model should learn XOR pattern."""
        model = ModuleMLP(input_dim=2, hidden_dim=8, activation='relu')

        # Generate XOR data
        X_np, y_np = generate_xor_data(n_samples=200, noise=0.05)
        X = torch.from_numpy(X_np.astype(np.float32))
        y = torch.from_numpy(y_np.astype(np.float32))

        # Train
        history = train_pytorch_model(
            model, X, y, n_epochs=200, learning_rate=0.5, verbose=False
        )

        # Evaluate
        with torch.no_grad():
            predictions = model.forward(X).numpy()
            binary_preds = (predictions >= 0.5).astype(int)
            accuracy = np.mean(binary_preds == y_np)

        # Should achieve reasonable accuracy
        assert accuracy > 0.7, f"XOR accuracy too low: {accuracy}"


class TestConvertNumpyToPyTorch:
    """Test parameter conversion utility."""

    def test_convert_to_manual_tensor(self):
        """Should convert NumPy params to ManualTensorMLP."""
        np_model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        pt_model = ManualTensorMLP(input_dim=2, hidden_dim=3)

        convert_numpy_to_pytorch(np_model, pt_model)

        # Check parameters match
        np_params = np_model.get_parameters()
        assert torch.allclose(
            pt_model.W1,
            torch.from_numpy(np_params['W1'].astype(np.float32))
        )

    def test_convert_to_module(self):
        """Should convert NumPy params to ModuleMLP."""
        np_model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        pt_model = ModuleMLP(input_dim=2, hidden_dim=3)

        convert_numpy_to_pytorch(np_model, pt_model)

        # Check parameters match
        np_params = np_model.get_parameters()
        assert torch.allclose(
            pt_model.layer1.weight,
            torch.from_numpy(np_params['W1'].astype(np.float32))
        )

    def test_convert_preserves_predictions(self):
        """Converted model should give same predictions."""
        np_model = OneHiddenLayerMLP(input_dim=2, hidden_dim=4, activation='relu')
        pt_model = ModuleMLP(input_dim=2, hidden_dim=4, activation='relu')

        # Before conversion - different predictions
        x_np = np.array([1.0, -1.0])
        pred_before = pt_model.forward(torch.from_numpy(x_np.astype(np.float32))).item()

        # Convert
        convert_numpy_to_pytorch(np_model, pt_model)

        # After conversion - should match NumPy
        pred_after = pt_model.forward(torch.from_numpy(x_np.astype(np.float32))).item()
        pred_numpy = np_model.forward(x_np)

        assert np.isclose(pred_after, pred_numpy, rtol=1e-5)


class TestActivationFunctions:
    """Test different activation functions."""

    def test_relu_activation(self):
        """Models should support ReLU activation."""
        models = [
            ManualTensorMLP(input_dim=2, hidden_dim=3, activation='relu'),
            AutogradMLP(input_dim=2, hidden_dim=3, activation='relu'),
            ModuleMLP(input_dim=2, hidden_dim=3, activation='relu'),
            SequentialMLP(input_dim=2, hidden_dim=3, activation='relu')
        ]

        x = torch.tensor([1.0, -1.0])
        for model in models:
            f = model.forward(x)
            assert isinstance(f, torch.Tensor)

    def test_sigmoid_activation(self):
        """Models should support sigmoid activation."""
        models = [
            ManualTensorMLP(input_dim=2, hidden_dim=3, activation='sigmoid'),
            AutogradMLP(input_dim=2, hidden_dim=3, activation='sigmoid'),
            ModuleMLP(input_dim=2, hidden_dim=3, activation='sigmoid'),
            SequentialMLP(input_dim=2, hidden_dim=3, activation='sigmoid')
        ]

        x = torch.tensor([1.0, -1.0])
        for model in models:
            f = model.forward(x)
            assert isinstance(f, torch.Tensor)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_activation_raises(self):
        """Invalid activation should raise error."""
        model = ManualTensorMLP(input_dim=2, hidden_dim=3, activation='tanh')
        x = torch.tensor([1.0, -1.0])
        with pytest.raises(ValueError, match="Unknown activation"):
            model.forward(x)

    def test_single_hidden_unit(self):
        """Models should work with single hidden unit."""
        model = ModuleMLP(input_dim=2, hidden_dim=1)
        x = torch.tensor([1.0, -1.0])
        f = model.forward(x)
        assert isinstance(f, torch.Tensor)

    def test_large_batch(self):
        """Models should handle large batches."""
        model = ModuleMLP(input_dim=2, hidden_dim=4)
        X = torch.randn(1000, 2)
        predictions = model.forward(X)
        assert predictions.shape == (1000,)

    def test_gradient_flow(self):
        """Gradients should flow through all parameters."""
        model = ModuleMLP(input_dim=2, hidden_dim=3)
        x = torch.tensor([1.0, -1.0])
        y = torch.tensor(1.0)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Forward and backward
        f = model.forward(x)
        loss = 0.5 * (f - y) ** 2
        optimizer.zero_grad()
        loss.backward()

        # All parameters should have gradients
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad))
