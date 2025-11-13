"""
Tests for complete neural network models.

Tests verify:
1. Model initialization
2. Forward pass correctness (single and batch)
3. Backward pass correctness
4. Integration with worked example from Appendix A
"""

import numpy as np
import pytest
from src.models import OneHiddenLayerMLP


class TestModelInitialization:
    """Test model initialization."""

    def test_model_creation(self):
        """Model should initialize with correct dimensions."""
        model = OneHiddenLayerMLP(input_dim=3, hidden_dim=5)
        assert model.input_dim == 3
        assert model.hidden_dim == 5
        assert model.activation_name == 'relu'

    def test_model_with_sigmoid(self):
        """Model should support sigmoid activation."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=4, activation='sigmoid')
        assert model.activation_name == 'sigmoid'

    def test_invalid_activation(self):
        """Model should raise error for invalid activation."""
        with pytest.raises(ValueError, match="Unknown activation"):
            OneHiddenLayerMLP(input_dim=2, hidden_dim=4, activation='tanh')

    def test_layers_initialized(self):
        """Model should have properly initialized layers."""
        model = OneHiddenLayerMLP(input_dim=3, hidden_dim=5)
        # Check layer1
        assert model.layer1.W.shape == (5, 3)
        assert model.layer1.b.shape == (5,)
        # Check layer2
        assert model.layer2.W.shape == (1, 5)
        assert model.layer2.b.shape == (1,)

    def test_repr(self):
        """Model should have informative string representation."""
        model = OneHiddenLayerMLP(input_dim=3, hidden_dim=5, activation='relu')
        repr_str = repr(model)
        assert 'OneHiddenLayerMLP' in repr_str
        assert 'input_dim=3' in repr_str
        assert 'hidden_dim=5' in repr_str
        assert 'relu' in repr_str


class TestForwardPass:
    """Test forward pass functionality."""

    def test_forward_single_sample(self):
        """Forward pass should work for single samples."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        x = np.array([1.0, -1.0])
        f = model.forward(x)
        assert isinstance(f, float)

    def test_forward_caches_intermediates(self):
        """Forward pass should cache a1 and h1 for backward pass."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        x = np.array([1.0, -1.0])
        model.forward(x)
        assert model.a1_cache is not None
        assert model.h1_cache is not None
        assert model.a1_cache.shape == (3,)
        assert model.h1_cache.shape == (3,)

    def test_forward_batch(self):
        """Forward pass should work for batches."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        X = np.random.randn(10, 2)
        predictions = model.forward_batch(X)
        assert predictions.shape == (10,)

    def test_forward_deterministic(self):
        """Same input should give same output with fixed parameters."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        x = np.array([1.0, -1.0])
        f1 = model.forward(x)
        f2 = model.forward(x)
        assert np.isclose(f1, f2)


class TestWorkedExampleAppendixA:
    """
    Test the worked example from Appendix A of the specification.

    Setup:
        d=2, h=2 (input_dim=2, hidden_dim=2)
        x = [1, -1]ᵀ
        y = 2

        W₁ = [[1, -2], [0.5, 1]], b₁ = [0, 0]
        W₂ = [[1, -1]], b₂ = [0]

        Activation: ReLU

    Expected forward pass:
        a₁ = W₁x + b₁ = [1×1 + (-2)×(-1), 0.5×1 + 1×(-1)] = [3, -0.5]
        h₁ = ReLU(a₁) = [3, 0]
        f = W₂h₁ + b₂ = 1×3 + (-1)×0 + 0 = 3
        L = (1/2)(f - y)² = (1/2)(3 - 2)² = 0.5

    Expected backward pass:
        δf = f - y = 3 - 2 = 1

        ∂L/∂W₂ = δf · h₁ᵀ = 1 × [3, 0] = [3, 0]
        ∂L/∂b₂ = δf = 1

        δh₁ = W₂ᵀ · δf = [[1], [-1]] × 1 = [1, -1]
        δa₁ = δh₁ ⊙ ReLU'(a₁) = [1, -1] ⊙ [1, 0] = [1, 0]

        ∂L/∂W₁ = δa₁ · xᵀ = [[1], [0]] × [[1, -1]] = [[1, -1], [0, 0]]
        ∂L/∂b₁ = δa₁ = [1, 0]
    """

    def test_forward_pass_appendix_a(self):
        """Test forward pass matches Appendix A worked example."""
        # Create model
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=2, activation='relu')

        # Set parameters from Appendix A
        model.set_parameters({
            'W1': np.array([[1.0, -2.0], [0.5, 1.0]]),
            'b1': np.array([0.0, 0.0]),
            'W2': np.array([[1.0, -1.0]]),
            'b2': np.array([0.0])
        })

        # Input and target from Appendix A
        x = np.array([1.0, -1.0])
        y = 2.0

        # Forward pass
        f = model.forward(x)

        # Check intermediate values
        # a₁ should be [3, -0.5]
        np.testing.assert_allclose(model.a1_cache, [3.0, -0.5], rtol=1e-10)

        # h₁ should be [3, 0] after ReLU
        np.testing.assert_allclose(model.h1_cache, [3.0, 0.0], rtol=1e-10)

        # f should be 3
        assert np.isclose(f, 3.0, rtol=1e-10)

        # Loss should be 0.5
        L = 0.5 * (f - y) ** 2
        assert np.isclose(L, 0.5, rtol=1e-10)

    def test_backward_pass_appendix_a(self):
        """Test backward pass matches Appendix A worked example."""
        # Create model and set parameters
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=2, activation='relu')
        model.set_parameters({
            'W1': np.array([[1.0, -2.0], [0.5, 1.0]]),
            'b1': np.array([0.0, 0.0]),
            'W2': np.array([[1.0, -1.0]]),
            'b2': np.array([0.0])
        })

        # Input and target
        x = np.array([1.0, -1.0])
        y = 2.0

        # Forward pass (required before backward)
        f = model.forward(x)

        # Backward pass
        grads = model.backward(x, y, f)

        # Check gradients match Appendix A
        # ∂L/∂W₂ = [3, 0]
        expected_dL_dW2 = np.array([[3.0, 0.0]])
        np.testing.assert_allclose(grads['dL_dW2'], expected_dL_dW2, rtol=1e-10)

        # ∂L/∂b₂ = 1
        expected_dL_db2 = np.array([1.0])
        np.testing.assert_allclose(grads['dL_db2'], expected_dL_db2, rtol=1e-10)

        # ∂L/∂W₁ = [[1, -1], [0, 0]]
        expected_dL_dW1 = np.array([[1.0, -1.0], [0.0, 0.0]])
        np.testing.assert_allclose(grads['dL_dW1'], expected_dL_dW1, rtol=1e-10)

        # ∂L/∂b₁ = [1, 0]
        expected_dL_db1 = np.array([1.0, 0.0])
        np.testing.assert_allclose(grads['dL_db1'], expected_dL_db1, rtol=1e-10)


class TestBackwardPass:
    """Test backward pass functionality."""

    def test_backward_requires_forward(self):
        """Backward pass should require forward pass first."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        x = np.array([1.0, -1.0])
        y = 1.0
        f = 0.5

        with pytest.raises(ValueError, match="Must call forward"):
            model.backward(x, y, f)

    def test_backward_returns_dict(self):
        """Backward pass should return dictionary with all gradients."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        x = np.array([1.0, -1.0])
        y = 1.0

        f = model.forward(x)
        grads = model.backward(x, y, f)

        assert isinstance(grads, dict)
        assert 'dL_dW2' in grads
        assert 'dL_db2' in grads
        assert 'dL_dW1' in grads
        assert 'dL_db1' in grads

    def test_backward_gradient_shapes(self):
        """Backward pass gradients should have correct shapes."""
        model = OneHiddenLayerMLP(input_dim=3, hidden_dim=5)
        x = np.random.randn(3)
        y = np.random.randn()

        f = model.forward(x)
        grads = model.backward(x, y, f)

        # Check shapes
        assert grads['dL_dW2'].shape == (1, 5)  # (1, h)
        assert grads['dL_db2'].shape == (1,)     # (1,)
        assert grads['dL_dW1'].shape == (5, 3)  # (h, d)
        assert grads['dL_db1'].shape == (5,)     # (h,)

    def test_backward_zero_gradient_when_prediction_matches(self):
        """Gradients should be zero when prediction matches target."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        x = np.array([1.0, -1.0])

        f = model.forward(x)
        y = f  # Target equals prediction
        grads = model.backward(x, y, f)

        # All gradients should be zero (within numerical precision)
        assert np.allclose(grads['dL_dW2'], 0.0)
        assert np.allclose(grads['dL_db2'], 0.0)
        assert np.allclose(grads['dL_dW1'], 0.0)
        assert np.allclose(grads['dL_db1'], 0.0)


class TestParameterManagement:
    """Test parameter get/set functionality."""

    def test_get_parameters(self):
        """Should be able to get all parameters."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        params = model.get_parameters()

        assert 'W1' in params
        assert 'b1' in params
        assert 'W2' in params
        assert 'b2' in params

        assert params['W1'].shape == (3, 2)
        assert params['b1'].shape == (3,)
        assert params['W2'].shape == (1, 3)
        assert params['b2'].shape == (1,)

    def test_set_parameters(self):
        """Should be able to set parameters."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)

        new_W1 = np.ones((3, 2))
        new_b1 = np.zeros(3)
        new_W2 = np.ones((1, 3)) * 0.5
        new_b2 = np.array([0.1])

        model.set_parameters({
            'W1': new_W1,
            'b1': new_b1,
            'W2': new_W2,
            'b2': new_b2
        })

        params = model.get_parameters()
        np.testing.assert_array_equal(params['W1'], new_W1)
        np.testing.assert_array_equal(params['b1'], new_b1)
        np.testing.assert_array_equal(params['W2'], new_W2)
        np.testing.assert_array_equal(params['b2'], new_b2)

    def test_set_partial_parameters(self):
        """Should be able to set subset of parameters."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)

        original_W1 = model.layer1.W.copy()
        new_b1 = np.ones(3)

        model.set_parameters({'b1': new_b1})

        # b1 should be updated
        np.testing.assert_array_equal(model.layer1.b, new_b1)
        # W1 should be unchanged
        np.testing.assert_array_equal(model.layer1.W, original_W1)


class TestSigmoidActivation:
    """Test model with sigmoid activation."""

    def test_sigmoid_forward(self):
        """Model should work with sigmoid activation."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3, activation='sigmoid')
        x = np.array([1.0, -1.0])
        f = model.forward(x)
        assert isinstance(f, float)
        assert model.h1_cache is not None
        # Sigmoid outputs should be in (0, 1)
        assert np.all((model.h1_cache > 0) & (model.h1_cache < 1))

    def test_sigmoid_backward(self):
        """Backward pass should work with sigmoid."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3, activation='sigmoid')
        x = np.array([1.0, -1.0])
        y = 1.0

        f = model.forward(x)
        grads = model.backward(x, y, f)

        # Check all gradients are computed
        assert all(key in grads for key in ['dL_dW2', 'dL_db2', 'dL_dW1', 'dL_db1'])

        # Gradients should be finite
        for grad in grads.values():
            assert np.all(np.isfinite(grad))
