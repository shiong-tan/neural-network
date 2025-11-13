"""
Tests for activation functions and their derivatives.

Tests verify:
1. Correct outputs for known values
2. Shape preservation
3. Boundary conditions
4. Derivative correctness
"""

import numpy as np
import pytest
from src.activations import ReLU, Sigmoid


class TestReLU:
    """Test ReLU activation function."""

    def test_relu_negative(self):
        """ReLU of negative values should be 0."""
        relu = ReLU()
        assert relu.forward(-1.0) == 0.0
        assert relu.forward(-100.0) == 0.0

    def test_relu_positive(self):
        """ReLU of positive values should be unchanged."""
        relu = ReLU()
        assert relu.forward(1.0) == 1.0
        assert relu.forward(5.5) == 5.5

    def test_relu_zero(self):
        """ReLU of zero should be zero."""
        relu = ReLU()
        assert relu.forward(0.0) == 0.0

    def test_relu_array(self):
        """ReLU should work on arrays and preserve shape."""
        relu = ReLU()
        x = np.array([-2, -1, 0, 1, 2])
        expected = np.array([0, 0, 0, 1, 2])
        result = relu.forward(x)
        np.testing.assert_array_equal(result, expected)

    def test_relu_multidim(self):
        """ReLU should preserve multidimensional array shapes."""
        relu = ReLU()
        x = np.array([[-1, 2], [3, -4]])
        expected = np.array([[0, 2], [3, 0]])
        result = relu.forward(x)
        np.testing.assert_array_equal(result, expected)
        assert result.shape == x.shape

    def test_relu_derivative_negative(self):
        """ReLU derivative should be 0 for negative values."""
        relu = ReLU()
        assert relu.derivative(-1.0) == 0.0
        assert relu.derivative(-100.0) == 0.0

    def test_relu_derivative_positive(self):
        """ReLU derivative should be 1 for positive values."""
        relu = ReLU()
        assert relu.derivative(1.0) == 1.0
        assert relu.derivative(100.0) == 1.0

    def test_relu_derivative_zero(self):
        """ReLU derivative at 0 should be 0 (subgradient convention)."""
        relu = ReLU()
        assert relu.derivative(0.0) == 0.0

    def test_relu_derivative_array(self):
        """ReLU derivative should work on arrays."""
        relu = ReLU()
        x = np.array([-2, -1, 0, 1, 2])
        expected = np.array([0, 0, 0, 1, 1])
        result = relu.derivative(x)
        np.testing.assert_array_equal(result, expected)


class TestSigmoid:
    """Test sigmoid activation function."""

    def test_sigmoid_zero(self):
        """Sigmoid of 0 should be 0.5."""
        sigmoid = Sigmoid()
        result = sigmoid.forward(0.0)
        assert np.isclose(result, 0.5)

    def test_sigmoid_large_positive(self):
        """Sigmoid of large positive values should approach 1."""
        sigmoid = Sigmoid()
        result = sigmoid.forward(10.0)
        assert result > 0.99
        assert result < 1.0

    def test_sigmoid_large_negative(self):
        """Sigmoid of large negative values should approach 0."""
        sigmoid = Sigmoid()
        result = sigmoid.forward(-10.0)
        assert result < 0.01
        assert result > 0.0

    def test_sigmoid_symmetry(self):
        """Sigmoid should satisfy σ(-x) = 1 - σ(x)."""
        sigmoid = Sigmoid()
        x = 2.5
        sigma_x = sigmoid.forward(x)
        sigma_neg_x = sigmoid.forward(-x)
        assert np.isclose(sigma_x + sigma_neg_x, 1.0)

    def test_sigmoid_array(self):
        """Sigmoid should work on arrays and preserve shape."""
        sigmoid = Sigmoid()
        x = np.array([-2, -1, 0, 1, 2])
        result = sigmoid.forward(x)
        assert result.shape == x.shape
        # All values should be in (0, 1)
        assert np.all(result > 0)
        assert np.all(result < 1)
        # Middle value should be 0.5
        assert np.isclose(result[2], 0.5)

    def test_sigmoid_multidim(self):
        """Sigmoid should preserve multidimensional array shapes."""
        sigmoid = Sigmoid()
        x = np.array([[-1, 0], [1, 2]])
        result = sigmoid.forward(x)
        assert result.shape == x.shape
        assert np.all((result > 0) & (result < 1))

    def test_sigmoid_numerical_stability(self):
        """Sigmoid should handle very large values without overflow."""
        sigmoid = Sigmoid()
        # These should not cause overflow errors
        result_pos = sigmoid.forward(100.0)
        result_neg = sigmoid.forward(-100.0)
        assert np.isfinite(result_pos)
        assert np.isfinite(result_neg)
        assert result_pos > 0.999
        assert result_neg < 0.001

    def test_sigmoid_derivative_zero(self):
        """Sigmoid derivative at 0 should be 0.25."""
        sigmoid = Sigmoid()
        result = sigmoid.derivative(0.0)
        # σ'(0) = σ(0)(1 - σ(0)) = 0.5 * 0.5 = 0.25
        assert np.isclose(result, 0.25)

    def test_sigmoid_derivative_formula(self):
        """Test that σ'(x) = σ(x)(1 - σ(x))."""
        sigmoid = Sigmoid()
        x_values = np.array([-2, -1, 0, 1, 2])
        for x in x_values:
            sigma_x = sigmoid.forward(x)
            derivative = sigmoid.derivative(x)
            expected = sigma_x * (1 - sigma_x)
            assert np.isclose(derivative, expected)

    def test_sigmoid_derivative_array(self):
        """Sigmoid derivative should work on arrays."""
        sigmoid = Sigmoid()
        x = np.array([-1, 0, 1])
        result = sigmoid.derivative(x)
        assert result.shape == x.shape
        # All derivatives should be positive
        assert np.all(result > 0)
        # Maximum derivative is at x=0 with value 0.25
        assert np.max(result) <= 0.25

    def test_sigmoid_derivative_maximum(self):
        """Sigmoid derivative should be maximized at x=0."""
        sigmoid = Sigmoid()
        x_values = np.linspace(-5, 5, 100)
        derivatives = sigmoid.derivative(x_values)
        max_idx = np.argmax(derivatives)
        # Maximum should be near x=0
        assert np.isclose(x_values[max_idx], 0.0, atol=0.1)


class TestActivationInterface:
    """Test that activations follow the expected interface."""

    def test_relu_is_activation(self):
        """ReLU should inherit from Activation."""
        from src.activations import Activation
        relu = ReLU()
        assert isinstance(relu, Activation)

    def test_sigmoid_is_activation(self):
        """Sigmoid should inherit from Activation."""
        from src.activations import Activation
        sigmoid = Sigmoid()
        assert isinstance(sigmoid, Activation)

    def test_activations_have_forward(self):
        """All activations should have forward method."""
        relu = ReLU()
        sigmoid = Sigmoid()
        assert hasattr(relu, 'forward')
        assert hasattr(sigmoid, 'forward')
        assert callable(relu.forward)
        assert callable(sigmoid.forward)

    def test_activations_have_derivative(self):
        """All activations should have derivative method."""
        relu = ReLU()
        sigmoid = Sigmoid()
        assert hasattr(relu, 'derivative')
        assert hasattr(sigmoid, 'derivative')
        assert callable(relu.derivative)
        assert callable(sigmoid.derivative)
