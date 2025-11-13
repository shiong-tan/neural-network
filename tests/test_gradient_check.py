"""
Tests for gradient checking functionality.

Tests verify:
1. Numerical gradient computation correctness
2. Gradient checking validation
3. Agreement between analytical and numerical gradients
4. Edge cases (zero gradients, different activations)
"""

import numpy as np
import pytest
from src.models import OneHiddenLayerMLP
from src.gradient_check import (
    compute_numerical_gradient,
    check_gradient,
    check_all_gradients,
    relative_error
)


class TestNumericalGradient:
    """Test numerical gradient computation."""

    def test_numerical_gradient_shape(self):
        """Numerical gradient should have same shape as parameter."""
        model = OneHiddenLayerMLP(input_dim=3, hidden_dim=5)
        x = np.random.randn(3)
        y = np.random.randn()

        # Check W1 gradient
        grad_W1 = compute_numerical_gradient(model, x, y, 'W1')
        assert grad_W1.shape == (5, 3)

        # Check b1 gradient
        grad_b1 = compute_numerical_gradient(model, x, y, 'b1')
        assert grad_b1.shape == (5,)

        # Check W2 gradient
        grad_W2 = compute_numerical_gradient(model, x, y, 'W2')
        assert grad_W2.shape == (1, 5)

        # Check b2 gradient
        grad_b2 = compute_numerical_gradient(model, x, y, 'b2')
        assert grad_b2.shape == (1,)

    def test_numerical_gradient_finite(self):
        """Numerical gradients should be finite."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        x = np.random.randn(2)
        y = np.random.randn()

        for param_name in ['W1', 'b1', 'W2', 'b2']:
            grad = compute_numerical_gradient(model, x, y, param_name)
            assert np.all(np.isfinite(grad))

    def test_numerical_gradient_deterministic(self):
        """Same inputs should give same numerical gradient."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        x = np.array([1.0, -1.0])
        y = 1.0

        grad1 = compute_numerical_gradient(model, x, y, 'W1')
        grad2 = compute_numerical_gradient(model, x, y, 'W1')

        np.testing.assert_allclose(grad1, grad2, rtol=1e-10)


class TestGradientChecking:
    """Test gradient checking validation."""

    def test_gradient_check_passes_random(self):
        """Gradient check should pass for random parameters."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        x = np.random.randn(2)
        y = np.random.randn()

        # Check all parameters
        for param_name in ['W1', 'b1', 'W2', 'b2']:
            passed, error, grad_anal, grad_num = check_gradient(
                model, x, y, param_name, threshold=1e-7
            )
            assert passed, f"Gradient check failed for {param_name} with error {error}"
            assert error < 1e-7

    def test_gradient_check_appendix_a(self):
        """Gradient check should pass for Appendix A example."""
        # Set up Appendix A example
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=2, activation='relu')
        model.set_parameters({
            'W1': np.array([[1.0, -2.0], [0.5, 1.0]]),
            'b1': np.array([0.0, 0.0]),
            'W2': np.array([[1.0, -1.0]]),
            'b2': np.array([0.0])
        })

        x = np.array([1.0, -1.0])
        y = 2.0

        # Check all parameters
        results = check_all_gradients(model, x, y, threshold=1e-7)

        for param_name, result in results.items():
            assert result['passed'], (
                f"Gradient check failed for {param_name} in Appendix A example "
                f"with error {result['error']}"
            )

    def test_gradient_check_with_sigmoid(self):
        """Gradient check should pass with sigmoid activation."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3, activation='sigmoid')
        x = np.random.randn(2)
        y = np.random.randn()

        results = check_all_gradients(model, x, y, threshold=1e-7)

        for param_name, result in results.items():
            assert result['passed'], (
                f"Gradient check failed for {param_name} with sigmoid "
                f"(error {result['error']})"
            )

    def test_gradient_check_zero_gradients(self):
        """Gradient check should handle zero gradients correctly."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        x = np.array([1.0, -1.0])

        # Make prediction equal target
        f = model.forward(x)
        y = f  # Target equals prediction → zero gradients

        results = check_all_gradients(model, x, y, threshold=1e-7)

        # All gradients should be near zero
        for param_name, result in results.items():
            assert result['passed']
            assert np.allclose(result['analytical'], 0.0, atol=1e-10)
            assert np.allclose(result['numerical'], 0.0, atol=1e-10)

    def test_gradient_check_returns_correct_structure(self):
        """check_gradient should return correct structure."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        x = np.random.randn(2)
        y = np.random.randn()

        passed, error, grad_analytical, grad_numerical = check_gradient(
            model, x, y, 'W1'
        )

        # Check types
        assert isinstance(passed, (bool, np.bool_))
        assert isinstance(error, (float, np.floating))
        assert isinstance(grad_analytical, np.ndarray)
        assert isinstance(grad_numerical, np.ndarray)

        # Check shapes match
        assert grad_analytical.shape == grad_numerical.shape


class TestCheckAllGradients:
    """Test batch gradient checking."""

    def test_check_all_gradients_returns_dict(self):
        """check_all_gradients should return dictionary with all parameters."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        x = np.random.randn(2)
        y = np.random.randn()

        results = check_all_gradients(model, x, y)

        assert isinstance(results, dict)
        assert set(results.keys()) == {'W1', 'b1', 'W2', 'b2'}

    def test_check_all_gradients_structure(self):
        """Each result should have correct structure."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        x = np.random.randn(2)
        y = np.random.randn()

        results = check_all_gradients(model, x, y)

        for param_name, result in results.items():
            assert 'passed' in result
            assert 'error' in result
            assert 'analytical' in result
            assert 'numerical' in result
            assert isinstance(result['passed'], (bool, np.bool_))
            assert isinstance(result['error'], (float, np.floating))

    def test_check_all_gradients_passes(self):
        """All gradients should pass for well-behaved model."""
        model = OneHiddenLayerMLP(input_dim=3, hidden_dim=4)
        x = np.random.randn(3)
        y = np.random.randn()

        results = check_all_gradients(model, x, y, threshold=1e-7)

        # All should pass
        for param_name, result in results.items():
            assert result['passed'], (
                f"Gradient check failed for {param_name} with error {result['error']}"
            )


class TestRelativeError:
    """Test relative error metric."""

    def test_relative_error_identical_arrays(self):
        """Relative error should be zero for identical arrays."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        error = relative_error(a, b)
        assert np.isclose(error, 0.0)

    def test_relative_error_zero_arrays(self):
        """Relative error should handle zero arrays."""
        a = np.zeros(5)
        b = np.zeros(5)
        error = relative_error(a, b)
        assert np.isclose(error, 0.0)

    def test_relative_error_symmetric(self):
        """Relative error should be symmetric."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.1, 2.1, 3.1])
        error_ab = relative_error(a, b)
        error_ba = relative_error(b, a)
        assert np.isclose(error_ab, error_ba)

    def test_relative_error_scale_invariant(self):
        """Relative error should be scale-invariant."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.1, 2.1, 3.1])

        # Scale both by same factor
        scale = 100.0
        error_original = relative_error(a, b)
        error_scaled = relative_error(a * scale, b * scale)

        assert np.isclose(error_original, error_scaled)

    def test_relative_error_small_difference(self):
        """Relative error should be small for small differences."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0001, 2.0001, 3.0001])
        error = relative_error(a, b)
        assert error < 1e-3


class TestGradientCheckWithDifferentSizes:
    """Test gradient checking with different model sizes."""

    def test_small_model(self):
        """Gradient check should work for small models."""
        model = OneHiddenLayerMLP(input_dim=1, hidden_dim=1)
        x = np.random.randn(1)
        y = np.random.randn()

        results = check_all_gradients(model, x, y, threshold=1e-7)
        assert all(r['passed'] for r in results.values())

    def test_large_model(self):
        """Gradient check should work for larger models."""
        model = OneHiddenLayerMLP(input_dim=10, hidden_dim=20)
        x = np.random.randn(10)
        y = np.random.randn()

        results = check_all_gradients(model, x, y, threshold=1e-7)
        assert all(r['passed'] for r in results.values())


class TestNumericalStability:
    """Test numerical stability of gradient checking."""

    def test_large_parameters(self):
        """Gradient check should handle large parameter values."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)

        # Set large parameters
        model.set_parameters({
            'W1': np.random.randn(3, 2) * 10,
            'b1': np.random.randn(3) * 10,
            'W2': np.random.randn(1, 3) * 10,
            'b2': np.random.randn(1) * 10
        })

        x = np.random.randn(2)
        y = np.random.randn()

        results = check_all_gradients(model, x, y, threshold=1e-6)
        assert all(r['passed'] for r in results.values())

    def test_small_parameters(self):
        """Gradient check should handle small parameter values."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)

        # Set small parameters
        model.set_parameters({
            'W1': np.random.randn(3, 2) * 0.01,
            'b1': np.random.randn(3) * 0.01,
            'W2': np.random.randn(1, 3) * 0.01,
            'b2': np.random.randn(1) * 0.01
        })

        x = np.random.randn(2)
        y = np.random.randn()

        results = check_all_gradients(model, x, y, threshold=1e-7)
        assert all(r['passed'] for r in results.values())

    def test_extreme_inputs(self):
        """Gradient check should handle extreme input values."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3, activation='sigmoid')

        # Extreme inputs that might cause sigmoid saturation
        x = np.array([10.0, -10.0])
        y = 0.5

        results = check_all_gradients(model, x, y, threshold=1e-6)

        # Gradients might be small due to saturation, but should still match
        for param_name, result in results.items():
            # Use looser threshold for saturated regions
            assert result['error'] < 1e-5, (
                f"Gradient check failed for {param_name} with extreme inputs "
                f"(error {result['error']})"
            )
