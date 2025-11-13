"""
Tests for forward pass components: layers, loss, and data generation.

Tests verify:
1. Shape consistency through layers
2. Loss computation correctness
3. Data generation validity
"""

import numpy as np
import pytest
from src.layers import AffineLayer
from src.loss import squared_error_loss, squared_error_gradient, squared_error_loss_gradient_batch
from src.data import generate_xor_data, generate_spiral_data, create_batches


class TestAffineLayer:
    """Test affine layer forward pass."""

    def test_layer_initialization(self):
        """Layer should initialize with correct shapes."""
        layer = AffineLayer(input_dim=3, output_dim=2)
        assert layer.W.shape == (2, 3)
        assert layer.b.shape == (2,)
        assert layer.input_dim == 3
        assert layer.output_dim == 2

    def test_forward_single_sample(self):
        """Forward pass should work for single samples."""
        layer = AffineLayer(input_dim=3, output_dim=2)
        x = np.array([1.0, 2.0, 3.0])
        z = layer.forward(x)
        assert z.shape == (2,)
        assert isinstance(z, np.ndarray)

    def test_forward_batch(self):
        """Forward pass should work for batches."""
        layer = AffineLayer(input_dim=3, output_dim=2)
        X = np.random.randn(10, 3)
        Z = layer.forward(X)
        assert Z.shape == (10, 2)

    def test_forward_computation(self):
        """Forward pass should compute z = Wx + b correctly."""
        layer = AffineLayer(input_dim=2, output_dim=2)
        # Set known parameters
        layer.W = np.array([[1.0, 2.0], [3.0, 4.0]])
        layer.b = np.array([0.5, 1.0])

        # Test single sample
        x = np.array([1.0, -1.0])
        z = layer.forward(x)
        # z = [[1, 2], [3, 4]] @ [1, -1] + [0.5, 1.0]
        #   = [1-2, 3-4] + [0.5, 1.0] = [-1, -1] + [0.5, 1.0] = [-0.5, 0.0]
        expected = np.array([-0.5, 0.0])
        np.testing.assert_allclose(z, expected)

    def test_forward_batch_computation(self):
        """Forward pass should compute Z = XW^T + b for batches."""
        layer = AffineLayer(input_dim=2, output_dim=2)
        layer.W = np.array([[1.0, 2.0], [3.0, 4.0]])
        layer.b = np.array([0.5, 1.0])

        X = np.array([[1.0, -1.0], [2.0, 3.0]])
        Z = layer.forward(X)
        # Row 0: [1, 2] @ [1, -1] + 0.5 = -1 + 0.5 = -0.5
        #        [3, 4] @ [1, -1] + 1.0 = -1 + 1.0 = 0.0
        # Row 1: [1, 2] @ [2, 3] + 0.5 = 8 + 0.5 = 8.5
        #        [3, 4] @ [2, 3] + 1.0 = 18 + 1.0 = 19.0
        expected = np.array([[-0.5, 0.0], [8.5, 19.0]])
        np.testing.assert_allclose(Z, expected)

    def test_forward_caches_input(self):
        """Forward pass should cache input for backward pass."""
        layer = AffineLayer(input_dim=3, output_dim=2)
        x = np.array([1.0, 2.0, 3.0])
        layer.forward(x)
        assert layer.x_cache is not None
        np.testing.assert_array_equal(layer.x_cache, x)

    def test_weights_initialized_small(self):
        """Weights should be initialized with small values."""
        layer = AffineLayer(input_dim=100, output_dim=50)
        # Most weights should be between -0.5 and 0.5 (3 sigma for 0.1 std)
        assert np.abs(layer.W).mean() < 0.2

    def test_biases_initialized_zero(self):
        """Biases should be initialized to zero."""
        layer = AffineLayer(input_dim=5, output_dim=3)
        np.testing.assert_array_equal(layer.b, np.zeros(3))


class TestLossFunctions:
    """Test loss functions and gradients."""

    def test_squared_error_zero_loss(self):
        """Loss should be zero when predictions match targets."""
        predictions = np.array([1.0, 2.0, 3.0])
        targets = np.array([1.0, 2.0, 3.0])
        loss = squared_error_loss(predictions, targets)
        assert np.isclose(loss, 0.0)

    def test_squared_error_known_value(self):
        """Loss should match hand-calculated value."""
        predictions = np.array([1.0, 2.0, 3.0])
        targets = np.array([1.5, 2.5, 2.5])
        # Errors: [-0.5, -0.5, 0.5]
        # Squared: [0.25, 0.25, 0.25]
        # Sum: 0.75
        # Loss = 0.75 / (2 * 3) = 0.125
        loss = squared_error_loss(predictions, targets)
        assert np.isclose(loss, 0.125)

    def test_squared_error_shape_mismatch(self):
        """Loss should raise error for mismatched shapes."""
        predictions = np.array([1.0, 2.0])
        targets = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            squared_error_loss(predictions, targets)

    def test_squared_error_gradient_positive(self):
        """Gradient should be positive when prediction > target."""
        grad = squared_error_gradient(3.0, 2.0)
        assert grad == 1.0

    def test_squared_error_gradient_negative(self):
        """Gradient should be negative when prediction < target."""
        grad = squared_error_gradient(1.5, 2.0)
        assert grad == -0.5

    def test_squared_error_gradient_zero(self):
        """Gradient should be zero when prediction == target."""
        grad = squared_error_gradient(2.0, 2.0)
        assert grad == 0.0

    def test_squared_error_gradient_batch(self):
        """Batch gradient should compute element-wise differences."""
        predictions = np.array([1.0, 2.0, 3.0])
        targets = np.array([1.5, 2.5, 2.5])
        grads = squared_error_loss_gradient_batch(predictions, targets)
        expected = np.array([-0.5, -0.5, 0.5])
        np.testing.assert_allclose(grads, expected)

    def test_squared_error_gradient_batch_shape(self):
        """Batch gradient should preserve input shape."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0])
        grads = squared_error_loss_gradient_batch(predictions, targets)
        assert grads.shape == predictions.shape


class TestDataGeneration:
    """Test data generation functions."""

    def test_xor_data_shape(self):
        """XOR data should have correct shapes."""
        X, y = generate_xor_data(n_samples=200)
        assert X.shape == (200, 2)
        assert y.shape == (200,)

    def test_xor_data_labels(self):
        """XOR data should have binary labels."""
        X, y = generate_xor_data(n_samples=200)
        unique_labels = np.unique(y)
        assert len(unique_labels) == 2
        assert set(unique_labels) == {0, 1}

    def test_xor_data_quadrants(self):
        """XOR data should respect quadrant labeling."""
        X, y = generate_xor_data(n_samples=400, noise=0.0)
        # Check that points in opposite quadrants have XOR pattern
        for i in range(len(X)):
            x1, x2 = X[i]
            expected_label = 1 if x1 * x2 < 0 else 0
            assert y[i] == expected_label

    def test_xor_data_balanced(self):
        """XOR data should be roughly balanced between classes."""
        X, y = generate_xor_data(n_samples=400)
        counts = np.bincount(y)
        # Should be exactly balanced since we generate equal per quadrant
        assert abs(counts[0] - counts[1]) <= 20  # Allow small noise-induced imbalance

    def test_spiral_data_shape(self):
        """Spiral data should have correct shapes."""
        X, y = generate_spiral_data(n_samples=200, n_classes=2)
        assert X.shape == (200, 2)
        assert y.shape == (200,)

    def test_spiral_data_classes(self):
        """Spiral data should have correct number of classes."""
        X, y = generate_spiral_data(n_samples=300, n_classes=3)
        unique_labels = np.unique(y)
        assert len(unique_labels) == 3
        assert set(unique_labels) == {0, 1, 2}

    def test_spiral_data_balanced(self):
        """Spiral data should be balanced among classes."""
        X, y = generate_spiral_data(n_samples=300, n_classes=3)
        counts = np.bincount(y)
        assert len(counts) == 3
        assert all(count == 100 for count in counts)

    def test_create_batches_count(self):
        """Batching should create correct number of batches."""
        X = np.random.randn(100, 2)
        y = np.random.randint(0, 2, 100)
        batches = create_batches(X, y, batch_size=32, shuffle=False)
        # 100 samples, batch_size 32 → 4 batches (32, 32, 32, 4)
        assert len(batches) == 4

    def test_create_batches_shapes(self):
        """Batches should have correct shapes."""
        X = np.random.randn(100, 2)
        y = np.random.randint(0, 2, 100)
        batches = create_batches(X, y, batch_size=32, shuffle=False)
        # First three batches should be full
        for i in range(3):
            X_batch, y_batch = batches[i]
            assert X_batch.shape == (32, 2)
            assert y_batch.shape == (32,)
        # Last batch should have remainder
        X_batch, y_batch = batches[-1]
        assert X_batch.shape == (4, 2)
        assert y_batch.shape == (4,)

    def test_create_batches_no_shuffle(self):
        """Without shuffle, batches should maintain order."""
        X = np.arange(10).reshape(10, 1)
        y = np.arange(10)
        batches = create_batches(X, y, batch_size=3, shuffle=False)
        # First batch should have first 3 samples
        X_batch, y_batch = batches[0]
        np.testing.assert_array_equal(X_batch.ravel(), [0, 1, 2])
        np.testing.assert_array_equal(y_batch, [0, 1, 2])

    def test_create_batches_covers_all_data(self):
        """All samples should appear in exactly one batch."""
        X = np.random.randn(100, 2)
        y = np.arange(100)
        batches = create_batches(X, y, batch_size=32, shuffle=True)
        # Collect all y values from batches
        all_y = np.concatenate([y_batch for _, y_batch in batches])
        # Should have all original indices (possibly reordered)
        assert len(all_y) == 100
        assert set(all_y) == set(y)

    def test_create_batches_shape_mismatch(self):
        """Should raise error if X and y have different lengths."""
        X = np.random.randn(10, 2)
        y = np.random.randint(0, 2, 8)
        with pytest.raises(ValueError):
            create_batches(X, y, batch_size=3)
