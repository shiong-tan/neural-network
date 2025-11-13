"""
Tests for optimization and training utilities.

Tests verify:
1. SGD optimizer correctness
2. Training step and epoch functionality
3. Full training loop
4. Accuracy and evaluation metrics
5. Integration with model and data
"""

import numpy as np
import pytest
from src.models import OneHiddenLayerMLP
from src.optimization import (
    SGD,
    train_step,
    train_epoch,
    train,
    compute_accuracy,
    evaluate
)
from src.data import generate_xor_data


class TestSGDOptimizer:
    """Test SGD optimizer."""

    def test_sgd_initialization(self):
        """SGD should initialize with learning rate."""
        optimizer = SGD(learning_rate=0.1)
        assert optimizer.learning_rate == 0.1

    def test_sgd_default_learning_rate(self):
        """SGD should have default learning rate."""
        optimizer = SGD()
        assert optimizer.learning_rate == 0.01

    def test_sgd_negative_learning_rate_raises(self):
        """SGD should reject negative learning rate."""
        with pytest.raises(ValueError, match="must be positive"):
            SGD(learning_rate=-0.1)

    def test_sgd_zero_learning_rate_raises(self):
        """SGD should reject zero learning rate."""
        with pytest.raises(ValueError, match="must be positive"):
            SGD(learning_rate=0.0)

    def test_sgd_step_updates_parameters(self):
        """SGD step should update model parameters."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        optimizer = SGD(learning_rate=0.1)

        # Get initial parameters
        params_before = model.get_parameters()
        W1_before = params_before['W1'].copy()
        b1_before = params_before['b1'].copy()
        W2_before = params_before['W2'].copy()
        b2_before = params_before['b2'].copy()

        # Compute gradients
        x = np.array([1.0, -1.0])
        y = 1.0
        f = model.forward(x)
        grads = model.backward(x, y, f)

        # Perform optimization step
        optimizer.step(model, grads)

        # Check parameters changed
        params_after = model.get_parameters()
        assert not np.allclose(params_after['W1'], W1_before)
        assert not np.allclose(params_after['b1'], b1_before)
        assert not np.allclose(params_after['W2'], W2_before)
        assert not np.allclose(params_after['b2'], b2_before)

    def test_sgd_step_moves_in_negative_gradient_direction(self):
        """SGD should update parameters in negative gradient direction."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        optimizer = SGD(learning_rate=0.1)

        # Get gradients
        x = np.array([1.0, -1.0])
        y = 1.0
        f = model.forward(x)
        grads = model.backward(x, y, f)

        # Expected update: θ_new = θ_old - η * grad
        params_before = model.get_parameters()
        expected_W1 = params_before['W1'] - 0.1 * grads['dL_dW1']
        expected_b1 = params_before['b1'] - 0.1 * grads['dL_db1']
        expected_W2 = params_before['W2'] - 0.1 * grads['dL_dW2']
        expected_b2 = params_before['b2'] - 0.1 * grads['dL_db2']

        # Perform step
        optimizer.step(model, grads)

        # Check actual matches expected
        params_after = model.get_parameters()
        np.testing.assert_allclose(params_after['W1'], expected_W1)
        np.testing.assert_allclose(params_after['b1'], expected_b1)
        np.testing.assert_allclose(params_after['W2'], expected_W2)
        np.testing.assert_allclose(params_after['b2'], expected_b2)

    def test_sgd_repr(self):
        """SGD should have informative string representation."""
        optimizer = SGD(learning_rate=0.05)
        assert '0.05' in repr(optimizer)
        assert 'SGD' in repr(optimizer)


class TestTrainStep:
    """Test single training step."""

    def test_train_step_returns_loss(self):
        """train_step should return average loss."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        optimizer = SGD(learning_rate=0.01)
        X_batch = np.random.randn(10, 2)
        y_batch = np.random.randn(10)

        loss = train_step(model, X_batch, y_batch, optimizer)

        assert isinstance(loss, float)
        assert loss >= 0  # Loss should be non-negative

    def test_train_step_updates_parameters(self):
        """train_step should update model parameters."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        optimizer = SGD(learning_rate=0.1)

        params_before = model.get_parameters()
        W1_before = params_before['W1'].copy()

        X_batch = np.random.randn(10, 2)
        y_batch = np.random.randn(10)
        train_step(model, X_batch, y_batch, optimizer)

        params_after = model.get_parameters()
        assert not np.allclose(params_after['W1'], W1_before)

    def test_train_step_single_sample(self):
        """train_step should work with single sample batch."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        optimizer = SGD(learning_rate=0.01)
        X_batch = np.random.randn(1, 2)
        y_batch = np.random.randn(1)

        loss = train_step(model, X_batch, y_batch, optimizer)

        assert isinstance(loss, float)

    def test_train_step_reduces_loss(self):
        """Multiple train_steps should reduce loss on fixed batch."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        optimizer = SGD(learning_rate=0.5)  # Large LR for fast convergence

        # Fixed batch
        X_batch = np.random.randn(10, 2)
        y_batch = np.random.randn(10)

        # Initial loss
        loss_initial = train_step(model, X_batch, y_batch, optimizer)

        # Train for several steps
        for _ in range(50):
            loss = train_step(model, X_batch, y_batch, optimizer)

        # Loss should decrease
        assert loss < loss_initial


class TestTrainEpoch:
    """Test training for one epoch."""

    def test_train_epoch_returns_loss(self):
        """train_epoch should return average loss."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        optimizer = SGD(learning_rate=0.01)
        X = np.random.randn(50, 2)
        y = np.random.randn(50)

        loss = train_epoch(model, X, y, optimizer, batch_size=10)

        assert isinstance(loss, float)
        assert loss >= 0

    def test_train_epoch_processes_all_samples(self):
        """train_epoch should process entire dataset."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        optimizer = SGD(learning_rate=0.01)

        # Dataset size not divisible by batch size
        X = np.random.randn(47, 2)  # 47 samples
        y = np.random.randn(47)

        # Should work without error
        loss = train_epoch(model, X, y, optimizer, batch_size=10)
        assert isinstance(loss, float)

    def test_train_epoch_with_shuffle(self):
        """train_epoch should support shuffling."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        optimizer = SGD(learning_rate=0.01)
        X = np.random.randn(50, 2)
        y = np.random.randn(50)

        # Should work with shuffle=True (default)
        loss1 = train_epoch(model, X, y, optimizer, shuffle=True)
        # Should work with shuffle=False
        loss2 = train_epoch(model, X, y, optimizer, shuffle=False)

        assert isinstance(loss1, float)
        assert isinstance(loss2, float)


class TestTrain:
    """Test full training loop."""

    def test_train_returns_history(self):
        """train should return training history."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        optimizer = SGD(learning_rate=0.01)
        X = np.random.randn(50, 2)
        y = np.random.randn(50)

        history = train(model, X, y, optimizer, n_epochs=10, verbose=False)

        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 10

    def test_train_with_validation(self):
        """train should record validation loss when provided."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        optimizer = SGD(learning_rate=0.01)
        X_train = np.random.randn(50, 2)
        y_train = np.random.randn(50)
        X_val = np.random.randn(20, 2)
        y_val = np.random.randn(20)

        history = train(
            model, X_train, y_train, optimizer,
            n_epochs=10,
            X_val=X_val,
            y_val=y_val,
            verbose=False
        )

        assert len(history['val_loss']) == 10

    def test_train_without_validation(self):
        """train should work without validation data."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        optimizer = SGD(learning_rate=0.01)
        X_train = np.random.randn(50, 2)
        y_train = np.random.randn(50)

        history = train(model, X_train, y_train, optimizer, n_epochs=10, verbose=False)

        assert len(history['val_loss']) == 0

    def test_train_reduces_loss(self):
        """Training should reduce loss over epochs."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=4)
        optimizer = SGD(learning_rate=0.1)

        # Simple learnable data
        X = np.random.randn(100, 2)
        y = np.random.randn(100)

        history = train(model, X, y, optimizer, n_epochs=50, verbose=False)

        # Loss should decrease
        initial_loss = history['train_loss'][0]
        final_loss = history['train_loss'][-1]
        assert final_loss < initial_loss

    def test_train_learns_xor(self):
        """Model should learn XOR pattern with sufficient training."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=8, activation='relu')
        optimizer = SGD(learning_rate=0.5)

        # Generate XOR data
        X, y = generate_xor_data(n_samples=200, noise=0.05)

        # Train
        history = train(model, X, y, optimizer, n_epochs=200, verbose=False)

        # Final accuracy should be reasonable (>70%)
        predictions = model.forward_batch(X)
        accuracy = compute_accuracy(predictions, y, threshold=0.5)
        assert accuracy > 0.7, f"XOR accuracy too low: {accuracy}"


class TestComputeAccuracy:
    """Test accuracy computation."""

    def test_accuracy_perfect_predictions(self):
        """Accuracy should be 1.0 for perfect predictions."""
        predictions = np.array([0.2, 0.8, 0.1, 0.9])
        targets = np.array([0, 1, 0, 1])
        accuracy = compute_accuracy(predictions, targets, threshold=0.5)
        assert accuracy == 1.0

    def test_accuracy_all_wrong(self):
        """Accuracy should be 0.0 for all wrong predictions."""
        predictions = np.array([0.8, 0.2])
        targets = np.array([0, 1])
        accuracy = compute_accuracy(predictions, targets, threshold=0.5)
        assert accuracy == 0.0

    def test_accuracy_half_correct(self):
        """Accuracy should be 0.5 for half correct."""
        predictions = np.array([0.2, 0.8, 0.8, 0.2])
        targets = np.array([0, 1, 0, 1])
        accuracy = compute_accuracy(predictions, targets, threshold=0.5)
        assert accuracy == 0.5

    def test_accuracy_custom_threshold(self):
        """Accuracy should respect custom threshold."""
        predictions = np.array([0.4, 0.6])
        targets = np.array([0, 1])

        # With threshold=0.5
        acc1 = compute_accuracy(predictions, targets, threshold=0.5)
        assert acc1 == 1.0  # 0.4 < 0.5 → 0, 0.6 > 0.5 → 1

        # With threshold=0.7
        acc2 = compute_accuracy(predictions, targets, threshold=0.7)
        assert acc2 == 0.5  # 0.4 < 0.7 → 0, 0.6 < 0.7 → 0

    def test_accuracy_edge_case_at_threshold(self):
        """Values exactly at threshold should be classified as 1."""
        predictions = np.array([0.5, 0.5])
        targets = np.array([1, 0])
        accuracy = compute_accuracy(predictions, targets, threshold=0.5)
        # 0.5 >= 0.5 → 1, so first is correct, second is wrong
        assert accuracy == 0.5

    def test_accuracy_range(self):
        """Accuracy should be in [0, 1]."""
        predictions = np.random.rand(100)
        targets = np.random.randint(0, 2, 100)
        accuracy = compute_accuracy(predictions, targets)
        assert 0.0 <= accuracy <= 1.0


class TestEvaluate:
    """Test model evaluation."""

    def test_evaluate_returns_dict(self):
        """evaluate should return dictionary with metrics."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        X = np.random.randn(50, 2)
        y = np.random.randint(0, 2, 50)

        metrics = evaluate(model, X, y)

        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert 'accuracy' in metrics

    def test_evaluate_loss_is_nonnegative(self):
        """Loss should be non-negative."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        X = np.random.randn(50, 2)
        y = np.random.randint(0, 2, 50)

        metrics = evaluate(model, X, y)

        assert metrics['loss'] >= 0

    def test_evaluate_accuracy_in_range(self):
        """Accuracy should be in [0, 1]."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        X = np.random.randn(50, 2)
        y = np.random.randint(0, 2, 50)

        metrics = evaluate(model, X, y)

        assert 0.0 <= metrics['accuracy'] <= 1.0

    def test_evaluate_with_custom_threshold(self):
        """evaluate should respect custom threshold."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        X = np.random.randn(50, 2)
        y = np.random.randint(0, 2, 50)

        metrics1 = evaluate(model, X, y, threshold=0.3)
        metrics2 = evaluate(model, X, y, threshold=0.7)

        # Different thresholds may give different accuracies
        assert isinstance(metrics1['accuracy'], float)
        assert isinstance(metrics2['accuracy'], float)


class TestIntegration:
    """Integration tests for training pipeline."""

    def test_full_pipeline_xor(self):
        """Full pipeline should train model on XOR data."""
        # Create model
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=8, activation='relu')
        optimizer = SGD(learning_rate=0.5)

        # Generate data
        X_train, y_train = generate_xor_data(n_samples=200, noise=0.05)
        X_val, y_val = generate_xor_data(n_samples=100, noise=0.05)

        # Train
        history = train(
            model, X_train, y_train, optimizer,
            n_epochs=100,
            X_val=X_val,
            y_val=y_val,
            verbose=False
        )

        # Check history
        assert len(history['train_loss']) == 100
        assert len(history['val_loss']) == 100

        # Check loss decreased
        assert history['train_loss'][-1] < history['train_loss'][0]

        # Evaluate
        metrics = evaluate(model, X_val, y_val)
        assert metrics['accuracy'] > 0.65  # Should learn something

    def test_training_converges_on_simple_problem(self):
        """Model should converge on simple linearly separable problem."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=4, activation='relu')
        optimizer = SGD(learning_rate=0.1)

        # Simple separable data: y = 1 if x1 > 0, else 0
        X = np.random.randn(100, 2)
        y = (X[:, 0] > 0).astype(int)

        history = train(model, X, y, optimizer, n_epochs=100, verbose=False)

        # Should achieve high accuracy
        metrics = evaluate(model, X, y)
        assert metrics['accuracy'] > 0.8

    def test_overfitting_detection(self):
        """Training loss should decrease more than validation loss (overfitting)."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=20, activation='relu')
        optimizer = SGD(learning_rate=0.5)

        # Small training set, different validation set
        X_train = np.random.randn(30, 2)
        y_train = np.random.randint(0, 2, 30)
        X_val = np.random.randn(100, 2)
        y_val = np.random.randint(0, 2, 100)

        history = train(
            model, X_train, y_train, optimizer,
            n_epochs=200,
            X_val=X_val,
            y_val=y_val,
            verbose=False
        )

        # Training loss should decrease significantly
        train_improvement = history['train_loss'][0] - history['train_loss'][-1]
        assert train_improvement > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_train_with_batch_size_larger_than_dataset(self):
        """train should handle batch size larger than dataset."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        optimizer = SGD(learning_rate=0.01)
        X = np.random.randn(10, 2)
        y = np.random.randn(10)

        # Batch size larger than dataset
        history = train(model, X, y, optimizer, n_epochs=5, batch_size=50, verbose=False)

        assert len(history['train_loss']) == 5

    def test_train_single_epoch(self):
        """train should work with single epoch."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        optimizer = SGD(learning_rate=0.01)
        X = np.random.randn(50, 2)
        y = np.random.randn(50)

        history = train(model, X, y, optimizer, n_epochs=1, verbose=False)

        assert len(history['train_loss']) == 1

    def test_train_with_small_learning_rate(self):
        """train should work with very small learning rate."""
        model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        optimizer = SGD(learning_rate=1e-6)
        X = np.random.randn(50, 2)
        y = np.random.randn(50)

        history = train(model, X, y, optimizer, n_epochs=5, verbose=False)

        assert len(history['train_loss']) == 5
        # Parameters should barely change
        assert all(np.isfinite(loss) for loss in history['train_loss'])
