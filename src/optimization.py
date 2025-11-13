"""
Optimization algorithms and training utilities.

This module implements Stochastic Gradient Descent (SGD) and provides
utilities for training neural networks on datasets.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from src.models import OneHiddenLayerMLP
from src.loss import squared_error_loss
from src.data import create_batches


class SGD:
    """
    Stochastic Gradient Descent optimizer.

    Updates parameters using the rule:
        θ ← θ - η∇L(θ)

    where:
        θ: parameters
        η: learning rate
        ∇L(θ): gradient of loss w.r.t. parameters

    Attributes:
        learning_rate: Step size for parameter updates
    """

    def __init__(self, learning_rate: float = 0.01):
        """
        Initialize SGD optimizer.

        Args:
            learning_rate: Learning rate (default 0.01)

        Examples:
            >>> optimizer = SGD(learning_rate=0.1)
            >>> optimizer.learning_rate
            0.1
        """
        if learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {learning_rate}")
        self.learning_rate = learning_rate

    def step(self, model: OneHiddenLayerMLP, gradients: Dict[str, np.ndarray]):
        """
        Perform one optimization step.

        Updates model parameters in-place using computed gradients:
            W₁ ← W₁ - η · ∂L/∂W₁
            b₁ ← b₁ - η · ∂L/∂b₁
            W₂ ← W₂ - η · ∂L/∂W₂
            b₂ ← b₂ - η · ∂L/∂b₂

        Args:
            model: OneHiddenLayerMLP to update
            gradients: Dictionary with keys 'dL_dW1', 'dL_db1', 'dL_dW2', 'dL_db2'

        Examples:
            >>> model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
            >>> optimizer = SGD(learning_rate=0.1)
            >>> x, y = np.array([1.0, -1.0]), 1.0
            >>> f = model.forward(x)
            >>> grads = model.backward(x, y, f)
            >>> optimizer.step(model, grads)
        """
        # Update layer 1 parameters
        model.layer1.W -= self.learning_rate * gradients['dL_dW1']
        model.layer1.b -= self.learning_rate * gradients['dL_db1']

        # Update layer 2 parameters
        model.layer2.W -= self.learning_rate * gradients['dL_dW2']
        model.layer2.b -= self.learning_rate * gradients['dL_db2']

    def __repr__(self) -> str:
        """String representation."""
        return f"SGD(learning_rate={self.learning_rate})"


def train_step(
    model: OneHiddenLayerMLP,
    X_batch: np.ndarray,
    y_batch: np.ndarray,
    optimizer: SGD
) -> float:
    """
    Perform one training step on a batch.

    Computes loss and gradients for each sample in batch, averages them,
    and performs one parameter update.

    Args:
        model: OneHiddenLayerMLP to train
        X_batch: Input batch, shape (batch_size, input_dim)
        y_batch: Target batch, shape (batch_size,)
        optimizer: SGD optimizer

    Returns:
        Average loss over the batch

    Examples:
        >>> model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        >>> optimizer = SGD(learning_rate=0.01)
        >>> X = np.random.randn(32, 2)
        >>> y = np.random.randn(32)
        >>> loss = train_step(model, X, y, optimizer)
        >>> isinstance(loss, float)
        True
    """
    batch_size = X_batch.shape[0]

    # Initialize gradient accumulators
    grad_W1 = np.zeros_like(model.layer1.W)
    grad_b1 = np.zeros_like(model.layer1.b)
    grad_W2 = np.zeros_like(model.layer2.W)
    grad_b2 = np.zeros_like(model.layer2.b)

    total_loss = 0.0

    # Accumulate gradients over batch
    for i in range(batch_size):
        x = X_batch[i]
        y = y_batch[i]

        # Forward pass
        f = model.forward(x)

        # Compute loss
        loss = 0.5 * (f - y) ** 2
        total_loss += loss

        # Backward pass
        grads = model.backward(x, y, f)

        # Accumulate gradients
        grad_W1 += grads['dL_dW1']
        grad_b1 += grads['dL_db1']
        grad_W2 += grads['dL_dW2']
        grad_b2 += grads['dL_db2']

    # Average gradients
    grad_W1 /= batch_size
    grad_b1 /= batch_size
    grad_W2 /= batch_size
    grad_b2 /= batch_size

    # Perform optimization step
    averaged_gradients = {
        'dL_dW1': grad_W1,
        'dL_db1': grad_b1,
        'dL_dW2': grad_W2,
        'dL_db2': grad_b2
    }
    optimizer.step(model, averaged_gradients)

    # Return average loss
    return total_loss / batch_size


def train_epoch(
    model: OneHiddenLayerMLP,
    X: np.ndarray,
    y: np.ndarray,
    optimizer: SGD,
    batch_size: int = 32,
    shuffle: bool = True
) -> float:
    """
    Train for one epoch (one pass through dataset).

    Args:
        model: OneHiddenLayerMLP to train
        X: Input data, shape (n_samples, input_dim)
        y: Target data, shape (n_samples,)
        optimizer: SGD optimizer
        batch_size: Size of mini-batches
        shuffle: Whether to shuffle data before batching

    Returns:
        Average loss over the epoch

    Examples:
        >>> model = OneHiddenLayerMLP(input_dim=2, hidden_dim=4)
        >>> optimizer = SGD(learning_rate=0.1)
        >>> X = np.random.randn(100, 2)
        >>> y = np.random.randn(100)
        >>> loss = train_epoch(model, X, y, optimizer, batch_size=32)
        >>> isinstance(loss, float)
        True
    """
    # Create batches
    batches = create_batches(X, y, batch_size, shuffle)

    # Train on each batch
    epoch_loss = 0.0
    for X_batch, y_batch in batches:
        batch_loss = train_step(model, X_batch, y_batch, optimizer)
        epoch_loss += batch_loss * len(X_batch)  # Weight by batch size

    # Return average loss
    return epoch_loss / len(X)


def train(
    model: OneHiddenLayerMLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    optimizer: SGD,
    n_epochs: int = 100,
    batch_size: int = 32,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    verbose: bool = True,
    print_every: int = 10
) -> Dict[str, List[float]]:
    """
    Train model for multiple epochs.

    Args:
        model: OneHiddenLayerMLP to train
        X_train: Training inputs, shape (n_train, input_dim)
        y_train: Training targets, shape (n_train,)
        optimizer: SGD optimizer
        n_epochs: Number of training epochs
        batch_size: Size of mini-batches
        X_val: Optional validation inputs
        y_val: Optional validation targets
        verbose: Whether to print progress
        print_every: Print progress every N epochs

    Returns:
        Dictionary with training history:
        {
            'train_loss': List of training losses per epoch,
            'val_loss': List of validation losses per epoch (if validation data provided)
        }

    Examples:
        >>> model = OneHiddenLayerMLP(input_dim=2, hidden_dim=4)
        >>> optimizer = SGD(learning_rate=0.1)
        >>> X_train = np.random.randn(100, 2)
        >>> y_train = np.random.randn(100)
        >>> history = train(model, X_train, y_train, optimizer, n_epochs=50, verbose=False)
        >>> len(history['train_loss'])
        50
    """
    history = {
        'train_loss': [],
        'val_loss': []
    }

    for epoch in range(n_epochs):
        # Train for one epoch
        train_loss = train_epoch(model, X_train, y_train, optimizer, batch_size)
        history['train_loss'].append(train_loss)

        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_predictions = model.forward_batch(X_val)
            val_loss = squared_error_loss(val_predictions, y_val)
            history['val_loss'].append(val_loss)
        else:
            val_loss = None

        # Print progress
        if verbose and (epoch + 1) % print_every == 0:
            msg = f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {train_loss:.6f}"
            if val_loss is not None:
                msg += f", Val Loss: {val_loss:.6f}"
            print(msg)

    return history


def compute_accuracy(predictions: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute classification accuracy.

    Converts continuous predictions to binary by thresholding,
    then computes fraction of correct predictions.

    Args:
        predictions: Model predictions, shape (n_samples,)
        targets: Target labels {0, 1}, shape (n_samples,)
        threshold: Decision boundary (default 0.5)

    Returns:
        Accuracy as fraction in [0, 1]

    Examples:
        >>> predictions = np.array([0.2, 0.7, 0.4, 0.9])
        >>> targets = np.array([0, 1, 0, 1])
        >>> compute_accuracy(predictions, targets, threshold=0.5)
        1.0
        >>> predictions = np.array([0.6, 0.3])
        >>> targets = np.array([0, 1])
        >>> compute_accuracy(predictions, targets, threshold=0.5)
        0.0
    """
    predictions = np.asarray(predictions).ravel()
    targets = np.asarray(targets).ravel()

    # Convert predictions to binary
    binary_predictions = (predictions >= threshold).astype(int)

    # Compute accuracy
    correct = np.sum(binary_predictions == targets)
    accuracy = correct / len(targets)

    return float(accuracy)


def evaluate(
    model: OneHiddenLayerMLP,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate model on dataset.

    Computes loss and accuracy metrics.

    Args:
        model: OneHiddenLayerMLP to evaluate
        X: Input data, shape (n_samples, input_dim)
        y: Target data, shape (n_samples,)
        threshold: Classification threshold for accuracy

    Returns:
        Dictionary with metrics:
        {
            'loss': squared error loss,
            'accuracy': classification accuracy
        }

    Examples:
        >>> model = OneHiddenLayerMLP(input_dim=2, hidden_dim=4)
        >>> X = np.random.randn(50, 2)
        >>> y = np.random.randint(0, 2, 50)
        >>> metrics = evaluate(model, X, y)
        >>> 'loss' in metrics and 'accuracy' in metrics
        True
    """
    # Get predictions
    predictions = model.forward_batch(X)

    # Compute loss
    loss = squared_error_loss(predictions, y)

    # Compute accuracy
    accuracy = compute_accuracy(predictions, y, threshold)

    return {
        'loss': loss,
        'accuracy': accuracy
    }
