"""
Loss functions for neural network training.

This module implements squared error loss (mean squared error) for regression
and binary classification tasks.
"""

import numpy as np


def squared_error_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute mean squared error loss.

    Mathematical definition:
        L = (1/2n) Σᵢ (fᵢ - yᵢ)²

    Where:
        n: number of samples
        fᵢ: prediction for sample i
        yᵢ: target for sample i

    The factor of 1/2 simplifies the gradient: ∂L/∂f = f - y

    Args:
        predictions: Model predictions, shape (n,) or (n, 1)
        targets: Ground truth targets, shape (n,) or (n, 1)

    Returns:
        Mean squared error (scalar)

    Examples:
        >>> predictions = np.array([1.0, 2.0, 3.0])
        >>> targets = np.array([1.5, 2.5, 2.5])
        >>> loss = squared_error_loss(predictions, targets)
        >>> # Loss = (1/6) * ((0.5)² + (0.5)² + (0.5)²) = 0.125
        >>> abs(loss - 0.125) < 1e-10
        True
    """
    # Ensure arrays are flattened for consistent computation
    predictions = np.asarray(predictions).ravel()
    targets = np.asarray(targets).ravel()

    if predictions.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
        )

    # L = (1/2n) Σᵢ (fᵢ - yᵢ)²
    n = len(predictions)
    squared_errors = (predictions - targets) ** 2
    loss = np.sum(squared_errors) / (2 * n)

    return float(loss)


def squared_error_gradient(prediction: float, target: float) -> float:
    """
    Gradient of squared error loss w.r.t. a single prediction.

    Mathematical definition:
        For L = (1/2)(f - y)²
        ∂L/∂f = f - y

    This is the "delta" term δf used in backpropagation.

    Args:
        prediction: Single prediction value
        target: Single target value

    Returns:
        Gradient scalar: prediction - target

    Examples:
        >>> grad = squared_error_gradient(3.0, 2.0)
        >>> grad
        1.0
        >>> grad = squared_error_gradient(1.5, 2.0)
        >>> grad
        -0.5
    """
    return float(prediction - target)


def squared_error_loss_gradient_batch(
    predictions: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """
    Compute gradient of squared error loss for a batch.

    For each sample i:
        ∂L/∂fᵢ = fᵢ - yᵢ

    Args:
        predictions: Batch of predictions, shape (n,) or (n, 1)
        targets: Batch of targets, shape (n,) or (n, 1)

    Returns:
        Gradients w.r.t. predictions, same shape as predictions

    Examples:
        >>> predictions = np.array([1.0, 2.0, 3.0])
        >>> targets = np.array([1.5, 2.5, 2.5])
        >>> grads = squared_error_loss_gradient_batch(predictions, targets)
        >>> np.allclose(grads, [-0.5, -0.5, 0.5])
        True
    """
    # Ensure arrays have consistent shape
    predictions = np.asarray(predictions).ravel()
    targets = np.asarray(targets).ravel()

    if predictions.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
        )

    # Element-wise: ∂L/∂f = f - y
    gradients = predictions - targets

    return gradients
