"""
Dataset generation utilities for neural network experiments.

This module provides functions to generate synthetic datasets for testing
and demonstrating neural network capabilities on nonlinear problems.
"""

import numpy as np
from typing import Tuple, List


def generate_xor_data(n_samples: int = 200, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate XOR pattern dataset: y = I{x₁ * x₂ < 0}

    Creates a classic nonlinearly separable binary classification problem.
    Points are in four quadrants:
    - Quadrant I (++): x₁ > 0, x₂ > 0 → label 0
    - Quadrant II (-+): x₁ < 0, x₂ > 0 → label 1
    - Quadrant III (--): x₁ < 0, x₂ < 0 → label 0
    - Quadrant IV (+-): x₁ > 0, x₂ < 0 → label 1

    Args:
        n_samples: Total number of samples to generate
        noise: Standard deviation of Gaussian noise added to coordinates

    Returns:
        Tuple of (X, y):
        - X: (n_samples, 2) input features in ℝ²
        - y: (n_samples,) binary labels {0, 1}

    Examples:
        >>> X, y = generate_xor_data(n_samples=4, noise=0.0)
        >>> X.shape
        (4, 2)
        >>> y.shape
        (4,)
        >>> len(np.unique(y))
        2
    """
    # Generate points uniformly in [-1, 1] × [-1, 1]
    # Divide samples equally among four quadrants
    samples_per_quadrant = n_samples // 4
    n_samples = samples_per_quadrant * 4  # Ensure even division

    X = np.zeros((n_samples, 2))
    y = np.zeros(n_samples, dtype=int)

    # Quadrant I (++): label 0
    X[0:samples_per_quadrant, 0] = np.random.uniform(0, 1, samples_per_quadrant)
    X[0:samples_per_quadrant, 1] = np.random.uniform(0, 1, samples_per_quadrant)
    y[0:samples_per_quadrant] = 0

    # Quadrant II (-+): label 1
    X[samples_per_quadrant:2*samples_per_quadrant, 0] = np.random.uniform(-1, 0, samples_per_quadrant)
    X[samples_per_quadrant:2*samples_per_quadrant, 1] = np.random.uniform(0, 1, samples_per_quadrant)
    y[samples_per_quadrant:2*samples_per_quadrant] = 1

    # Quadrant III (--): label 0
    X[2*samples_per_quadrant:3*samples_per_quadrant, 0] = np.random.uniform(-1, 0, samples_per_quadrant)
    X[2*samples_per_quadrant:3*samples_per_quadrant, 1] = np.random.uniform(-1, 0, samples_per_quadrant)
    y[2*samples_per_quadrant:3*samples_per_quadrant] = 0

    # Quadrant IV (+-): label 1
    X[3*samples_per_quadrant:4*samples_per_quadrant, 0] = np.random.uniform(0, 1, samples_per_quadrant)
    X[3*samples_per_quadrant:4*samples_per_quadrant, 1] = np.random.uniform(-1, 0, samples_per_quadrant)
    y[3*samples_per_quadrant:4*samples_per_quadrant] = 1

    # Add Gaussian noise
    if noise > 0:
        X += np.random.randn(n_samples, 2) * noise

    return X, y


def generate_spiral_data(n_samples: int = 200, n_classes: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate spiral dataset for visualization.

    Creates interleaved spirals, one for each class. This is another classic
    nonlinearly separable problem useful for demonstrating neural network
    capacity for learning complex decision boundaries.

    Args:
        n_samples: Total number of samples (divided among classes)
        n_classes: Number of spiral arms (classes)

    Returns:
        Tuple of (X, y):
        - X: (n_samples, 2) input features in ℝ²
        - y: (n_samples,) class labels {0, 1, ..., n_classes-1}

    Examples:
        >>> X, y = generate_spiral_data(n_samples=100, n_classes=2)
        >>> X.shape
        (100, 2)
        >>> y.shape
        (100,)
        >>> len(np.unique(y))
        2
    """
    samples_per_class = n_samples // n_classes
    n_samples = samples_per_class * n_classes  # Ensure even division

    X = np.zeros((n_samples, 2))
    y = np.zeros(n_samples, dtype=int)

    for class_idx in range(n_classes):
        # Radius: linearly increase from 0 to 1
        r = np.linspace(0.0, 1.0, samples_per_class)

        # Angle: create spiral effect
        # Each class starts at different angle offset
        theta = np.linspace(
            class_idx * 4 * np.pi / n_classes,  # Start angle
            (class_idx + 1) * 4 * np.pi / n_classes + 4 * np.pi,  # End angle (one full spiral)
            samples_per_class
        )

        # Add noise to radius
        r_noisy = r + np.random.randn(samples_per_class) * 0.1

        # Convert polar to Cartesian coordinates
        start_idx = class_idx * samples_per_class
        end_idx = (class_idx + 1) * samples_per_class

        X[start_idx:end_idx, 0] = r_noisy * np.cos(theta)
        X[start_idx:end_idx, 1] = r_noisy * np.sin(theta)
        y[start_idx:end_idx] = class_idx

    return X, y


def create_batches(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create mini-batches from data.

    Splits dataset into batches for mini-batch gradient descent.
    Optionally shuffles data before batching.

    Args:
        X: Input features, shape (n_samples, n_features)
        y: Targets, shape (n_samples,)
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data before batching

    Returns:
        List of (X_batch, y_batch) tuples, where each batch has size
        batch_size (except possibly the last batch)

    Examples:
        >>> X = np.random.randn(10, 2)
        >>> y = np.random.randint(0, 2, 10)
        >>> batches = create_batches(X, y, batch_size=3, shuffle=False)
        >>> len(batches)
        4
        >>> batches[0][0].shape
        (3, 2)
        >>> batches[-1][0].shape  # Last batch may be smaller
        (1, 2)
    """
    n_samples = X.shape[0]

    if n_samples != len(y):
        raise ValueError(
            f"X and y must have same number of samples: {n_samples} vs {len(y)}"
        )

    # Shuffle indices if requested
    if shuffle:
        indices = np.random.permutation(n_samples)
    else:
        indices = np.arange(n_samples)

    # Create batches
    batches = []
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        batches.append((X_batch, y_batch))

    return batches
