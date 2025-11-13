"""
Layer implementations for neural networks.

This module implements affine (linear) transformations and will be extended
with backpropagation in Phase 4.
"""

import numpy as np
from typing import Optional, Tuple


class AffineLayer:
    """
    Affine transformation layer: z = Wx + b

    Performs a linear transformation of the input by multiplying with a weight
    matrix W and adding a bias vector b.

    Mathematical form:
        z = Wx + b

    Where:
        W ∈ ℝ^(output_dim × input_dim): weight matrix
        b ∈ ℝ^(output_dim): bias vector
        x ∈ ℝ^(input_dim): input vector
        z ∈ ℝ^(output_dim): output vector
    """

    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize affine layer with random weights and zero biases.

        Args:
            input_dim: Dimensionality of input features
            output_dim: Dimensionality of output features

        Note:
            Weights are initialized with small random values (scaled by 0.1)
            to break symmetry while avoiding saturation.
            Biases are initialized to zero.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Cache for backward pass (will be used in Phase 4)
        self.x_cache: Optional[np.ndarray] = None

        # Initialize parameters
        self.initialize_parameters()

    def initialize_parameters(self):
        """
        Initialize weights and biases.

        Uses small random initialization for weights (Gaussian with σ=0.1)
        and zero initialization for biases.
        """
        # W ∈ ℝ^(output_dim × input_dim)
        self.W = np.random.randn(self.output_dim, self.input_dim) * 0.1

        # b ∈ ℝ^(output_dim)
        self.b = np.zeros(self.output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: compute z = Wx + b.

        Supports both single samples and batches:
        - Single sample: x ∈ ℝ^d → z ∈ ℝ^m
        - Batch: X ∈ ℝ^(n×d) → Z ∈ ℝ^(n×m)

        Args:
            x: Input array
               - Shape (input_dim,) for single sample
               - Shape (batch_size, input_dim) for batch

        Returns:
            z: Output array
               - Shape (output_dim,) for single sample
               - Shape (batch_size, output_dim) for batch

        Examples:
            >>> layer = AffineLayer(input_dim=3, output_dim=2)
            >>> x_single = np.array([1.0, 2.0, 3.0])
            >>> z_single = layer.forward(x_single)
            >>> z_single.shape
            (2,)
            >>> X_batch = np.random.randn(10, 3)
            >>> Z_batch = layer.forward(X_batch)
            >>> Z_batch.shape
            (10, 2)
        """
        # Cache input for backward pass (Phase 4)
        self.x_cache = x

        # Handle both single samples and batches
        if x.ndim == 1:
            # Single sample: z = Wx + b
            # W: (m, d), x: (d,) → Wx: (m,), b: (m,) → z: (m,)
            z = self.W @ x + self.b
        else:
            # Batch: Z = XW^T + b
            # X: (n, d), W: (m, d) → XW^T: (n, m), b: (m,) → Z: (n, m)
            z = x @ self.W.T + self.b

        return z

    def backward(self, dL_dz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass: compute gradients using chain rule.

        Given gradient of loss w.r.t. layer output (dL/dz), compute:
        - dL/dx: gradient w.r.t. input (for backprop to previous layer)
        - dL/dW: gradient w.r.t. weights (for parameter update)
        - dL/db: gradient w.r.t. bias (for parameter update)

        Mathematical derivations:
            Given z = Wx + b, the Jacobians are:
            - ∂z/∂x = W^T
            - ∂z/∂W = x^T (outer product structure)
            - ∂z/∂b = I (identity)

            By chain rule:
            - dL/dx = W^T · dL/dz
            - dL/dW = dL/dz · x^T (outer product)
            - dL/db = dL/dz

        Args:
            dL_dz: Gradient of loss w.r.t. layer output
                   - Shape (output_dim,) for single sample
                   - Shape (batch_size, output_dim) for batch

        Returns:
            Tuple of (dL_dx, dL_dW, dL_db):
            - dL_dx: Gradient w.r.t. input, same shape as cached input
            - dL_dW: Gradient w.r.t. weights, shape (output_dim, input_dim)
            - dL_db: Gradient w.r.t. bias, shape (output_dim,)

        Note:
            This method will be implemented in Phase 4. For now, it serves
            as a placeholder with the correct interface.
        """
        if self.x_cache is None:
            raise ValueError("Must call forward() before backward()")

        x = self.x_cache

        # Handle both single samples and batches
        if x.ndim == 1:
            # Single sample
            # dL/dx = W^T @ dL/dz
            # W: (m, d), dL/dz: (m,) → W^T @ dL/dz: (d,)
            dL_dx = self.W.T @ dL_dz

            # dL/dW = dL/dz ⊗ x (outer product)
            # dL/dz: (m,), x: (d,) → dL/dW: (m, d)
            dL_dW = dL_dz[:, None] @ x[None, :]

            # dL/db = dL/dz
            dL_db = dL_dz
        else:
            # Batch: average gradients over batch
            batch_size = x.shape[0]

            # dL/dx = dL/dZ @ W
            # dL/dZ: (n, m), W: (m, d) → dL/dX: (n, d)
            dL_dx = dL_dz @ self.W

            # dL/dW = (1/n) * dL/dZ^T @ X
            # dL/dZ: (n, m), X: (n, d) → dL/dW: (m, d)
            dL_dW = (dL_dz.T @ x) / batch_size

            # dL/db = (1/n) * sum over batch
            # dL/dZ: (n, m) → dL/db: (m,)
            dL_db = np.mean(dL_dz, axis=0)

        return dL_dx, dL_dW, dL_db
