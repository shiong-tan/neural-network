"""
Activation functions and their derivatives for neural networks.

This module implements ReLU and Sigmoid activations with their analytical
derivatives for backpropagation.
"""

import numpy as np
from typing import Union


class Activation:
    """Base class for activation functions."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        """
        Compute activation function.

        Args:
            z: Input array of any shape

        Returns:
            Activated output with same shape as input
        """
        raise NotImplementedError

    def derivative(self, z: np.ndarray) -> np.ndarray:
        """
        Compute elementwise derivative of activation function.

        Args:
            z: Input array (pre-activation values)

        Returns:
            Derivative with same shape as input
        """
        raise NotImplementedError


class ReLU(Activation):
    """
    Rectified Linear Unit activation.

    Mathematical definition:
        Forward: relu(u) = max{0, u}
        Derivative: relu'(u) = 1 if u > 0, else 0 (subdifferential convention: relu'(0) = 0)
    """

    def forward(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Apply ReLU activation: max(0, z).

        Args:
            z: Input value(s)

        Returns:
            ReLU(z) with same shape as input. Always returns float dtype.

        Note:
            - np.inf becomes np.inf (since max(0, inf) = inf)
            - -np.inf becomes 0.0 (since max(0, -inf) = 0)
            - np.nan propagates as np.nan

        Examples:
            >>> relu = ReLU()
            >>> relu.forward(-1.0)
            0.0
            >>> relu.forward(2.5)
            2.5
            >>> relu.forward(np.array([-1, 0, 2]))
            array([0., 0., 2.])
        """
        if isinstance(z, (int, float)):
            return max(0.0, float(z))
        return np.maximum(0.0, z)

    def derivative(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute ReLU derivative.

        Args:
            z: Pre-activation values

        Returns:
            1 where z > 0, 0 elsewhere (including at z=0)

        Examples:
            >>> relu = ReLU()
            >>> relu.derivative(-1.0)
            0.0
            >>> relu.derivative(2.5)
            1.0
            >>> relu.derivative(0.0)
            0.0
        """
        if isinstance(z, (int, float)):
            return 1.0 if z > 0 else 0.0
        return (z > 0).astype(float)


class Sigmoid(Activation):
    """
    Sigmoid (logistic) activation function.

    Mathematical definition:
        Forward: σ(u) = 1 / (1 + exp(-u))
        Derivative: σ'(u) = σ(u)(1 - σ(u))

    The derivative has a convenient form in terms of the forward output,
    which allows efficient computation during backpropagation.
    """

    def forward(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Apply sigmoid activation: 1 / (1 + exp(-z)).

        Uses numerically stable implementation to avoid overflow.

        Args:
            z: Input value(s)

        Returns:
            Sigmoid(z) ∈ (0, 1) with same shape as input

        Examples:
            >>> sigmoid = Sigmoid()
            >>> sigmoid.forward(0.0)
            0.5
            >>> sigmoid.forward(10.0) > 0.99
            True
            >>> sigmoid.forward(-10.0) < 0.01
            True
        """
        # Numerically stable sigmoid implementation
        # For large positive z, exp(-z) → 0, so σ(z) → 1
        # For large negative z, use σ(z) = exp(z)/(1 + exp(z))
        if isinstance(z, (int, float)):
            if z >= 0:
                return 1.0 / (1.0 + np.exp(-z))
            else:
                exp_z = np.exp(z)
                return exp_z / (1.0 + exp_z)

        # Vectorized version
        result = np.zeros_like(z, dtype=float)
        positive_mask = z >= 0

        # For positive z: standard formula σ(z) = 1/(1 + exp(-z))
        # This avoids overflow since exp(-z) is small when z is large and positive
        result[positive_mask] = 1.0 / (1.0 + np.exp(-z[positive_mask]))

        # For negative z: equivalent form σ(z) = exp(z)/(1 + exp(z))
        # This avoids underflow since exp(z) is computable when z is large and negative
        exp_z = np.exp(z[~positive_mask])
        result[~positive_mask] = exp_z / (1.0 + exp_z)

        return result

    def derivative(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute sigmoid derivative: σ'(z) = σ(z)(1 - σ(z)).

        Args:
            z: Pre-activation values

        Returns:
            Derivative with same shape as input

        Examples:
            >>> sigmoid = Sigmoid()
            >>> abs(sigmoid.derivative(0.0) - 0.25) < 1e-10
            True
        """
        sigma_z = self.forward(z)
        return sigma_z * (1 - sigma_z)
