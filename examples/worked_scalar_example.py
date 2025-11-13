"""
Worked Scalar Example from Appendix A.

This script walks through the complete forward and backward pass for the
specific example given in the specification's Appendix A.

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

import numpy as np
from src.models import OneHiddenLayerMLP


def print_section(title: str):
    """Print section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def print_step(step: str, value, expected=None):
    """Print computation step."""
    print(f"\n{step}")
    if isinstance(value, np.ndarray):
        print(f"  Computed: {value}")
    else:
        print(f"  Computed: {value:.6f}")

    if expected is not None:
        if isinstance(expected, np.ndarray):
            print(f"  Expected: {expected}")
            match = np.allclose(value, expected, rtol=1e-10)
        else:
            print(f"  Expected: {expected:.6f}")
            match = np.isclose(value, expected, rtol=1e-10)

        status = "✓ MATCH" if match else "✗ MISMATCH"
        print(f"  Status:   {status}")


def main():
    """Run Appendix A worked example."""

    print_section("Appendix A: Worked Scalar Example")

    print("\nProblem Setup:")
    print("  Input dimension (d): 2")
    print("  Hidden dimension (h): 2")
    print("  Activation: ReLU")
    print("\nInput and target:")
    print("  x = [1, -1]ᵀ")
    print("  y = 2")

    print("\nParameters:")
    print("  W₁ = [[1.0, -2.0],")
    print("       [0.5,  1.0]]")
    print("  b₁ = [0.0, 0.0]")
    print("  W₂ = [[1.0, -1.0]]")
    print("  b₂ = [0.0]")

    # Create model
    model = OneHiddenLayerMLP(input_dim=2, hidden_dim=2, activation='relu')

    # Set parameters from Appendix A
    model.set_parameters({
        'W1': np.array([[1.0, -2.0], [0.5, 1.0]]),
        'b1': np.array([0.0, 0.0]),
        'W2': np.array([[1.0, -1.0]]),
        'b2': np.array([0.0])
    })

    # Input and target
    x = np.array([1.0, -1.0])
    y = 2.0

    # ==========================================================================
    # Forward Pass
    # ==========================================================================

    print_section("Forward Pass")

    print("\nStep 1: First affine transformation")
    print("  a₁ = W₁x + b₁")
    print("     = [[1, -2], [0.5, 1]] @ [1, -1] + [0, 0]")
    print("     = [1×1 + (-2)×(-1), 0.5×1 + 1×(-1)]")
    print("     = [1 + 2, 0.5 - 1]")
    print("     = [3, -0.5]")

    # Forward pass
    f = model.forward(x)

    print_step("Computed a₁:", model.a1_cache, np.array([3.0, -0.5]))

    print("\nStep 2: ReLU activation")
    print("  h₁ = ReLU(a₁) = max(0, a₁)")
    print("     = [max(0, 3), max(0, -0.5)]")
    print("     = [3, 0]")

    print_step("Computed h₁:", model.h1_cache, np.array([3.0, 0.0]))

    print("\nStep 3: Second affine transformation")
    print("  f = W₂h₁ + b₂")
    print("    = [[1, -1]] @ [3, 0] + [0]")
    print("    = [1×3 + (-1)×0]")
    print("    = [3]")

    print_step("Computed f:", f, 3.0)

    print("\nStep 4: Loss computation")
    print("  L = (1/2)(f - y)²")
    print("    = (1/2)(3 - 2)²")
    print("    = (1/2)(1)²")
    print("    = 0.5")

    L = 0.5 * (f - y) ** 2
    print_step("Computed L:", L, 0.5)

    # ==========================================================================
    # Backward Pass
    # ==========================================================================

    print_section("Backward Pass")

    # Backward pass
    grads = model.backward(x, y, f)

    print("\nStep 1: Loss gradient")
    print("  δf = ∂L/∂f = f - y")
    print("     = 3 - 2")
    print("     = 1")

    delta_f = f - y
    print_step("Computed δf:", delta_f, 1.0)

    print("\nStep 2: Layer 2 gradients")
    print("  ∂L/∂W₂ = δf · h₁ᵀ")
    print("         = 1 × [3, 0]")
    print("         = [3, 0]")

    print_step("Computed ∂L/∂W₂:", grads['dL_dW2'], np.array([[3.0, 0.0]]))

    print("\n  ∂L/∂b₂ = δf")
    print("         = 1")

    print_step("Computed ∂L/∂b₂:", grads['dL_db2'], np.array([1.0]))

    print("\nStep 3: Backprop to hidden layer")
    print("  δh₁ = W₂ᵀ · δf")
    print("      = [[1], [-1]] × 1")
    print("      = [1, -1]")

    # Compute manually for display
    delta_h1 = model.layer2.W.T.ravel() * delta_f
    print_step("Computed δh₁:", delta_h1, np.array([1.0, -1.0]))

    print("\nStep 4: Backprop through activation")
    print("  ReLU'(a₁) = [a₁ > 0]")
    print("            = [3 > 0, -0.5 > 0]")
    print("            = [1, 0]")
    print("\n  δa₁ = δh₁ ⊙ ReLU'(a₁)")
    print("      = [1, -1] ⊙ [1, 0]")
    print("      = [1, 0]")

    relu_derivative = (model.a1_cache > 0).astype(float)
    delta_a1 = delta_h1 * relu_derivative
    print_step("Computed δa₁:", delta_a1, np.array([1.0, 0.0]))

    print("\nStep 5: Layer 1 gradients")
    print("  ∂L/∂W₁ = δa₁ · xᵀ")
    print("         = [[1], [0]] × [[1, -1]]")
    print("         = [[1×1, 1×(-1)],")
    print("            [0×1, 0×(-1)]]")
    print("         = [[1, -1],")
    print("            [0,  0]]")

    print_step("Computed ∂L/∂W₁:", grads['dL_dW1'], np.array([[1.0, -1.0], [0.0, 0.0]]))

    print("\n  ∂L/∂b₁ = δa₁")
    print("         = [1, 0]")

    print_step("Computed ∂L/∂b₁:", grads['dL_db1'], np.array([1.0, 0.0]))

    # ==========================================================================
    # Summary
    # ==========================================================================

    print_section("Summary")

    print("\nForward Pass Results:")
    print(f"  a₁ = {model.a1_cache}")
    print(f"  h₁ = {model.h1_cache}")
    print(f"  f  = {f:.6f}")
    print(f"  L  = {L:.6f}")

    print("\nBackward Pass Results:")
    print(f"  ∂L/∂W₂ = {grads['dL_dW2']}")
    print(f"  ∂L/∂b₂ = {grads['dL_db2']}")
    print(f"  ∂L/∂W₁ = {grads['dL_dW1']}")
    print(f"  ∂L/∂b₁ = {grads['dL_db1']}")

    print("\nAll computations match Appendix A specifications! ✓")

    # Verify all values match
    assert np.allclose(model.a1_cache, [3.0, -0.5], rtol=1e-10)
    assert np.allclose(model.h1_cache, [3.0, 0.0], rtol=1e-10)
    assert np.isclose(f, 3.0, rtol=1e-10)
    assert np.isclose(L, 0.5, rtol=1e-10)
    assert np.allclose(grads['dL_dW2'], [[3.0, 0.0]], rtol=1e-10)
    assert np.allclose(grads['dL_db2'], [1.0], rtol=1e-10)
    assert np.allclose(grads['dL_dW1'], [[1.0, -1.0], [0.0, 0.0]], rtol=1e-10)
    assert np.allclose(grads['dL_db1'], [1.0, 0.0], rtol=1e-10)

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
