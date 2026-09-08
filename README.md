# PINN solver for the 1D Burgers equation

Physics-Informed Neural Network (PyTorch) solving the viscous Burgers equation:

∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²,  ν = 0.01/π

- Domain: x ∈ [−1, 1], t ∈ [0, 1]
- Initial condition: u(x, 0) = −sin(πx)
- Boundary conditions: u(−1, t) = u(1, t) = 0

## Method
Fully connected network u_θ(x, t); loss = PDE residual (autograd, second derivatives)
+ initial + boundary terms. Optimizer: Adam.

## Result
Final total loss ≈ 2.3·10⁻³ (`day20.py`).

## Usage
