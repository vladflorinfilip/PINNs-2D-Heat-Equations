"""
Simple PINN training script matching Rust test cases
Trains on the same three test cases, then tests on an unseen example
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pinn_heat_equation import HeatEquationPINN
import os

# Create output directory
os.makedirs('pinn_results', exist_ok=True)

def test_case_1_gaussian():
    """Gaussian pulse with Dirichlet BC (u=0 on boundaries)"""
    print("\n" + "="*60)
    print("Test Case 1: Gaussian Pulse (Dirichlet BC)")
    print("="*60)
    
    # Match Rust parameters
    nx, ny = 100, 100
    dx, dy = 0.05, 0.05
    alpha = 1.0
    num_steps = 500
    dt = 0.0001
    t_final = num_steps * dt  # 0.05
    
    x_max = (nx - 1) * dx  # 4.95
    y_max = (ny - 1) * dy  # 4.95
    x0 = x_max / 2.0
    y0 = y_max / 2.0
    sigma = 0.3
    
    # Initial condition
    def initial_condition(x, y):
        return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    
    # Boundary condition (Dirichlet: u=0)
    def boundary_condition(x, y, t):
        return 0.0
    
    # Create PINN (small network for fast training)
    pinn = HeatEquationPINN(
        x_domain=(0.0, x_max),
        y_domain=(0.0, y_max),
        t_domain=(0.0, t_final),
        alpha=alpha,
        layers=[3, 20, 20, 1]  # Small network for speed
    )
    
    # Set training data (reduced points for speed)
    pinn.set_training_data(
        n_collocation=2000,  # Reduced for speed
        n_initial=200,
        n_boundary=200,
        initial_condition=initial_condition,
        boundary_condition=boundary_condition
    )
    
    # Train (fewer epochs for speed)
    pinn.train(
        epochs=1000,  # Reduced for speed
        lr=1e-3,
        weights={'physics': 1.0, 'initial': 100.0, 'boundary': 100.0},
        print_every=250
    )
    
    # Test on grid
    X, Y, U_pred = pinn.predict_grid(nx=50, ny=50, t=t_final)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Initial condition
    x_ic = np.linspace(0, x_max, 50)
    y_ic = np.linspace(0, y_max, 50)
    X_ic, Y_ic = np.meshgrid(x_ic, y_ic)
    U_ic = np.array([[initial_condition(x, y) for x in x_ic] for y in y_ic])
    
    im1 = axes[0].contourf(X_ic, Y_ic, U_ic, levels=20, cmap='hot')
    axes[0].set_title('Initial Condition (t=0)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].contourf(X, Y, U_pred, levels=20, cmap='hot')
    axes[1].set_title(f'PINN Prediction (t={t_final:.4f})')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('pinn_results/gaussian_test.png', dpi=150)
    print(f"  Results saved to pinn_results/gaussian_test.png")
    plt.close()
    
    return pinn


def test_case_2_sine():
    """Sine wave with Dirichlet BC (simplified from Neumann for PINN)"""
    print("\n" + "="*60)
    print("Test Case 2: Sine Wave (Dirichlet BC)")
    print("="*60)
    
    # Match Rust parameters
    nx, ny = 80, 80
    dx, dy = 0.1, 0.1
    alpha = 0.5
    num_steps = 200
    dt = 0.001
    t_final = num_steps * dt  # 0.2
    
    x_max = (nx - 1) * dx  # 7.9
    y_max = (ny - 1) * dy  # 7.9
    lx = x_max
    ly = y_max
    
    # Initial condition
    def initial_condition(x, y):
        return np.sin(np.pi * x / lx) * np.sin(np.pi * y / ly)
    
    # Boundary condition (Dirichlet: u=0)
    def boundary_condition(x, y, t):
        return 0.0
    
    # Create PINN
    pinn = HeatEquationPINN(
        x_domain=(0.0, x_max),
        y_domain=(0.0, y_max),
        t_domain=(0.0, t_final),
        alpha=alpha,
        layers=[3, 32, 32, 32, 1]
    )
    
    # Set training data
    pinn.set_training_data(
        n_collocation=5000,
        n_initial=500,
        n_boundary=500,
        initial_condition=initial_condition,
        boundary_condition=boundary_condition
    )
    
    # Train
    pinn.train(
        epochs=2000,
        lr=1e-3,
        weights={'physics': 1.0, 'initial': 100.0, 'boundary': 100.0},
        print_every=500
    )
    
    # Test on grid
    X, Y, U_pred = pinn.predict_grid(nx=50, ny=50, t=t_final)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x_ic = np.linspace(0, x_max, 50)
    y_ic = np.linspace(0, y_max, 50)
    X_ic, Y_ic = np.meshgrid(x_ic, y_ic)
    U_ic = np.array([[initial_condition(x, y) for x in x_ic] for y in y_ic])
    
    im1 = axes[0].contourf(X_ic, Y_ic, U_ic, levels=20, cmap='coolwarm')
    axes[0].set_title('Initial Condition (t=0)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].contourf(X, Y, U_pred, levels=20, cmap='coolwarm')
    axes[1].set_title(f'PINN Prediction (t={t_final:.4f})')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('pinn_results/sine_test.png', dpi=150)
    print(f"  Results saved to pinn_results/sine_test.png")
    plt.close()
    
    return pinn


def test_case_3_checkerboard():
    """Checkerboard pattern with Dirichlet BC (simplified from periodic)"""
    print("\n" + "="*60)
    print("Test Case 3: Checkerboard Pattern (Dirichlet BC)")
    print("="*60)
    
    # Match Rust parameters
    nx, ny = 60, 60
    dx, dy = 0.1, 0.1
    alpha = 0.8
    num_steps = 300
    dt = 0.0005
    t_final = num_steps * dt  # 0.15
    
    x_max = (nx - 1) * dx  # 5.9
    y_max = (ny - 1) * dy  # 5.9
    
    # Initial condition (checkerboard)
    def initial_condition(x, y):
        i = int(x / dx)
        j = int(y / dy)
        return 1.0 if (i + j) % 2 == 0 else 0.0
    
    # Boundary condition (Dirichlet: u=0)
    def boundary_condition(x, y, t):
        return 0.0
    
    # Create PINN
    pinn = HeatEquationPINN(
        x_domain=(0.0, x_max),
        y_domain=(0.0, y_max),
        t_domain=(0.0, t_final),
        alpha=alpha,
        layers=[3, 32, 32, 32, 1]
    )
    
    # Set training data
    pinn.set_training_data(
        n_collocation=5000,
        n_initial=500,
        n_boundary=500,
        initial_condition=initial_condition,
        boundary_condition=boundary_condition
    )
    
    # Train
    pinn.train(
        epochs=2000,
        lr=1e-3,
        weights={'physics': 1.0, 'initial': 100.0, 'boundary': 100.0},
        print_every=500
    )
    
    # Test on grid
    X, Y, U_pred = pinn.predict_grid(nx=50, ny=50, t=t_final)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x_ic = np.linspace(0, x_max, 50)
    y_ic = np.linspace(0, y_max, 50)
    X_ic, Y_ic = np.meshgrid(x_ic, y_ic)
    U_ic = np.array([[initial_condition(x, y) for x in x_ic] for y in y_ic])
    
    im1 = axes[0].contourf(X_ic, Y_ic, U_ic, levels=20, cmap='viridis')
    axes[0].set_title('Initial Condition (t=0)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].contourf(X, Y, U_pred, levels=20, cmap='viridis')
    axes[1].set_title(f'PINN Prediction (t={t_final:.4f})')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('pinn_results/checkerboard_test.png', dpi=150)
    print(f"  Results saved to pinn_results/checkerboard_test.png")
    plt.close()
    
    return pinn


def unseen_test_case():
    """Unseen test case: Double Gaussian with non-zero boundary"""
    print("\n" + "="*60)
    print("Unseen Test Case: Double Gaussian with Non-Zero Boundary")
    print("="*60)
    
    # New parameters
    x_max, y_max = 5.0, 5.0
    alpha = 1.2
    t_final = 0.1
    
    # Initial condition: two Gaussian pulses
    def initial_condition(x, y):
        x1, y1 = 1.5, 1.5
        x2, y2 = 3.5, 3.5
        sigma = 0.4
        g1 = np.exp(-((x - x1)**2 + (y - y1)**2) / (2 * sigma**2))
        g2 = np.exp(-((x - x2)**2 + (y - y2)**2) / (2 * sigma**2))
        return g1 + 0.8 * g2
    
    # Boundary condition: non-zero (decaying with time)
    def boundary_condition(x, y, t):
        return 0.1 * np.exp(-t) * (x + y) / (x_max + y_max)
    
    # Create PINN
    pinn = HeatEquationPINN(
        x_domain=(0.0, x_max),
        y_domain=(0.0, y_max),
        t_domain=(0.0, t_final),
        alpha=alpha,
        layers=[3, 32, 32, 32, 1]
    )
    
    # Set training data
    pinn.set_training_data(
        n_collocation=5000,
        n_initial=500,
        n_boundary=500,
        initial_condition=initial_condition,
        boundary_condition=boundary_condition
    )
    
    # Train
    pinn.train(
        epochs=2000,
        lr=1e-3,
        weights={'physics': 1.0, 'initial': 100.0, 'boundary': 100.0},
        print_every=500
    )
    
    # Test on grid
    X, Y, U_pred = pinn.predict_grid(nx=50, ny=50, t=t_final)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x_ic = np.linspace(0, x_max, 50)
    y_ic = np.linspace(0, y_max, 50)
    X_ic, Y_ic = np.meshgrid(x_ic, y_ic)
    U_ic = np.array([[initial_condition(x, y) for x in x_ic] for y in y_ic])
    
    im1 = axes[0].contourf(X_ic, Y_ic, U_ic, levels=20, cmap='plasma')
    axes[0].set_title('Initial Condition (t=0)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].contourf(X, Y, U_pred, levels=20, cmap='plasma')
    axes[1].set_title(f'PINN Prediction (t={t_final:.4f})')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('pinn_results/unseen_test.png', dpi=150)
    print(f"  Results saved to pinn_results/unseen_test.png")
    plt.close()
    
    return pinn


def main():
    """Run all test cases"""
    print("="*60)
    print("PINN Training: Matching Rust Test Cases")
    print("="*60)
    
    # Test on same cases as Rust
    print("\n>>> Training on Rust test cases...")
    pinn1 = test_case_1_gaussian()
    pinn2 = test_case_2_sine()
    pinn3 = test_case_3_checkerboard()
    
    # Test on unseen case
    print("\n>>> Testing on unseen example...")
    pinn4 = unseen_test_case()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("Results saved in pinn_results/ directory")
    print("="*60)


if __name__ == "__main__":
    main()

