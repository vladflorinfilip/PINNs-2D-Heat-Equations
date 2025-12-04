"""
Quick Start Script for PINN Heat Equation Solver

This script provides a simple, fast example to verify your installation
and get started with PINNs quickly.
"""

import numpy as np
import torch
import time
import os
from pinn_heat_equation import HeatEquationPINN
from pinn_visualization import plot_temperature_field_2d, plot_initial_condition


def quick_demo():
    """
    A quick demonstration that trains a small PINN in ~30 seconds.
    """
    print("="*70)
    print("Quick Start Demo: PINN for 2D Heat Equation")
    print("="*70)
    print()
    print("This is a fast demo with a small problem to verify your installation.")
    print("Training time: ~30-60 seconds")
    print()
    
    # Create output directories
    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Small problem for fast training
    print("Setting up problem...")
    x_domain = (0.0, 2.0)
    y_domain = (0.0, 2.0)
    t_domain = (0.0, 0.1)
    alpha = 1.0
    
    # Simple Gaussian initial condition
    x0, y0 = 1.0, 1.0
    sigma = 0.3
    
    def initial_condition(x, y):
        return np.exp(-((x - x0)**2 + (y - y0)**2) / (2.0 * sigma**2))
    
    def boundary_condition(x, y, t):
        return 0.0
    
    # Plot initial condition
    print("Plotting initial condition...")
    plot_initial_condition(
        initial_condition,
        x_domain,
        y_domain,
        nx=50,
        ny=50,
        title="Quick Demo - Initial Condition",
        save_path="plots/quickstart_initial.png"
    )
    
    # Create PINN with small network
    print("\nInitializing PINN...")
    print(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    solver = HeatEquationPINN(
        x_domain=x_domain,
        y_domain=y_domain,
        t_domain=t_domain,
        alpha=alpha,
        layers=[3, 32, 32, 32, 1]  # Smaller network for speed
    )
    
    # Generate training data (fewer points for speed)
    print("\nGenerating training data...")
    solver.set_training_data(
        n_collocation=3000,
        n_initial=500,
        n_boundary=500,
        initial_condition=initial_condition,
        boundary_condition=boundary_condition
    )
    
    # Train with fewer epochs
    print("\nTraining PINN (this will take ~30-60 seconds)...")
    print("Progress will be printed every 1000 epochs.")
    print()
    
    start_time = time.time()
    solver.train(
        epochs=5000,
        lr=1e-3,
        weights={'physics': 1.0, 'initial': 100.0, 'boundary': 100.0},
        print_every=1000,
        use_scheduler=True
    )
    elapsed_time = time.time() - start_time
    
    print(f"\n✓ Training completed in {elapsed_time:.2f} seconds!")
    
    # Generate and plot solution
    print("\nGenerating solution at final time...")
    X, Y, U_final = solver.predict_grid(50, 50, t_domain[1])
    
    print(f"  Max temperature: {U_final.max():.6f}")
    print(f"  Min temperature: {U_final.min():.6f}")
    print(f"  Average temperature: {U_final.mean():.6f}")
    
    plot_temperature_field_2d(
        X, Y, U_final,
        title="Quick Demo - Final Solution",
        save_path="plots/quickstart_final.png"
    )
    
    # Plot loss history
    print("\nPlotting training loss...")
    solver.plot_loss_history(save_path="plots/quickstart_loss.png")
    
    # Save model
    print("\nSaving trained model...")
    solver.save_model("models/quickstart.pth")
    
    print("\n" + "="*70)
    print("Quick Start Demo Completed Successfully!")
    print("="*70)
    print("\nGenerated files:")
    print("  📊 plots/quickstart_initial.png  - Initial temperature distribution")
    print("  📊 plots/quickstart_final.png    - Final temperature distribution")
    print("  📈 plots/quickstart_loss.png     - Training loss history")
    print("  💾 models/quickstart.pth         - Trained PINN model")
    print()
    print("Next steps:")
    print("  1. Run full examples: python pinn_examples.py")
    print("  2. Read PINN_README.md for detailed documentation")
    print("  3. Experiment with custom initial/boundary conditions")
    print()
    print("=" * 70)


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        quick_demo()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("  2. Check that PyTorch is properly installed: python -c 'import torch; print(torch.__version__)'")
        print("  3. See PINN_README.md for more help")

