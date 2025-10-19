"""
Examples for solving 2D heat equation with PINNs.

This script provides examples that match the original finite difference implementation:
1. Gaussian pulse with Dirichlet boundary conditions
2. Sine wave with Neumann boundary conditions (approximated)
3. Checkerboard pattern with periodic boundary conditions (approximated)
4. Performance comparison

Note: PINNs naturally handle complex boundary conditions through the loss function.
"""

import numpy as np
import torch
import time
from pinn_heat_equation import HeatEquationPINN
from pinn_visualization import (
    plot_temperature_field_2d,
    plot_temperature_field_3d,
    plot_initial_condition,
    plot_time_evolution
)


def example_gaussian_pulse():
    """
    Example 1: Gaussian pulse with Dirichlet boundary conditions.
    
    Initial condition: u(x,y,0) = exp(-((x-x0)² + (y-y0)²)/(2σ²))
    Boundary condition: u = 0 on all boundaries
    """
    print("\n" + "="*70)
    print("Example 1: Gaussian pulse with Dirichlet boundary conditions")
    print("="*70)
    print("Initial condition: u(x,y,0) = exp(-((x-x0)² + (y-y0)²)/(2σ²))")
    print("Boundary condition: u = 0 on all boundaries")
    print()
    
    # Problem parameters (matching Rust FD example)
    nx = 100
    ny = 100
    dx = 0.05
    dy = 0.05
    alpha = 1.0
    
    # Domain
    x_domain = (0.0, (nx - 1) * dx)
    y_domain = (0.0, (ny - 1) * dy)
    t_domain = (0.0, 0.05)  # Final time (500 steps * 0.0001)
    
    # Initial condition parameters
    x0 = (nx * dx) / 2.0
    y0 = (ny * dy) / 2.0
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
        title="Gaussian Pulse Initial Condition",
        save_path="plots/pinn_gaussian_initial.png"
    )
    
    # Create PINN solver
    print("\nInitializing PINN solver...")
    solver = HeatEquationPINN(
        x_domain=x_domain,
        y_domain=y_domain,
        t_domain=t_domain,
        alpha=alpha,
        layers=[3, 64, 64, 64, 64, 1]  # 4 hidden layers with 64 neurons each
    )
    
    # Generate training data
    print("Generating training data...")
    solver.set_training_data(
        n_collocation=10000,   # Physics loss points
        n_initial=2000,        # Initial condition points
        n_boundary=2000,       # Boundary condition points
        initial_condition=initial_condition,
        boundary_condition=boundary_condition
    )
    
    # Train the PINN
    print("\nTraining PINN...")
    start_time = time.time()
    solver.train(
        epochs=15000,
        lr=1e-3,
        weights={'physics': 1.0, 'initial': 100.0, 'boundary': 100.0},
        print_every=1000,
        use_scheduler=True
    )
    elapsed_time = time.time() - start_time
    
    print(f"\nTraining completed in {elapsed_time:.2f} seconds")
    
    # Plot final solution
    print("\nGenerating final solution...")
    X, Y, U_final = solver.predict_grid(nx, ny, t_domain[1])
    
    print(f"Max temperature: {U_final.max():.6f}")
    print(f"Min temperature: {U_final.min():.6f}")
    print(f"Average temperature: {U_final.mean():.6f}")
    
    plot_temperature_field_2d(
        X, Y, U_final,
        title="Gaussian Pulse Final Solution (PINN)",
        save_path="plots/pinn_gaussian_final.png"
    )
    
    plot_temperature_field_3d(
        X, Y, U_final,
        title="Gaussian Pulse Final Solution 3D (PINN)",
        save_path="plots/pinn_gaussian_final_3d.png"
    )
    
    # Plot time evolution
    print("\nPlotting time evolution...")
    times = np.linspace(0, t_domain[1], 6)
    plot_time_evolution(
        solver,
        times=times.tolist(),
        nx=nx,
        ny=ny,
        title="Gaussian Pulse Temperature Evolution",
        save_path="plots/pinn_gaussian_evolution.png"
    )
    
    # Plot loss history
    print("\nPlotting loss history...")
    solver.plot_loss_history(save_path="plots/pinn_gaussian_loss.png")
    
    # Save model
    print("\nSaving trained model...")
    solver.save_model("models/pinn_gaussian.pth")
    
    print("\n✓ Example 1 completed successfully!")
    print(f"  Plots saved to plots/pinn_gaussian_*.png")
    print(f"  Model saved to models/pinn_gaussian.pth")


def example_sine_wave():
    """
    Example 2: Sine wave pattern.
    
    Initial condition: u(x,y,0) = sin(πx/Lx) * sin(πy/Ly)
    Boundary condition: u = 0 (Dirichlet boundaries)
    
    Note: For true Neumann boundaries, we would add derivative constraints to the loss.
    """
    print("\n" + "="*70)
    print("Example 2: Sine wave with boundary conditions")
    print("="*70)
    print("Initial condition: u(x,y,0) = sin(πx/Lx) * sin(πy/Ly)")
    print("Boundary condition: u = 0 at boundaries")
    print()
    
    # Problem parameters
    nx = 80
    ny = 80
    dx = 0.1
    dy = 0.1
    alpha = 0.5
    
    # Domain
    x_domain = (0.0, (nx - 1) * dx)
    y_domain = (0.0, (ny - 1) * dy)
    t_domain = (0.0, 0.2)  # Final time (200 steps * 0.001)
    
    lx = (nx - 1) * dx
    ly = (ny - 1) * dy
    
    def initial_condition(x, y):
        return np.sin(np.pi * x / lx) * np.sin(np.pi * y / ly)
    
    def boundary_condition(x, y, t):
        return 0.0
    
    # Plot initial condition
    print("Plotting initial condition...")
    plot_initial_condition(
        initial_condition,
        x_domain,
        y_domain,
        title="Sine Wave Initial Condition",
        save_path="plots/pinn_sine_initial.png"
    )
    
    # Create PINN solver
    print("\nInitializing PINN solver...")
    solver = HeatEquationPINN(
        x_domain=x_domain,
        y_domain=y_domain,
        t_domain=t_domain,
        alpha=alpha,
        layers=[3, 64, 64, 64, 64, 1]
    )
    
    # Generate training data
    print("Generating training data...")
    solver.set_training_data(
        n_collocation=10000,
        n_initial=2000,
        n_boundary=2000,
        initial_condition=initial_condition,
        boundary_condition=boundary_condition
    )
    
    # Train the PINN
    print("\nTraining PINN...")
    start_time = time.time()
    solver.train(
        epochs=12000,
        lr=1e-3,
        weights={'physics': 1.0, 'initial': 100.0, 'boundary': 100.0},
        print_every=1000,
        use_scheduler=True
    )
    elapsed_time = time.time() - start_time
    
    print(f"\nTraining completed in {elapsed_time:.2f} seconds")
    
    # Plot final solution
    print("\nGenerating final solution...")
    X, Y, U_final = solver.predict_grid(nx, ny, t_domain[1])
    
    print(f"Max temperature: {U_final.max():.6f}")
    print(f"Min temperature: {U_final.min():.6f}")
    print(f"Average temperature: {U_final.mean():.6f}")
    
    plot_temperature_field_2d(
        X, Y, U_final,
        title="Sine Wave Final Solution (PINN)",
        save_path="plots/pinn_sine_final.png"
    )
    
    plot_temperature_field_3d(
        X, Y, U_final,
        title="Sine Wave Final Solution 3D (PINN)",
        save_path="plots/pinn_sine_final_3d.png"
    )
    
    # Plot time evolution
    print("\nPlotting time evolution...")
    times = np.linspace(0, t_domain[1], 6)
    plot_time_evolution(
        solver,
        times=times.tolist(),
        nx=nx,
        ny=ny,
        title="Sine Wave Temperature Evolution",
        save_path="plots/pinn_sine_evolution.png"
    )
    
    # Save model
    print("\nSaving trained model...")
    solver.save_model("models/pinn_sine.pth")
    
    print("\n✓ Example 2 completed successfully!")
    print(f"  Plots saved to plots/pinn_sine_*.png")
    print(f"  Model saved to models/pinn_sine.pth")


def example_checkerboard():
    """
    Example 3: Checkerboard pattern.
    
    Initial condition: alternating high/low temperature squares
    Boundary condition: u = 0 at boundaries
    
    Note: For true periodic boundaries, we would add periodic constraints to the loss.
    """
    print("\n" + "="*70)
    print("Example 3: Checkerboard pattern")
    print("="*70)
    print("Initial condition: alternating high/low temperature squares")
    print("Boundary condition: u = 0 at boundaries")
    print()
    
    # Problem parameters
    nx = 60
    ny = 60
    dx = 0.1
    dy = 0.1
    alpha = 0.8
    
    # Domain
    x_domain = (0.0, (nx - 1) * dx)
    y_domain = (0.0, (ny - 1) * dy)
    t_domain = (0.0, 0.15)  # Final time (300 steps * 0.0005)
    
    def initial_condition(x, y):
        i = int(x / dx)
        j = int(y / dy)
        if (i + j) % 2 == 0:
            return 1.0  # High temperature
        else:
            return 0.0  # Low temperature
    
    def boundary_condition(x, y, t):
        return 0.0
    
    # Plot initial condition
    print("Plotting initial condition...")
    plot_initial_condition(
        initial_condition,
        x_domain,
        y_domain,
        title="Checkerboard Initial Condition",
        save_path="plots/pinn_checkerboard_initial.png"
    )
    
    # Create PINN solver
    print("\nInitializing PINN solver...")
    solver = HeatEquationPINN(
        x_domain=x_domain,
        y_domain=y_domain,
        t_domain=t_domain,
        alpha=alpha,
        layers=[3, 64, 64, 64, 64, 1]
    )
    
    # Generate training data
    print("Generating training data...")
    solver.set_training_data(
        n_collocation=10000,
        n_initial=2000,
        n_boundary=2000,
        initial_condition=initial_condition,
        boundary_condition=boundary_condition
    )
    
    # Train the PINN
    print("\nTraining PINN...")
    start_time = time.time()
    solver.train(
        epochs=12000,
        lr=1e-3,
        weights={'physics': 1.0, 'initial': 100.0, 'boundary': 100.0},
        print_every=1000,
        use_scheduler=True
    )
    elapsed_time = time.time() - start_time
    
    print(f"\nTraining completed in {elapsed_time:.2f} seconds")
    
    # Plot final solution
    print("\nGenerating final solution...")
    X, Y, U_final = solver.predict_grid(nx, ny, t_domain[1])
    
    print(f"Max temperature: {U_final.max():.6f}")
    print(f"Min temperature: {U_final.min():.6f}")
    print(f"Average temperature: {U_final.mean():.6f}")
    
    plot_temperature_field_2d(
        X, Y, U_final,
        title="Checkerboard Final Solution (PINN)",
        save_path="plots/pinn_checkerboard_final.png"
    )
    
    plot_temperature_field_3d(
        X, Y, U_final,
        title="Checkerboard Final Solution 3D (PINN)",
        save_path="plots/pinn_checkerboard_final_3d.png"
    )
    
    # Plot time evolution
    print("\nPlotting time evolution...")
    times = np.linspace(0, t_domain[1], 6)
    plot_time_evolution(
        solver,
        times=times.tolist(),
        nx=nx,
        ny=ny,
        title="Checkerboard Temperature Evolution",
        save_path="plots/pinn_checkerboard_evolution.png"
    )
    
    # Save model
    print("\nSaving trained model...")
    solver.save_model("models/pinn_checkerboard.pth")
    
    print("\n✓ Example 3 completed successfully!")
    print(f"  Plots saved to plots/pinn_checkerboard_*.png")
    print(f"  Model saved to models/pinn_checkerboard.pth")


def performance_benchmark():
    """
    Performance benchmark: Compare training time for different configurations.
    """
    print("\n" + "="*70)
    print("Performance Benchmark: Different problem configurations")
    print("="*70)
    print()
    
    configs = [
        {"name": "Small (50×50)", "nx": 50, "ny": 50, "epochs": 5000},
        {"name": "Medium (100×100)", "nx": 100, "ny": 100, "epochs": 5000},
        {"name": "Large (200×200)", "nx": 200, "ny": 200, "epochs": 5000},
    ]
    
    results = []
    
    for config in configs:
        print(f"\nBenchmark: {config['name']}")
        print("-" * 50)
        
        nx = config['nx']
        ny = config['ny']
        dx = 0.1
        dy = 0.1
        alpha = 1.0
        
        x_domain = (0.0, (nx - 1) * dx)
        y_domain = (0.0, (ny - 1) * dy)
        t_domain = (0.0, 0.1)
        
        # Simple initial condition
        def initial_condition(x, y):
            return x + y
        
        def boundary_condition(x, y, t):
            return 0.0
        
        # Create solver
        solver = HeatEquationPINN(
            x_domain=x_domain,
            y_domain=y_domain,
            t_domain=t_domain,
            alpha=alpha,
            layers=[3, 32, 32, 32, 1]  # Smaller network for benchmark
        )
        
        # Generate training data
        solver.set_training_data(
            n_collocation=5000,
            n_initial=1000,
            n_boundary=1000,
            initial_condition=initial_condition,
            boundary_condition=boundary_condition
        )
        
        # Train
        start_time = time.time()
        solver.train(
            epochs=config['epochs'],
            lr=1e-3,
            weights={'physics': 1.0, 'initial': 50.0, 'boundary': 50.0},
            print_every=5000,
            use_scheduler=False
        )
        elapsed_time = time.time() - start_time
        
        # Get final solution
        X, Y, U = solver.predict_grid(nx, ny, t_domain[1])
        
        total_points = nx * ny
        time_per_epoch = elapsed_time / config['epochs']
        
        result = {
            'name': config['name'],
            'points': total_points,
            'elapsed': elapsed_time,
            'time_per_epoch': time_per_epoch,
            'final_loss': solver.loss_history['total'][-1]
        }
        results.append(result)
        
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Time per epoch: {time_per_epoch:.4f} seconds")
        print(f"Final loss: {result['final_loss']:.6e}")
    
    # Print summary
    print("\n" + "="*70)
    print("Benchmark Summary")
    print("="*70)
    print(f"{'Configuration':<20} {'Points':<12} {'Time (s)':<12} {'Time/Epoch (s)':<15} {'Final Loss':<12}")
    print("-" * 70)
    for result in results:
        print(f"{result['name']:<20} {result['points']:<12} {result['elapsed']:<12.2f} "
              f"{result['time_per_epoch']:<15.4f} {result['final_loss']:<12.6e}")
    
    print("\n✓ Benchmark completed!")


def main():
    """
    Main function to run all examples.
    """
    print("="*70)
    print("2D Heat Equation Solver using Physics-Informed Neural Networks")
    print("="*70)
    print()
    print("This implementation replaces the finite difference method with PINNs.")
    print("PINNs learn the solution by minimizing a loss function that includes:")
    print("  - Physics loss: PDE residual at collocation points")
    print("  - Initial condition loss: Match u(x,y,0)")
    print("  - Boundary condition loss: Match boundary values")
    print()
    
    # Create output directories
    import os
    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Run examples
    try:
        # Example 1: Gaussian pulse
        example_gaussian_pulse()
        
        # Example 2: Sine wave
        example_sine_wave()
        
        # Example 3: Checkerboard
        example_checkerboard()
        
        # Performance benchmark
        performance_benchmark()
        
        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70)
        print("\nGenerated files:")
        print("  - plots/pinn_*.png: Visualization of solutions")
        print("  - models/pinn_*.pth: Trained PINN models")
        print()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()

