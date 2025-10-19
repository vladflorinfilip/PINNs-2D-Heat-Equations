"""
Visualization utilities for PINN solutions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable, Optional
import os


def plot_temperature_field_2d(X: np.ndarray, 
                              Y: np.ndarray, 
                              U: np.ndarray,
                              title: str = "Temperature Field",
                              save_path: Optional[str] = None,
                              show_colorbar: bool = True,
                              cmap: str = 'hot'):
    """
    Plot 2D temperature field as a heatmap.
    
    Args:
        X, Y: Meshgrid coordinates
        U: Temperature field
        title: Plot title
        save_path: Path to save figure (optional)
        show_colorbar: Whether to show colorbar
        cmap: Colormap name
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.contourf(X, Y, U, levels=50, cmap=cmap)
    
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Temperature', rotation=270, labelpad=20)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_temperature_field_3d(X: np.ndarray,
                              Y: np.ndarray,
                              U: np.ndarray,
                              title: str = "Temperature Field",
                              save_path: Optional[str] = None,
                              cmap: str = 'hot',
                              elev: float = 30,
                              azim: float = 45):
    """
    Plot 3D surface of temperature field.
    
    Args:
        X, Y: Meshgrid coordinates
        U: Temperature field
        title: Plot title
        save_path: Path to save figure (optional)
        cmap: Colormap name
        elev: Elevation angle
        azim: Azimuthal angle
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create surface plot
    surf = ax.plot_surface(X, Y, U, cmap=cmap, linewidth=0, 
                          antialiased=True, alpha=0.9)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Temperature')
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_comparison(X: np.ndarray,
                   Y: np.ndarray,
                   U_pinn: np.ndarray,
                   U_exact: np.ndarray,
                   title: str = "PINN vs Exact Solution",
                   save_path: Optional[str] = None,
                   cmap: str = 'hot'):
    """
    Plot comparison between PINN solution and exact/reference solution.
    
    Args:
        X, Y: Meshgrid coordinates
        U_pinn: PINN predicted temperature
        U_exact: Exact or reference temperature
        title: Plot title
        save_path: Path to save figure
        cmap: Colormap name
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # PINN solution
    im1 = axes[0].contourf(X, Y, U_pinn, levels=50, cmap=cmap)
    axes[0].set_title('PINN Solution')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0])
    
    # Exact solution
    im2 = axes[1].contourf(X, Y, U_exact, levels=50, cmap=cmap)
    axes[1].set_title('Reference/Exact Solution')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[1])
    
    # Error
    error = np.abs(U_pinn - U_exact)
    im3 = axes[2].contourf(X, Y, error, levels=50, cmap='viridis')
    axes[2].set_title(f'Absolute Error (Mean: {np.mean(error):.2e})')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_aspect('equal')
    plt.colorbar(im3, ax=axes[2])
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()


def plot_time_evolution(solver,
                       times: list,
                       nx: int = 100,
                       ny: int = 100,
                       title: str = "Temperature Evolution",
                       save_path: Optional[str] = None,
                       cmap: str = 'hot'):
    """
    Plot temperature field at multiple time snapshots.
    
    Args:
        solver: Trained HeatEquationPINN solver
        times: List of time values to plot
        nx, ny: Grid resolution
        title: Plot title
        save_path: Path to save figure
        cmap: Colormap name
    """
    n_times = len(times)
    ncols = min(n_times, 4)
    nrows = (n_times + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    # Find global min/max for consistent colorbar
    vmin, vmax = float('inf'), float('-inf')
    solutions = []
    
    for t in times:
        X, Y, U = solver.predict_grid(nx, ny, t)
        solutions.append((X, Y, U))
        vmin = min(vmin, U.min())
        vmax = max(vmax, U.max())
    
    # Plot each time snapshot
    for idx, (t, (X, Y, U)) in enumerate(zip(times, solutions)):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        
        im = ax.contourf(X, Y, U, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f't = {t:.4f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        
        if col == ncols - 1 or idx == n_times - 1:
            plt.colorbar(im, ax=ax)
    
    # Hide extra subplots
    for idx in range(n_times, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Time evolution plot saved to {save_path}")
    
    plt.show()


def plot_cross_section(solver,
                      t: float,
                      y_value: float,
                      nx: int = 100,
                      title: Optional[str] = None,
                      save_path: Optional[str] = None):
    """
    Plot cross-section of temperature field at fixed y.
    
    Args:
        solver: Trained HeatEquationPINN solver
        t: Time value
        y_value: Fixed y coordinate for cross-section
        nx: Number of x points
        title: Plot title
        save_path: Path to save figure
    """
    x = np.linspace(solver.x_domain[0], solver.x_domain[1], nx)
    y = np.full_like(x, y_value)
    t_array = np.full_like(x, t)
    
    u = solver.predict(x, y, t_array)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, u, 'b-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('Temperature')
    
    if title is None:
        title = f'Temperature Cross-Section at y={y_value:.2f}, t={t:.4f}'
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Cross-section plot saved to {save_path}")
    
    plt.show()


def plot_initial_condition(initial_func: Callable,
                          x_domain: tuple,
                          y_domain: tuple,
                          nx: int = 100,
                          ny: int = 100,
                          title: str = "Initial Condition",
                          save_path: Optional[str] = None,
                          cmap: str = 'hot'):
    """
    Plot the initial condition function.
    
    Args:
        initial_func: Function u(x, y) at t=0
        x_domain: (x_min, x_max)
        y_domain: (y_min, y_max)
        nx, ny: Grid resolution
        title: Plot title
        save_path: Path to save figure
        cmap: Colormap name
    """
    x = np.linspace(x_domain[0], x_domain[1], nx)
    y = np.linspace(y_domain[0], y_domain[1], ny)
    X, Y = np.meshgrid(x, y)
    
    U = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            U[i, j] = initial_func(X[i, j], Y[i, j])
    
    plot_temperature_field_2d(X, Y, U, title=title, save_path=save_path, cmap=cmap)


def create_animation(solver,
                    t_start: float,
                    t_end: float,
                    n_frames: int = 50,
                    nx: int = 100,
                    ny: int = 100,
                    save_path: Optional[str] = None,
                    fps: int = 10,
                    cmap: str = 'hot'):
    """
    Create animation of temperature field evolution.
    
    Args:
        solver: Trained HeatEquationPINN solver
        t_start, t_end: Time range
        n_frames: Number of frames
        nx, ny: Grid resolution
        save_path: Path to save animation (e.g., 'animation.gif')
        fps: Frames per second
        cmap: Colormap name
    """
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        print("Error: matplotlib.animation not available. Install pillow for GIF support.")
        return
    
    times = np.linspace(t_start, t_end, n_frames)
    
    # Precompute all solutions for consistent colorbar
    solutions = []
    vmin, vmax = float('inf'), float('-inf')
    
    print("Generating frames...")
    for t in times:
        X, Y, U = solver.predict_grid(nx, ny, t)
        solutions.append((X, Y, U))
        vmin = min(vmin, U.min())
        vmax = max(vmax, U.max())
    
    # Create animation
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def update(frame):
        ax.clear()
        X, Y, U = solutions[frame]
        im = ax.contourf(X, Y, U, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Temperature Field at t = {times[frame]:.4f}')
        ax.set_aspect('equal')
        return [im]
    
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
        print(f"Animation saved to {save_path}")
    
    plt.show()

