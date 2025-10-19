"""
Physics-Informed Neural Network (PINN) for 2D Heat Equation

This module implements a PINN to solve the 2D heat equation:
∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)

where:
- u(x,y,t) is the temperature at position (x,y) and time t
- α is the thermal diffusivity coefficient
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Tuple, Optional
import matplotlib.pyplot as plt
from torch.autograd import grad


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for solving the 2D heat equation.
    
    The network takes (x, y, t) as input and outputs u(x, y, t).
    """
    
    def __init__(self, layers: list, activation=nn.Tanh()):
        """
        Initialize the PINN.
        
        Args:
            layers: List of layer sizes [input_dim, hidden1, hidden2, ..., output_dim]
            activation: Activation function to use between layers
        """
        super(PINN, self).__init__()
        
        self.activation = activation
        self.layers_list = nn.ModuleList()
        
        # Create neural network layers
        for i in range(len(layers) - 1):
            self.layers_list.append(nn.Linear(layers[i], layers[i + 1]))
            
            # Xavier initialization
            nn.init.xavier_normal_(self.layers_list[-1].weight)
            nn.init.zeros_(self.layers_list[-1].bias)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Args:
            x, y, t: Input coordinates (each is a tensor of shape [N, 1])
        
        Returns:
            u: Network output (temperature field) of shape [N, 1]
        """
        # Concatenate inputs
        inputs = torch.cat([x, y, t], dim=1)
        
        # Pass through layers
        out = inputs
        for i, layer in enumerate(self.layers_list[:-1]):
            out = self.activation(layer(out))
        
        # Output layer (no activation)
        out = self.layers_list[-1](out)
        
        return out
    
    def physics_loss(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, 
                    alpha: float) -> torch.Tensor:
        """
        Compute the physics loss based on the PDE residual.
        
        The PDE is: ∂u/∂t - α(∂²u/∂x² + ∂²u/∂y²) = 0
        
        Args:
            x, y, t: Collocation points
            alpha: Thermal diffusivity
        
        Returns:
            Physics loss (MSE of PDE residual)
        """
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)
        
        # Forward pass
        u = self.forward(x, y, t)
        
        # First-order derivatives
        u_t = grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        # Second-order derivatives
        u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        
        # PDE residual
        pde_residual = u_t - alpha * (u_xx + u_yy)
        
        return torch.mean(pde_residual ** 2)
    
    def initial_condition_loss(self, x: torch.Tensor, y: torch.Tensor, 
                              u_initial: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for initial condition at t=0.
        
        Args:
            x, y: Spatial coordinates
            u_initial: True initial condition values
        
        Returns:
            Initial condition loss (MSE)
        """
        t = torch.zeros_like(x)
        u_pred = self.forward(x, y, t)
        return torch.mean((u_pred - u_initial) ** 2)
    
    def boundary_condition_loss(self, x_bc: torch.Tensor, y_bc: torch.Tensor, 
                               t_bc: torch.Tensor, u_bc: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for boundary conditions.
        
        Args:
            x_bc, y_bc, t_bc: Boundary coordinates
            u_bc: True boundary values
        
        Returns:
            Boundary condition loss (MSE)
        """
        u_pred = self.forward(x_bc, y_bc, t_bc)
        return torch.mean((u_pred - u_bc) ** 2)


class HeatEquationPINN:
    """
    Solver for 2D heat equation using Physics-Informed Neural Networks.
    """
    
    def __init__(self, 
                 x_domain: Tuple[float, float],
                 y_domain: Tuple[float, float],
                 t_domain: Tuple[float, float],
                 alpha: float,
                 layers: Optional[list] = None,
                 device: Optional[str] = None):
        """
        Initialize the PINN solver.
        
        Args:
            x_domain: (x_min, x_max) spatial domain in x
            y_domain: (y_min, y_max) spatial domain in y
            t_domain: (t_min, t_max) temporal domain
            alpha: Thermal diffusivity
            layers: Network architecture [input, hidden..., output]
            device: 'cuda' or 'cpu'
        """
        self.x_domain = x_domain
        self.y_domain = y_domain
        self.t_domain = t_domain
        self.alpha = alpha
        
        # Default architecture: 3 inputs (x,y,t) -> hidden layers -> 1 output (u)
        if layers is None:
            layers = [3, 64, 64, 64, 64, 1]
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Create network
        self.model = PINN(layers).to(self.device)
        
        # Optimizer
        self.optimizer = None
        self.scheduler = None
        
        # Training history
        self.loss_history = {
            'total': [],
            'physics': [],
            'initial': [],
            'boundary': []
        }
    
    def set_training_data(self,
                         n_collocation: int = 10000,
                         n_initial: int = 1000,
                         n_boundary: int = 1000,
                         initial_condition: Optional[Callable] = None,
                         boundary_condition: Optional[Callable] = None):
        """
        Generate training data points.
        
        Args:
            n_collocation: Number of collocation points for physics loss
            n_initial: Number of points for initial condition
            n_boundary: Number of points for boundary condition
            initial_condition: Function u(x, y) at t=0
            boundary_condition: Function u(x, y, t) at boundaries
        """
        # Collocation points (interior domain)
        x_col = torch.rand(n_collocation, 1, device=self.device) * (self.x_domain[1] - self.x_domain[0]) + self.x_domain[0]
        y_col = torch.rand(n_collocation, 1, device=self.device) * (self.y_domain[1] - self.y_domain[0]) + self.y_domain[0]
        t_col = torch.rand(n_collocation, 1, device=self.device) * (self.t_domain[1] - self.t_domain[0]) + self.t_domain[0]
        
        self.collocation_points = (x_col, y_col, t_col)
        
        # Initial condition points (t=0)
        x_ic = torch.rand(n_initial, 1, device=self.device) * (self.x_domain[1] - self.x_domain[0]) + self.x_domain[0]
        y_ic = torch.rand(n_initial, 1, device=self.device) * (self.y_domain[1] - self.y_domain[0]) + self.y_domain[0]
        
        if initial_condition is not None:
            u_ic = torch.tensor([[initial_condition(x.item(), y.item())] 
                               for x, y in zip(x_ic, y_ic)], 
                               dtype=torch.float32, device=self.device)
        else:
            u_ic = torch.zeros_like(x_ic)
        
        self.initial_data = (x_ic, y_ic, u_ic)
        
        # Boundary condition points
        n_per_boundary = n_boundary // 8  # 4 boundaries × 2 edges per boundary
        
        x_bc_list = []
        y_bc_list = []
        t_bc_list = []
        u_bc_list = []
        
        # Generate boundary points for all four edges at various times
        for _ in range(2):  # Multiple time samples
            t_sample = torch.rand(n_per_boundary, 1, device=self.device) * (self.t_domain[1] - self.t_domain[0]) + self.t_domain[0]
            
            # Left boundary (x = x_min)
            y_temp = torch.rand(n_per_boundary, 1, device=self.device) * (self.y_domain[1] - self.y_domain[0]) + self.y_domain[0]
            x_bc_list.append(torch.full_like(y_temp, self.x_domain[0]))
            y_bc_list.append(y_temp)
            t_bc_list.append(t_sample.clone())
            
            # Right boundary (x = x_max)
            y_temp = torch.rand(n_per_boundary, 1, device=self.device) * (self.y_domain[1] - self.y_domain[0]) + self.y_domain[0]
            x_bc_list.append(torch.full_like(y_temp, self.x_domain[1]))
            y_bc_list.append(y_temp)
            t_bc_list.append(t_sample.clone())
            
            # Bottom boundary (y = y_min)
            x_temp = torch.rand(n_per_boundary, 1, device=self.device) * (self.x_domain[1] - self.x_domain[0]) + self.x_domain[0]
            x_bc_list.append(x_temp)
            y_bc_list.append(torch.full_like(x_temp, self.y_domain[0]))
            t_bc_list.append(t_sample.clone())
            
            # Top boundary (y = y_max)
            x_temp = torch.rand(n_per_boundary, 1, device=self.device) * (self.x_domain[1] - self.x_domain[0]) + self.x_domain[0]
            x_bc_list.append(x_temp)
            y_bc_list.append(torch.full_like(x_temp, self.y_domain[1]))
            t_bc_list.append(t_sample.clone())
        
        x_bc = torch.cat(x_bc_list, dim=0)
        y_bc = torch.cat(y_bc_list, dim=0)
        t_bc = torch.cat(t_bc_list, dim=0)
        
        if boundary_condition is not None:
            u_bc = torch.tensor([[boundary_condition(x.item(), y.item(), t.item())] 
                               for x, y, t in zip(x_bc, y_bc, t_bc)],
                               dtype=torch.float32, device=self.device)
        else:
            u_bc = torch.zeros_like(x_bc)
        
        self.boundary_data = (x_bc, y_bc, t_bc, u_bc)
    
    def train(self, 
              epochs: int = 10000,
              lr: float = 1e-3,
              weights: Optional[dict] = None,
              print_every: int = 1000,
              use_scheduler: bool = True):
        """
        Train the PINN.
        
        Args:
            epochs: Number of training epochs
            lr: Learning rate
            weights: Dictionary of loss weights {'physics': w1, 'initial': w2, 'boundary': w3}
            print_every: Print progress every N epochs
            use_scheduler: Use learning rate scheduler
        """
        # Default weights
        if weights is None:
            weights = {'physics': 1.0, 'initial': 100.0, 'boundary': 100.0}
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Learning rate scheduler
        if use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=1000, verbose=False
            )
        
        print(f"\nTraining PINN for {epochs} epochs...")
        print(f"Loss weights: {weights}")
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Physics loss (PDE residual)
            x_col, y_col, t_col = self.collocation_points
            loss_physics = self.model.physics_loss(x_col, y_col, t_col, self.alpha)
            
            # Initial condition loss
            x_ic, y_ic, u_ic = self.initial_data
            loss_initial = self.model.initial_condition_loss(x_ic, y_ic, u_ic)
            
            # Boundary condition loss
            x_bc, y_bc, t_bc, u_bc = self.boundary_data
            loss_boundary = self.model.boundary_condition_loss(x_bc, y_bc, t_bc, u_bc)
            
            # Total loss
            loss = (weights['physics'] * loss_physics + 
                   weights['initial'] * loss_initial + 
                   weights['boundary'] * loss_boundary)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Learning rate scheduling
            if use_scheduler and self.scheduler is not None:
                self.scheduler.step(loss)
            
            # Store history
            self.loss_history['total'].append(loss.item())
            self.loss_history['physics'].append(loss_physics.item())
            self.loss_history['initial'].append(loss_initial.item())
            self.loss_history['boundary'].append(loss_boundary.item())
            
            # Print progress
            if (epoch + 1) % print_every == 0 or epoch == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1}/{epochs} | "
                      f"Loss: {loss.item():.6e} | "
                      f"Physics: {loss_physics.item():.6e} | "
                      f"Initial: {loss_initial.item():.6e} | "
                      f"Boundary: {loss_boundary.item():.6e} | "
                      f"LR: {current_lr:.2e}")
        
        print("\nTraining completed!")
    
    def predict(self, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Predict temperature field at given points.
        
        Args:
            x, y, t: Coordinates (numpy arrays)
        
        Returns:
            u: Predicted temperature (numpy array)
        """
        self.model.eval()
        
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device).reshape(-1, 1)
            y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).reshape(-1, 1)
            t_tensor = torch.tensor(t, dtype=torch.float32, device=self.device).reshape(-1, 1)
            
            u_tensor = self.model(x_tensor, y_tensor, t_tensor)
            u = u_tensor.cpu().numpy().flatten()
        
        self.model.train()
        return u
    
    def predict_grid(self, nx: int, ny: int, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict temperature on a regular grid at time t.
        
        Args:
            nx, ny: Grid resolution
            t: Time
        
        Returns:
            X, Y, U: Meshgrid coordinates and temperature field
        """
        x = np.linspace(self.x_domain[0], self.x_domain[1], nx)
        y = np.linspace(self.y_domain[0], self.y_domain[1], ny)
        X, Y = np.meshgrid(x, y)
        
        x_flat = X.flatten()
        y_flat = Y.flatten()
        t_flat = np.full_like(x_flat, t)
        
        u_flat = self.predict(x_flat, y_flat, t_flat)
        U = u_flat.reshape(X.shape)
        
        return X, Y, U
    
    def plot_loss_history(self, save_path: Optional[str] = None):
        """
        Plot training loss history.
        
        Args:
            save_path: Path to save figure (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].semilogy(self.loss_history['total'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        axes[0, 1].semilogy(self.loss_history['physics'])
        axes[0, 1].set_title('Physics Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        axes[1, 0].semilogy(self.loss_history['initial'])
        axes[1, 0].set_title('Initial Condition Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
        
        axes[1, 1].semilogy(self.loss_history['boundary'])
        axes[1, 1].set_title('Boundary Condition Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Loss history saved to {save_path}")
        
        plt.show()
    
    def save_model(self, path: str):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'loss_history': self.loss_history,
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_history = checkpoint['loss_history']
        print(f"Model loaded from {path}")

