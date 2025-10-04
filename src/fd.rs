/// 2D Heat Equation Solver using Finite Difference
/// 
/// The heat equation in 2D is: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
/// where:
/// - u(x,y,t) is the temperature at position (x,y) and time t
/// - α is the thermal diffusivity coefficient
/// - We use explicit finite difference scheme with forward Euler in time
///   and central difference in space
pub struct HeatEquation2D {
    /// Grid dimensions
    nx: usize,
    ny: usize,
    
    /// Spatial step sizes
    dx: f64,
    dy: f64,
    
    /// Time step size
    dt: f64,
    
    /// Thermal diffusivity
    alpha: f64,
    
    /// Current temperature field (2D grid)
    u: Vec<Vec<f64>>,
    
    /// Previous temperature field for time stepping
    u_prev: Vec<Vec<f64>>,
    
    /// Grid coordinates
    x: Vec<f64>,
    y: Vec<f64>,
}

impl HeatEquation2D {
    /// Create a new 2D heat equation solver
    pub fn new(nx: usize, ny: usize, dx: f64, dy: f64, dt: f64, alpha: f64) -> Self {
        let mut solver = Self {
            nx,
            ny,
            dx,
            dy,
            dt,
            alpha,
            u: vec![vec![0.0; ny]; nx],
            u_prev: vec![vec![0.0; ny]; nx],
            x: vec![0.0; nx],
            y: vec![0.0; ny],
        };
        
        // Initialize grid coordinates
        for i in 0..nx {
            solver.x[i] = i as f64 * dx;
        }
        for j in 0..ny {
            solver.y[j] = j as f64 * dy;
        }
        
        solver
    }
    
    /// Set initial temperature distribution
    pub fn set_initial_condition<F>(&mut self, initial_func: F)
    where
        F: Fn(f64, f64) -> f64,
    {
        for i in 0..self.nx {
            for j in 0..self.ny {
                self.u[i][j] = initial_func(self.x[i], self.y[j]);
                self.u_prev[i][j] = self.u[i][j];
            }
        }
    }
    
    /// Set boundary conditions
    pub fn set_boundary_conditions<F>(&mut self, boundary_func: F)
    where
        F: Fn(f64, f64) -> f64,
    {
        // Left boundary (x = 0)
        for j in 0..self.ny {
            self.u[0][j] = boundary_func(self.x[0], self.y[j]);
            self.u_prev[0][j] = self.u[0][j];
        }
        
        // Right boundary (x = Lx)
        for j in 0..self.ny {
            self.u[self.nx - 1][j] = boundary_func(self.x[self.nx - 1], self.y[j]);
            self.u_prev[self.nx - 1][j] = self.u[self.nx - 1][j];
        }
        
        // Bottom boundary (y = 0)
        for i in 0..self.nx {
            self.u[i][0] = boundary_func(self.x[i], self.y[0]);
            self.u_prev[i][0] = self.u[i][0];
        }
        
        // Top boundary (y = Ly)
        for i in 0..self.nx {
            self.u[i][self.ny - 1] = boundary_func(self.x[i], self.y[self.ny - 1]);
            self.u_prev[i][self.ny - 1] = self.u[i][self.ny - 1];
        }
    }
    
    /// Perform one time step using explicit finite difference
    pub fn step(&mut self) {
        // Copy current solution to previous
        for i in 0..self.nx {
            for j in 0..self.ny {
                self.u_prev[i][j] = self.u[i][j];
            }
        }
        
        // Update interior points using finite difference scheme
        for i in 1..self.nx - 1 {
            for j in 1..self.ny - 1 {
                let d2u_dx2 = (self.u_prev[i + 1][j] - 2.0 * self.u_prev[i][j] + self.u_prev[i - 1][j]) / (self.dx * self.dx);
                let d2u_dy2 = (self.u_prev[i][j + 1] - 2.0 * self.u_prev[i][j] + self.u_prev[i][j - 1]) / (self.dy * self.dy);
                
                self.u[i][j] = self.u_prev[i][j] + self.alpha * self.dt * (d2u_dx2 + d2u_dy2);
            }
        }
    }
    
    /// Solve for a given number of time steps
    pub fn solve(&mut self, num_steps: usize) {
        for _ in 0..num_steps {
            self.step();
        }
    }
    
    /// Get current temperature field
    pub fn get_temperature(&self) -> &Vec<Vec<f64>> {
        &self.u
    }
    
    /// Get grid coordinates
    #[allow(dead_code)]
    pub fn get_grid(&self) -> (&Vec<f64>, &Vec<f64>) {
        (&self.x, &self.y)
    }
    
    /// Check if the solution is stable (CFL condition)
    pub fn is_stable(&self) -> bool {
        let stability_criterion = self.alpha * self.dt * (1.0 / (self.dx * self.dx) + 1.0 / (self.dy * self.dy));
        stability_criterion <= 0.5
    }
    
    /// Calculate the maximum temperature
    pub fn max_temperature(&self) -> f64 {
        self.u.iter()
            .flat_map(|row| row.iter())
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    }
    
    /// Calculate the minimum temperature
    pub fn min_temperature(&self) -> f64 {
        self.u.iter()
            .flat_map(|row| row.iter())
            .fold(f64::INFINITY, |a, &b| a.min(b))
    }
    
    /// Calculate the average temperature
    pub fn average_temperature(&self) -> f64 {
        let sum: f64 = self.u.iter()
            .flat_map(|row| row.iter())
            .sum();
        sum / ((self.nx * self.ny) as f64)
    }
}

/// Example usage and test functions
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_initialization() {
        let solver = HeatEquation2D::new(10, 10, 0.1, 0.1, 0.001, 1.0);
        assert_eq!(solver.nx, 10);
        assert_eq!(solver.ny, 10);
        assert_eq!(solver.dx, 0.1);
        assert_eq!(solver.dy, 0.1);
        assert_eq!(solver.dt, 0.001);
        assert_eq!(solver.alpha, 1.0);
    }
    
    #[test]
    fn test_stability() {
        let solver = HeatEquation2D::new(10, 10, 0.1, 0.1, 0.001, 1.0);
        assert!(solver.is_stable());
        
        let unstable_solver = HeatEquation2D::new(10, 10, 0.1, 0.1, 0.01, 1.0);
        assert!(!unstable_solver.is_stable());
    }
    
    #[test]
    fn test_initial_condition() {
        let mut solver = HeatEquation2D::new(5, 5, 1.0, 1.0, 0.1, 1.0);
        
        // Set initial condition: u(x,y) = x + y
        solver.set_initial_condition(|x, y| x + y);
        
        assert_eq!(solver.u[0][0], 0.0);  // u(0,0) = 0
        assert_eq!(solver.u[1][1], 2.0);  // u(1,1) = 2
        assert_eq!(solver.u[4][4], 8.0);  // u(4,4) = 8
    }
    
    #[test]
    fn test_boundary_conditions() {
        let mut solver = HeatEquation2D::new(5, 5, 1.0, 1.0, 0.1, 1.0);
        
        // Set boundary condition: u(x,y) = x^2 + y^2
        solver.set_boundary_conditions(|x, y| x * x + y * y);
        
        // Check left boundary
        assert_eq!(solver.u[0][0], 0.0);   // u(0,0) = 0
        assert_eq!(solver.u[0][1], 1.0);   // u(0,1) = 1
        assert_eq!(solver.u[0][4], 16.0);  // u(0,4) = 16
        
        // Check bottom boundary
        assert_eq!(solver.u[1][0], 1.0);   // u(1,0) = 1
        assert_eq!(solver.u[4][0], 16.0);  // u(4,0) = 16
    }
}

/// Example main function showing how to use the solver
#[allow(dead_code)]
pub fn example_usage() {
    // Problem parameters
    let nx = 50;
    let ny = 50;
    let dx = 0.1;
    let dy = 0.1;
    let dt = 0.001;
    let alpha = 1.0;
    let num_steps = 1000;
    
    // Create solver
    let mut solver = HeatEquation2D::new(nx, ny, dx, dy, dt, alpha);
    
    // Check stability
    if !solver.is_stable() {
        println!("Warning: Solution may be unstable!");
        println!("Consider reducing dt or increasing dx, dy");
    }
    
    // Set initial condition: Gaussian pulse at center
    solver.set_initial_condition(|x, y| {
        let x0 = (nx as f64 * dx) / 2.0;
        let y0 = (ny as f64 * dy) / 2.0;
        let sigma = 0.5;
        (-((x - x0).powi(2) + (y - y0).powi(2)) / (2.0 * sigma * sigma)).exp()
    });
    
    // Set boundary conditions: zero temperature at boundaries
    solver.set_boundary_conditions(|_x, _y| 0.0);
    
    // Solve
    println!("Solving 2D heat equation...");
    solver.solve(num_steps);
    
    // Print results
    println!("Solution completed!");
    println!("Max temperature: {:.6}", solver.max_temperature());
    println!("Min temperature: {:.6}", solver.min_temperature());
    println!("Average temperature: {:.6}", solver.average_temperature());
}
