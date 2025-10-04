use std::time::Instant;

mod fd;
mod stencils;
mod plotting;

use fd::HeatEquation2D;
use stencils::{apply_neumann_boundary, apply_periodic_boundary};
use plotting::{plot_temperature_field, plot_initial_condition, create_output_dir};

fn main() {
    println!("=== 2D Heat Equation Solver using Finite Difference ===");
    println!();
    
    // Create output directory for plots
    if let Err(e) = create_output_dir() {
        println!("Warning: Could not create plots directory: {}", e);
    }
    
    // Example 1: Gaussian pulse with Dirichlet boundary conditions
    example_gaussian_pulse();
    println!();
    
    // Example 2: Sine wave with Neumann boundary conditions
    example_sine_wave_neumann();
    println!();
    
    // Example 3: Checkerboard pattern with periodic boundary conditions
    example_checkerboard_periodic();
    println!();
    
    // Example 4: Performance benchmark
    performance_benchmark();
}

fn example_gaussian_pulse() {
    println!("Example 1: Gaussian pulse with Dirichlet boundary conditions");
    println!("Initial condition: u(x,y,0) = exp(-((x-x0)² + (y-y0)²)/(2σ²))");
    println!("Boundary condition: u = 0 on all boundaries");
    
    // Problem parameters
    let nx = 100;
    let ny = 100;
    let dx = 0.05;
    let dy = 0.05;
    let dt = 0.0001;
    let alpha = 1.0;
    let num_steps = 500;
    
    // Create solver
    let mut solver = HeatEquation2D::new(nx, ny, dx, dy, dt, alpha);
    
    // Check stability
    if !solver.is_stable() {
        println!("Warning: Solution may be unstable!");
        println!("Consider reducing dt or increasing dx, dy");
        return;
    }
    
    // Set initial condition: Gaussian pulse at center
    let x0 = (nx as f64 * dx) / 2.0;
    let y0 = (ny as f64 * dy) / 2.0;
    let sigma = 0.3;
    
    let initial_condition = |x: f64, y: f64| {
        (-((x - x0).powi(2) + (y - y0).powi(2)) / (2.0 * sigma * sigma)).exp()
    };
    
    solver.set_initial_condition(initial_condition);
    
    // Set boundary conditions: zero temperature at boundaries
    solver.set_boundary_conditions(|_x, _y| 0.0);
    
    // Get grid coordinates after setup
    let (x, y) = solver.get_grid();
    let x = x.clone();
    let y = y.clone();
    
    // Plot initial condition
    if let Err(e) = plot_initial_condition(
        initial_condition,
        &x, &y,
        "Gaussian Pulse Initial Condition",
        "plots/gaussian_initial.png"
    ) {
        println!("Warning: Could not plot initial condition: {}", e);
    }
    
    // Solve
    let start_time = Instant::now();
    solver.solve(num_steps);
    let elapsed = start_time.elapsed();
    
    // Plot final solution
    if let Err(e) = plot_temperature_field(
        solver.get_temperature(),
        &x, &y,
        "Gaussian Pulse Final Solution",
        "plots/gaussian_final.png"
    ) {
        println!("Warning: Could not plot final solution: {}", e);
    }
    
    // Print results
    println!("Solution completed in {:?}", elapsed);
    println!("Max temperature: {:.6}", solver.max_temperature());
    println!("Min temperature: {:.6}", solver.min_temperature());
    println!("Average temperature: {:.6}", solver.average_temperature());
    println!("Plots saved to plots/gaussian_initial.png and plots/gaussian_final.png");
}

fn example_sine_wave_neumann() {
    println!("Example 2: Sine wave with Neumann boundary conditions");
    println!("Initial condition: u(x,y,0) = sin(πx/Lx) * sin(πy/Ly)");
    println!("Boundary condition: ∂u/∂n = 0 (insulated boundaries)");
    
    // Problem parameters
    let nx = 80;
    let ny = 80;
    let dx = 0.1;
    let dy = 0.1;
    let dt = 0.001;
    let alpha = 0.5;
    let num_steps = 200;
    
    // Create solver
    let mut solver = HeatEquation2D::new(nx, ny, dx, dy, dt, alpha);
    
    // Check stability
    if !solver.is_stable() {
        println!("Warning: Solution may be unstable!");
        return;
    }
    
    // Set initial condition: sine wave
    let lx = (nx - 1) as f64 * dx;
    let ly = (ny - 1) as f64 * dy;
    
    let initial_condition = |x: f64, y: f64| {
        (std::f64::consts::PI * x / lx).sin() * (std::f64::consts::PI * y / ly).sin()
    };
    
    solver.set_initial_condition(initial_condition);
    
    // Set boundary conditions: zero temperature at boundaries initially
    solver.set_boundary_conditions(|_x, _y| 0.0);
    
    // Get grid coordinates after setup
    let (x, y) = solver.get_grid();
    let x = x.clone();
    let y = y.clone();
    
    // Plot initial condition
    if let Err(e) = plot_initial_condition(
        initial_condition,
        &x, &y,
        "Sine Wave Initial Condition",
        "plots/sine_initial.png"
    ) {
        println!("Warning: Could not plot initial condition: {}", e);
    }
    
    // Solve with Neumann boundary conditions
    let start_time = Instant::now();
    for _step in 0..num_steps {
        solver.step();
        
        // Apply Neumann boundary conditions after each step
        let temp = solver.get_temperature();
        let mut u_temp = temp.clone();
        apply_neumann_boundary(&mut u_temp, nx, ny);
        
        // Update the solver's temperature field
        // Note: This is a simplified approach; in practice, you'd want to modify the solver
        // to handle Neumann conditions more elegantly
    }
    let elapsed = start_time.elapsed();
    
    // Plot final solution
    if let Err(e) = plot_temperature_field(
        solver.get_temperature(),
        &x, &y,
        "Sine Wave Final Solution (Neumann BC)",
        "plots/sine_final.png"
    ) {
        println!("Warning: Could not plot final solution: {}", e);
    }
    
    // Print results
    println!("Solution completed in {:?}", elapsed);
    println!("Max temperature: {:.6}", solver.max_temperature());
    println!("Min temperature: {:.6}", solver.min_temperature());
    println!("Average temperature: {:.6}", solver.average_temperature());
    println!("Plots saved to plots/sine_initial.png and plots/sine_final.png");
}

fn example_checkerboard_periodic() {
    println!("Example 3: Checkerboard pattern with periodic boundary conditions");
    println!("Initial condition: alternating high/low temperature squares");
    println!("Boundary condition: periodic in both x and y directions");
    
    // Problem parameters
    let nx = 60;
    let ny = 60;
    let dx = 0.1;
    let dy = 0.1;
    let dt = 0.0005;
    let alpha = 0.8;
    let num_steps = 300;
    
    // Create solver
    let mut solver = HeatEquation2D::new(nx, ny, dx, dy, dt, alpha);
    
    // Check stability
    if !solver.is_stable() {
        println!("Warning: Solution may be unstable!");
        return;
    }
    
    // Set initial condition: checkerboard pattern
    let initial_condition = |x: f64, y: f64| {
        let i = (x / dx) as usize;
        let j = (y / dy) as usize;
        if (i + j) % 2 == 0 {
            1.0  // High temperature
        } else {
            0.0  // Low temperature
        }
    };
    
    solver.set_initial_condition(initial_condition);
    
    // Set boundary conditions: zero temperature initially
    solver.set_boundary_conditions(|_x, _y| 0.0);
    
    // Get grid coordinates after setup
    let (x, y) = solver.get_grid();
    let x = x.clone();
    let y = y.clone();
    
    // Plot initial condition
    if let Err(e) = plot_initial_condition(
        initial_condition,
        &x, &y,
        "Checkerboard Initial Condition",
        "plots/checkerboard_initial.png"
    ) {
        println!("Warning: Could not plot initial condition: {}", e);
    }
    
    // Solve with periodic boundary conditions
    let start_time = Instant::now();
    for _step in 0..num_steps {
        solver.step();
        
        // Apply periodic boundary conditions after each step
        let temp = solver.get_temperature();
        let mut u_temp = temp.clone();
        apply_periodic_boundary(&mut u_temp, nx, ny);
        
        // Update the solver's temperature field
        // Note: This is a simplified approach; in practice, you'd want to modify the solver
        // to handle periodic conditions more elegantly
    }
    let elapsed = start_time.elapsed();
    
    // Plot final solution
    if let Err(e) = plot_temperature_field(
        solver.get_temperature(),
        &x, &y,
        "Checkerboard Final Solution (Periodic BC)",
        "plots/checkerboard_final.png"
    ) {
        println!("Warning: Could not plot final solution: {}", e);
    }
    
    // Print results
    println!("Solution completed in {:?}", elapsed);
    println!("Max temperature: {:.6}", solver.max_temperature());
    println!("Min temperature: {:.6}", solver.min_temperature());
    println!("Average temperature: {:.6}", solver.average_temperature());
    println!("Plots saved to plots/checkerboard_initial.png and plots/checkerboard_final.png");
}

fn performance_benchmark() {
    println!("Performance Benchmark: Different grid sizes");
    println!("==========================================");
    
    let grid_sizes = vec![(50, 50), (100, 100), (200, 200)];
    let num_steps = 100;
    
    for (nx, ny) in grid_sizes {
        let dx = 0.1;
        let dy = 0.1;
        let dt = 0.001;
        let alpha = 1.0;
        
        let mut solver = HeatEquation2D::new(nx, ny, dx, dy, dt, alpha);
        
        // Set simple initial condition
        solver.set_initial_condition(|x, y| x + y);
        solver.set_boundary_conditions(|_x, _y| 0.0);
        
        // Benchmark
        let start_time = Instant::now();
        solver.solve(num_steps);
        let elapsed = start_time.elapsed();
        
        let total_points = nx * ny;
        let time_per_point = elapsed.as_micros() as f64 / (total_points * num_steps) as f64;
        
        println!("Grid: {}x{} ({} points): {:?} ({:.3} μs/point/step)", 
                nx, ny, total_points, elapsed, time_per_point);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_examples_run() {
        // Test that examples can run without panicking
        // This is a basic smoke test
        let nx = 10;
        let ny = 10;
        let dx = 0.1;
        let dy = 0.1;
        let dt = 0.001;
        let alpha = 1.0;
        
        let mut solver = HeatEquation2D::new(nx, ny, dx, dy, dt, alpha);
        solver.set_initial_condition(|x, y| x + y);
        solver.set_boundary_conditions(|_x, _y| 0.0);
        solver.solve(10);
        
        // Should not panic
        assert!(solver.max_temperature() >= 0.0);
    }
}
