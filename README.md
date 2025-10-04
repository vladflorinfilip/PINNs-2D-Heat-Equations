# 2D Heat Equation Solver using Finite Difference Methods

This Rust project implements a numerical solver for the 2D heat equation using finite difference methods. The heat equation is a fundamental partial differential equation that describes how heat diffuses through a medium over time.

## Mathematical Background

The 2D heat equation is:

```
∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
```

where:
- `u(x,y,t)` is the temperature at position `(x,y)` and time `t`
- `α` is the thermal diffusivity coefficient
- The spatial derivatives are approximated using finite differences
- Time integration uses the explicit forward Euler method

## Features

- **Explicit Finite Difference Scheme**: Second-order accurate in space, first-order in time
- **Flexible Boundary Conditions**: Support for Dirichlet, Neumann, and periodic boundary conditions
- **Stability Checking**: Automatic verification of the CFL stability condition
- **High-Order Stencils**: Fourth-order accurate finite difference approximations where possible
- **Performance Optimized**: Efficient memory layout and computation
- **Comprehensive Testing**: Unit tests and benchmarks included

## Project Structure

```
src/
├── main.rs          # Main program with examples
├── fd.rs            # Core heat equation solver
└── stencils.rs      # Finite difference stencils and utilities
```

## Installation and Usage

### Prerequisites

- Rust 1.70+ (install from [rustup.rs](https://rustup.rs/))

### Building and Running

1. **Clone and build the project:**
   ```bash
   git clone <repository-url>
   cd PINNs-Cylinder-Flow
   cargo build --release
   ```

2. **Run the examples:**
   ```bash
   cargo run --release
   ```

3. **Run tests:**
   ```bash
   cargo test
   ```

4. **Run benchmarks:**
   ```bash
   cargo bench
   ```

## Usage Examples

### Basic Usage

```rust
use fd::HeatEquation2D;

// Create solver with 100x100 grid
let mut solver = HeatEquation2D::new(100, 100, 0.1, 0.1, 0.001, 1.0);

// Set initial condition (Gaussian pulse)
solver.set_initial_condition(|x, y| {
    let x0 = 5.0;
    let y0 = 5.0;
    let sigma = 0.5;
    (-((x - x0).powi(2) + (y - y0).powi(2)) / (2.0 * sigma * sigma)).exp()
});

// Set boundary conditions (zero temperature)
solver.set_boundary_conditions(|_x, _y| 0.0);

// Solve for 1000 time steps
solver.solve(1000);

// Get results
println!("Max temperature: {}", solver.max_temperature());
println!("Min temperature: {}", solver.min_temperature());
```

### Different Boundary Conditions

#### Dirichlet (Fixed Temperature)
```rust
// Fixed temperature at boundaries
solver.set_boundary_conditions(|x, y| x + y);
```

#### Neumann (Insulated Boundaries)
```rust
use stencils::apply_neumann_boundary;

// After each time step
solver.step();
let temp = solver.get_temperature();
let mut u_temp = temp.clone();
apply_neumann_boundary(&mut u_temp, nx, ny);
```

#### Periodic Boundaries
```rust
use stencils::apply_periodic_boundary;

// After each time step
solver.step();
let temp = solver.get_temperature();
let mut u_temp = temp.clone();
apply_periodic_boundary(&mut u_temp, nx, ny);
```

## Examples Included

1. **Gaussian Pulse**: Heat diffusion from a central hot spot
2. **Sine Wave**: Oscillating temperature pattern with insulated boundaries
3. **Checkerboard**: Alternating high/low temperature squares with periodic boundaries
4. **Performance Benchmark**: Timing comparison for different grid sizes

## Stability and Accuracy

### Stability Condition
The explicit scheme is stable when:
```
α * dt * (1/dx² + 1/dy²) ≤ 0.5
```

The solver automatically checks this condition and warns if violated.

### Accuracy
- **Spatial**: Second-order accurate (O(dx², dy²))
- **Temporal**: First-order accurate (O(dt))
- **Fourth-order**: Available for interior points using `laplacian_4th()`

## Performance Considerations

- **Memory Layout**: 2D arrays stored as `Vec<Vec<f64>>` for clarity
- **Optimization**: Release builds use aggressive optimization flags
- **Benchmarking**: Built-in performance measurement tools
- **Scalability**: Linear scaling with grid size and time steps

## Extending the Code

### Adding New Initial Conditions
```rust
// Custom initial condition function
solver.set_initial_condition(|x, y| {
    // Your custom function here
    x.powi(2) + y.powi(2)
});
```

### Custom Boundary Conditions
```rust
// Time-dependent boundary conditions
solver.set_boundary_conditions(|x, y| {
    let time = current_time;
    x * y * time
});
```

### Alternative Time Stepping
```rust
// Custom time integration
for step in 0..num_steps {
    solver.step();
    
    // Apply custom modifications
    let temp = solver.get_temperature();
    // ... custom logic ...
}
```

## Mathematical Details

### Finite Difference Stencils

#### Second-Order Laplacian
```
∂²u/∂x² ≈ (u[i+1,j] - 2u[i,j] + u[i-1,j]) / dx²
∂²u/∂y² ≈ (u[i,j+1] - 2u[i,j] + u[i,j-1]) / dy²
```

#### Fourth-Order Laplacian
```
∂²u/∂x² ≈ (-u[i+2,j] + 16u[i+1,j] - 30u[i,j] + 16u[i-1,j] - u[i-2,j]) / (12dx²)
```

### Time Integration
```
u[i,j]^{n+1} = u[i,j]^n + α * dt * (∂²u/∂x² + ∂²u/∂y²)
```

## Troubleshooting

### Common Issues

1. **Unstable Solution**: Reduce `dt` or increase `dx`, `dy`
2. **Slow Performance**: Use release builds (`cargo build --release`)
3. **Memory Issues**: Reduce grid size for large problems

### Debugging
- Enable debug builds: `cargo build`
- Run tests: `cargo test`
- Check stability: `solver.is_stable()`

## Contributing

Contributions are welcome! Areas for improvement:
- Implicit time integration schemes
- Adaptive mesh refinement
- Parallel computing support
- Visualization tools
- More boundary condition types

## License

This project is open source. Please check the LICENSE file for details.

## References

- LeVeque, R. J. (2007). Finite Difference Methods for Ordinary and Partial Differential Equations
- Morton, K. W., & Mayers, D. F. (2005). Numerical Solution of Partial Differential Equations
- Press, W. H., et al. (2007). Numerical Recipes: The Art of Scientific Computing
