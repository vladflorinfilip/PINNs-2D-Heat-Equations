/// Finite Difference Stencils for 2D Heat Equation
/// 
/// This module provides various finite difference approximations for
/// spatial derivatives used in solving partial differential equations.

/// First-order forward difference for ∂u/∂x
#[allow(dead_code)]
pub fn forward_diff_x(u: &[Vec<f64>], i: usize, j: usize, dx: f64) -> f64 {
    if i + 1 < u.len() {
        (u[i + 1][j] - u[i][j]) / dx
    } else {
        // Use backward difference at boundary
        (u[i][j] - u[i - 1][j]) / dx
    }
}

/// First-order backward difference for ∂u/∂x
#[allow(dead_code)]
pub fn backward_diff_x(u: &[Vec<f64>], i: usize, j: usize, dx: f64) -> f64 {
    if i > 0 {
        (u[i][j] - u[i - 1][j]) / dx
    } else {
        // Use forward difference at boundary
        (u[i + 1][j] - u[i][j]) / dx
    }
}

/// Second-order central difference for ∂²u/∂x²
#[allow(dead_code)]
pub fn central_diff_xx(u: &[Vec<f64>], i: usize, j: usize, dx: f64) -> f64 {
    if i > 0 && i + 1 < u.len() {
        (u[i + 1][j] - 2.0 * u[i][j] + u[i - 1][j]) / (dx * dx)
    } else if i == 0 {
        // Forward difference at left boundary
        (u[i + 2][j] - 2.0 * u[i + 1][j] + u[i][j]) / (dx * dx)
    } else {
        // Backward difference at right boundary
        (u[i][j] - 2.0 * u[i - 1][j] + u[i - 2][j]) / (dx * dx)
    }
}

/// Second-order central difference for ∂²u/∂y²
#[allow(dead_code)]
pub fn central_diff_yy(u: &[Vec<f64>], i: usize, j: usize, dy: f64) -> f64 {
    if j > 0 && j + 1 < u[0].len() {
        (u[i][j + 1] - 2.0 * u[i][j] + u[i][j - 1]) / (dy * dy)
    } else if j == 0 {
        // Forward difference at bottom boundary
        (u[i][j + 2] - 2.0 * u[i][j + 1] + u[i][j]) / (dy * dy)
    } else {
        // Backward difference at top boundary
        (u[i][j] - 2.0 * u[i][j - 1] + u[i][j - 2]) / (dy * dy)
    }
}

/// Fourth-order central difference for ∂²u/∂x²
#[allow(dead_code)]
pub fn central_diff_xx_4th(u: &[Vec<f64>], i: usize, j: usize, dx: f64) -> f64 {
    if i >= 2 && i + 2 < u.len() {
        (-u[i + 2][j] + 16.0 * u[i + 1][j] - 30.0 * u[i][j] + 16.0 * u[i - 1][j] - u[i - 2][j]) / (12.0 * dx * dx)
    } else {
        // Fall back to second-order
        central_diff_xx(u, i, j, dx)
    }
}

/// Fourth-order central difference for ∂²u/∂y²
#[allow(dead_code)]
pub fn central_diff_yy_4th(u: &[Vec<f64>], i: usize, j: usize, dy: f64) -> f64 {
    if j >= 2 && j + 2 < u[0].len() {
        (-u[i][j + 2] + 16.0 * u[i][j + 1] - 30.0 * u[i][j] + 16.0 * u[i][j - 1] - u[i][j - 2]) / (12.0 * dy * dy)
    } else {
        // Fall back to second-order
        central_diff_yy(u, i, j, dy)
    }
}

/// Mixed derivative ∂²u/∂x∂y using central differences
#[allow(dead_code)]
pub fn central_diff_xy(u: &[Vec<f64>], i: usize, j: usize, dx: f64, dy: f64) -> f64 {
    if i > 0 && i + 1 < u.len() && j > 0 && j + 1 < u[0].len() {
        (u[i + 1][j + 1] - u[i + 1][j - 1] - u[i - 1][j + 1] + u[i - 1][j - 1]) / (4.0 * dx * dy)
    } else {
        0.0 // Fall back to zero for boundary points
    }
}

/// Neumann boundary condition treatment (∂u/∂n = 0)
pub fn apply_neumann_boundary(u: &mut [Vec<f64>], nx: usize, ny: usize) {
    // Left boundary: ∂u/∂x = 0
    for j in 0..ny {
        u[0][j] = u[1][j];
    }
    
    // Right boundary: ∂u/∂x = 0
    for j in 0..ny {
        u[nx - 1][j] = u[nx - 2][j];
    }
    
    // Bottom boundary: ∂u/∂y = 0
    for i in 0..nx {
        u[i][0] = u[i][1];
    }
    
    // Top boundary: ∂u/∂y = 0
    for i in 0..nx {
        u[i][ny - 1] = u[i][ny - 2];
    }
}

/// Periodic boundary conditions
pub fn apply_periodic_boundary(u: &mut [Vec<f64>], nx: usize, ny: usize) {
    // Periodic in x-direction
    for j in 0..ny {
        u[0][j] = u[nx - 2][j];
        u[nx - 1][j] = u[1][j];
    }
    
    // Periodic in y-direction
    for i in 0..nx {
        u[i][0] = u[i][ny - 2];
        u[i][ny - 1] = u[i][1];
    }
}

/// Calculate the Laplacian ∇²u = ∂²u/∂x² + ∂²u/∂y²
#[allow(dead_code)]
pub fn laplacian(u: &[Vec<f64>], i: usize, j: usize, dx: f64, dy: f64) -> f64 {
    central_diff_xx(u, i, j, dx) + central_diff_yy(u, i, j, dy)
}

/// Calculate the Laplacian using fourth-order accuracy where possible
#[allow(dead_code)]
pub fn laplacian_4th(u: &[Vec<f64>], i: usize, j: usize, dx: f64, dy: f64) -> f64 {
    central_diff_xx_4th(u, i, j, dx) + central_diff_yy_4th(u, i, j, dy)
}

/// Calculate gradient magnitude |∇u| = √((∂u/∂x)² + (∂u/∂y)²)
#[allow(dead_code)]
pub fn gradient_magnitude(u: &[Vec<f64>], i: usize, j: usize, dx: f64, dy: f64) -> f64 {
    let du_dx = if i > 0 && i + 1 < u.len() {
        (u[i + 1][j] - u[i - 1][j]) / (2.0 * dx)
    } else if i == 0 {
        (u[i + 1][j] - u[i][j]) / dx
    } else {
        (u[i][j] - u[i - 1][j]) / dx
    };
    
    let du_dy = if j > 0 && j + 1 < u[0].len() {
        (u[i][j + 1] - u[i][j - 1]) / (2.0 * dy)
    } else if j == 0 {
        (u[i][j + 1] - u[i][j]) / dy
    } else {
        (u[i][j] - u[i][j - 1]) / dy
    };
    
    (du_dx * du_dx + du_dy * du_dy).sqrt()
}

/// Calculate the maximum gradient magnitude over the entire domain
#[allow(dead_code)]
pub fn max_gradient_magnitude(u: &[Vec<f64>], dx: f64, dy: f64) -> f64 {
    let mut max_grad = 0.0;
    
    for i in 0..u.len() {
        for j in 0..u[0].len() {
            let grad = gradient_magnitude(u, i, j, dx, dy);
            if grad > max_grad {
                max_grad = grad;
            }
        }
    }
    
    max_grad
}

/// Calculate the L2 norm of the solution
#[allow(dead_code)]
pub fn l2_norm(u: &[Vec<f64>]) -> f64 {
    let sum_squares: f64 = u.iter()
        .flat_map(|row| row.iter())
        .map(|&val| val * val)
        .sum();
    
    sum_squares.sqrt()
}

/// Calculate the L∞ norm (maximum absolute value) of the solution
#[allow(dead_code)]
pub fn linf_norm(u: &[Vec<f64>]) -> f64 {
    u.iter()
        .flat_map(|row| row.iter())
        .map(|&val| val.abs())
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_central_diff_xx() {
        let u = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        
        // Test interior point
        let result = central_diff_xx(&u, 1, 1, 1.0);
        assert!((result - 0.0).abs() < 1e-10); // Should be 0 for linear function
        
        // Test boundary points
        let result_left = central_diff_xx(&u, 0, 1, 1.0);
        let result_right = central_diff_xx(&u, 2, 1, 1.0);
        assert!(result_left.abs() < 1e-10);
        assert!(result_right.abs() < 1e-10);
    }
    
    #[test]
    fn test_gradient_magnitude() {
        let u = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
        ];
        
        let grad = gradient_magnitude(&u, 1, 1, 1.0, 1.0);
        assert!((grad - 2.0_f64.sqrt()).abs() < 1e-10);
    }
    
    #[test]
    fn test_neumann_boundary() {
        let mut u = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        
        apply_neumann_boundary(&mut u, 3, 3);
        
        // Check that boundary values match their neighbors
        assert_eq!(u[0][1], u[1][1]);
        assert_eq!(u[2][1], u[1][1]);
        assert_eq!(u[1][0], u[1][1]);
        assert_eq!(u[1][2], u[1][1]);
    }
}
