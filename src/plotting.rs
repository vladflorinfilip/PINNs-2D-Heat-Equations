/// Plotting utilities for 2D heat equation solutions
/// 
/// This module provides functions to visualize temperature fields,
/// initial conditions, and solution evolution over time.

use plotters::prelude::*;

/// Plot a 2D temperature field as a heatmap
pub fn plot_temperature_field(
    u: &[Vec<f64>],
    x: &[f64],
    y: &[f64],
    title: &str,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let nx = x.len();
    let ny = y.len();
    
    // Find min and max values for color scaling
    let min_temp = u.iter()
        .flat_map(|row| row.iter())
        .fold(f64::INFINITY, |a, &b| a.min(b));
    let max_temp = u.iter()
        .flat_map(|row| row.iter())
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    // Create the chart
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 20).into_font())
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x[0]..x[nx-1], y[0]..y[ny-1])?;

    chart.configure_mesh()
        .x_labels(5)
        .y_labels(5)
        .x_label_formatter(&|x| format!("{:.2}", x))
        .y_label_formatter(&|y| format!("{:.2}", y))
        .draw()?;

    // Create the heatmap
    let mut data_points = Vec::new();
    for i in 0..nx {
        for j in 0..ny {
            data_points.push((x[i], y[j], u[i][j]));
        }
    }

    // Draw the heatmap using rectangles
    for i in 0..nx-1 {
        for j in 0..ny-1 {
            let temp = u[i][j];
            let normalized_temp = (temp - min_temp) / (max_temp - min_temp);
            
            // Create color based on temperature
            let color = if normalized_temp < 0.0 {
                RGBColor(0, 0, 255) // Blue for cold
            } else if normalized_temp < 0.5 {
                let intensity = (normalized_temp * 2.0 * 255.0) as u8;
                RGBColor(0, intensity, 255 - intensity) // Blue to cyan
            } else if normalized_temp < 1.0 {
                let intensity = ((normalized_temp - 0.5) * 2.0 * 255.0) as u8;
                RGBColor(intensity, 255, 0) // Cyan to yellow
            } else {
                RGBColor(255, 255, 0) // Yellow for hot
            };
            
            let rect = Rectangle::new(
                [(x[i], y[j]), (x[i+1], y[j+1])],
                ShapeStyle::from(&color).filled(),
            );
            chart.draw_series(std::iter::once(rect))?;
        }
    }

    // Add colorbar
    let colorbar = create_colorbar(min_temp, max_temp);
    chart.draw_series(colorbar)?;

    root.present()?;
    Ok(())
}

/// Plot initial conditions
pub fn plot_initial_condition<F>(
    initial_func: F,
    x: &[f64],
    y: &[f64],
    title: &str,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: Fn(f64, f64) -> f64,
{
    // Create temperature field from initial condition
    let mut u = vec![vec![0.0; y.len()]; x.len()];
    for i in 0..x.len() {
        for j in 0..y.len() {
            u[i][j] = initial_func(x[i], y[j]);
        }
    }
    
    plot_temperature_field(&u, x, y, title, filename)
}

/// Create a colorbar for the heatmap
fn create_colorbar(min_temp: f64, max_temp: f64) -> Vec<Rectangle<(f64, f64)>> {
    let mut colorbar = Vec::new();
    let bar_width = 0.1;
    let bar_height = max_temp - min_temp;
    let bar_x = 0.9; // Position on the right side
    
    // Create colorbar with 20 segments
    let num_segments = 20;
    for i in 0..num_segments {
        let normalized_temp = i as f64 / (num_segments - 1) as f64;
        let _temp = min_temp + normalized_temp * (max_temp - min_temp);
        
        let color = if normalized_temp < 0.0 {
            RGBColor(0, 0, 255)
        } else if normalized_temp < 0.5 {
            let intensity = (normalized_temp * 2.0 * 255.0) as u8;
            RGBColor(0, intensity, 255 - intensity)
        } else if normalized_temp < 1.0 {
            let intensity = ((normalized_temp - 0.5) * 2.0 * 255.0) as u8;
            RGBColor(intensity, 255, 0)
        } else {
            RGBColor(255, 255, 0)
        };
        
        let y_start = min_temp + (i as f64 / num_segments as f64) * bar_height;
        let y_end = min_temp + ((i + 1) as f64 / num_segments as f64) * bar_height;
        
        let rect = Rectangle::new(
            [(bar_x, y_start), (bar_x + bar_width, y_end)],
            ShapeStyle::from(&color).filled(),
        );
        colorbar.push(rect);
    }
    
    colorbar
}

/// Plot solution evolution over time (multiple snapshots)
#[allow(dead_code)]
pub fn plot_solution_evolution(
    solutions: &[Vec<Vec<f64>>],
    x: &[f64],
    y: &[f64],
    times: &[f64],
    base_filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    for (_i, (solution, &time)) in solutions.iter().zip(times.iter()).enumerate() {
        let filename = format!("{}_t{:.3}.png", base_filename, time);
        let title = format!("Temperature at t = {:.3}", time);
        plot_temperature_field(solution, x, y, &title, &filename)?;
    }
    Ok(())
}

/// Plot a 1D slice of the solution along x-axis at a specific y position
#[allow(dead_code)]
pub fn plot_x_slice(
    u: &[Vec<f64>],
    x: &[f64],
    y_index: usize,
    title: &str,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let nx = x.len();
    let _y_val = if y_index < u[0].len() { y_index as f64 } else { 0.0 };
    
    // Extract the slice
    let mut data_points = Vec::new();
    for i in 0..nx {
        if y_index < u[i].len() {
            data_points.push((x[i], u[i][y_index]));
        }
    }

    let min_temp = data_points.iter().map(|(_, temp)| *temp).fold(f64::INFINITY, f64::min);
    let max_temp = data_points.iter().map(|(_, temp)| *temp).fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 20).into_font())
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x[0]..x[nx-1], min_temp..max_temp)?;

    chart.configure_mesh()
        .x_labels(5)
        .y_labels(5)
        .x_label_formatter(&|x| format!("{:.2}", x))
        .y_label_formatter(&|y| format!("{:.2}", y))
        .draw()?;

    chart.draw_series(LineSeries::new(
        data_points,
        RED.stroke_width(2),
    ))?;

    root.present()?;
    Ok(())
}

/// Plot a 1D slice of the solution along y-axis at a specific x position
#[allow(dead_code)]
pub fn plot_y_slice(
    u: &[Vec<f64>],
    y: &[f64],
    x_index: usize,
    title: &str,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let ny = y.len();
    let _x_val = if x_index < u.len() { x_index as f64 } else { 0.0 };
    
    // Extract the slice
    let mut data_points = Vec::new();
    for j in 0..ny {
        if x_index < u.len() && j < u[x_index].len() {
            data_points.push((y[j], u[x_index][j]));
        }
    }

    let min_temp = data_points.iter().map(|(_, temp)| *temp).fold(f64::INFINITY, f64::min);
    let max_temp = data_points.iter().map(|(_, temp)| *temp).fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 20).into_font())
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(y[0]..y[ny-1], min_temp..max_temp)?;

    chart.configure_mesh()
        .x_labels(5)
        .y_labels(5)
        .x_label_formatter(&|y| format!("{:.2}", y))
        .y_label_formatter(&|x| format!("{:.2}", x))
        .draw()?;

    chart.draw_series(LineSeries::new(
        data_points,
        BLUE.stroke_width(2),
    ))?;

    root.present()?;
    Ok(())
}

/// Create a directory for output plots
pub fn create_output_dir() -> Result<(), std::io::Error> {
    std::fs::create_dir_all("plots")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plot_temperature_field() {
        // Create a simple test case
        let u = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0],
        ];
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 2.0];
        
        // This should not panic
        let result = plot_temperature_field(&u, &x, &y, "Test", "test_plot.png");
        assert!(result.is_ok());
        
        // Clean up
        let _ = std::fs::remove_file("test_plot.png");
    }
}
