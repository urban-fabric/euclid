//! `Dense`'s [`TensorWindow`](crate::TensorWindow) implementation.

use super::super::dense::Dense;
use super::*;
use crate::TensorWindow;

impl TensorWindow for Dense {
    fn reduce_window_sum(
        &self,
        window_dimensions: &[usize],
        window_strides: &[usize],
        padding: &[(usize, usize)],
    ) -> Self {
        let (shape, cells) = windows(self, window_dimensions, window_strides, padding);
        Dense::new(&shape, cells.iter().map(|c| c.iter().sum()).collect())
    }

    fn reduce_window_max(
        &self,
        window_dimensions: &[usize],
        window_strides: &[usize],
        padding: &[(usize, usize)],
    ) -> Self {
        let (shape, cells) = windows(self, window_dimensions, window_strides, padding);
        Dense::new(
            &shape,
            cells
                .iter()
                .map(|c| c.iter().copied().fold(f64::NEG_INFINITY, f64::max))
                .collect(),
        )
    }

    /// The windowed sum over the window's element count. Padded-away positions are not counted,
    /// so an edge window averages only the elements it actually covers.
    fn reduce_window_mean(
        &self,
        window_dimensions: &[usize],
        window_strides: &[usize],
        padding: &[(usize, usize)],
    ) -> Self {
        let (shape, cells) = windows(self, window_dimensions, window_strides, padding);
        Dense::new(&shape, cells.iter().map(|c| mean_of(c)).collect())
    }
}
