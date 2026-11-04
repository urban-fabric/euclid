//! `Dense`'s [`TensorFft`](crate::TensorFft) implementation.

use super::super::dense::Dense;
use crate::{FftKind, TensorFft};

impl TensorFft for Dense {
    /// A direct (naive) DFT over the final axis, on the interleaved complex layout described in
    /// this module's docs: the last axis holds `re, im, re, im, …`, so a transform of logical
    /// length `m` occupies `2m` slots and the shape is preserved.
    ///
    /// `lengths` must name a single transform length; multi-axis transforms return an empty
    /// tensor rather than a wrong answer.
    fn fft(&self, kind: FftKind, lengths: &[usize]) -> Self {
        let [m] = lengths else {
            return Dense::new(&[0], Vec::new());
        };
        let m = *m;
        let Some(&last) = self.dims().last() else {
            return Dense::new(&[0], Vec::new());
        };
        if m == 0 || last != 2 * m {
            return Dense::new(&[0], Vec::new());
        }

        let (sign, scale) = match kind {
            // A real input is the same computation with zero imaginary parts, which the
            // interleaved layout already supplies.
            FftKind::Forward | FftKind::RealForward => (-1.0, 1.0),
            FftKind::Inverse | FftKind::RealInverse => (1.0, 1.0 / m as f64),
        };

        self.map_lanes(self.dims().len() - 1, |lane| {
            let mut spectrum = vec![0.0; 2 * m];
            for k in 0..m {
                let (mut re, mut im) = (0.0, 0.0);
                for t in 0..m {
                    let angle = sign * core::f64::consts::TAU * (k * t) as f64 / m as f64;
                    let (s, c) = angle.sin_cos();
                    let (xr, xi) = (lane[2 * t], lane[2 * t + 1]);
                    re += xr * c - xi * s;
                    im += xr * s + xi * c;
                }
                spectrum[2 * k] = re * scale;
                spectrum[2 * k + 1] = im * scale;
            }
            spectrum
        })
    }
}
