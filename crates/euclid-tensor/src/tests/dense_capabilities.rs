//! The eight capability extensions, implemented for [`Dense`] with real semantics.
//!
//! Two conventions are worth stating, because the traits leave them to the implementation:
//!
//! - **Indices** are ordinary `Dense` tensors holding integral values, since this is a
//!   single-dtype carrier.
//! - **Complex values** (needed only by [`TensorFft`]) are carried *interleaved* along the last
//!   axis as `re, im, re, im, …`. A real-only element type cannot otherwise represent a
//!   spectrum, and this keeps `fft` shape-preserving so `ifft(fft(x)) ≈ x` is checkable.

use super::dense::Dense;
use crate::{
    FftKind, Tensor, TensorContract, TensorConv, TensorFft, TensorIndexing, TensorReduce,
    TensorSort, TensorTransform, TensorWindow,
};
use euclid_numeric::Real;

/// Mean of a slice, `0.0` for an empty one.
fn mean_of(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

/// Stable `ln Σ exp` via the max-shift trick.
fn logsumexp_of(xs: &[f64]) -> f64 {
    let m = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    m + xs.iter().map(|&x| (x - m).exp()).sum::<f64>().ln()
}

impl TensorReduce for Dense {
    fn cumsum(self, axis: usize) -> Self {
        self.map_lanes(axis, |lane| {
            lane.iter()
                .scan(0.0, |acc, &x| {
                    *acc += x;
                    Some(*acc)
                })
                .collect()
        })
    }

    fn var(self, axes: &[usize]) -> Self {
        self.reduce_groups(axes, |g| {
            let m = mean_of(g);
            mean_of(&g.iter().map(|&x| (x - m) * (x - m)).collect::<Vec<_>>())
        })
    }

    fn std(self, axes: &[usize]) -> Self {
        TensorReduce::var(self, axes).map_values(f64::sqrt)
    }

    fn logsumexp(self, axes: &[usize]) -> Self {
        self.reduce_groups(axes, logsumexp_of)
    }

    fn softmax(self, axis: usize) -> Self {
        self.map_lanes(axis, |lane| {
            let lse = logsumexp_of(lane);
            lane.iter().map(|&x| (x - lse).exp()).collect()
        })
    }

    fn log_softmax(self, axis: usize) -> Self {
        self.map_lanes(axis, |lane| {
            let lse = logsumexp_of(lane);
            lane.iter().map(|&x| x - lse).collect()
        })
    }
}

/// Order a lane's indices, breaking ties by position so the sort is deterministic.
fn ordered_indices(lane: &[f64], descending: bool) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..lane.len()).collect();
    idx.sort_by(|&a, &b| {
        let ord = lane[a].total_cmp(&lane[b]);
        let ord = if descending { ord.reverse() } else { ord };
        ord.then(a.cmp(&b))
    });
    idx
}

impl TensorSort for Dense {
    fn sort(self, axis: usize, descending: bool) -> Self {
        self.map_lanes(axis, |lane| {
            ordered_indices(lane, descending)
                .into_iter()
                .map(|i| lane[i])
                .collect()
        })
    }

    fn argsort(&self, axis: usize, descending: bool) -> Self::Index {
        self.map_lanes(axis, |lane| {
            ordered_indices(lane, descending)
                .into_iter()
                .map(|i| i as f64)
                .collect()
        })
    }

    fn argmax(&self, axis: usize) -> Self::Index {
        self.reduce_lanes(axis, |lane| {
            ordered_indices(lane, true).first().copied().unwrap_or(0) as f64
        })
    }

    fn argmin(&self, axis: usize) -> Self::Index {
        self.reduce_lanes(axis, |lane| {
            ordered_indices(lane, false).first().copied().unwrap_or(0) as f64
        })
    }

    fn top_k(&self, k: usize, largest: bool) -> (Self, Self::Index) {
        let rank = self.dims().len();
        if rank == 0 {
            return (self.clone(), self.clone());
        }
        let axis = rank - 1;
        let len = self.dims()[axis];
        let k = k.min(len);

        let mut out_shape = self.dims().to_vec();
        out_shape[axis] = k;

        let mut values = Vec::new();
        let mut indices = Vec::new();
        // The last axis is contiguous, so lanes are simple chunks.
        for chunk in self.values().chunks(len) {
            let order = ordered_indices(chunk, largest);
            for &i in order.iter().take(k) {
                values.push(chunk[i]);
                indices.push(i as f64);
            }
        }

        (
            Dense::new(&out_shape, values),
            Dense::new(&out_shape, indices),
        )
    }
}

impl TensorContract for Dense {
    fn dot_general(
        &self,
        rhs: &Self,
        lhs_contracting: &[usize],
        rhs_contracting: &[usize],
        lhs_batch: &[usize],
        rhs_batch: &[usize],
    ) -> Self {
        let lhs_shape = self.dims();
        let rhs_shape = rhs.dims();

        let lhs_free: Vec<usize> = (0..lhs_shape.len())
            .filter(|a| !lhs_contracting.contains(a) && !lhs_batch.contains(a))
            .collect();
        let rhs_free: Vec<usize> = (0..rhs_shape.len())
            .filter(|a| !rhs_contracting.contains(a) && !rhs_batch.contains(a))
            .collect();

        // Output layout is batch dims, then lhs free dims, then rhs free dims.
        let batch_dims: Vec<usize> = lhs_batch.iter().map(|&a| lhs_shape[a]).collect();
        let lhs_free_dims: Vec<usize> = lhs_free.iter().map(|&a| lhs_shape[a]).collect();
        let rhs_free_dims: Vec<usize> = rhs_free.iter().map(|&a| rhs_shape[a]).collect();
        let contract_dims: Vec<usize> = lhs_contracting.iter().map(|&a| lhs_shape[a]).collect();

        let out_shape: Vec<usize> = batch_dims
            .iter()
            .chain(&lhs_free_dims)
            .chain(&rhs_free_dims)
            .copied()
            .collect();
        let out_len: usize = out_shape.iter().product::<usize>().max(1);
        let contract_len: usize = contract_dims.iter().product::<usize>().max(1);

        let mut data = vec![0.0; out_len];
        for (slot, cell) in data.iter_mut().enumerate() {
            let out_coords = Dense::unravel(slot, &out_shape);
            let (batch_c, rest) = out_coords.split_at(batch_dims.len());
            let (lhs_free_c, rhs_free_c) = rest.split_at(lhs_free_dims.len());

            let mut total = 0.0;
            for c in 0..contract_len {
                let contract_c = Dense::unravel(c, &contract_dims);

                let mut lhs_coords = vec![0usize; lhs_shape.len()];
                for (i, &a) in lhs_batch.iter().enumerate() {
                    lhs_coords[a] = batch_c[i];
                }
                for (i, &a) in lhs_free.iter().enumerate() {
                    lhs_coords[a] = lhs_free_c[i];
                }
                for (i, &a) in lhs_contracting.iter().enumerate() {
                    lhs_coords[a] = contract_c[i];
                }

                let mut rhs_coords = vec![0usize; rhs_shape.len()];
                for (i, &a) in rhs_batch.iter().enumerate() {
                    rhs_coords[a] = batch_c[i];
                }
                for (i, &a) in rhs_free.iter().enumerate() {
                    rhs_coords[a] = rhs_free_c[i];
                }
                for (i, &a) in rhs_contracting.iter().enumerate() {
                    rhs_coords[a] = contract_c[i];
                }

                let l = self.values()[Dense::ravel(&lhs_coords, lhs_shape)];
                let r = rhs.values()[Dense::ravel(&rhs_coords, rhs_shape)];
                total += l * r;
            }
            *cell = total;
        }

        Dense::new(&out_shape, data)
    }

    fn einsum1(&self, config: &str) -> Self {
        einsum(&[self], config)
    }

    fn einsum2(&self, rhs: &Self, config: &str) -> Self {
        einsum(&[self, rhs], config)
    }
}

/// A general Einstein summation over one or two operands.
///
/// Handles repeated letters within an operand (`"ii->i"` takes a diagonal), letters dropped
/// from the output (summed over), and permutation (`"ij->ji"`).
fn einsum(operands: &[&Dense], config: &str) -> Dense {
    let Some((inputs, output)) = config.split_once("->") else {
        return Dense::point(f64::NAN);
    };
    let input_specs: Vec<Vec<char>> = inputs
        .split(',')
        .map(|s| s.trim().chars().collect())
        .collect();
    let out_spec: Vec<char> = output.trim().chars().collect();

    if input_specs.len() != operands.len() {
        return Dense::point(f64::NAN);
    }

    // Letter -> extent, read off whichever operand axis mentions it.
    let mut letters: Vec<char> = Vec::new();
    let mut extents: Vec<usize> = Vec::new();
    for (spec, operand) in input_specs.iter().zip(operands) {
        for (axis, &letter) in spec.iter().enumerate() {
            let extent = operand.dims().get(axis).copied().unwrap_or(0);
            match letters.iter().position(|&l| l == letter) {
                Some(i) => {
                    if extents[i] != extent {
                        return Dense::point(f64::NAN);
                    }
                }
                None => {
                    letters.push(letter);
                    extents.push(extent);
                }
            }
        }
    }

    let out_shape: Vec<usize> = out_spec
        .iter()
        .map(|l| {
            letters
                .iter()
                .position(|x| x == l)
                .map(|i| extents[i])
                .unwrap_or(0)
        })
        .collect();
    let out_len: usize = out_shape.iter().product::<usize>().max(1);
    let total: usize = extents.iter().product::<usize>().max(1);

    let mut data = vec![0.0; out_len];
    // Walk every assignment of letters to index values, accumulating into the output slot.
    for flat in 0..total {
        let assignment = Dense::unravel(flat, &extents);
        let value: f64 = input_specs
            .iter()
            .zip(operands)
            .map(|(spec, operand)| {
                let coords: Vec<usize> = spec
                    .iter()
                    .map(|l| {
                        letters
                            .iter()
                            .position(|x| x == l)
                            .map(|i| assignment[i])
                            .unwrap_or(0)
                    })
                    .collect();
                operand
                    .values()
                    .get(Dense::ravel(&coords, operand.dims()))
                    .copied()
                    .unwrap_or(0.0)
            })
            .product();

        let out_coords: Vec<usize> = out_spec
            .iter()
            .map(|l| {
                letters
                    .iter()
                    .position(|x| x == l)
                    .map(|i| assignment[i])
                    .unwrap_or(0)
            })
            .collect();
        data[Dense::ravel(&out_coords, &out_shape)] += value;
    }

    Dense::new(&out_shape, data)
}

impl TensorIndexing for Dense {
    fn iota(len: usize) -> Self::Index {
        Dense::vector(&(0..len).map(|i| i as f64).collect::<Vec<_>>())
    }

    fn one_hot(indices: &Self::Index, depth: usize) -> Self {
        let mut out_shape = indices.dims().to_vec();
        out_shape.push(depth);
        let mut data = vec![0.0; indices.values().len() * depth];
        for (row, &idx) in indices.values().iter().enumerate() {
            let col = idx as usize;
            if idx >= 0.0 && col < depth {
                data[row * depth + col] = 1.0;
            }
        }
        Dense::new(&out_shape, data)
    }

    fn bincount(indices: &Self::Index, length: usize) -> Self {
        let mut counts = vec![0.0; length];
        for &idx in indices.values() {
            let slot = idx as usize;
            if idx >= 0.0 && slot < length {
                counts[slot] += 1.0;
            }
        }
        Dense::new(&[length], counts)
    }
}

impl TensorTransform for Dense {
    fn roll(&self, shift: isize, axis: usize) -> Self {
        let Some(&len) = self.dims().get(axis) else {
            return self.clone();
        };
        if len == 0 {
            return self.clone();
        }
        let len_i = len as isize;
        self.map_lanes(axis, |lane| {
            (0..len)
                .map(|i| {
                    // Positive shift moves elements toward higher indices.
                    let src = ((i as isize - shift) % len_i + len_i) % len_i;
                    lane[src as usize]
                })
                .collect()
        })
    }

    fn tile(&self, reps: usize, axis: usize) -> Self {
        let copies: Vec<Self> = (0..reps).map(|_| self.clone()).collect();
        Tensor::concatenate(&copies, axis)
    }

    fn clip(self, lo: Self, hi: Self) -> Self {
        Real::clamp(self, lo, hi)
    }
}

/// Walk every window position: yields the output shape and, per output cell, the (possibly
/// padded) values the window covers.
fn windows(
    input: &Dense,
    window_dimensions: &[usize],
    window_strides: &[usize],
    padding: &[(usize, usize)],
) -> (Vec<usize>, Vec<Vec<f64>>) {
    let shape = input.dims();
    let rank = shape.len();
    let dim = |v: &[usize], a: usize| v.get(a).copied().unwrap_or(1);
    let pad = |a: usize| padding.get(a).copied().unwrap_or((0, 0));

    let out_shape: Vec<usize> = (0..rank)
        .map(|a| {
            let (lo, hi) = pad(a);
            let padded = shape[a] + lo + hi;
            let w = dim(window_dimensions, a);
            if padded < w {
                0
            } else {
                (padded - w) / dim(window_strides, a) + 1
            }
        })
        .collect();

    let out_len: usize = out_shape.iter().product::<usize>().max(1);
    let window_len: usize = (0..rank).map(|a| dim(window_dimensions, a)).product();

    let mut cells = Vec::with_capacity(out_len);
    for slot in 0..out_len {
        let out_coords = Dense::unravel(slot, &out_shape);
        let mut covered = Vec::with_capacity(window_len);
        let window_shape: Vec<usize> = (0..rank).map(|a| dim(window_dimensions, a)).collect();

        for w in 0..window_len {
            let w_coords = Dense::unravel(w, &window_shape);
            let mut in_coords = vec![0usize; rank];
            let mut inside = true;
            for a in 0..rank {
                let (lo, _) = pad(a);
                let start = out_coords[a] * dim(window_strides, a);
                let padded_idx = start + w_coords[a];
                if padded_idx < lo || padded_idx - lo >= shape[a] {
                    inside = false;
                    break;
                }
                in_coords[a] = padded_idx - lo;
            }
            if inside {
                covered.push(input.values()[Dense::ravel(&in_coords, shape)]);
            }
        }
        cells.push(covered);
    }

    (out_shape, cells)
}

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

impl TensorConv for Dense {
    /// Implemented for the `NCHW` input / `OIHW` kernel layout that
    /// [`conv2d_nchw`](TensorConv::conv2d_nchw) delegates to — batch/feature/spatial axes
    /// `0`/`1`/`[2,3]` on the input and output/input-feature axes `0`/`1` on the kernel, with
    /// strides, padding, both dilations, and feature grouping all honored. Other dimension
    /// numberings return an empty tensor rather than silently computing something else.
    #[allow(clippy::too_many_arguments)]
    fn conv_general_dilated(
        &self,
        rhs: &Self,
        window_strides: &[usize],
        padding: &[(usize, usize)],
        lhs_dilation: &[usize],
        rhs_dilation: &[usize],
        input_batch_dimension: usize,
        input_feature_dimension: usize,
        input_spatial_dimensions: &[usize],
        kernel_input_feature_dimension: usize,
        kernel_output_feature_dimension: usize,
        kernel_spatial_dimensions: &[usize],
        feature_group_count: usize,
        batch_group_count: usize,
    ) -> Self {
        let nchw = input_batch_dimension == 0
            && input_feature_dimension == 1
            && input_spatial_dimensions == [2, 3]
            && kernel_output_feature_dimension == 0
            && kernel_input_feature_dimension == 1
            && kernel_spatial_dimensions == [2, 3]
            && batch_group_count == 1;
        let ([n, c_in, h, w], [o, i_k, kh, kw]) = (self.dims(), rhs.dims()) else {
            return Dense::new(&[0], Vec::new());
        };
        if !nchw || feature_group_count == 0 || c_in % feature_group_count != 0 {
            return Dense::new(&[0], Vec::new());
        }
        let (n, c_in, h, w) = (*n, *c_in, *h, *w);
        let (o, i_k, kh, kw) = (*o, *i_k, *kh, *kw);
        if i_k != c_in / feature_group_count || o % feature_group_count != 0 {
            return Dense::new(&[0], Vec::new());
        }

        let stride = |a: usize| window_strides.get(a).copied().unwrap_or(1).max(1);
        let pad = |a: usize| padding.get(a).copied().unwrap_or((0, 0));
        let ldil = |a: usize| lhs_dilation.get(a).copied().unwrap_or(1).max(1);
        let rdil = |a: usize| rhs_dilation.get(a).copied().unwrap_or(1).max(1);

        // Dilation inflates the logical extents before the window slides.
        let eff = |len: usize, d: usize| if len == 0 { 0 } else { (len - 1) * d + 1 };
        let h_in = eff(h, ldil(0)) + pad(0).0 + pad(0).1;
        let w_in = eff(w, ldil(1)) + pad(1).0 + pad(1).1;
        let kh_eff = eff(kh, rdil(0));
        let kw_eff = eff(kw, rdil(1));
        if h_in < kh_eff || w_in < kw_eff {
            return Dense::new(&[0], Vec::new());
        }
        let h_out = (h_in - kh_eff) / stride(0) + 1;
        let w_out = (w_in - kw_eff) / stride(1) + 1;

        let per_group_out = o / feature_group_count;
        let mut data = vec![0.0; n * o * h_out * w_out];

        for b in 0..n {
            for oc in 0..o {
                let group = oc / per_group_out;
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut acc = 0.0;
                        for ic in 0..i_k {
                            let c = group * i_k + ic;
                            for r in 0..kh {
                                for s in 0..kw {
                                    let ph = oh * stride(0) + r * rdil(0);
                                    let pw = ow * stride(1) + s * rdil(1);
                                    // Undo padding, then undo input dilation.
                                    if ph < pad(0).0 || pw < pad(1).0 {
                                        continue;
                                    }
                                    let dh = ph - pad(0).0;
                                    let dw = pw - pad(1).0;
                                    if dh % ldil(0) != 0 || dw % ldil(1) != 0 {
                                        continue;
                                    }
                                    let (ih, iw) = (dh / ldil(0), dw / ldil(1));
                                    if ih >= h || iw >= w {
                                        continue;
                                    }
                                    let x =
                                        self.values()[Dense::ravel(&[b, c, ih, iw], self.dims())];
                                    let k = rhs.values()[Dense::ravel(&[oc, ic, r, s], rhs.dims())];
                                    acc += x * k;
                                }
                            }
                        }
                        data[Dense::ravel(&[b, oc, oh, ow], &[n, o, h_out, w_out])] = acc;
                    }
                }
            }
        }

        Dense::new(&[n, o, h_out, w_out], data)
    }
}

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
