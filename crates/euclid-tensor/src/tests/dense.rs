//! A rank-n dense `f64` tensor, used to prove the capability extensions are implementable
//! with real semantics rather than only type-checking.
//!
//! [`ToyTensor`](super::toy_tensor::ToyTensor) is deliberately rank-0 and stops at [`Ring`], so
//! it cannot reach the many extension methods bounded `where Self: Real` — on those, the
//! tensor *itself* must implement the whole 98-method real surface. `Dense` does: it is a
//! unified-dtype carrier in the mould of a JAX array, where values, masks, and indices are all
//! the same type, and every scalar operation is the `f64` one applied elementwise.
//!
//! Conventions chosen where the contract leaves them open, since a reference implementation has
//! to pick something:
//!
//! - Reductions **remove** the reduced axes; an empty `axes` slice reduces over *all* axes,
//!   which is the only way this signature can express a full reduction.
//! - A mask is a tensor of `0.0`/`1.0`; any non-zero element is true.
//! - Binary operations broadcast a rank-0 operand against a rank-n one. Equal shapes pair
//!   elementwise. Anything else is a shape error, reported as described below.
//! - Shape errors cannot be reported by panicking (the crate denies `clippy::panic`), so they
//!   degrade to an empty tensor, which any assertion will catch.
//!
//! ## Layout
//!
//! - this module: the `Dense` struct and its inherent index/reduce/lane helpers.
//! - [`scalar`] — the operator, mask, and 98-method `Real` implementations.
//! - [`array`] — the rank-n `Tensor` core.
//!
//! The capability extensions are in [`super::dense_capabilities`].

mod array;
mod scalar;

/// A row-major dense tensor over `f64`.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Dense {
    shape: Vec<usize>,
    data: Vec<f64>,
}

impl Dense {
    pub(super) fn new(shape: &[usize], data: Vec<f64>) -> Self {
        debug_assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "shape/data mismatch"
        );
        Self {
            shape: shape.to_vec(),
            data,
        }
    }

    /// A rank-0 tensor.
    pub(super) fn point(x: f64) -> Self {
        Self {
            shape: Vec::new(),
            data: vec![x],
        }
    }

    /// A rank-1 tensor.
    pub(super) fn vector(xs: &[f64]) -> Self {
        Self {
            shape: vec![xs.len()],
            data: xs.to_vec(),
        }
    }

    /// A rank-2 tensor from row-major data.
    pub(super) fn matrix(rows: usize, cols: usize, xs: &[f64]) -> Self {
        Self::new(&[rows, cols], xs.to_vec())
    }

    pub(super) fn values(&self) -> &[f64] {
        &self.data
    }

    pub(super) fn dims(&self) -> &[usize] {
        &self.shape
    }

    /// The sole element of a rank-0 (or single-element) tensor.
    pub(super) fn item(&self) -> f64 {
        self.data.first().copied().unwrap_or(f64::NAN)
    }

    fn empty() -> Self {
        Self {
            shape: vec![0],
            data: Vec::new(),
        }
    }

    fn count(&self) -> usize {
        self.data.len()
    }

    fn map(&self, f: impl Fn(f64) -> f64) -> Self {
        Self {
            shape: self.shape.clone(),
            data: self.data.iter().copied().map(f).collect(),
        }
    }

    /// Elementwise map, for the capability extensions in the sibling module.
    pub(super) fn map_values(&self, f: impl Fn(f64) -> f64) -> Self {
        self.map(f)
    }

    /// Pair two tensors elementwise, broadcasting a rank-0 operand against a rank-n one.
    fn zip(&self, other: &Self, f: impl Fn(f64, f64) -> f64) -> Self {
        match (self.shape.is_empty(), other.shape.is_empty()) {
            (false, true) => {
                let s = other.item();
                self.map(|x| f(x, s))
            }
            (true, false) => {
                let s = self.item();
                other.map(|y| f(s, y))
            }
            _ if self.shape == other.shape => Self {
                shape: self.shape.clone(),
                data: self
                    .data
                    .iter()
                    .zip(&other.data)
                    .map(|(&x, &y)| f(x, y))
                    .collect(),
            },
            _ => Self::empty(),
        }
    }

    /// Read element `i`, treating a single-element tensor as a broadcast scalar.
    fn lane(&self, i: usize) -> f64 {
        if self.count() == 1 {
            self.item()
        } else {
            self.data.get(i).copied().unwrap_or(f64::NAN)
        }
    }

    /// The broadcast result shape across operands: the first non-rank-0 shape present.
    fn broadcast_shape<'a>(operands: impl IntoIterator<Item = &'a Self>) -> Vec<usize> {
        operands
            .into_iter()
            .map(|t| &t.shape)
            .find(|s| !s.is_empty())
            .cloned()
            .unwrap_or_default()
    }

    fn zip3(&self, a: &Self, b: &Self, f: impl Fn(f64, f64, f64) -> f64) -> Self {
        let shape = Self::broadcast_shape([self, a, b]);
        let n = shape.iter().product::<usize>().max(1);
        let data = (0..n)
            .map(|i| f(self.lane(i), a.lane(i), b.lane(i)))
            .collect();
        Self { shape, data }
    }

    // --- index arithmetic -------------------------------------------------

    pub(super) fn strides_of(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    pub(super) fn unravel(mut flat: usize, shape: &[usize]) -> Vec<usize> {
        let strides = Self::strides_of(shape);
        let mut coords = vec![0usize; shape.len()];
        for (axis, stride) in strides.iter().enumerate() {
            coords[axis] = flat / stride;
            flat %= stride;
        }
        coords
    }

    pub(super) fn ravel(coords: &[usize], shape: &[usize]) -> usize {
        Self::strides_of(shape)
            .iter()
            .zip(coords)
            .map(|(s, c)| s * c)
            .sum()
    }

    fn at(&self, coords: &[usize]) -> f64 {
        self.data
            .get(Self::ravel(coords, &self.shape))
            .copied()
            .unwrap_or(f64::NAN)
    }

    /// Normalize an `axes` argument: empty means every axis.
    fn resolve_axes(&self, axes: &[usize]) -> Vec<usize> {
        if axes.is_empty() {
            (0..self.shape.len()).collect()
        } else {
            let mut a = axes.to_vec();
            a.sort_unstable();
            a.dedup();
            a
        }
    }

    /// Reduce `axes` away with `f`, starting from `init`.
    pub(super) fn reduce(&self, axes: &[usize], init: f64, f: impl Fn(f64, f64) -> f64) -> Self {
        let axes = self.resolve_axes(axes);
        let kept: Vec<usize> = (0..self.shape.len())
            .filter(|a| !axes.contains(a))
            .collect();
        let out_shape: Vec<usize> = kept.iter().map(|&a| self.shape[a]).collect();
        let out_len: usize = out_shape.iter().product();
        let mut out = vec![init; out_len.max(1)];

        for flat in 0..self.count() {
            let coords = Self::unravel(flat, &self.shape);
            let out_coords: Vec<usize> = kept.iter().map(|&a| coords[a]).collect();
            let slot = Self::ravel(&out_coords, &out_shape);
            out[slot] = f(out[slot], self.data[flat]);
        }

        Self {
            shape: out_shape,
            data: out,
        }
    }

    /// Collect, for every output position, the full set of values that reduce into it.
    ///
    /// Reductions that need all members at once (variance, log-sum-exp) cannot be written as a
    /// running fold, so they go through this instead of [`reduce`](Self::reduce).
    pub(super) fn reduce_groups(&self, axes: &[usize], f: impl Fn(&[f64]) -> f64) -> Self {
        let axes = self.resolve_axes(axes);
        let kept: Vec<usize> = (0..self.shape.len())
            .filter(|a| !axes.contains(a))
            .collect();
        let out_shape: Vec<usize> = kept.iter().map(|&a| self.shape[a]).collect();
        let out_len: usize = out_shape.iter().product::<usize>().max(1);
        let mut groups = vec![Vec::new(); out_len];

        for flat in 0..self.count() {
            let coords = Self::unravel(flat, &self.shape);
            let out_coords: Vec<usize> = kept.iter().map(|&a| coords[a]).collect();
            groups[Self::ravel(&out_coords, &out_shape)].push(self.data[flat]);
        }

        Self {
            shape: out_shape,
            data: groups.iter().map(|g| f(g)).collect(),
        }
    }

    /// Apply `f` to each 1-D lane along `axis`, writing a lane of the same length back.
    pub(super) fn map_lanes(&self, axis: usize, f: impl Fn(&[f64]) -> Vec<f64>) -> Self {
        let Some(&len) = self.shape.get(axis) else {
            return self.clone();
        };
        let mut out = self.data.clone();
        let stride = Self::strides_of(&self.shape)[axis];

        for flat in 0..self.count() {
            let coords = Self::unravel(flat, &self.shape);
            if coords[axis] != 0 {
                continue;
            }
            let lane: Vec<f64> = (0..len).map(|i| self.data[flat + i * stride]).collect();
            let replaced = f(&lane);
            for (i, v) in replaced.into_iter().take(len).enumerate() {
                out[flat + i * stride] = v;
            }
        }

        Self {
            shape: self.shape.clone(),
            data: out,
        }
    }

    /// Reduce each 1-D lane along `axis` to one value, removing the axis.
    pub(super) fn reduce_lanes(&self, axis: usize, f: impl Fn(&[f64]) -> f64) -> Self {
        let Some(&len) = self.shape.get(axis) else {
            return self.clone();
        };
        let out_shape: Vec<usize> = self
            .shape
            .iter()
            .enumerate()
            .filter(|(a, _)| *a != axis)
            .map(|(_, &d)| d)
            .collect();
        let stride = Self::strides_of(&self.shape)[axis];
        let mut data = Vec::new();

        for flat in 0..self.count() {
            let coords = Self::unravel(flat, &self.shape);
            if coords[axis] != 0 {
                continue;
            }
            let lane: Vec<f64> = (0..len).map(|i| self.data[flat + i * stride]).collect();
            data.push(f(&lane));
        }

        Self {
            shape: out_shape,
            data,
        }
    }
}
