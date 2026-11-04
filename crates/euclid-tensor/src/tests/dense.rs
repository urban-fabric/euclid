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

use core::ops::{Add, Div, Mul, Neg, Sub};

use euclid_numeric::{Compare, Element, NormedDivisionAlgebra, Real};

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

// --- operators ----------------------------------------------------------

macro_rules! binop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl $trait for Dense {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self {
                self.zip(&rhs, |x, y| x $op y)
            }
        }
    };
}

binop!(Add, add, +);
binop!(Sub, sub, -);
binop!(Mul, mul, *);
binop!(Div, div, /);

impl Neg for Dense {
    type Output = Self;
    fn neg(self) -> Self {
        self.map(|x| -x)
    }
}

impl Element for Dense {
    fn zero() -> Self {
        Self::point(0.0)
    }
    fn one() -> Self {
        Self::point(1.0)
    }
}

// --- masks and comparison ------------------------------------------------

/// Masks are `0.0`/`1.0` tensors, so this carrier is single-dtype like a numeric array
/// backend: `Mask == Self`.
impl euclid_numeric::Boolean for Dense {
    fn and(&self, other: &Self) -> Self {
        self.zip(other, |x, y| f64::from(x != 0.0 && y != 0.0))
    }
    fn or(&self, other: &Self) -> Self {
        self.zip(other, |x, y| f64::from(x != 0.0 || y != 0.0))
    }
    fn xor(&self, other: &Self) -> Self {
        self.zip(other, |x, y| f64::from((x != 0.0) != (y != 0.0)))
    }
    fn not(&self) -> Self {
        self.map(|x| f64::from(x == 0.0))
    }
}

impl Compare for Dense {
    type Mask = Self;

    fn lt(&self, other: &Self) -> Self {
        self.zip(other, |x, y| f64::from(x < y))
    }
    fn le(&self, other: &Self) -> Self {
        self.zip(other, |x, y| f64::from(x <= y))
    }
    fn gt(&self, other: &Self) -> Self {
        self.zip(other, |x, y| f64::from(x > y))
    }
    fn ge(&self, other: &Self) -> Self {
        self.zip(other, |x, y| f64::from(x >= y))
    }
    fn eq(&self, other: &Self) -> Self {
        self.zip(other, |x, y| f64::from(x == y))
    }
    fn ne(&self, other: &Self) -> Self {
        self.zip(other, |x, y| f64::from(x != y))
    }

    fn select(mask: &Self, on_true: Self, on_false: Self) -> Self {
        let shape = Self::broadcast_shape([&on_true, &on_false, mask]);
        let n = shape.iter().product::<usize>().max(1);
        let data = (0..n)
            .map(|i| {
                if mask.lane(i) != 0.0 {
                    on_true.lane(i)
                } else {
                    on_false.lane(i)
                }
            })
            .collect();
        Self { shape, data }
    }

    /// A host loop: iterate while every lane of the predicate mask is true.
    fn while_loop<C, P, B>(init: C, cond: P, body: B) -> C
    where
        P: Fn(&C) -> Self::Mask,
        B: Fn(C) -> C,
    {
        let mut state = init;
        let mut guard = 0u32;
        while cond(&state).data.iter().all(|&m| m != 0.0) {
            state = body(state);
            guard += 1;
            if guard > 1_000_000 {
                break;
            }
        }
        state
    }
}

// --- the real scalar surface, applied elementwise ------------------------

/// `Semiring` and `Ring` are blanket-implemented over their operators, but `Arithmetic` is
/// not, so a division-bearing carrier opts in explicitly.
impl euclid_numeric::Arithmetic for Dense {}

impl NormedDivisionAlgebra for Dense {
    type RealPart = Self;

    fn conjugate(&self) -> Self {
        self.clone()
    }
    fn norm_sqr(&self) -> Self {
        self.map(|x| x * x)
    }
    fn real_part(&self) -> Self {
        self.clone()
    }
    fn from_real(re: Self) -> Self {
        re
    }
    fn magnitude(&self) -> Self {
        self.map(f64::abs)
    }
}

/// Generate the `Real` methods, each delegating to `f64`'s implementation elementwise.
macro_rules! real_const {
    ($($name:ident),* $(,)?) => {
        $(fn $name() -> Self { Self::point(<f64 as Real>::$name()) })*
    };
}
macro_rules! real_unary_ref {
    ($($name:ident),* $(,)?) => {
        $(fn $name(&self) -> Self { self.map(|x| <f64 as Real>::$name(&x)) })*
    };
}
macro_rules! real_unary_own {
    ($($name:ident),* $(,)?) => {
        $(fn $name(self) -> Self { self.map(|x| <f64 as Real>::$name(x)) })*
    };
}
macro_rules! real_binary_own {
    ($($name:ident),* $(,)?) => {
        $(fn $name(self, other: Self) -> Self {
            self.zip(&other, |x, y| <f64 as Real>::$name(x, y))
        })*
    };
}

impl Real for Dense {
    type Integer = i64;

    fn scalar(x: f64) -> Self {
        Self::point(x)
    }

    /// Same-shape fill, so a lifted literal keeps the operand's shape context.
    fn scalar_like(&self, x: f64) -> Self {
        Self {
            shape: self.shape.clone(),
            data: vec![x; self.count()],
        }
    }

    real_const!(
        two,
        half,
        pi,
        tau,
        frac_pi_2,
        frac_pi_4,
        sqrt_2,
        frac_1_sqrt_2,
        ln_2,
        ln_pi,
        ln_tau,
        half_ln_tau,
        e,
        log2_e,
        log10_e,
        ln_10,
        log2_10,
        log10_2,
        euler_gamma,
        infinity,
        neg_infinity,
        nan,
    );

    fn is_nan(&self) -> Self {
        self.map(|x| f64::from(x.is_nan()))
    }
    fn is_infinite(&self) -> Self {
        self.map(|x| f64::from(x.is_infinite()))
    }
    fn is_finite(&self) -> Self {
        self.map(|x| f64::from(x.is_finite()))
    }

    real_unary_ref!(
        exp,
        ln,
        ln_1p,
        exp_m1,
        sqrt,
        abs,
        square,
        ln_1p_exp,
        ln_1m_exp,
        ln_1m,
        ln_exp_m1,
        sigmoid,
        ln_sigmoid,
        ln_1m_sigmoid,
        logit,
    );

    real_unary_own!(
        sin,
        cos,
        tan,
        asin,
        acos,
        atan,
        sinh,
        cosh,
        tanh,
        asinh,
        acosh,
        atanh,
        cbrt,
        recip,
        log2,
        floor,
        ceil,
        round,
        trunc,
        fract,
        signum,
        to_radians,
        to_degrees,
        erf,
        erfc,
        erf_inv,
        ln_gamma,
        digamma,
        trigamma,
        erfcx,
        bessel_i0e,
        bessel_i1e,
        gamma,
        ln_bessel_i0,
        bessel_i1_over_i0,
    );

    real_binary_own!(
        atan2, hypot, copysign, rem_euclid, min, max, gamma_lr, gamma_ur, xlogy, xlog1py, xlog1my,
        ln_add_exp, ln_sub_exp, ln_beta, beta,
    );

    fn powf(&self, p: Self) -> Self {
        self.zip(&p, |x, y| <f64 as Real>::powf(&x, y))
    }

    fn sin_cos(self) -> (Self, Self) {
        (Real::sin(self.clone()), Real::cos(self))
    }

    fn powi(self, n: i64) -> Self {
        self.map(|x| <f64 as Real>::powi(x, n))
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        self.zip3(&a, &b, <f64 as Real>::mul_add)
    }

    fn clamp(self, lo: Self, hi: Self) -> Self {
        self.zip3(&lo, &hi, <f64 as Real>::clamp)
    }

    fn beta_reg(self, a: Self, b: Self) -> Self {
        self.zip3(&a, &b, <f64 as Real>::beta_reg)
    }
}

// --- the rank-n array core ----------------------------------------------

impl crate::Tensor for Dense {
    /// Single-dtype carrier: indices are tensors holding integral `f64` values.
    type Index = Self;

    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn reshape(self, shape: &[usize]) -> Self {
        if shape.iter().product::<usize>() == self.count() {
            Self {
                shape: shape.to_vec(),
                data: self.data,
            }
        } else {
            Self::empty()
        }
    }

    fn full(shape: &[usize], value: Self) -> Self {
        Self {
            shape: shape.to_vec(),
            data: vec![value.item(); shape.iter().product()],
        }
    }

    fn from_host(data: &[f64], shape: &[usize]) -> Self {
        if shape.iter().product::<usize>() == data.len() {
            Self::new(shape, data.to_vec())
        } else {
            Self::empty()
        }
    }

    fn broadcast_in_dim(&self, out_dims: &[usize], broadcast_dims: &[usize]) -> Self {
        let out_len: usize = out_dims.iter().product();
        let data = (0..out_len)
            .map(|flat| {
                let out_coords = Self::unravel(flat, out_dims);
                // Each input axis maps to one output axis; size-1 input axes broadcast.
                let in_coords: Vec<usize> = broadcast_dims
                    .iter()
                    .enumerate()
                    .map(|(in_axis, &out_axis)| {
                        if self.shape.get(in_axis) == Some(&1) {
                            0
                        } else {
                            out_coords.get(out_axis).copied().unwrap_or(0)
                        }
                    })
                    .collect();
                self.at(&in_coords)
            })
            .collect();
        Self {
            shape: out_dims.to_vec(),
            data,
        }
    }

    fn transpose_axes(&self, axes: &[usize]) -> Self {
        let out_shape: Vec<usize> = axes.iter().map(|&a| self.shape[a]).collect();
        let out_len: usize = out_shape.iter().product();
        let data = (0..out_len)
            .map(|flat| {
                let out_coords = Self::unravel(flat, &out_shape);
                let mut in_coords = vec![0usize; self.shape.len()];
                for (i, &a) in axes.iter().enumerate() {
                    in_coords[a] = out_coords[i];
                }
                self.at(&in_coords)
            })
            .collect();
        Self {
            shape: out_shape,
            data,
        }
    }

    fn squeeze_axis(&self, axis: isize) -> Self {
        let rank = self.shape.len() as isize;
        let axis = if axis < 0 { rank + axis } else { axis };
        let Ok(axis) = usize::try_from(axis) else {
            return self.clone();
        };
        if self.shape.get(axis) != Some(&1) {
            return self.clone();
        }
        let mut shape = self.shape.clone();
        shape.remove(axis);
        Self {
            shape,
            data: self.data.clone(),
        }
    }

    fn slice_in_dim(&self, start: usize, limit: usize, stride: usize, axis: usize) -> Self {
        if stride == 0 || axis >= self.shape.len() {
            return Self::empty();
        }
        let limit = limit.min(self.shape[axis]);
        let taken = if limit > start {
            (limit - start).div_ceil(stride)
        } else {
            0
        };
        let mut out_shape = self.shape.clone();
        out_shape[axis] = taken;
        let out_len: usize = out_shape.iter().product();
        let data = (0..out_len)
            .map(|flat| {
                let mut coords = Self::unravel(flat, &out_shape);
                coords[axis] = start + coords[axis] * stride;
                self.at(&coords)
            })
            .collect();
        Self {
            shape: out_shape,
            data,
        }
    }

    fn concatenate(inputs: &[Self], axis: usize) -> Self {
        let Some(first) = inputs.first() else {
            return Self::empty();
        };
        if axis >= first.shape.len() {
            return Self::empty();
        }
        let mut out_shape = first.shape.clone();
        out_shape[axis] = inputs.iter().map(|t| t.shape[axis]).sum();
        let out_len: usize = out_shape.iter().product();
        let data = (0..out_len)
            .map(|flat| {
                let mut coords = Self::unravel(flat, &out_shape);
                let mut remaining = coords[axis];
                for part in inputs {
                    if remaining < part.shape[axis] {
                        coords[axis] = remaining;
                        return part.at(&coords);
                    }
                    remaining -= part.shape[axis];
                }
                f64::NAN
            })
            .collect();
        Self {
            shape: out_shape,
            data,
        }
    }

    fn stack(inputs: &[Self], axis: usize) -> Self {
        // Insert a length-1 axis in each input, then concatenate along it.
        let lifted: Vec<Self> = inputs
            .iter()
            .map(|t| {
                let mut shape = t.shape.clone();
                shape.insert(axis.min(shape.len()), 1);
                Self {
                    shape,
                    data: t.data.clone(),
                }
            })
            .collect();
        Self::concatenate(&lifted, axis)
    }

    fn pad(
        &self,
        value: Self,
        low_padding: &[usize],
        high_padding: &[usize],
        interior_padding: &[usize],
    ) -> Self {
        let fill = value.item();
        let out_shape: Vec<usize> = self
            .shape
            .iter()
            .enumerate()
            .map(|(a, &d)| {
                let interior = interior_padding.get(a).copied().unwrap_or(0);
                let spread = d + d.saturating_sub(1) * interior;
                low_padding.get(a).copied().unwrap_or(0)
                    + spread
                    + high_padding.get(a).copied().unwrap_or(0)
            })
            .collect();
        let out_len: usize = out_shape.iter().product();
        let data = (0..out_len)
            .map(|flat| {
                let coords = Self::unravel(flat, &out_shape);
                let mut in_coords = vec![0usize; self.shape.len()];
                for (a, &c) in coords.iter().enumerate() {
                    let low = low_padding.get(a).copied().unwrap_or(0);
                    let interior = interior_padding.get(a).copied().unwrap_or(0);
                    if c < low {
                        return fill;
                    }
                    let offset = c - low;
                    if offset % (interior + 1) != 0 {
                        return fill; // lands in an interior gap
                    }
                    let idx = offset / (interior + 1);
                    if idx >= self.shape[a] {
                        return fill;
                    }
                    in_coords[a] = idx;
                }
                self.at(&in_coords)
            })
            .collect();
        Self {
            shape: out_shape,
            data,
        }
    }

    fn reverse(&self, axes: &[usize]) -> Self {
        let data = (0..self.count())
            .map(|flat| {
                let mut coords = Self::unravel(flat, &self.shape);
                for &a in axes {
                    if a < self.shape.len() {
                        coords[a] = self.shape[a] - 1 - coords[a];
                    }
                }
                self.at(&coords)
            })
            .collect();
        Self {
            shape: self.shape.clone(),
            data,
        }
    }

    /// Restricted to the `take`-shaped case this reference exercises: gathering whole slices
    /// along axis 0 with a rank-1 index vector. The full XLA dimension-numbering surface is
    /// beyond a reference carrier; see `take` for the ergonomic form.
    fn gather(
        &self,
        indices: &Self::Index,
        _offset_dims: &[usize],
        _collapsed_slice_dims: &[usize],
        _start_index_map: &[usize],
        _index_vector_dim: Option<usize>,
        _slice_sizes: &[usize],
    ) -> Self {
        self.take(indices, 0)
    }

    fn take(&self, indices: &Self::Index, axis: usize) -> Self {
        if axis >= self.shape.len() {
            return Self::empty();
        }
        let picks: Vec<usize> = indices.data.iter().map(|&i| i as usize).collect();
        let mut out_shape = self.shape.clone();
        out_shape[axis] = picks.len();
        let out_len: usize = out_shape.iter().product();
        let data = (0..out_len)
            .map(|flat| {
                let mut coords = Self::unravel(flat, &out_shape);
                let pick = picks.get(coords[axis]).copied().unwrap_or(0);
                if pick >= self.shape[axis] {
                    return f64::NAN;
                }
                coords[axis] = pick;
                self.at(&coords)
            })
            .collect();
        Self {
            shape: out_shape,
            data,
        }
    }

    fn dynamic_slice(&self, start_indices: &[Self::Index], slice_sizes: &[usize]) -> Self {
        let starts: Vec<usize> = start_indices.iter().map(|s| s.item() as usize).collect();
        let out_len: usize = slice_sizes.iter().product();
        let data = (0..out_len)
            .map(|flat| {
                let coords = Self::unravel(flat, slice_sizes);
                let shifted: Vec<usize> = coords
                    .iter()
                    .enumerate()
                    .map(|(a, &c)| {
                        // Clamp the window into bounds, as XLA's dynamic_slice does.
                        let start = starts.get(a).copied().unwrap_or(0);
                        let max_start = self.shape[a].saturating_sub(slice_sizes[a]);
                        start.min(max_start) + c
                    })
                    .collect();
                self.at(&shifted)
            })
            .collect();
        Self {
            shape: slice_sizes.to_vec(),
            data,
        }
    }

    fn dynamic_update_slice(&self, update: &Self, start_indices: &[Self::Index]) -> Self {
        let starts: Vec<usize> = start_indices.iter().map(|s| s.item() as usize).collect();
        let mut out = self.data.clone();
        for flat in 0..update.count() {
            let coords = Self::unravel(flat, &update.shape);
            let target: Vec<usize> = coords
                .iter()
                .enumerate()
                .map(|(a, &c)| {
                    let start = starts.get(a).copied().unwrap_or(0);
                    let max_start = self.shape[a].saturating_sub(update.shape[a]);
                    start.min(max_start) + c
                })
                .collect();
            out[Self::ravel(&target, &self.shape)] = update.data[flat];
        }
        Self {
            shape: self.shape.clone(),
            data: out,
        }
    }

    fn scatter_update_1d(&self, indices: &Self::Index, updates: &Self) -> Self {
        let mut out = self.data.clone();
        for (slot, &idx) in indices.data.iter().enumerate() {
            let idx = idx as usize;
            if idx < out.len() {
                out[idx] = updates.lane(slot);
            }
        }
        Self {
            shape: self.shape.clone(),
            data: out,
        }
    }

    fn scatter_add_1d(&self, indices: &Self::Index, updates: &Self) -> Self {
        let mut out = self.data.clone();
        for (slot, &idx) in indices.data.iter().enumerate() {
            let idx = idx as usize;
            if idx < out.len() {
                out[idx] += updates.lane(slot);
            }
        }
        Self {
            shape: self.shape.clone(),
            data: out,
        }
    }

    /// Restricted to the 1-D case, as with `gather`.
    fn scatter_update(
        &self,
        scatter_indices: &Self::Index,
        updates: &Self,
        _update_window_dims: &[usize],
        _inserted_window_dims: &[usize],
        _scatter_dims_to_operand_dims: &[usize],
        _index_vector_dim: Option<usize>,
        _indices_are_sorted: bool,
        _unique_indices: bool,
    ) -> Self {
        self.scatter_update_1d(scatter_indices, updates)
    }

    /// Restricted to the 1-D case, as with `gather`.
    fn scatter_add(
        &self,
        scatter_indices: &Self::Index,
        updates: &Self,
        _update_window_dims: &[usize],
        _inserted_window_dims: &[usize],
        _scatter_dims_to_operand_dims: &[usize],
        _index_vector_dim: Option<usize>,
        _indices_are_sorted: bool,
        _unique_indices: bool,
    ) -> Self {
        self.scatter_add_1d(scatter_indices, updates)
    }

    fn sum(self, axes: &[usize]) -> Self {
        self.reduce(axes, 0.0, |a, b| a + b)
    }

    fn prod(self, axes: &[usize]) -> Self {
        self.reduce(axes, 1.0, |a, b| a * b)
    }

    fn max(self, axes: &[usize]) -> Self {
        self.reduce(axes, f64::NEG_INFINITY, f64::max)
    }

    fn min(self, axes: &[usize]) -> Self {
        self.reduce(axes, f64::INFINITY, f64::min)
    }

    fn mean(self, axes: &[usize]) -> Self {
        let resolved = self.resolve_axes(axes);
        let divisor: usize = resolved.iter().map(|&a| self.shape[a]).product();
        let summed = self.reduce(axes, 0.0, |a, b| a + b);
        summed.map(|x| x / divisor as f64)
    }

    fn any(self, axes: &[usize]) -> Self {
        self.reduce(axes, 0.0, |a, b| f64::from(a != 0.0 || b != 0.0))
    }

    fn all(self, axes: &[usize]) -> Self {
        self.reduce(axes, 1.0, |a, b| f64::from(a != 0.0 && b != 0.0))
    }

    /// Rank-2 `(m,k)·(k,n)` matrix product; rank-1 operands contract as a dot product.
    fn matmul(self, rhs: Self) -> Self {
        match (self.shape.as_slice(), rhs.shape.as_slice()) {
            ([k_l], [k_r]) if k_l == k_r => Self::point(
                self.data
                    .iter()
                    .zip(&rhs.data)
                    .map(|(&a, &b)| a * b)
                    .sum::<f64>(),
            ),
            ([m, k_l], [k_r, n]) if k_l == k_r => {
                let (m, k, n) = (*m, *k_l, *n);
                let mut data = vec![0.0; m * n];
                for i in 0..m {
                    for j in 0..n {
                        data[i * n + j] = (0..k)
                            .map(|p| self.data[i * k + p] * rhs.data[p * n + j])
                            .sum();
                    }
                }
                Self::new(&[m, n], data)
            }
            _ => Self::empty(),
        }
    }
}
