//! Opt-in capability extensions on [`Tensor`]. Each trait groups operations an implementation
//! *may* offer beyond the rank-n core, named so a caller can ask for the narrowest bound it
//! needs: statistical reductions, order statistics, general contraction, index/categorical
//! encoding, and structural conveniences.

use super::Tensor;
use euclid_numeric::{NormedDivisionAlgebra, Real, Ring, Semiring};

/// Statistical and scan reductions beyond the additive/ordered core on [`Tensor`].
///
/// Each composes from the base reductions, but an implementation may provide it as one stable kernel
/// (the max-shift trick for [`logsumexp`](Self::logsumexp), a fused scan for
/// [`cumsum`](Self::cumsum)), so they are named operations rather than caller-side formulas.
/// The bounds mirror the base trait: [`cumsum`](Self::cumsum) only needs a [`Ring`]; the
/// statistics need a real, ordered scalar ([`Real`]).
pub trait TensorReduce: Tensor {
    /// Inclusive cumulative sum along `axis`.
    fn cumsum(self, axis: usize) -> Self
    where
        Self: Ring;

    /// Population variance over `axes` (divisor `n`).
    ///
    /// **Do not implement this as `mean(x²) − mean(x)²`.** That form subtracts two
    /// nearly-equal large numbers whenever the mean dominates the spread, so it loses every
    /// significant digit and can even return a negative variance: for `[1e9, 1e9+1, …]` the
    /// true variance is 2 and the cancelling form returns garbage.
    ///
    /// Use a two-pass formulation — `mean(x)` first, then `mean((x − mean)²)` — or Welford's
    /// online recurrence where a single pass is required. Both cost one more traversal and
    /// remove the cancellation entirely.
    fn var(self, axes: &[usize]) -> Self
    where
        Self: Real;

    /// Population standard deviation over `axes` — the square root of [`var`](Self::var), and
    /// subject to the same stability requirement.
    fn std(self, axes: &[usize]) -> Self
    where
        Self: Real;

    /// Numerically stable `ln Σ exp(x)` over `axes` (the max-shift trick). The tensor peer of
    /// [`Real::ln_add_exp`](euclid_numeric::Real::ln_add_exp).
    fn logsumexp(self, axes: &[usize]) -> Self
    where
        Self: Real;

    /// Softmax along `axis` — `exp(x) / Σ exp(x)`, normalized so the axis sums to one.
    fn softmax(self, axis: usize) -> Self
    where
        Self: Real;

    /// Log-softmax along `axis` — `x − logsumexp(x)`, the stable log of
    /// [`softmax`](Self::softmax).
    fn log_softmax(self, axis: usize) -> Self
    where
        Self: Real;
}

/// Order statistics: sorting and the index-returning argmin/argmax/top-k family.
///
/// Value results stay in `Self`; index results use [`Self::Index`](Tensor::Index), the
/// implementation's integer index tensor — the same type [`take`](Tensor::take) and
/// [`gather`](Tensor::gather) consume. Ordering needs a real scalar, so every method asks
/// for [`Real`].
pub trait TensorSort: Tensor {
    /// Sort the values along `axis`, ascending unless `descending`.
    fn sort(self, axis: usize, descending: bool) -> Self
    where
        Self: Real;

    /// The indices that would sort `self` along `axis`.
    fn argsort(&self, axis: usize, descending: bool) -> Self::Index
    where
        Self: Real;

    /// The index of the maximum along `axis`.
    fn argmax(&self, axis: usize) -> Self::Index
    where
        Self: Real;

    /// The index of the minimum along `axis`.
    fn argmin(&self, axis: usize) -> Self::Index
    where
        Self: Real;

    /// The `k` largest values (or smallest, when not `largest`) along the last axis and their
    /// indices, as `(values, indices)`.
    fn top_k(&self, k: usize, largest: bool) -> (Self, Self::Index)
    where
        Self: Real;
}

/// General tensor contraction beyond the single [`matmul`](Tensor::matmul): explicit
/// contracting/batch axes and the Einstein-summation spelling. A contraction is a sum of
/// products, so every method asks for [`Ring`].
pub trait TensorContract: Tensor {
    /// Contract `self` against `rhs` over the named contracting axes, batching over the named
    /// batch axes (the general dot operation).
    fn dot_general(
        &self,
        rhs: &Self,
        lhs_contracting: &[usize],
        rhs_contracting: &[usize],
        lhs_batch: &[usize],
        rhs_batch: &[usize],
    ) -> Self
    where
        Self: Ring;

    /// Einstein summation over a single tensor (e.g. `"ii->i"` for the diagonal).
    fn einsum1(&self, config: &str) -> Self
    where
        Self: Ring;

    /// Einstein summation over two tensors (e.g. `"ij,jk->ik"` for matmul).
    fn einsum2(&self, rhs: &Self, config: &str) -> Self
    where
        Self: Ring;
}

/// Index construction and categorical encoding — the seedstock of gather/scatter and the
/// one-hot / histogram primitives that bridge integer indices to numeric tensors.
pub trait TensorIndexing: Tensor {
    /// The integer index vector `[0, 1, …, len − 1]`. The base of ranges, indexing, and
    /// one-hot encoding.
    fn iota(len: usize) -> Self::Index;

    /// One-hot encode an integer index tensor into a tensor of `1`s and `0`s: `1` where the
    /// index equals the column and `0` elsewhere, with `depth` columns. It only needs a
    /// semiring's `0`/`1` ([`Semiring`]), so the output may be integer, real, or complex.
    fn one_hot(indices: &Self::Index, depth: usize) -> Self
    where
        Self: Semiring;

    /// Count occurrences of each value in `[0, length)` across an integer index vector,
    /// returning the counts (the histogram / contingency primitive). A sum of one-hot rows, so
    /// it too needs only a [`Semiring`] — counts are naturally integer.
    fn bincount(indices: &Self::Index, length: usize) -> Self
    where
        Self: Semiring;
}

/// Structural conveniences beyond the [`Tensor`] core: circular shift, repetition, and
/// value clamping. ([`reverse`](Tensor::reverse) and
/// [`slice_in_dim`](Tensor::slice_in_dim) already cover flip/slice.)
pub trait TensorTransform: Tensor {
    /// Circularly shift elements along `axis` by `shift` (wrapping around; negative shifts
    /// toward index zero).
    fn roll(&self, shift: isize, axis: usize) -> Self;

    /// Repeat `self` `reps` times along `axis`.
    fn tile(&self, reps: usize, axis: usize) -> Self;

    /// Clamp every element into `[lo, hi]`, with scalar-like `lo`/`hi`.
    fn clip(self, lo: Self, hi: Self) -> Self
    where
        Self: Real;
}

/// Sliding-window reductions and the pooling conveniences built on them — the local
/// (neighbourhood) counterpart to the whole-axis reductions on the base [`Tensor`].
///
/// A window of `window_dimensions`, stepped by `window_strides`, slides over the tensor and is
/// reduced at each position; `padding` gives the `(low, high)` pad per axis. The bounds mirror
/// the base reductions — sum needs a [`Ring`], max a real ordered scalar, mean a field to
/// divide by the window count. The pooling defaults delegate to the windowed reductions, so an
/// implementation only supplies the three kernels.
pub trait TensorWindow: Tensor {
    /// Windowed sum (`jax.lax.reduce_window` with addition).
    fn reduce_window_sum(
        &self,
        window_dimensions: &[usize],
        window_strides: &[usize],
        padding: &[(usize, usize)],
    ) -> Self
    where
        Self: Ring;

    /// Windowed maximum (`jax.lax.reduce_window` with max).
    fn reduce_window_max(
        &self,
        window_dimensions: &[usize],
        window_strides: &[usize],
        padding: &[(usize, usize)],
    ) -> Self
    where
        Self: Real;

    /// Windowed mean — the windowed sum divided by the window's element count.
    fn reduce_window_mean(
        &self,
        window_dimensions: &[usize],
        window_strides: &[usize],
        padding: &[(usize, usize)],
    ) -> Self
    where
        Self: NormedDivisionAlgebra;

    /// 2-D max-pooling over the trailing spatial axes of an `NCHW` tensor — a max window over
    /// the `H`/`W` axes only, the everyday case of [`reduce_window_max`](Self::reduce_window_max).
    fn max_pool2d_nchw(
        &self,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: [(usize, usize); 2],
    ) -> Self
    where
        Self: Real,
    {
        self.reduce_window_max(
            &[1, 1, kernel.0, kernel.1],
            &[1, 1, stride.0, stride.1],
            &[(0, 0), (0, 0), padding[0], padding[1]],
        )
    }

    /// 2-D average-pooling over the trailing spatial axes of an `NCHW` tensor — the mean
    /// counterpart of [`max_pool2d_nchw`](Self::max_pool2d_nchw).
    fn avg_pool2d_nchw(
        &self,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: [(usize, usize); 2],
    ) -> Self
    where
        Self: NormedDivisionAlgebra,
    {
        self.reduce_window_mean(
            &[1, 1, kernel.0, kernel.1],
            &[1, 1, stride.0, stride.1],
            &[(0, 0), (0, 0), padding[0], padding[1]],
        )
    }
}

/// N-dimensional convolution — the structured contraction behind every convolutional network.
/// A convolution is a sum of products, so it asks for [`Ring`]; like the other capability
/// extensions it sits outside the core because not every carrier supplies it.
pub trait TensorConv: Tensor {
    /// General batched, grouped, dilated convolution (`jax.lax.conv_general_dilated` /
    /// XLA `ConvGeneralDilated`). `self` is the input and `rhs` the kernel; the dimension
    /// arguments name which axis plays the batch / feature / spatial role on each side, and
    /// `padding` gives the `(low, high)` pad per spatial axis.
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
    ) -> Self
    where
        Self: Ring;

    /// 2-D convolution for the common `NCHW` input / `OIHW` kernel layout — the everyday case
    /// of [`conv_general_dilated`](Self::conv_general_dilated).
    fn conv2d_nchw(
        &self,
        kernel: &Self,
        stride: (usize, usize),
        padding: [(usize, usize); 2],
        lhs_dilation: (usize, usize),
        rhs_dilation: (usize, usize),
        feature_group_count: usize,
    ) -> Self
    where
        Self: Ring,
    {
        self.conv_general_dilated(
            kernel,
            &[stride.0, stride.1],
            &padding,
            &[lhs_dilation.0, lhs_dilation.1],
            &[rhs_dilation.0, rhs_dilation.1],
            0,
            1,
            &[2, 3],
            1,
            0,
            &[2, 3],
            feature_group_count,
            1,
        )
    }
}

/// Which discrete Fourier transform [`TensorFft::fft`] computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FftKind {
    /// Forward complex→complex transform.
    Forward,
    /// Inverse complex→complex transform.
    Inverse,
    /// Forward real→complex transform (`rfft`).
    RealForward,
    /// Inverse complex→real transform (`irfft`).
    RealInverse,
}

/// The discrete Fourier transform over the trailing axes — the spectral primitive. The single
/// required [`fft`](Self::fft) takes an [`FftKind`]; the named methods are conveniences.
pub trait TensorFft: Tensor {
    /// Transform `self` over its trailing `lengths.len()` axes, `lengths` giving the transform
    /// size per axis, with [`FftKind`] selecting forward/inverse and real/complex.
    fn fft(&self, kind: FftKind, lengths: &[usize]) -> Self;

    /// Forward complex FFT — [`fft`](Self::fft) with [`FftKind::Forward`].
    fn fft_forward(&self, lengths: &[usize]) -> Self {
        self.fft(FftKind::Forward, lengths)
    }

    /// Inverse complex FFT — [`fft`](Self::fft) with [`FftKind::Inverse`].
    fn ifft(&self, lengths: &[usize]) -> Self {
        self.fft(FftKind::Inverse, lengths)
    }

    /// Forward real FFT — [`fft`](Self::fft) with [`FftKind::RealForward`].
    fn rfft(&self, lengths: &[usize]) -> Self {
        self.fft(FftKind::RealForward, lengths)
    }

    /// Inverse real FFT — [`fft`](Self::fft) with [`FftKind::RealInverse`].
    fn irfft(&self, lengths: &[usize]) -> Self {
        self.fft(FftKind::RealInverse, lengths)
    }
}
