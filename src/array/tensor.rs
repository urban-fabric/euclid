use super::control_flow;
use crate::numeric::{LazyBool, LazyCompare, LazyReal, LazyRing, LazySemiring};

/// **Rank n.** An n-dimensional array over a lazy element type: the dtype-agnostic array
/// notions — shape, reshape, indexing/gather, and tensor control flow. `LazyTensor: Clone`,
/// so boolean tensors can participate without pretending to be rings. The element kind is a
/// refinement: additive reductions and contractions require [`LazyRing`], a real tensor
/// also implements [`LazyReal`], a complex one [`LazyComplex`](crate::numeric::LazyComplex),
/// an integer one [`LazyInteger`](crate::numeric::LazyInteger), and a boolean tensor can
/// implement [`LazyBool`](crate::numeric::LazyBool). A rank-1 tensor over a field is also
/// a [`LazyVector`](super::LazyVector), a rank-2 one a [`LazyMatrix`](super::LazyMatrix).
///
/// The bound is [`Clone`], not [`LazyRing`], so shape/reshape/control-flow carry no
/// arithmetic demand; [`sum`](LazyTensor::sum), [`matmul`](LazyTensor::matmul),
/// [`max`](LazyTensor::max), and the rank-1/2 linear algebra ask for the algebra they need.
/// Axis naming, mask representation, broadcast/gather/scatter details, and precision policy
/// are outside this contract.
pub trait LazyTensor: Clone {
    /// The backend's integer index tensor. Unified-dtype backends can use `Self`;
    /// typed backends can point at their integer tensor peer.
    type Index: Clone;

    /// The axis lengths, returned owned — a tensor's shape is dynamic metadata a backend
    /// computes, not necessarily a stored slice.
    fn shape(&self) -> Vec<usize>;
    fn rank(&self) -> usize {
        self.shape().len()
    }

    /// Reshape to a compatible shape.
    fn reshape(self, shape: &[usize]) -> Self;

    // --- constructors -------------------------------------------------------

    /// A tensor with `shape`, filled from a scalar-like `value`.
    fn full(shape: &[usize], value: Self) -> Self;

    /// A real tensor of zeros.
    fn zeros(shape: &[usize]) -> Self
    where
        Self: LazyReal,
    {
        Self::full(shape, Self::zero())
    }

    /// A tensor with the same shape as `self`, filled from a scalar-like `value`.
    fn full_like(&self, value: Self) -> Self {
        Self::full(&self.shape(), value)
    }

    /// A real tensor of zeros with the same shape as `self`.
    fn zeros_like(&self) -> Self
    where
        Self: LazyReal,
    {
        Self::zeros(&self.shape())
    }

    // --- structural indexing and shape ops ---------------------------------

    /// Broadcast to `out_dims`, mapping each input axis to the output axis named
    /// in `broadcast_dims`.
    fn broadcast_in_dim(&self, out_dims: &[usize], broadcast_dims: &[usize]) -> Self;

    /// General axis permutation.
    fn transpose_axes(&self, axes: &[usize]) -> Self;

    /// Remove a size-1 axis.
    fn squeeze_axis(&self, axis: isize) -> Self;

    /// Slice one axis with host-known bounds and stride.
    fn slice_in_dim(&self, start: usize, limit: usize, stride: usize, axis: usize) -> Self;

    /// Concatenate tensors along `axis`.
    fn concatenate(inputs: &[Self], axis: usize) -> Self;

    /// Stack tensors along a new `axis`.
    fn stack(inputs: &[Self], axis: usize) -> Self;

    /// Pad with scalar-like `value`, low/high edge padding, and interior padding per axis.
    fn pad(
        &self,
        value: Self,
        low_padding: &[usize],
        high_padding: &[usize],
        interior_padding: &[usize],
    ) -> Self;

    /// Reverse the given axes.
    fn reverse(&self, axes: &[usize]) -> Self;

    /// Full backend gather with explicit dimension numbers.
    #[allow(clippy::too_many_arguments)]
    fn gather(
        &self,
        indices: &Self::Index,
        offset_dims: &[usize],
        collapsed_slice_dims: &[usize],
        start_index_map: &[usize],
        index_vector_dim: Option<usize>,
        slice_sizes: &[usize],
    ) -> Self;

    /// Gather slices along a single axis using integer `indices`.
    fn take(&self, indices: &Self::Index, axis: usize) -> Self;

    /// Dynamic slice with runtime scalar integer starts.
    fn dynamic_slice(&self, start_indices: &[Self::Index], slice_sizes: &[usize]) -> Self;

    /// Functional dynamic update of a rectangular slice.
    fn dynamic_update_slice(&self, update: &Self, start_indices: &[Self::Index]) -> Self;

    /// Vector scatter-update: `out[indices[i]] = updates[i]`.
    fn scatter_update_1d(&self, indices: &Self::Index, updates: &Self) -> Self;

    /// Vector scatter-add: `out[indices[i]] += updates[i]`.
    fn scatter_add_1d(&self, indices: &Self::Index, updates: &Self) -> Self
    where
        Self: LazyRing;

    // --- reductions over the given axes ---
    fn sum(self, axes: &[usize]) -> Self
    where
        Self: LazyRing;
    /// Product over axes.
    fn prod(self, axes: &[usize]) -> Self
    where
        Self: LazySemiring;
    /// Reduce-max (the kernel behind log-sum-exp). Requires a real, ordered scalar.
    fn max(self, axes: &[usize]) -> Self
    where
        Self: LazyReal;
    /// Reduce-min. Requires a real, ordered scalar.
    fn min(self, axes: &[usize]) -> Self
    where
        Self: LazyReal;
    /// Arithmetic mean over axes.
    fn mean(self, axes: &[usize]) -> Self
    where
        Self: LazyReal;
    /// Boolean OR reduction over axes.
    fn any(self, axes: &[usize]) -> Self
    where
        Self: LazyBool;
    /// Boolean AND reduction over axes.
    fn all(self, axes: &[usize]) -> Self
    where
        Self: LazyBool;

    /// The tensor contraction (matmul / batched-matmul / einsum); the backend names the axes.
    fn matmul(self, rhs: Self) -> Self
    where
        Self: LazyRing;

    // --- tensor control flow ------------------------------------------------

    /// Elementwise tensor conditional: `mask ? on_true : on_false`.
    ///
    /// This is the tensor-level spelling of [`LazyCompare::select`]. Backends
    /// with vector masks lower this to an elementwise select/where operation.
    fn where_cond(mask: &<Self as LazyCompare>::Mask, on_true: Self, on_false: Self) -> Self
    where
        Self: LazyCompare,
    {
        control_flow::where_cond(mask, on_true, on_false)
    }

    /// Alias for [`where_cond`](Self::where_cond), named like an ordinary conditional
    /// expression.
    fn if_then_else(mask: &<Self as LazyCompare>::Mask, on_true: Self, on_false: Self) -> Self
    where
        Self: LazyCompare,
    {
        Self::where_cond(mask, on_true, on_false)
    }
}

/// Explicit graph control-flow contract for tensor carriers.
pub trait LazyTensorControlFlow: LazyTensor + LazyCompare {
    /// Graph-level conditional over tensor values.
    fn cond<T, F>(
        pred: &Self::Mask,
        true_arg: Self,
        true_branch: T,
        false_arg: Self,
        false_branch: F,
    ) -> Self
    where
        T: FnOnce(Self) -> Self,
        F: FnOnce(Self) -> Self;

    /// Graph-level while loop over tensor state.
    fn while_loop<C, B>(init: Self, cond: C, body: B) -> Self
    where
        C: Fn(&Self) -> Self::Mask,
        B: Fn(Self) -> Self;
}
