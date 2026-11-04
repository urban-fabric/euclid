use euclid_numeric::{Boolean, Compare, Element, NormedDivisionAlgebra, Real, Ring, Semiring};

/// **Rank n.** An n-dimensional array over a carrier element type: the dtype-agnostic array
/// notions — shape, reshape, indexing/gather, and tensor selection. A vector is just a rank-1
/// tensor and a matrix a rank-2 tensor; there are no separate vector/matrix contracts.
///
/// The bound is [`Clone`], not [`Ring`], so shape/reshape/select carry no arithmetic demand
/// and boolean tensors can participate without pretending to be rings. The element kind is a
/// refinement that individual methods ask for: [`sum`](Tensor::sum) and
/// [`matmul`](Tensor::matmul) require [`Ring`], [`max`](Tensor::max) requires [`Real`], and a
/// boolean tensor can implement [`Boolean`].
///
/// Axis naming, mask representation, broadcast/gather/scatter details, and precision policy
/// are outside this contract.
pub trait Tensor: Clone {
    /// The integer index tensor. Unified-dtype implementations can use `Self`; typed
    /// implementations can point at their integer tensor peer.
    type Index: Clone;

    /// The axis lengths, returned owned — a tensor's shape may be dynamic metadata an implementation
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

    /// A tensor of zeros. Bounded by [`Element`] rather than a division algebra, so it serves
    /// boolean, natural, integer, real, and complex tensors alike.
    fn zeros(shape: &[usize]) -> Self
    where
        Self: Element,
    {
        Self::full(shape, Self::zero())
    }

    /// A tensor with the same shape as `self`, filled from a scalar-like `value`.
    fn full_like(&self, value: Self) -> Self {
        Self::full(&self.shape(), value)
    }

    /// A tensor of zeros with the same shape as `self` (see [`zeros`](Self::zeros)).
    fn zeros_like(&self) -> Self
    where
        Self: Element,
    {
        Self::zeros(&self.shape())
    }

    /// A tensor with `shape`, materialized from row-major host `data`. The way a literal array
    /// — weights read from disk, a fixed kernel, a test fixture — enters the contract, since
    /// every other constructor only broadcasts a single scalar.
    ///
    /// `data.len()` must equal the product of `shape`. The element type is `f64`, hence the
    /// [`Real`] bound; index/integer carriers build their literals through
    /// [`iota`](super::TensorIndexing::iota) and the gather/scatter family instead.
    fn from_host(data: &[f64], shape: &[usize]) -> Self
    where
        Self: Real;

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

    /// Full gather with explicit dimension numbers.
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

    /// 1-D scatter-update: `out[indices[i]] = updates[i]`.
    fn scatter_update_1d(&self, indices: &Self::Index, updates: &Self) -> Self;

    /// 1-D scatter-add: `out[indices[i]] += updates[i]`.
    fn scatter_add_1d(&self, indices: &Self::Index, updates: &Self) -> Self
    where
        Self: Ring;

    /// General scatter that *replaces* slices of `self` at the positions named by
    /// `scatter_indices` with `updates`, the structural inverse of [`gather`](Self::gather)
    /// and its mirror in dimension numbering: `update_window_dims` name the update axes that
    /// map onto operand windows, `inserted_window_dims` the operand axes collapsed away,
    /// `scatter_dims_to_operand_dims` how index components address operand axes, and
    /// `index_vector_dim` which axis of `scatter_indices` carries an index vector
    /// (`None` for an implicit trailing scalar). `indices_are_sorted` / `unique_indices`
    /// are correctness-preserving hints an implementation may exploit. The 1-D
    /// [`scatter_update_1d`](Self::scatter_update_1d) is the common special case.
    #[allow(clippy::too_many_arguments)]
    fn scatter_update(
        &self,
        scatter_indices: &Self::Index,
        updates: &Self,
        update_window_dims: &[usize],
        inserted_window_dims: &[usize],
        scatter_dims_to_operand_dims: &[usize],
        index_vector_dim: Option<usize>,
        indices_are_sorted: bool,
        unique_indices: bool,
    ) -> Self;

    /// General scatter that *adds* `updates` into `self` at the scattered positions,
    /// accumulating where indices collide (the additive combiner, hence [`Ring`]). The
    /// dimension numbers match [`scatter_update`](Self::scatter_update); the 1-D
    /// [`scatter_add_1d`](Self::scatter_add_1d) is the common special case.
    #[allow(clippy::too_many_arguments)]
    fn scatter_add(
        &self,
        scatter_indices: &Self::Index,
        updates: &Self,
        update_window_dims: &[usize],
        inserted_window_dims: &[usize],
        scatter_dims_to_operand_dims: &[usize],
        index_vector_dim: Option<usize>,
        indices_are_sorted: bool,
        unique_indices: bool,
    ) -> Self
    where
        Self: Ring;

    // --- reductions over the given axes ---
    fn sum(self, axes: &[usize]) -> Self
    where
        Self: Ring;
    /// Product over axes.
    fn prod(self, axes: &[usize]) -> Self
    where
        Self: Semiring;
    /// Reduce-max (the kernel behind log-sum-exp). Requires a real, ordered scalar.
    fn max(self, axes: &[usize]) -> Self
    where
        Self: Real;
    /// Reduce-min. Requires a real, ordered scalar.
    fn min(self, axes: &[usize]) -> Self
    where
        Self: Real;
    /// Arithmetic mean over axes — `sum / count`. Real **or** complex: mean needs a field to
    /// divide by the count, not the ordering that [`max`](Self::max)/[`min`](Self::min) require.
    fn mean(self, axes: &[usize]) -> Self
    where
        Self: NormedDivisionAlgebra;
    /// Boolean OR reduction over axes.
    fn any(self, axes: &[usize]) -> Self
    where
        Self: Boolean;
    /// Boolean AND reduction over axes.
    fn all(self, axes: &[usize]) -> Self
    where
        Self: Boolean;

    /// The tensor contraction (matmul / batched-matmul / einsum); the implementation names the axes.
    fn matmul(self, rhs: Self) -> Self
    where
        Self: Ring;

    // --- tensor select ------------------------------------------------------

    /// Elementwise tensor conditional: `mask ? on_true : on_false`.
    ///
    /// This is the tensor-level spelling of [`Compare::if_then_else`]. Implementations
    /// with vector masks lower this to an elementwise select operation.
    fn if_then_else(mask: &<Self as Compare>::Mask, on_true: Self, on_false: Self) -> Self
    where
        Self: Compare,
    {
        <Self as Compare>::if_then_else(mask, on_true, on_false)
    }
}
