//! Lightweight dtype and tensor-spec metadata.
//!
//! These contracts let generic math code describe and inspect abstract values without taking
//! ownership of buffers, devices, or execution. Backends remain responsible for storage and
//! lowering; euclid only names the portable metadata.

/// Backend-neutral element dtype metadata.
///
/// The variants are the names themselves and carry no doc comments, which is why `missing_docs`
/// is allowed here. `/// 32-bit float.` above `F32` is not information a reader lacks, and
/// fifteen such lines would push the variant list off a screen it currently fits on. Where a
/// variant *does* need saying — `Bf16`'s exponent range, or that `C64` means two `f32`s rather
/// than a 64-bit component — it is said on the method that observes it
/// ([`bit_width`](Self::bit_width)) rather than repeated per variant.
#[allow(missing_docs)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DType {
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    Bf16,
    F32,
    F64,
    C64,
    C128,
}

impl DType {
    /// The lowercase portable name — `"f64"`, `"bf16"`, `"c128"`.
    ///
    /// Stable across versions, because backends use it in serialised plans and error messages
    /// where a renamed string is a breaking change rather than a cosmetic one.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::C64 => "c64",
            Self::C128 => "c128",
        }
    }

    /// The boolean dtype, which is the mask dtype for a unified-dtype carrier.
    pub const fn is_bool(self) -> bool {
        matches!(self, Self::Bool)
    }

    /// One of `I8`…`I64`. `Bool` is not an integer here, deliberately — it is a mask.
    pub const fn is_signed_integer(self) -> bool {
        matches!(self, Self::I8 | Self::I16 | Self::I32 | Self::I64)
    }

    /// One of `U8`…`U64`.
    pub const fn is_unsigned_integer(self) -> bool {
        matches!(self, Self::U8 | Self::U16 | Self::U32 | Self::U64)
    }

    /// Signed or unsigned, excluding `Bool`.
    pub const fn is_integer(self) -> bool {
        self.is_signed_integer() || self.is_unsigned_integer()
    }

    /// A real floating dtype, including the two 16-bit formats. **Not** the complex ones, whose
    /// components are floats — use [`is_complex`](Self::is_complex) for those.
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F16 | Self::Bf16 | Self::F32 | Self::F64)
    }

    /// `C64` or `C128`.
    pub const fn is_complex(self) -> bool {
        matches!(self, Self::C64 | Self::C128)
    }

    /// Width of one element in bits, counting **both components** of a complex dtype: `C64` is
    /// 64 bits total (two `f32`s), matching the name rather than the component width.
    pub const fn bit_width(self) -> u32 {
        match self {
            Self::Bool | Self::I8 | Self::U8 => 8,
            Self::I16 | Self::U16 | Self::F16 | Self::Bf16 => 16,
            Self::I32 | Self::U32 | Self::F32 => 32,
            Self::I64 | Self::U64 | Self::F64 | Self::C64 => 64,
            Self::C128 => 128,
        }
    }

    /// Width of one element in bytes. `Bool` reports 1, since no portable backend addresses a
    /// bit — a bit-packed mask is a backend's storage choice and not visible here.
    pub const fn element_size_in_bytes(self) -> usize {
        (self.bit_width() as usize) / 8
    }
}

impl core::fmt::Display for DType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Shape and dtype for one dense tensor value.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorSpec {
    dtype: DType,
    shape: Vec<usize>,
}

impl TensorSpec {
    /// A spec from a dtype and a shape. An empty shape is rank 0, i.e. a scalar.
    pub fn new(dtype: DType, shape: impl Into<Vec<usize>>) -> Self {
        Self {
            dtype,
            shape: shape.into(),
        }
    }

    /// The element dtype.
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// The axis lengths.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Take the shape, consuming the spec — for a caller building a new spec from this one
    /// without a copy.
    pub fn into_shape(self) -> Vec<usize> {
        self.shape
    }

    /// Number of axes.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Rank 0. Note this is *not* "holds one element": a shape of `[1, 1]` holds one element at
    /// rank 2, and the distinction matters to every structural operation.
    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }

    /// The product of the axis lengths, or `None` if that **overflows `usize`**.
    ///
    /// `Option` rather than a saturating or panicking product because a shape can arrive from a
    /// deserialised plan or a user request, where a nonsensical product is input rather than a
    /// bug. A rank-0 spec gives `Some(1)`, the empty product, which is what makes a scalar
    /// behave as one element in a generic size calculation.
    pub fn num_elements(&self) -> Option<usize> {
        if self.shape.contains(&0) {
            return Some(0);
        }
        self.shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
    }
}

/// Values that can expose best-effort tensor metadata.
///
/// Metadata may be unavailable for deferred-error values, abstract carriers, or backends that do
/// not track dtype. This is inspection only: it does not imply host materialization.
pub trait HasTensorSpec {
    /// The element dtype, or `None` where the carrier does not track one.
    fn dtype(&self) -> Option<DType>;

    /// The axis lengths. Returned owned, since a shape may be metadata the carrier computes
    /// rather than stores — the same reason [`Tensor::shape`](crate::Tensor::shape) does.
    fn shape(&self) -> Vec<usize>;

    /// Dtype and shape together, or `None` when the dtype is unavailable.
    ///
    /// Defaulted, and an implementor should override only to avoid recomputing a shape it
    /// already holds alongside its dtype.
    fn spec(&self) -> Option<TensorSpec> {
        self.dtype()
            .map(|dtype| TensorSpec::new(dtype, self.shape()))
    }
}

/// Optional dtype conversion capability.
///
/// Implementations decide the exact conversion semantics. For graph/tensor backends this usually
/// records a cast node; for eager carriers it may convert immediately.
pub trait Cast: Sized {
    /// Reinterpret or convert to `dtype`.
    ///
    /// **Rounding and out-of-range behaviour are the backend's to define** and are deliberately
    /// not fixed here: a narrowing float cast, or a float-to-integer cast of a value past the
    /// integer's range, has no portable answer worth imposing. A caller needing a specific
    /// convention should clamp or round before casting rather than rely on this.
    fn to_dtype(&self, dtype: DType) -> Self;
}

#[cfg(test)]
mod tests {
    use super::{DType, HasTensorSpec, TensorSpec};

    #[test]
    fn dtype_classification_is_small_and_backend_neutral() {
        let signed_integers = [DType::I8, DType::I16, DType::I32, DType::I64];
        let unsigned_integers = [DType::U8, DType::U16, DType::U32, DType::U64];
        let floats = [DType::F16, DType::Bf16, DType::F32, DType::F64];
        let complex = [DType::C64, DType::C128];
        let all = [
            DType::Bool,
            DType::I8,
            DType::I16,
            DType::I32,
            DType::I64,
            DType::U8,
            DType::U16,
            DType::U32,
            DType::U64,
            DType::F16,
            DType::Bf16,
            DType::F32,
            DType::F64,
            DType::C64,
            DType::C128,
        ];

        for dtype in all {
            assert_eq!(dtype.is_bool(), dtype == DType::Bool);
            assert_eq!(dtype.is_signed_integer(), signed_integers.contains(&dtype));
            assert_eq!(
                dtype.is_unsigned_integer(),
                unsigned_integers.contains(&dtype)
            );
            assert_eq!(
                dtype.is_integer(),
                signed_integers.contains(&dtype) || unsigned_integers.contains(&dtype)
            );
            assert_eq!(dtype.is_float(), floats.contains(&dtype));
            assert_eq!(dtype.is_complex(), complex.contains(&dtype));
        }

        assert_eq!(DType::Bf16.bit_width(), 16);
        assert_eq!(DType::C64.element_size_in_bytes(), 8);
        assert_eq!(DType::F64.to_string(), "f64");
    }

    #[test]
    fn tensor_spec_tracks_shape_dtype_and_element_count() {
        let spec = TensorSpec::new(DType::F32, [2, 3, 4]);
        assert_eq!(spec.dtype(), DType::F32);
        assert_eq!(spec.shape(), &[2, 3, 4]);
        assert_eq!(spec.rank(), 3);
        assert_eq!(spec.num_elements(), Some(24));
        assert!(!spec.is_scalar());
        assert_eq!(spec.into_shape(), vec![2, 3, 4]);
        assert!(TensorSpec::new(DType::I64, []).is_scalar());
    }

    #[test]
    fn element_count_is_invariant_under_permuting_empty_axes() {
        for shape in [[0, usize::MAX, 2], [usize::MAX, 0, 2], [usize::MAX, 2, 0]] {
            assert_eq!(TensorSpec::new(DType::F64, shape).num_elements(), Some(0));
        }
        assert_eq!(
            TensorSpec::new(DType::F64, [usize::MAX, 2]).num_elements(),
            None
        );
        assert_eq!(TensorSpec::new(DType::F64, []).num_elements(), Some(1));
    }

    #[test]
    fn default_metadata_combines_the_reported_shape_and_dtype() {
        struct Metadata(Option<DType>, Vec<usize>);
        impl HasTensorSpec for Metadata {
            fn dtype(&self) -> Option<DType> {
                self.0
            }

            fn shape(&self) -> Vec<usize> {
                self.1.clone()
            }
        }
        for shape in [vec![], vec![1], vec![0, 2], vec![3, 4, 5]] {
            let expected = TensorSpec::new(DType::U16, shape.clone());
            assert_eq!(
                Metadata(Some(DType::U16), shape.clone()).spec(),
                Some(expected)
            );
            assert_eq!(Metadata(None, shape).spec(), None);
        }
    }
}
