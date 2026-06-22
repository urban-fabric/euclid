//! Lightweight dtype and tensor-spec metadata.
//!
//! These contracts let generic math code describe and inspect abstract values without taking
//! ownership of buffers, devices, or execution. Backends remain responsible for storage and
//! lowering; euclid only names the portable metadata.

/// Backend-neutral element dtype metadata.
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

    pub const fn is_bool(self) -> bool {
        matches!(self, Self::Bool)
    }

    pub const fn is_signed_integer(self) -> bool {
        matches!(self, Self::I8 | Self::I16 | Self::I32 | Self::I64)
    }

    pub const fn is_unsigned_integer(self) -> bool {
        matches!(self, Self::U8 | Self::U16 | Self::U32 | Self::U64)
    }

    pub const fn is_integer(self) -> bool {
        self.is_signed_integer() || self.is_unsigned_integer()
    }

    pub const fn is_float(self) -> bool {
        matches!(self, Self::F16 | Self::Bf16 | Self::F32 | Self::F64)
    }

    pub const fn is_complex(self) -> bool {
        matches!(self, Self::C64 | Self::C128)
    }

    pub const fn bit_width(self) -> u32 {
        match self {
            Self::Bool | Self::I8 | Self::U8 => 8,
            Self::I16 | Self::U16 | Self::F16 | Self::Bf16 => 16,
            Self::I32 | Self::U32 | Self::F32 => 32,
            Self::I64 | Self::U64 | Self::F64 | Self::C64 => 64,
            Self::C128 => 128,
        }
    }

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
    pub fn new(dtype: DType, shape: impl Into<Vec<usize>>) -> Self {
        Self {
            dtype,
            shape: shape.into(),
        }
    }

    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn into_shape(self) -> Vec<usize> {
        self.shape
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }

    pub fn num_elements(&self) -> Option<usize> {
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
    fn dtype(&self) -> Option<DType>;

    fn shape(&self) -> Vec<usize>;

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
    fn to_dtype(&self, dtype: DType) -> Self;
}

#[cfg(test)]
mod tests {
    use super::{DType, TensorSpec};

    #[test]
    fn dtype_classification_is_small_and_backend_neutral() {
        assert!(DType::Bool.is_bool());
        assert!(DType::I32.is_signed_integer());
        assert!(DType::U64.is_unsigned_integer());
        assert!(DType::F32.is_float());
        assert!(DType::C128.is_complex());
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
        assert!(TensorSpec::new(DType::I64, []).is_scalar());
    }
}
