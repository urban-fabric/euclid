use crate::DType;

/// Random-bit-generator algorithm identifiers when an implementation exposes one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RandomAlgorithm {
    /// Implementation-defined algorithm and state shape.
    Default,
    /// Counter-based ThreeFry family.
    ThreeFry,
    /// Counter-based Philox family.
    Philox,
}

/// Backend-neutral dtype tags for random samplers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RandomDType {
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
    F32,
    Bf16,
    F64,
}

impl RandomDType {
    pub const fn is_bool(self) -> bool {
        self as u8 == Self::Bool as u8
    }

    pub const fn is_signed_integer(self) -> bool {
        let dtype = self as u8;
        dtype >= Self::I8 as u8 && dtype <= Self::I64 as u8
    }

    pub const fn is_unsigned_integer(self) -> bool {
        let dtype = self as u8;
        dtype >= Self::U8 as u8 && dtype <= Self::U64 as u8
    }

    pub const fn is_integer(self) -> bool {
        self.is_signed_integer() || self.is_unsigned_integer()
    }

    pub const fn is_float(self) -> bool {
        let dtype = self as u8;
        dtype >= Self::F16 as u8 && dtype <= Self::F64 as u8
    }

    pub const fn bit_width(self) -> u32 {
        match self {
            Self::Bool | Self::I8 | Self::U8 => 8,
            Self::I16 | Self::U16 | Self::F16 | Self::Bf16 => 16,
            Self::I32 | Self::U32 | Self::F32 => 32,
            Self::I64 | Self::U64 | Self::F64 => 64,
        }
    }
}

impl From<RandomDType> for DType {
    fn from(value: RandomDType) -> Self {
        match value {
            RandomDType::Bool => Self::Bool,
            RandomDType::I8 => Self::I8,
            RandomDType::I16 => Self::I16,
            RandomDType::I32 => Self::I32,
            RandomDType::I64 => Self::I64,
            RandomDType::U8 => Self::U8,
            RandomDType::U16 => Self::U16,
            RandomDType::U32 => Self::U32,
            RandomDType::U64 => Self::U64,
            RandomDType::F16 => Self::F16,
            RandomDType::F32 => Self::F32,
            RandomDType::Bf16 => Self::Bf16,
            RandomDType::F64 => Self::F64,
        }
    }
}

impl TryFrom<DType> for RandomDType {
    type Error = DType;

    fn try_from(value: DType) -> Result<Self, Self::Error> {
        match value {
            DType::Bool => Ok(Self::Bool),
            DType::I8 => Ok(Self::I8),
            DType::I16 => Ok(Self::I16),
            DType::I32 => Ok(Self::I32),
            DType::I64 => Ok(Self::I64),
            DType::U8 => Ok(Self::U8),
            DType::U16 => Ok(Self::U16),
            DType::U32 => Ok(Self::U32),
            DType::U64 => Ok(Self::U64),
            DType::F16 => Ok(Self::F16),
            DType::Bf16 => Ok(Self::Bf16),
            DType::F32 => Ok(Self::F32),
            DType::F64 => Ok(Self::F64),
            DType::C64 | DType::C128 => Err(value),
        }
    }
}
