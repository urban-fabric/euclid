use crate::Factor;
use crate::conversion::dim;
use bytemuck::{Pod, Zeroable};

#[repr(transparent)]
#[derive(
    Copy,
    Clone,
    Debug,
    Default,
    PartialEq,
    PartialOrd,
    Pod,
    Zeroable,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(transparent)]
pub struct Percentage(pub(crate) f64);

impl Percentage {
    pub const ZERO: Self = Self(0.0);

    pub const FULL: Self = Self(100.0);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub fn from_fraction(f: Factor) -> Self {
        Self(f.as_f64() * 100.0)
    }

    pub fn to_fraction(self) -> Factor {
        Factor::new(self.0 / 100.0)
    }

    pub const fn as_percent(self) -> f64 {
        self.0
    }

    pub fn is_zero(self) -> bool {
        self.0 == 0.0
    }

    pub fn is_non_negative(self) -> bool {
        self.0 >= 0.0
    }

    pub fn is_positive(self) -> bool {
        self.0 > 0.0
    }

    pub fn is_in_range(self) -> bool {
        self.0.is_finite() && (0.0..=100.0).contains(&self.0)
    }
}

impl crate::conversion::UnitOfMeasure for Percentage {
    type Dim = dim::Dimensionless;
    const TO_PRIMARY: f64 = 1.0 / 100.0;
}

const _: () = assert!(
    <Percentage as euclid_units::Measure>::DIMENSION
        .same(<dim::Dimensionless as euclid_units::Dimensioned>::DIMENSION),
    concat!("dimension mismatch for ", stringify!(Percentage)),
);

impl core::fmt::Display for Percentage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        crate::conversion::fmt_quantity(*self, f)
    }
}

impl euclid_units::Measure for Percentage {
    type Repr = f64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Dimensionless;
    const UNIT_NAME: &'static str = "%";
    fn raw(&self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl euclid_units::ScalarQuantity for Percentage {}

impl euclid_units::LinearQuantity for Percentage {}

impl euclid_units::Dimensionless for Percentage {
    fn unit() -> Self {
        Self::FULL
    }

    fn times(self, other: Self) -> Self {
        Self::from_fraction(self.to_fraction() * other.to_fraction())
    }

    fn recip(self) -> Self {
        Self::from_fraction(Factor::new(self.to_fraction().as_f64().recip()))
    }
}

impl ::core::ops::Add for Percentage {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        crate::quantity::scalar_add(self, rhs)
    }
}

impl ::core::ops::Sub for Percentage {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        crate::quantity::scalar_sub(self, rhs)
    }
}

impl ::core::ops::AddAssign for Percentage {
    fn add_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_add(*self, rhs);
    }
}

impl ::core::ops::SubAssign for Percentage {
    fn sub_assign(&mut self, rhs: Self) {
        *self = crate::quantity::scalar_sub(*self, rhs);
    }
}

impl ::core::ops::Neg for Percentage {
    type Output = Self;
    fn neg(self) -> Self {
        crate::quantity::scalar_neg(self)
    }
}

impl ::core::ops::Mul<f64> for Percentage {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        crate::quantity::scalar_mul(self, k)
    }
}

impl ::core::ops::Mul<Percentage> for f64 {
    type Output = Percentage;
    fn mul(self, x: Percentage) -> Percentage {
        crate::quantity::scalar_mul(x, self)
    }
}

impl ::core::ops::Div<f64> for Percentage {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        crate::quantity::scalar_div(self, k)
    }
}

impl ::core::iter::Sum for Percentage {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        crate::quantity::scalar_sum(iter)
    }
}
