//! Coverage over the `f64` carrier with toy unit carriers that implement the dimension
//! contracts. `f64: Real`, so this exercises the same generic code paths used by other real
//! carriers, and `<f64 as Compare>::Mask = bool`, so the mask-returning predicates collapse to
//! plain `bool` here. euclid ships no concrete unit type, so the carriers below play the role a
//! consumer crate's units would.

use super::{
    Angle, Area, Dimension, Length, LinearQuantity, Quantity, ScalarQuantity, Speed, Time,
};

const EPS: f64 = 1e-9;

macro_rules! unit {
    ($name:ident, $dim:expr, $sym:literal) => {
        #[derive(Clone, Debug, PartialEq)]
        struct $name(f64);

        impl Quantity for $name {
            type Repr = f64;
            const DIMENSION: Dimension = $dim;
            const UNIT_NAME: &'static str = $sym;
            fn raw(&self) -> f64 {
                self.0
            }
            fn from_raw(v: f64) -> Self {
                Self(v)
            }
        }

        impl ScalarQuantity for $name {
            fn min(self, other: Self) -> Self {
                Self(self.0.min(other.0))
            }

            fn max(self, other: Self) -> Self {
                Self(self.0.max(other.0))
            }

            fn clamp(self, lo: Self, hi: Self) -> Self {
                Self(self.0.clamp(lo.0, hi.0))
            }

            fn floor(self) -> Self {
                Self(self.0.floor())
            }

            fn ceil(self) -> Self {
                Self(self.0.ceil())
            }

            fn round(self) -> Self {
                Self(self.0.round())
            }

            fn trunc(self) -> Self {
                Self(self.0.trunc())
            }

            fn is_finite(&self) -> bool {
                self.0.is_finite()
            }

            fn is_nan(&self) -> bool {
                self.0.is_nan()
            }

            fn is_infinite(&self) -> bool {
                self.0.is_infinite()
            }

            fn approx_eq(&self, other: &Self, tolerance: &Self) -> bool {
                (self.0 - other.0).abs() <= tolerance.0.abs()
            }
        }

        impl LinearQuantity for $name {
            fn add(self, other: Self) -> Self {
                Self(self.0 + other.0)
            }

            fn sub(self, other: Self) -> Self {
                Self(self.0 - other.0)
            }

            fn neg(self) -> Self {
                Self(-self.0)
            }

            fn abs(self) -> Self {
                Self(self.0.abs())
            }

            fn signum(self) -> f64 {
                self.0.signum()
            }

            fn ratio(self, other: Self) -> f64 {
                self.0 / other.0
            }

            fn lerp(self, other: Self, t: f64) -> Self {
                Self(self.0 * (1.0 - t) + other.0 * t)
            }

            fn scale(self, factor: f64) -> Self {
                Self(self.0 * factor)
            }

            fn scaled(self, k: f64) -> Self {
                Self(self.0 * k)
            }
        }
    };
}

unit!(Len, Dimension::Length, "m");
unit!(Tim, Dimension::Time, "s");
unit!(Spd, Dimension::Speed, "m/s");
unit!(Ar, Dimension::Area, "m^2");
unit!(Ang, Dimension::Angle, "rad");

impl Length for Len {
    type Area = Ar;
    type Time = Tim;
    type Speed = Spd;

    fn times(self, width: Self) -> Ar {
        Ar(self.0 * width.0)
    }

    fn per(self, elapsed: Tim) -> Spd {
        Spd(self.0 / elapsed.0)
    }

    fn over(self, speed: Spd) -> Tim {
        Tim(self.0 / speed.0)
    }

    fn squared(self) -> Ar {
        Ar(self.0 * self.0)
    }
}

impl Time for Tim {
    type Length = Len;
    type Speed = Spd;

    fn times(self, speed: Spd) -> Len {
        Len(self.0 * speed.0)
    }
}

impl Speed for Spd {
    type Length = Len;
    type Time = Tim;

    fn times(self, elapsed: Tim) -> Len {
        Len(self.0 * elapsed.0)
    }
}

impl Area for Ar {
    type Length = Len;

    fn over(self, side: Len) -> Len {
        Len(self.0 / side.0)
    }
}

impl Angle for Ang {
    fn from_radians(radians: f64) -> Self {
        Self(radians)
    }

    fn from_degrees(degrees: f64) -> Self {
        Self(degrees.to_radians())
    }

    fn half_turn() -> Self {
        Self(std::f64::consts::PI)
    }

    fn full_turn() -> Self {
        Self(std::f64::consts::TAU)
    }

    fn radians(&self) -> f64 {
        self.0
    }

    fn degrees(&self) -> f64 {
        self.0.to_degrees()
    }

    fn sin(&self) -> f64 {
        self.0.sin()
    }

    fn cos(&self) -> f64 {
        self.0.cos()
    }

    fn tan(&self) -> f64 {
        self.0.tan()
    }

    fn normalized_unsigned(self) -> Self {
        Self(self.0.rem_euclid(std::f64::consts::TAU))
    }

    fn normalized_signed(self) -> Self {
        let r = self.0.rem_euclid(std::f64::consts::TAU);
        if r > std::f64::consts::PI {
            Self(r - std::f64::consts::TAU)
        } else {
            Self(r)
        }
    }
}

#[test]
fn linear_ops_over_f64() {
    let a = Len(3.0);
    let b = Len(4.0);

    assert!((a.clone().add(b.clone()).raw() - 7.0).abs() < EPS);
    assert!((b.clone().sub(a.clone()).raw() - 1.0).abs() < EPS);
    assert!((a.clone().neg().raw() + 3.0).abs() < EPS);
    assert!((a.clone().scaled(2.5).raw() - 7.5).abs() < EPS);
    assert!((a.clone().scale(2.0).raw() - 6.0).abs() < EPS);
    assert!((a.clone().lerp(b.clone(), 0.5).raw() - 3.5).abs() < EPS);
    assert!((b.ratio(a) - 4.0 / 3.0).abs() < EPS);
}

#[test]
fn selectors_and_masks_collapse_to_bool() {
    let a = Len(2.0);
    let b = Len(5.0);

    assert!((a.clone().min(b.clone()).raw() - 2.0).abs() < EPS);
    assert!((a.clone().max(b.clone()).raw() - 5.0).abs() < EPS);
    assert!((Len(9.0).clamp(Len(0.0), Len(5.0)).raw() - 5.0).abs() < EPS);

    let finite: bool = a.is_finite();
    assert!(finite);
    assert!(!a.is_nan());
    assert!(a.approx_eq(&Len(2.0 + 1e-12), &Len(EPS)));
}

#[test]
fn dimensional_algebra() {
    let v: Spd = Len(100.0).per(Tim(4.0));
    assert!((v.clone().raw() - 25.0).abs() < EPS);

    let back: Len = v.clone().times(Tim(4.0));
    assert!((back.raw() - 100.0).abs() < EPS);

    let elapsed: Tim = Len(100.0).over(v);
    assert!((elapsed.raw() - 4.0).abs() < EPS);

    let area = Len(3.0).times(Len(4.0));
    assert!((area.clone().raw() - 12.0).abs() < EPS);
    assert!((area.over(Len(4.0)).raw() - 3.0).abs() < EPS);

    assert!((Len(3.0).squared().raw() - 9.0).abs() < EPS);
}

#[test]
fn angle_conversion_and_normalization() {
    let half_turn = Ang::from_degrees(180.0);
    assert!((half_turn.radians() - std::f64::consts::PI).abs() < EPS);
    assert!((half_turn.degrees() - 180.0).abs() < EPS);

    let wrapped = Ang::from_degrees(450.0).normalized_unsigned();
    assert!((wrapped.degrees() - 90.0).abs() < EPS);

    let signed = Ang::from_degrees(270.0).normalized_signed();
    assert!((signed.degrees() + 90.0).abs() < EPS);
}

#[test]
fn unit_metadata() {
    assert_eq!(Len::UNIT_NAME, "m");
    assert_eq!(Spd::DIMENSION, Dimension::Speed);
    assert_eq!(Ang::DIMENSION, Dimension::Angle);
    assert!(Dimension::Length.same(Dimension::Length));
    assert!(!Dimension::Length.same(Dimension::Time));
}
