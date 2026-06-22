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

        impl LinearQuantity for $name {}
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

impl Angle for Ang {}

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
