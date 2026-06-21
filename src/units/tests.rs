//! Coverage over the `f64` carrier. `f64: LazyReal`, so this exercises the exact generic
//! code paths a polars `Expr` or backend `Tensor` carrier will run — only the carrier type
//! differs. `<f64 as LazyCompare>::Mask = bool`, so the mask-returning predicates collapse
//! to plain `bool` here.
//!
//! `LazyReal` is implemented for both `f32` and `f64`, so a bare literal carrier is
//! ambiguous — these tests pin `f64` via helpers (a real backend carrier names its own type).

use super::{
    Dimension, LazyKm, LazyLinearQuantity, LazyMeters, LazyQuantity, LazyScalarQuantity,
    LazySeconds, LazySpeedMs,
};

const EPS: f64 = 1e-9;

fn m(v: f64) -> LazyMeters<f64> {
    LazyMeters::new(v)
}
fn s(v: f64) -> LazySeconds<f64> {
    LazySeconds::new(v)
}

#[test]
fn linear_ops_over_f64() {
    let a = m(3.0);
    let b = m(4.0);

    assert!(((a.clone() + b.clone()).value() - 7.0).abs() < EPS);
    assert!(((b.clone() - a.clone()).value() - 1.0).abs() < EPS);
    assert!(((-a.clone()).value() + 3.0).abs() < EPS);
    assert!((a.clone().scaled(2.5).value() - 7.5).abs() < EPS);
    assert!((a.clone().scale(2.0).value() - 6.0).abs() < EPS);
    assert!(((a.clone() * 10.0).value() - 30.0).abs() < EPS);
    assert!((a.clone().lerp(b.clone(), 0.5).value() - 3.5).abs() < EPS);
    assert!((b.ratio(a) - 4.0 / 3.0).abs() < EPS);
}

#[test]
fn assign_ops() {
    let mut a = m(10.0);
    a += m(5.0);
    assert!((a.clone().value() - 15.0).abs() < EPS);
    a -= m(2.0);
    assert!((a.value() - 13.0).abs() < EPS);
}

#[test]
fn selectors_and_masks_collapse_to_bool() {
    let a = m(2.0);
    let b = m(5.0);

    assert!((a.clone().min(b.clone()).value() - 2.0).abs() < EPS);
    assert!((a.clone().max(b.clone()).value() - 5.0).abs() < EPS);
    assert!((m(9.0).clamp(m(0.0), m(5.0)).value() - 5.0).abs() < EPS);

    let finite: bool = a.is_finite();
    assert!(finite);
    assert!(!a.is_nan());
    assert!(a.approx_eq(&m(2.0 + 1e-12), &m(EPS)));
}

#[test]
fn dimensional_algebra() {
    let v: LazySpeedMs = m(100.0) / s(4.0);
    assert!((v.clone().value() - 25.0).abs() < EPS);

    let back: LazyMeters = v.clone() * s(4.0);
    assert!((back.value() - 100.0).abs() < EPS);

    let elapsed: LazySeconds = m(100.0) / v;
    assert!((elapsed.value() - 4.0).abs() < EPS);

    let area = m(3.0) * m(4.0);
    assert!((area.clone().value() - 12.0).abs() < EPS);
    assert!(((area / m(4.0)).value() - 3.0).abs() < EPS);
}

#[test]
fn km_conversion_lifts_factor() {
    let five_km: LazyKm = LazyKm::new(5.0);
    assert!((five_km.to_meters().value() - 5_000.0).abs() < EPS);
}

#[test]
fn unit_metadata() {
    assert_eq!(LazyMeters::<f64>::UNIT_NAME, "m");
    assert_eq!(LazySpeedMs::<f64>::DIMENSION, Dimension::Speed);
    assert!(Dimension::Length.same(Dimension::Length));
    assert!(!Dimension::Length.same(Dimension::Time));
}
