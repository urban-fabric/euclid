//! Direct coverage of the [`ScalarQuantity`](crate::ScalarQuantity) and
//! [`LinearQuantity`](crate::LinearQuantity) default bodies.
//!
//! These methods used to be required, so each carrier supplied its own body and the
//! [`algebra`](super::algebra) tests exercised them incidentally through whichever carrier they
//! happened to use. Now one default body serves every carrier in every consuming crate, which
//! makes each one a single point of failure — so they are tested here on their own rather than
//! only as a side effect of testing the dimensional algebra.
//!
//! `Len` is linear and `Tmp` is not, which is the pairing that matters: it confirms the
//! `ScalarQuantity` defaults reach a carrier that deliberately withholds `LinearQuantity`.

use crate::{LinearQuantity, Measure, ScalarQuantity};

use super::toy_units::{Len, Tmp};

fn m(v: f64) -> Len {
    Len::from_raw(v)
}

// --- ScalarQuantity -----------------------------------------------------------------------

#[test]
fn ordering_selectors_stay_in_unit() {
    assert_eq!(m(2.0).min(m(5.0)), m(2.0));
    assert_eq!(m(2.0).max(m(5.0)), m(5.0));
    assert_eq!(m(7.0).clamp(m(2.0), m(5.0)), m(5.0));
    assert_eq!(m(1.0).clamp(m(2.0), m(5.0)), m(2.0));
    assert_eq!(m(3.0).clamp(m(2.0), m(5.0)), m(3.0));
}

#[test]
fn rounding_family_stays_in_unit() {
    assert_eq!(m(2.6).floor(), m(2.0));
    assert_eq!(m(2.6).ceil(), m(3.0));
    assert_eq!(m(2.6).round(), m(3.0));
    assert_eq!(m(2.6).trunc(), m(2.0));
    // Truncation goes toward zero and flooring toward negative infinity: the one input that
    // distinguishes them.
    assert_eq!(m(-2.6).trunc(), m(-2.0));
    assert_eq!(m(-2.6).floor(), m(-3.0));
}

#[test]
fn finiteness_predicates() {
    assert!(m(1.0).is_finite());
    assert!(!m(f64::INFINITY).is_finite());
    assert!(m(f64::INFINITY).is_infinite());
    assert!(m(f64::NEG_INFINITY).is_infinite());
    assert!(m(f64::NAN).is_nan());
    assert!(!m(1.0).is_nan());
}

#[test]
fn approx_eq_uses_a_typed_tolerance() {
    let a = m(1.000_1);
    let b = m(1.000_2);
    assert!(a.clone().approx_eq(&b, &m(0.001)));
    assert!(!a.clone().approx_eq(&b, &m(0.000_01)));
    // The tolerance's magnitude is what counts, so a negative one does not reject everything.
    assert!(a.approx_eq(&b, &m(-0.001)));
}

#[test]
fn scalar_defaults_reach_a_carrier_that_is_not_linear() {
    // Tmp withholds LinearQuantity, but the ScalarQuantity defaults still apply to it.
    let k = Tmp::from_raw(293.15);
    assert_eq!(k.clone().round(), Tmp::from_raw(293.0));
    assert!(k.is_finite());
}

// --- LinearQuantity -----------------------------------------------------------------------

#[test]
fn additive_group_operations() {
    assert_eq!(m(2.0).add(m(3.0)), m(5.0));
    assert_eq!(m(5.0).sub(m(3.0)), m(2.0));
    assert_eq!(m(2.0).neg(), m(-2.0));
    assert_eq!(m(-4.0).abs(), m(4.0));
}

#[test]
fn signum_and_ratio_drop_to_the_carrier() {
    assert_eq!(m(7.0).signum(), 1.0);
    assert_eq!(m(-7.0).signum(), -1.0);
    assert_eq!(m(10.0).ratio(m(4.0)), 2.5);
}

#[test]
fn lerp_hits_both_endpoints() {
    let a = m(0.0);
    let b = m(10.0);
    assert_eq!(a.clone().lerp(b.clone(), 0.0), a);
    assert_eq!(a.clone().lerp(b.clone(), 1.0), b);
    assert_eq!(a.lerp(b, 0.25), m(2.5));
}

#[test]
fn scale_and_scaled_agree_on_a_host_literal() {
    // `scale` takes a carrier value and `scaled` lifts a host literal through
    // `Real::scalar_like`. For `f64` the lift is the identity, so the two must agree.
    assert_eq!(m(3.0).scale(2.0), m(6.0));
    assert_eq!(m(3.0).scaled(2.0), m(6.0));
}

// --- the public conformance suites ----------------------------------------------------------
//
// The laws are `crate::conformance`, public API a carrier author calls. Running them against
// the toy carriers proves the suites are satisfiable and that the defaults above satisfy them.

#[test]
fn toy_carriers_satisfy_the_quantity_laws() {
    let xs = [m(0.0), m(1.0), m(-2.5), m(4.0), m(0.5)];
    crate::conformance::quantity_laws(&xs, &m(1e-9), "Len");
}

#[test]
fn an_absolute_scalar_satisfies_everything_but_the_linear_laws() {
    // Tmp withholds LinearQuantity, so it gets the measure and scalar laws only — which is the
    // asymmetry the tier split exists to express.
    let xs = [
        Tmp::from_raw(0.0),
        Tmp::from_raw(273.15),
        Tmp::from_raw(-40.0),
    ];
    crate::conformance::measure_laws(&xs, "Tmp");
    crate::conformance::scalar_laws(&xs, &Tmp::from_raw(1e-9), "Tmp");
}
