//! Unit tests for the inverse complementary error function.
//!
//! Beside its siblings rather than with the CDF's own tests, because what they reach is
//! [`super::ErfcInvRoute`] and [`super::HalvingCoordinate`] — the domain arms and the coordinate
//! the half-probability is formed in. The subnormal cases are the point: below `f64::MIN_POSITIVE`
//! halving `y` rounds or vanishes, and the log-domain route is what keeps the answer ordered out
//! there.

use super::*;
use crate::kernels::erf::{erfc, erfcx};

fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol * b.abs().max(1.0),
        "{a:e} vs {b:e} (tol {tol:e})"
    );
}

/// `erfc_inv` must reject its out-of-domain arguments, including the ones that look like
/// rounding noise.
///
/// Weakening the guard's `||` to `&&` makes
/// `erfc_inv(-5e-324)` return **`+inf`** instead of `NaN`, which is worse than a wrong
/// number: `0.5 * -5e-324` underflows to `-0.0`, `ndtri(-0.0)` takes the `p == 0.0` branch
/// and returns `-inf`, and the closing negation turns that into a confident `+inf` for an
/// argument outside the function's domain.
///
/// The existing domain tests used `-0.1` and `2.5`, which the weakened guard still rejects
/// by a different route — `ln` of a negative reaching `NaN` on its own. Only a value whose
/// *half underflows to a signed zero* reaches the branch that invents an answer, which is
/// why this needs the subnormals rather than a round number outside the range.
#[test]
fn erfc_inv_rejects_arguments_whose_half_underflows() {
    for &y in &[
        -5e-324,
        -1e-320,
        -f64::MIN_POSITIVE,
        -1e-300,
        -0.1,
        2.0 + f64::EPSILON * 2.0,
        2.5,
        f64::NAN,
    ] {
        let got = erfc_inv(y);
        assert!(
            got.is_nan(),
            "erfc_inv({y:e}) is outside [0, 2] and must be NaN, got {got:e}"
        );
    }
    // The endpoints themselves stay exactly as documented.
    assert_eq!(erfc_inv(0.0), f64::INFINITY);
    assert_eq!(erfc_inv(2.0), f64::NEG_INFINITY);
    // Inside the domain it solves, including well into the subnormals.
    for &y in &[1e-300, 1e-310, 1e-320] {
        let x = erfc_inv(y);
        assert!(
            x.is_finite() && x > 0.0,
            "erfc_inv({y:e}) is in the domain and must solve, got {x:e}"
        );
    }

    let roots = [1, 2, 3, 4, 5].map(|bits| erfc_inv(f64::from_bits(bits)));
    assert!(roots.iter().all(|x| x.is_finite()));
    assert!(roots.windows(2).all(|pair| pair[0] > pair[1]));
}

#[test]
fn erfc_inv_inverts_erfc_including_small_arguments() {
    for i in 1..40 {
        let y = f64::from(i) * 0.05;
        close(erfc(erfc_inv(y)), y, 1e-12);
    }
    // Small y is the regime `erf_inv(1 - y)` cannot reach.
    for &y in &[1e-10f64, 1e-50, 1e-200] {
        let x = erfc_inv(y);
        assert!(x.is_finite() && x > 0.0, "erfc_inv({y}) = {x}");
        // Check through the scaled form, since erfc(x) itself has underflowed.
        close(erfcx(x).ln() - x * x, y.ln(), 1e-9);
    }
}

/// The four routes that invert `erfc`, and the coordinate the halving is formed in.
///
/// `erfc` maps the line onto `(0, 2)`, so `0` and `2` are limits rather than attained values
/// and everything outside `[0, 2]` is off the domain — `NaN` included, which `contains`
/// rejects without a separate test.
#[test]
fn the_erfc_inverse_routes_cover_its_range() {
    use ErfcInvRoute::{AtTwo, AtZero, Interior, Undefined};
    use HalvingCoordinate::{Exact, Logarithmic};

    for y in [
        f64::NAN,
        -f64::MIN_POSITIVE,
        -0.1,
        2.0_f64.next_up(),
        2.5,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        assert_eq!(ErfcInvRoute::for_probability(y), Undefined, "at {y:e}");
        assert!(erfc_inv(y).is_nan(), "at {y:e}");
    }
    assert_eq!(ErfcInvRoute::for_probability(0.0), AtZero);
    assert_eq!(ErfcInvRoute::for_probability(-0.0), AtZero);
    assert_eq!(erfc_inv(0.0), f64::INFINITY);
    assert_eq!(ErfcInvRoute::for_probability(2.0), AtTwo);
    assert_eq!(erfc_inv(2.0), f64::NEG_INFINITY);
    for y in [f64::MIN_POSITIVE, 0.5, 1.0, 1.5, 2.0_f64.next_down()] {
        assert_eq!(ErfcInvRoute::for_probability(y), Interior, "at {y:e}");
    }

    // Halving is exact for every normal `y` — the exponent just drops — and rounds once `y`
    // is subnormal, where the mantissa has no bit left to shift into.
    for y in [1.0_f64, 0.5, 1.5, f64::MIN_POSITIVE] {
        assert_eq!(
            HalvingCoordinate::for_probability(y, 0.5 * y),
            Exact,
            "at {y:e}"
        );
    }
    let subnormal = f64::from_bits(1);
    assert_eq!(0.5 * subnormal, 0.0, "the least subnormal halves to zero");
    assert_eq!(
        HalvingCoordinate::for_probability(subnormal, 0.5 * subnormal),
        Logarithmic
    );
    // And the log route still returns a finite, ordered answer out there, which is the whole
    // reason it exists.
    assert!(erfc_inv(subnormal).is_finite());
    assert!(erfc_inv(subnormal) > erfc_inv(f64::from_bits(2)) - 1.0);
}
