//! **What happens at the boundaries?** Behavior at `0`, `±∞`, `NaN`, the domain edges, and the
//! representable limits.
//!
//! These are the inputs a formula hits in production and never in a notebook: a zero
//! probability, a saturated log-weight, an out-of-support quantile, an uninitialized value
//! that arrived as `NaN`. The contract is only useful if the behavior there is *defined*, so
//! these tests pin it.

use crate::{Compare, Real};

// --- NaN propagation ----------------------------------------------------

#[test]
fn nan_propagates_through_the_elementwise_surface() {
    let n = f64::NAN;
    assert!(Real::exp(&n).is_nan());
    assert!(Real::ln(&n).is_nan());
    assert!(Real::sqrt(&n).is_nan());
    assert!(Real::erf(n).is_nan());
    assert!(Real::erfc(n).is_nan());
    assert!(Real::ln_gamma(n).is_nan());
    assert!(Real::sigmoid(&n).is_nan());
    assert!(Real::sin(n).is_nan());
}

#[test]
fn nan_is_detectable_through_the_mask_surface() {
    let n = f64::NAN;
    // The predicates are the supported way to find a NaN; `==` cannot.
    assert!(Real::is_nan(&n));
    assert!(!Real::is_finite(&n));
    assert!(!Real::is_infinite(&n));
    #[allow(clippy::eq_op)]
    {
        assert!(!Compare::eq(&n, &n), "NaN is not equal to itself");
    }
}

#[test]
fn nan_to_num_and_finite_or_replace_only_what_they_claim() {
    // `nan_to_num` replaces NaN and leaves infinities alone.
    assert_eq!(Real::nan_to_num(f64::NAN, -1.0), -1.0);
    assert_eq!(Real::nan_to_num(f64::INFINITY, -1.0), f64::INFINITY);
    assert_eq!(Real::nan_to_num(f64::NEG_INFINITY, -1.0), f64::NEG_INFINITY);
    assert_eq!(Real::nan_to_num(2.5, -1.0), 2.5);

    // `finite_or` replaces anything non-finite.
    assert_eq!(Real::finite_or(f64::NAN, 7.0), 7.0);
    assert_eq!(Real::finite_or(f64::INFINITY, 7.0), 7.0);
    assert_eq!(Real::finite_or(f64::NEG_INFINITY, 7.0), 7.0);
    assert_eq!(Real::finite_or(2.5, 7.0), 2.5);
}

#[test]
fn mask_invalid_substitutes_on_the_out_of_support_lane() {
    // The documented "double where" output half: keep where valid, else fill.
    let in_support = true;
    assert_eq!(Real::mask_invalid(3.0, in_support, f64::NEG_INFINITY), 3.0);
    assert_eq!(
        Real::mask_invalid(3.0, false, f64::NEG_INFINITY),
        f64::NEG_INFINITY
    );
    // Even a NaN in the invalid lane is replaced, which is the point.
    assert_eq!(Real::mask_invalid(f64::NAN, false, -1.0), -1.0);
}

// --- infinities ---------------------------------------------------------

#[test]
fn infinities_map_to_their_limits() {
    let inf = f64::INFINITY;
    let ninf = f64::NEG_INFINITY;

    assert_eq!(Real::exp(&ninf), 0.0);
    assert_eq!(Real::exp(&inf), inf);
    assert_eq!(Real::ln(&inf), inf);

    // The error function saturates *in one direction only*. See
    // `infinite_inputs_are_handled_inconsistently` below for the rest.
    assert_eq!(Real::erfc(inf), 0.0);
    assert_eq!(Real::erfcx(inf), 0.0);
    assert_eq!(Real::bessel_i0e(inf), 0.0);

    // The logistic saturates without overflowing.
    assert_eq!(Real::sigmoid(&inf), 1.0);
    assert_eq!(Real::sigmoid(&ninf), 0.0);

    // softplus: +inf passes through, -inf collapses to 0.
    assert_eq!(Real::ln_1p_exp(&inf), inf);
    assert_eq!(Real::ln_1p_exp(&ninf), 0.0);
}

/// **KNOWN DEVIATION — not the mathematically correct behavior.**
///
/// These functions return `NaN` at an infinite argument even though the limit is well defined.
/// C99 and CPython's `math` both return the limit: `erf(+∞) = 1`, `erf(−∞) = −1`,
/// `erfc(−∞) = 2`, `lnΓ(+∞) = +∞`, `P(a, +∞) = 1`.
///
/// It matters because an infinite argument arrives in practice — an overflowed log-weight, a
/// normalization that divided by zero — and `NaN` poisons every downstream value, whereas the
/// correct limit would let the formula carry on. The inconsistency is the clearest sign this
/// is an oversight rather than a policy: `erfc(+∞)`, `erfcx(+∞)` and `bessel_i0e(+∞)` all do
/// return their limits.
///
/// This test pins what the code *currently does*, so the suite is honest and the gap is
/// recorded. When the guards are added, this test will fail and should be replaced by
/// assertions on the correct limits above.
#[test]
fn infinite_inputs_are_handled_inconsistently() {
    let inf = f64::INFINITY;
    let ninf = f64::NEG_INFINITY;

    // Correct today:
    assert_eq!(Real::erfc(inf), 0.0, "erfc(+inf) = 0, correct");

    // Should be 1.0:
    assert!(Real::erf(inf).is_nan(), "erf(+inf) should be 1.0");
    // Should be -1.0:
    assert!(Real::erf(ninf).is_nan(), "erf(-inf) should be -1.0");
    // Should be 2.0:
    assert!(Real::erfc(ninf).is_nan(), "erfc(-inf) should be 2.0");
    // Should be +inf:
    assert!(
        Real::ln_gamma(inf).is_nan(),
        "ln_gamma(+inf) should be +inf"
    );
    // Should be 1.0:
    assert!(
        Real::gamma_lr(inf, 2.0).is_nan(),
        "gamma_lr(+inf, a) should be 1.0"
    );
}

/// **KNOWN DEVIATION.** `erf_inv` has no real value outside `[-1, 1]`, but returns `±∞`
/// rather than `NaN`, so an out-of-range probability passes silently instead of being
/// detectable via `is_nan`. Pinned here; the behavior is currently undocumented on the trait.
#[test]
fn erf_inv_outside_its_domain_returns_infinity_rather_than_nan() {
    assert_eq!(Real::erf_inv(1.5f64), f64::INFINITY);
    assert_eq!(Real::erf_inv(-1.5f64), f64::NEG_INFINITY);
    assert_eq!(Real::erf_inv(2.0f64), f64::INFINITY);
}

#[test]
fn log_of_zero_is_negative_infinity_not_nan() {
    // Formulas rely on this: a zero weight becomes -inf log-weight, which then drops out of a
    // logsumexp rather than poisoning it.
    assert_eq!(Real::ln(&0.0), f64::NEG_INFINITY);
    assert_eq!(Real::ln_sigmoid(&f64::NEG_INFINITY), f64::NEG_INFINITY);
    // ln_add_exp with a -inf operand returns the other operand.
    assert_eq!(Real::ln_add_exp(f64::NEG_INFINITY, 3.0), 3.0);
    assert_eq!(Real::ln_add_exp(3.0, f64::NEG_INFINITY), 3.0);
    // Both -inf stays -inf, i.e. an all-zero mixture has zero total mass.
    assert_eq!(
        Real::ln_add_exp(f64::NEG_INFINITY, f64::NEG_INFINITY),
        f64::NEG_INFINITY
    );
}

// --- exact values at zero -----------------------------------------------

#[test]
fn functions_hit_their_exact_values_at_zero() {
    assert_eq!(Real::erf(0.0), 0.0);
    assert_eq!(Real::erfc(0.0), 1.0);
    assert_eq!(Real::erfcx(0.0), 1.0);
    assert_eq!(Real::sigmoid(&0.0), 0.5);
    assert_eq!(Real::ln_1p(&0.0), 0.0);
    assert_eq!(Real::exp_m1(&0.0), 0.0);
    assert_eq!(Real::bessel_i0e(0.0), 1.0);
    assert_eq!(Real::bessel_i1e(0.0), 0.0);
    assert_eq!(Real::bessel_i1_over_i0(0.0), 0.0);
    assert_eq!(Real::ln_bessel_i0(0.0), 0.0);
    // softplus(0) = ln 2
    assert!((Real::ln_1p_exp(&0.0) - core::f64::consts::LN_2).abs() < 1e-16);
}

#[test]
fn signed_zero_is_handled_consistently() {
    assert_eq!(Real::erf(-0.0), 0.0);
    assert_eq!(Real::sigmoid(&-0.0), 0.5);
    // copysign is the documented way to observe the sign bit.
    assert_eq!(Real::copysign(1.0, -0.0), -1.0);
    assert_eq!(Real::copysign(1.0, 0.0), 1.0);
}

// --- domain edges -------------------------------------------------------

#[test]
fn cdf_inverses_saturate_at_their_domain_edges() {
    // erf_inv at ±1 is ±inf, not NaN: a probability of exactly 1 is a legitimate input.
    assert_eq!(Real::erf_inv(1.0), f64::INFINITY);
    assert_eq!(Real::erf_inv(-1.0), f64::NEG_INFINITY);
    assert_eq!(Real::erf_inv(0.0), 0.0);
    // Out-of-domain behavior is pinned in
    // `erf_inv_outside_its_domain_returns_infinity_rather_than_nan`.
}

#[test]
fn logit_saturates_at_the_probability_endpoints() {
    assert_eq!(Real::logit(&0.0), f64::NEG_INFINITY);
    assert_eq!(Real::logit(&1.0), f64::INFINITY);
    assert_eq!(Real::logit(&0.5), 0.0);
    // Unlike `erf_inv`, `logit` does report an out-of-range probability as NaN.
    assert!(Real::logit(&1.5f64).is_nan());
}

#[test]
fn incomplete_gamma_and_beta_are_defined_at_their_endpoints() {
    for &a in &[0.5f64, 1.0, 10.0] {
        assert_eq!(Real::gamma_lr(0.0, a), 0.0);
        assert_eq!(Real::gamma_ur(0.0, a), 1.0);
    }
    for &(a, b) in &[(0.5f64, 0.5f64), (1.0, 1.0), (3.0, 7.0)] {
        assert_eq!(Real::beta_reg(0.0, a, b), 0.0);
        assert!((Real::beta_reg(1.0, a, b) - 1.0).abs() < 1e-14);
    }
}

#[test]
fn sqrt_and_ln_of_negative_inputs_are_nan_not_a_silent_value() {
    assert!(Real::sqrt(&-1.0f64).is_nan());
    assert!(Real::ln(&-1.0f64).is_nan());
    assert!(Real::ln_1p(&-2.0f64).is_nan());
    assert!(Real::asin(2.0f64).is_nan());
    assert!(Real::acos(2.0f64).is_nan());
    assert!(Real::acosh(0.5f64).is_nan());
    assert!(Real::atanh(2.0f64).is_nan());
}

// --- representable limits -----------------------------------------------

#[test]
fn subnormal_inputs_do_not_break_the_elementwise_surface() {
    let tiny = f64::MIN_POSITIVE / 4.0; // subnormal
    assert!(tiny > 0.0 && tiny.is_finite());
    assert_eq!(Real::ln_1p(&tiny), tiny);
    assert_eq!(Real::exp_m1(&tiny), tiny);
    assert!(Real::sqrt(&tiny) > 0.0);
    assert_eq!(Real::erf(tiny), 0.0f64.max(Real::erf(tiny)));
    assert!(Real::sigmoid(&tiny) > 0.0);
    assert!(Real::ln(&tiny).is_finite() && Real::ln(&tiny) < 0.0);
}

#[test]
fn extreme_magnitudes_stay_finite_where_the_math_allows() {
    // Operations that must not overflow even near the f64 ceiling.
    assert!(Real::hypot(f64::MAX / 2.0, f64::MAX / 2.0).is_finite());
    assert!(Real::ln(&f64::MAX).is_finite());
    assert!(Real::ln_1p_exp(&f64::MAX).is_finite());
    assert!(Real::sigmoid(&f64::MAX).is_finite());
    assert!(Real::erfcx(1e300f64).is_finite());
    assert!(Real::bessel_i0e(1e300f64) >= 0.0);

    // And at the bottom.
    assert!(Real::ln(&f64::MIN_POSITIVE).is_finite());
    assert!(Real::ln_1p_exp(&-f64::MAX) >= 0.0);
}

#[test]
fn clamp_requires_ordered_non_nan_bounds() {
    // The documented contract: `clamp` delegates to `f64::clamp`, which panics on bad bounds.
    // Here we only pin the well-formed behavior.
    assert_eq!(Real::clamp(5.0, 0.0, 1.0), 1.0);
    assert_eq!(Real::clamp(-5.0, 0.0, 1.0), 0.0);
    assert_eq!(Real::clamp(0.5, 0.0, 1.0), 0.5);
    // A NaN value (not bound) passes through as NaN.
    assert!(Real::clamp(f64::NAN, 0.0, 1.0).is_nan());
}

#[test]
fn min_max_propagate_nan_consistently_with_the_mask_surface() {
    // Documented behavior of the comparison-free selectors on a NaN operand.
    let n = f64::NAN;
    // f64::min/max return the non-NaN operand; this is IEEE minNum/maxNum behavior.
    assert_eq!(Real::min(n, 1.0), 1.0);
    assert_eq!(Real::max(n, 1.0), 1.0);
    assert_eq!(Real::min(1.0, n), 1.0);
    assert_eq!(Real::max(1.0, n), 1.0);
}

// --- integer and natural carriers ---------------------------------------

#[test]
fn integer_real_conversions_saturate_rather_than_wrap() {
    use crate::Integer;
    // Documented: out-of-range float->int saturates.
    assert_eq!(<i32 as Integer>::from_real(1e30_f32), i32::MAX);
    assert_eq!(<i32 as Integer>::from_real(-1e30_f32), i32::MIN);
    assert_eq!(<i64 as Integer>::from_real(1e300), i64::MAX);
    assert_eq!(<i64 as Integer>::from_real(-1e300), i64::MIN);
    // NaN maps to zero, as Rust's `as` cast does.
    assert_eq!(<i64 as Integer>::from_real(f64::NAN), 0);
}

#[test]
fn natural_conversions_clamp_negatives_to_zero() {
    use crate::Natural;
    // Documented: negative inputs clamp to zero, out-of-range saturates.
    assert_eq!(<u32 as Natural>::from_real(-5.0_f32), 0);
    assert_eq!(<u64 as Natural>::from_real(-5.0), 0);
    assert_eq!(<u64 as Natural>::from_real(1e300), u64::MAX);
    assert_eq!(<u64 as Natural>::from_real(f64::NAN), 0);
    // `from_real` *rounds* (it is documented as "round a real back into this carrier"),
    // so 7.5 and 7.9 both land on 8 rather than truncating toward zero.
    assert_eq!(<u64 as Natural>::from_real(7.9), 8);
    assert_eq!(<u64 as Natural>::from_real(7.5), 8);
    assert_eq!(<u64 as Natural>::from_real(7.4), 7);
}
