//! The log-domain family at its boundaries, and the compensated primitives at theirs.
//!
//! Part of the [poles suite](super). These functions exist precisely to be used where the
//! naive form fails, which means they are *routinely* called at the edges of their domains —
//! a log-sum-exp over a mask of `−∞` lanes, a logit at a probability that has rounded to 1.
//! So their boundary conventions are part of the contract rather than an afterthought.
//!
//! Two inconsistencies were found here:
//!
//! - **`ln_1m_exp(0)` returned `NaN`** where the limit is `−∞`. The same mathematical point
//!   reached the other way, `ln_1m(1)`, already returned `−∞`, so the two disagreed about
//!   `ln(0)`. `ln_exp_m1(0)` inherited it, being `x + ln_1m_exp(−x)`.
//! - **`two_sum` and `two_product` returned a `NaN` or `±∞` error term on overflow**, which
//!   poisons a compensated accumulator permanently — long after the offending term has gone
//!   past. Zero keeps `value + error == value`, so the reduction saturates instead.

use crate::Real;

#[test]
fn the_log_domain_boundaries_agree_with_each_other() {
    // `ln(0) = -inf` must be reached consistently, by whichever route.
    assert_eq!(Real::ln_1m_exp(&0.0f64), f64::NEG_INFINITY);
    assert_eq!(Real::ln_1m_exp(&-0.0f64), f64::NEG_INFINITY);
    assert_eq!(Real::ln_1m(&1.0f64), f64::NEG_INFINITY);
    assert_eq!(Real::ln_exp_m1(&0.0f64), f64::NEG_INFINITY);
    assert_eq!(Real::ln_1m_exp(&0.0f32), f32::NEG_INFINITY);
    assert_eq!(Real::ln_exp_m1(&0.0f32), f32::NEG_INFINITY);

    // Above the boundary there is no real value, and NaN is the statement of that.
    for &x in &[1e-17f64, 1e-8, 0.5, 1.0, 1e300] {
        assert!(Real::ln_1m_exp(&x).is_nan(), "ln_1m_exp({x:e}) must be NaN");
    }
    for &x in &[-1e-17f64, -0.5, -1e300] {
        assert!(Real::ln_exp_m1(&x).is_nan(), "ln_exp_m1({x:e}) must be NaN");
    }
    // `ln_1m(x) = ln(1 − x)` has no real value above 1. The arguments have to be *genuinely*
    // above 1 to say so: `1.0 + 1e-17` rounds straight back to 1, where `−∞` is correct.
    for &x in &[1.5f64, 2.0, 1e300] {
        assert!(Real::ln_1m(&x).is_nan(), "ln_1m({x:e}) must be NaN");
    }

    // Approaching the boundary from inside, each must diverge rather than flatten. `ln_1m_exp`
    // near zero is `ln(-x)`, which is the regime its `ln(-expm1(x))` form exists to serve.
    for &d in &[1e-4f64, 1e-8, 1e-12, 1e-17, 1e-300] {
        let value = Real::ln_1m_exp(&-d);
        assert!(
            value.is_finite() && value < 0.0,
            "ln_1m_exp(-{d:e}) = {value}"
        );
        // ln(1 - e^{-d}) = ln(d) + O(d), so the leading term is ln d.
        assert!(
            (value - d.ln()).abs() < 1.0,
            "ln_1m_exp(-{d:e}) = {value:e} should be near ln({d:e}) = {:e}",
            d.ln()
        );
    }
}

#[test]
fn the_log_sum_exp_pair_handle_infinite_operands() {
    let inf = f64::INFINITY;
    let neg = f64::NEG_INFINITY;
    // ln(e^a + e^b) with an absent term: a -inf operand contributes nothing.
    assert_eq!(Real::ln_add_exp(neg, 0.0), 0.0);
    assert_eq!(Real::ln_add_exp(0.0, neg), 0.0);
    assert_eq!(Real::ln_add_exp(neg, neg), neg, "ln(0 + 0) = -inf");
    assert_eq!(Real::ln_add_exp(inf, 0.0), inf);
    assert_eq!(Real::ln_add_exp(inf, inf), inf);
    // Equal large operands: the answer is a + ln 2, and at 1e300 that addend is below the ulp.
    assert_eq!(Real::ln_add_exp(1e300f64, 1e300), 1e300);
    assert_eq!(Real::ln_add_exp(1.0f64, 1.0), 1.0 + core::f64::consts::LN_2);

    // ln(e^a - e^b) is defined for a >= b, and vanishes to -inf when they are equal.
    assert_eq!(Real::ln_sub_exp(0.0f64, 0.0), neg);
    assert_eq!(Real::ln_sub_exp(1e300f64, 1e300), neg);
    assert_eq!(Real::ln_sub_exp(0.0f64, neg), 0.0, "e^0 - 0 = 1");
    assert!(
        Real::ln_sub_exp(0.0f64, 1.0).is_nan(),
        "a < b has no real log"
    );
    assert!(Real::ln_sub_exp(inf, inf).is_nan(), "inf - inf");
}

#[test]
fn the_absorbing_zero_survives_an_infinite_multiplicand() {
    // `xlogy` and its relatives exist so that `0·ln 0 = 0` rather than `NaN`. The convention
    // has to hold against an infinite `y` too, which is where a plain `x * y.ln()` gives
    // `0 · ±inf = NaN` — the exact failure these functions are named for.
    for &y in &[0.0f64, -1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        assert_eq!(Real::xlogy(0.0, y), 0.0, "xlogy(0, {y:e}) must be 0");
        assert_eq!(Real::xlog1py(0.0, y), 0.0, "xlog1py(0, {y:e}) must be 0");
        assert_eq!(Real::xlog1my(0.0, y), 0.0, "xlog1my(0, {y:e}) must be 0");
    }
    // A nonzero multiplier does not get the exemption: the logarithm speaks for itself.
    assert_eq!(Real::xlogy(1.0f64, 0.0), f64::NEG_INFINITY);
    assert_eq!(Real::xlogy(-1.0f64, 0.0), f64::INFINITY);
}

#[test]
fn logit_and_sigmoid_round_trip_to_the_ends_of_the_unit_interval() {
    assert_eq!(Real::logit(&0.0f64), f64::NEG_INFINITY);
    assert_eq!(Real::logit(&1.0f64), f64::INFINITY);
    assert_eq!(Real::sigmoid(&f64::NEG_INFINITY), 0.0);
    assert_eq!(Real::sigmoid(&f64::INFINITY), 1.0);
    // Through the round trip, including probabilities at the bottom of the format where
    // `ln(p/(1-p))` must not lose `p` to the subtraction.
    for &p in &[1e-320f64, 1e-300, 1e-100, 1e-17, 0.25, 0.5, 0.75] {
        let back = Real::sigmoid(&Real::logit(&p));
        assert!(
            ((back - p) / p).abs() < 1e-12,
            "sigmoid(logit({p:e})) = {back:e}"
        );
    }
}

#[test]
fn the_compensated_primitives_saturate_rather_than_poisoning_the_accumulator() {
    // On overflow the exact sum is unrepresentable, so no error term can satisfy
    // `value + error == exact`. Reporting zero preserves `value + error == value`.
    for &(a, b) in &[
        (f64::MAX, f64::MAX),
        (f64::MAX, 1e308),
        (f64::MIN, f64::MIN),
        (1.0, f64::NAN),
        (f64::INFINITY, 1.0),
    ] {
        let (value, error) = Real::two_sum(a, b);
        if !value.is_finite() {
            assert_eq!(error, 0.0, "two_sum({a:e}, {b:e}) error must be zero");
        }
        let (product, perror) = Real::two_product(a, b);
        if !product.is_finite() {
            assert_eq!(perror, 0.0, "two_product({a:e}, {b:e}) error must be zero");
        }
    }
    // The point of it: a compensated reduction that overflows must still saturate. The Knuth
    // sequence used to produce `inf - inf = NaN` for the error, and that NaN then survived in
    // the correction accumulator for the rest of the reduction.
    let mut total = 0.0f64;
    let mut correction = 0.0f64;
    for _ in 0..8 {
        let (sum, error) = Real::two_sum(total, f64::MAX);
        total = sum;
        correction += error;
    }
    assert_eq!(
        total + correction,
        f64::INFINITY,
        "a compensated sum of eight f64::MAX should be +inf, not NaN"
    );

    // Where the result *is* finite the exactness contract holds, including at the subnormal
    // end where the product underflows and the remainder is all there is.
    for &(a, b) in &[
        (1.0f64, 1e-20),
        (1e300, 1e-300),
        (f64::MIN_POSITIVE, 0.5),
        (1.0, f64::EPSILON),
        (0.1, 0.2),
    ] {
        let (value, error) = Real::two_sum(a, b);
        assert_eq!(value + error, a + b, "two_sum exactness at ({a:e}, {b:e})");
        let (product, perror) = Real::two_product(a, b);
        assert!(
            product.is_finite() && perror.is_finite(),
            "two_product({a:e}, {b:e}) = ({product:e}, {perror:e})"
        );
    }
}
