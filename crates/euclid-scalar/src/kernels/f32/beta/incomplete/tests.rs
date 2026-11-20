//! Unit tests for the single-precision incomplete beta.
//!
//! Beside [`super`] rather than with `B(a, b)`'s own tests, because what they reach is this
//! module's: the [`BetaRegMethod`] table, the [`EdgeworthTail`] cut, the three Edgeworth
//! components and the continued fraction in both orientations.
//!
//! The Edgeworth components are asserted **as the beta's own moments** rather than through
//! `beta_reg`. Each is a closed form in `a` and `b`, so a wrong one is unambiguous there, while
//! through the CDF it would sit inside a tolerance wide enough to absorb it.

#![allow(clippy::panic)]

use super::{
    BETA_EDGEWORTH_MIN, BetaRegMethod, EDGEWORTH_Z_MAX, EdgeworthTail, beta_cf, beta_edgeworth,
    beta_fraction_cdf, beta_gamma_rate, beta_mean_residual, beta_reg, edgeworth_correction,
    edgeworth_inverse_width, edgeworth_skew,
};
use crate::Real;

fn close(a: f32, b: f32, tol: f32) {
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
}

fn agrees(a: f32, b: f64, tol: f32) {
    let b = b as f32;
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
}

#[test]
fn beta_reg_matches_f64_reference() {
    assert_eq!(beta_reg(2.0, 3.0, 0.0), 0.0);
    assert_eq!(beta_reg(2.0, 3.0, 1.0), 1.0);
    for &(a, b, x) in &[(2.0, 3.0, 0.4), (0.5, 0.5, 0.3), (5.0, 2.0, 0.7)] {
        agrees(
            beta_reg(a, b, x),
            crate::kernels::beta::beta_reg(a as f64, b as f64, x as f64),
            1e-5,
        );
    }
}

#[test]
fn beta_reg_endpoints_and_symmetry() {
    // `I_x(a,b) = 1 − I_{1−x}(b,a)`, which constrains both fraction orientations at once.
    for &(a, b, x) in &[
        (2.0_f32, 3.0_f32, 0.4_f32),
        (0.5, 0.5, 0.3),
        (5.0, 2.0, 0.7),
    ] {
        close(beta_reg(a, b, x), 1.0 - beta_reg(b, a, 1.0 - x), 1e-5);
    }
    close(beta_reg(1.0, 1.0, 0.37), 0.37, 1e-6);
    assert!(beta_reg(0.0, 1.0, 0.5).is_nan());
}

#[test]
fn the_regularized_method_table_is_ordered_as_written() {
    use BetaRegMethod::{
        Edgeworth, Fraction, GammaLimit, GammaLimitReflected, Invalid, UnitLower, UnitUpper, Value,
    };
    // The order of these arms is load-bearing: the endpoint checks precede the shape checks, and
    // the two gamma limits precede Edgeworth, so a shape pair that qualifies for both takes the
    // gamma route. Reading the table is the only way that ordering is ever checked.
    for (a, b, x, want) in [
        (2.0_f32, 3.0_f32, 0.5_f32, Fraction),
        (1.0, 3.0, 0.5, UnitLower),
        (3.0, 1.0, 0.5, UnitUpper),
        (1.0, 1.0, 0.5, UnitLower),
        (BETA_EDGEWORTH_MIN, BETA_EDGEWORTH_MIN, 0.5, Edgeworth),
        (
            BETA_EDGEWORTH_MIN.next_down(),
            BETA_EDGEWORTH_MIN,
            0.5,
            Fraction,
        ),
        (2.0, 1e5, 0.5, GammaLimit),
        (1e5, 2.0, 0.5, GammaLimitReflected),
        (2.0, 3.0, 0.0, Value(0.0)),
        (2.0, 3.0, -0.5, Value(0.0)),
        (2.0, 3.0, 1.0, Value(1.0)),
        (2.0, 3.0, 2.0, Value(1.0)),
        (1e-20, 1e-20, 0.5, Value(0.5)),
        (0.0, 3.0, 0.5, Invalid),
        (-1.0, 3.0, 0.5, Invalid),
        (2.0, 0.0, 0.5, Invalid),
        (f32::INFINITY, 3.0, 0.5, Invalid),
        (2.0, f32::INFINITY, 0.5, Invalid),
        (f32::NAN, 3.0, 0.5, Invalid),
        // `x = NaN` is checked with the shapes: every comparison below it is false for a NaN, so
        // without that check it would reach the fraction and return a number.
        (2.0, 3.0, f32::NAN, Invalid),
    ] {
        assert_eq!(
            BetaRegMethod::for_arguments(a, b, x),
            want,
            "at a={a}, b={b}, x={x}"
        );
    }
}

#[test]
fn beta_fraction_agrees_with_the_integrated_polynomial() {
    // For `(2,3)` the CDF is `x²(6 − 8x + 3x²)` and its prefactor divided by `a` is `6x²(1−x)³`.
    for x in [0.007_812_5_f32, 0.0625, 0.125, 0.25, 0.4] {
        let expected = (6.0 - 8.0 * x + 3.0 * x * x) / (6.0 * (1.0 - x).powi(3));
        let Some(actual) = beta_cf(2.0, 3.0, x) else {
            panic!("fraction failed to converge at {x}");
        };
        close(actual, expected, 1e-5);
    }
}

#[test]
fn both_fraction_orientations_match_the_integrated_cdf() {
    // The integral of `20x(1−x)³`, since `B(2,4) = 1/20`. These `x` straddle the orientation
    // boundary at `3/8`, so both arms of `beta_fraction_cdf` are exercised.
    for x in [0.0625_f32, 0.25, 0.375, 0.5, 0.875] {
        let expected = x * x * (10.0 + x * (-20.0 + x * (15.0 - 4.0 * x)));
        let Some(actual) = beta_fraction_cdf(2.0, 4.0, x) else {
            panic!("integer-shape fraction failed to converge at x={x}");
        };
        close(actual, expected, 1e-5);
    }
}

#[test]
fn gamma_approximation_uses_the_shifted_rate_after_the_crossover() {
    // The gate is `b/(a+1)^{3/2} ≥ 1e4`, this format's scaling of the `f64` peer's `5e7`, and the
    // rate carries a half-shift in `a`. Both the gate and the shift are below the resolution of
    // any whole-function comparison, which is why they are asserted here.
    assert_eq!(beta_gamma_rate(2.0, 1e4), None);
    assert_eq!(beta_gamma_rate(2.0, 5e4), None);
    assert_eq!(beta_gamma_rate(2.0, 1e5), Some(100_000.5));
    assert_eq!(beta_gamma_rate(0.5, 1e5), Some(99_999.75));
    // Once gamma itself would use Edgeworth, there is no point standardising in beta coordinates.
    let edgeworth_min = crate::kernels::f32::gamma::GAMMA_EDGEWORTH_MIN;
    assert_eq!(beta_gamma_rate(edgeworth_min, 1e30), None);
    assert!(beta_gamma_rate(edgeworth_min.next_down(), 1e30).is_some());
}

#[test]
fn edgeworth_at_the_mean_uses_the_beta_skewness() {
    // At the mean `z = 0`, so the normal contributes exactly one half and what is left is the
    // skewness correction alone — which makes this the one argument where that term is the whole
    // signal rather than a perturbation.
    let (a, b) = (1e3_f32, 3e3_f32);
    let n = a + b;
    let skewness = 2.0 * (b - a) * (n + 1.0).sqrt() / ((n + 2.0) * (a * b).sqrt());
    let expected = 0.5 + skewness / (6.0 * (2.0 * core::f32::consts::PI).sqrt());
    // The mean is `a/n = 0.25`, exactly representable, so `x·n − a` is exactly zero here.
    close(beta_edgeworth(a, b, 0.25), expected, 4.0 * f32::EPSILON);
    assert!((beta_edgeworth(a, b, 0.25) - 0.502_427).abs() < 1e-5);
}

#[test]
fn edgeworth_returns_certainty_outside_the_reachable_tail() {
    // Past `|z| = 10` the standard normal tail has left this format in both directions, and
    // forming `z·z` would overflow the correction into a `NaN`. At `a = b = 512` the standardised
    // argument is `64(x − ½)`, so these two sit at `∓19.2`.
    let (a, b) = (BETA_EDGEWORTH_MIN, BETA_EDGEWORTH_MIN);
    assert_eq!(beta_edgeworth(a, b, 0.2), 0.0);
    assert_eq!(beta_edgeworth(a, b, 0.8), 1.0);
    // Symmetric shapes put the median at one half, where the correction vanishes.
    close(beta_edgeworth(a, b, 0.5), 0.5, 1e-6);
    // The result is a probability on either side of the cut.
    for x in [0.15_f32, 0.3, 0.45, 0.55, 0.7, 0.85] {
        let value = beta_edgeworth(a, b, x);
        assert!((0.0..=1.0).contains(&value), "at {x}: {value}");
    }
}

#[test]
fn edgeworth_saturates_where_the_shape_sum_leaves_the_format() {
    // `a + b` overflows from `a ≈ 1.7e38` in this format, which `f32::MAX` as a shape hits
    // directly — `f64` does not reach the equivalent until `2^970`, so this branch is far more
    // than a formality here. Halving keeps the sum finite and preserves the residual's sign, and
    // at shapes this large the transition is narrower than one ulp of the mean: the answer is a
    // certainty either side of it.
    let (a, b) = (f32::MAX, f32::MAX);
    assert!(!(a + b).is_finite());
    assert_eq!(beta_edgeworth(a, b, 0.25), 0.0);
    assert_eq!(beta_edgeworth(a, b, 0.75), 1.0);
    // Exactly at the mean the residual is zero, which is the one argument that is not a
    // certainty, and the halving is what keeps its sign recoverable at all.
    assert_eq!(beta_edgeworth(a, b, 0.5), 0.5);
    assert!(beta_edgeworth(a, b, f32::NAN).is_nan());
    // Unequal enormous shapes put the mean away from one half and the certainty follows it: the
    // mean is `a/(a+b)`, so these two sit at `0.75` and `0.25` and `x = 0.5` falls on opposite
    // sides of them. That the sign tracks the mean rather than the midpoint is the whole content
    // of keeping the residual compensated through the halving.
    assert_eq!(beta_edgeworth(f32::MAX, f32::MAX / 3.0, 0.5), 0.0);
    assert_eq!(beta_edgeworth(f32::MAX / 3.0, f32::MAX, 0.5), 1.0);
    assert_eq!(beta_edgeworth(f32::MAX, f32::MAX / 3.0, 0.8), 1.0);
    assert_eq!(beta_edgeworth(f32::MAX / 3.0, f32::MAX, 0.2), 0.0);
}

#[test]
fn beta_mean_residual_retains_the_low_bits_of_the_offset() {
    // `x(a+b) − a` near the mean is a difference of two nearly-equal `O(a)` quantities, so the
    // naive form loses the residual and `z` comes back as noise. These three values are chosen so
    // the compensated result is **correctly rounded** while the naive one is a full ulp away.
    let a = f32::from_bits(0x3f80_0006);
    let b = f32::from_bits(0x3f5b_bc37);
    let x = f32::from_bits(0x3f32_ddaf);
    let (n, n_error) = a.two_sum(b);
    // Rounded once from the exact binary rationals `x(a+b) − a`.
    let expected = f32::from_bits(0x3e98_c9a3);
    assert_eq!(beta_mean_residual(a, n, n_error, x), expected);
    // What it is being compared against: the uncompensated form lands one ulp high.
    let naive = x * (a + b) - a;
    assert_eq!(naive.to_bits(), 0x3e98_c9a4);
    assert_ne!(naive, expected);

    // A second triple, chosen so the **final** compensation term is what decides the answer:
    // adding `offset_error` gives the correctly rounded value and subtracting it lands one step
    // away, so this is the case that pins the sign of that last addition rather than its
    // presence. The naive form misses by two steps.
    let a = f32::from_bits(0x3f1d_5898);
    let b = f32::from_bits(0x3f7a_185d);
    let x = f32::from_bits(0x3ec5_6918);
    let (n, n_error) = a.two_sum(b);
    assert_eq!(beta_mean_residual(a, n, n_error, x).to_bits(), 0xba7f_ffe0);
    assert_eq!((x * (a + b) - a).to_bits(), 0xba80_0000);

    // And the residual is exactly zero at a representable mean, which the certainty branches and
    // the skewness test both depend on.
    let (a, b) = (1e3_f32, 3e3_f32);
    let (n, n_error) = a.two_sum(b);
    assert_eq!(beta_mean_residual(a, n, n_error, 0.25), 0.0);
}

#[test]
fn the_edgeworth_tail_cut_is_strict_on_both_sides() {
    use EdgeworthTail::{CertainLower, CertainUpper, Corrected};
    // Both boundaries belong to the corrected arm. Reached directly: `z` is a product of two
    // rounded quantities inside `beta_edgeworth`, so `|z| = 10` exactly is not constructible
    // from any `(a, b, x)` — which is what left the `>=` narrowing alive.
    assert_eq!(EdgeworthTail::for_standardised(EDGEWORTH_Z_MAX), Corrected);
    assert_eq!(EdgeworthTail::for_standardised(-EDGEWORTH_Z_MAX), Corrected);
    assert_eq!(
        EdgeworthTail::for_standardised(EDGEWORTH_Z_MAX.next_up()),
        CertainUpper
    );
    assert_eq!(
        EdgeworthTail::for_standardised((-EDGEWORTH_Z_MAX).next_down()),
        CertainLower
    );
    assert_eq!(EdgeworthTail::for_standardised(0.0), Corrected);
    assert_eq!(EdgeworthTail::for_standardised(f32::INFINITY), CertainUpper);
    assert_eq!(
        EdgeworthTail::for_standardised(f32::NEG_INFINITY),
        CertainLower
    );
    // A `NaN` compares false either way and reaches the correction, where it propagates.
    assert_eq!(EdgeworthTail::for_standardised(f32::NAN), Corrected);
}

#[test]
fn the_edgeworth_components_are_the_beta_moments() {
    // `√(n+1)/√(ab)`, chosen so `n + 1 = 400` is a perfect square and the answer is exactly
    // `20/199.5`. Dropping the `+ 1` would give `√399/199.5`, which differs in the fourth digit
    // — under any whole-function tolerance at the shapes this arm is selected for.
    let (a, n) = (199.5_f32, 399.0_f32);
    assert_eq!(edgeworth_inverse_width(a, a, n), 20.0 / 199.5);
    assert_ne!(
        edgeworth_inverse_width(a, a, n),
        n.sqrt() / a.sqrt() / a.sqrt()
    );

    // The skewness, against the textbook `γ₁ = 2(b−a)√(n+1)/((n+2)√(ab))`.
    let (a, b) = (1e3_f32, 3e3_f32);
    let n = a + b;
    let inverse_width = edgeworth_inverse_width(a, b, n);
    let skew = edgeworth_skew(a, b, n, inverse_width);
    let textbook = 2.0 * (b - a) * (n + 1.0).sqrt() / ((n + 2.0) * (a * b).sqrt());
    assert!(
        (skew - textbook).abs() <= 4.0 * f32::EPSILON * textbook,
        "{skew}"
    );
    // Symmetric shapes have no skewness at all, so the correction vanishes identically.
    assert_eq!(edgeworth_skew(a, a, 2.0 * a, inverse_width), 0.0);

    // `φ(z)·γ₁·(z²−1)/6` at the points that separate its factors: it is largest at the mean,
    // **exactly zero at `z = ±1`** where the quadratic vanishes, and even in `z` apart from the
    // density. A test at the mean alone pins none of that.
    assert_eq!(edgeworth_correction(1.0, skew), 0.0);
    assert_eq!(edgeworth_correction(-1.0, skew), 0.0);
    for (z, want) in [
        (0.0_f32, -0.002_426_975_4_f32),
        (0.5, -0.001_606_348_6),
        (2.0, 0.000_985_366_2),
        (-2.0, 0.000_985_366_2),
        (3.0, 0.000_215_690_1),
    ] {
        let got = edgeworth_correction(z, skew);
        assert!((got - want).abs() <= 1e-9, "at z={z}: {got} vs {want}");
    }
    // The density carries the sign of `−½z²`: dropping it turns a decay into a growth, which is
    // invisible at `z = 0` and four orders wrong by `z = 2`.
    assert!(edgeworth_correction(2.0, skew).abs() < edgeworth_correction(0.0, skew).abs());
    assert_eq!(edgeworth_correction(0.0, 0.0), 0.0);
}

#[test]
fn beta_edgeworth_matches_the_moment_formula_away_from_the_mean() {
    // The mean pins only `γ₁`; these `x` put `z` at roughly `∓2` and `∓1`, where the density and
    // the `z² − 1` factor both carry signal. The expectation is rebuilt from the moments here,
    // so it is independent of how the kernel associates them.
    let (a, b) = (1e3_f32, 3e3_f32);
    let n = a + b;
    let inverse_width = (n + 1.0).sqrt() / a.sqrt() / b.sqrt();
    let skew = 2.0 * ((b - a) / (n + 2.0)) * inverse_width;
    for x in [0.236_f32, 0.243, 0.25, 0.257, 0.264, 0.3] {
        let z = (x * n - a) * inverse_width;
        let density = (-0.5 * z * z).exp() / crate::constants::f32::SQRT_TAU;
        let expected = 0.5 * crate::kernels::f32::erf::erfc(-z * core::f32::consts::FRAC_1_SQRT_2)
            - density * skew * (z * z - 1.0) / 6.0;
        let got = beta_edgeworth(a, b, x);
        assert!(
            (got - expected).abs() <= 1e-5 * expected.max(1e-6),
            "at x={x} (z={z}): {got} vs {expected}"
        );
    }
}

#[test]
fn edgeworth_returns_certainty_where_the_standardised_square_would_overflow() {
    // `a = 512`, `b = 1e38` puts `z` at `2.21e36`, whose square leaves the format. Without the
    // cut the correction would be `NaN` and so would the answer; this is the case that makes the
    // cut a correctness guard rather than a shortcut.
    let (a, b) = (512.0_f32, 1e38_f32);
    let n = a + b;
    let inverse_width = edgeworth_inverse_width(a, b, n);
    let z = (0.5 * n - a) * inverse_width;
    assert!(z > 1e36, "{z}");
    assert!(!(z * z).is_finite());
    assert_eq!(beta_edgeworth(a, b, 0.5), 1.0);
    assert_eq!(beta_edgeworth(b, a, 0.5), 0.0);
}

#[test]
fn the_gamma_limit_arms_match_the_wide_kernel() {
    // `Beta(2, 1e5)` has its mass at `x ≈ 2e-5`, and this is the only route that reaches it: the
    // method table sends it to `GammaLimit` before Edgeworth. The arm's argument is
    // `−rate·ln(1−x)`, where every operator survived until this test — a sign or an association
    // there moves the answer by far more than the comparison below allows.
    for &x in &[1e-5_f32, 2e-5, 5e-5, 1e-4, 1e-3] {
        assert_eq!(
            BetaRegMethod::for_arguments(2.0, 1e5, x),
            BetaRegMethod::GammaLimit
        );
        agrees(
            beta_reg(2.0, 1e5, x),
            crate::kernels::beta::beta_reg(2.0, 1e5, f64::from(x)),
            1e-3,
        );
    }
    // And the reflection, which preserves the small upper tail instead of a complement.
    for &x in &[0.999_9_f32, 0.999_95, 0.999_99] {
        assert_eq!(
            BetaRegMethod::for_arguments(1e5, 2.0, x),
            BetaRegMethod::GammaLimitReflected
        );
        agrees(
            beta_reg(1e5, 2.0, x),
            crate::kernels::beta::beta_reg(1e5, 2.0, f64::from(x)),
            1e-3,
        );
    }
}
