//! What the branch can still *resolve* once the shapes leave the format.
//!
//! Part of the [`beta_large_shape`](super) suite. These are the cases where the mean itself has
//! rounded: the shape sum overflows, or the transition is finer than the spacing of the rounded
//! mean. What a caller needs there is not accuracy but **ordering** — which side of the mean an
//! argument falls on — and that is what these assert.

use super::*;

#[test]
fn an_overflowing_shape_sum_puts_its_point_mass_at_the_mean() {
    // The `a + b` overflow path has three arms — below the mean, above it, and exactly on it —
    // and the test above reaches only the third, because a symmetric shape puts the mean at
    // `0.5` and that is the `x` it asks about. Two mutants survived in precisely that blind
    // spot: swapping either comparison went unnoticed while `I_½(a, a)` stayed right.
    //
    // So this sweeps either side as well, and at an asymmetric ratio, where the mean is not at
    // the midpoint and getting the comparison backwards cannot coincidentally agree.
    for &(a, b) in &[
        (1.5e308f64, 1.5e308f64),
        (f64::MAX, f64::MAX),
        (1.5e308, 3.0e307),
        (3.0e307, 1.5e308),
    ] {
        assert!(!(a + b).is_finite(), "({a:e}, {b:e}) was meant to overflow");
        let mean = 1.0 / (1.0 + b / a);
        assert!(
            (0.0..=1.0).contains(&mean),
            "the ratio-form mean must stay a probability: {mean}"
        );
        for &x in &[0.0, 1e-9, 0.25, 0.5, 0.75, 1.0 - f64::EPSILON, 1.0] {
            let want = if x <= 0.0 {
                0.0 // the `x <= 0` guard in `beta_reg` takes this before the branch
            } else if x >= 1.0 {
                1.0
            } else if x < mean {
                0.0
            } else if x > mean {
                1.0
            } else {
                0.5
            };
            assert_exact(
                Real::beta_reg(x, a, b),
                want,
                &format!("I_{x}({a:e}, {b:e}) against a point mass at {mean}"),
            );
        }
        // And the arms are actually distinguishable at this ratio, or the loop proves nothing.
        assert!(
            mean > 1e-9 && mean < 1.0 - f64::EPSILON,
            "the sample must straddle the mean at ({a:e}, {b:e}), which sits at {mean}"
        );
    }
}

#[test]
fn beta_reg_resolves_the_side_of_an_extreme_mean() {
    let a = f64::MAX;
    let b = a.next_down();
    // The means straddle 1/2. The opposite tail is below 1e-270 in
    // each shape order by the one-sided variance bound.
    assert_eq!(Real::beta_reg(0.5, a, b), 0.0);
    assert_eq!(Real::beta_reg(0.5, b, a), 1.0);
    assert!(Real::beta_reg(f64::NAN, a, b).is_nan());

    let unit = 2.0f64.powi(1020);
    let (a, b) = (3.0 * unit, 15.0 * unit);
    // Binary64 rounds 1/6 down and 5/6 up. The adjacent value on
    // the other side of each fraction lies in the opposite tail.
    let x = 1.0f64 / 6.0;
    let y = 5.0f64 / 6.0;
    assert_eq!(Real::beta_reg(x, a, b), 0.0);
    assert_eq!(Real::beta_reg(x.next_up(), a, b), 1.0);
    assert_eq!(Real::beta_reg(y.next_down(), b, a), 0.0);
    assert_eq!(Real::beta_reg(y, b, a), 1.0);

    // The exact mean is representable for this asymmetric pair.
    assert_eq!(Real::beta_reg(0.25, 4.0 * unit, 12.0 * unit), 0.5);
    assert_eq!(Real::beta_reg(0.75, 12.0 * unit, 4.0 * unit), 0.5);

    // x = 1/2 + 2^-53 and b = a(1 - 4·2^-53), so x(a+b) - a
    // is negative even though the rounded product equals a.
    let a = 2.0f64.powi(1022);
    let b = f64::from_bits(a.to_bits() - 4);
    assert_eq!(Real::beta_reg(0.5f64.next_up(), a, b), 0.0);
}

#[test]
fn beta_reg_resolves_a_transition_finer_than_the_rounded_mean() {
    let a = 2.0f64.powi(104);
    let b = a.next_up();
    // Normal-limit references use the exact binary64 inputs and the
    // beta variance, evaluated with 100-digit Decimal arithmetic
    // before calling erfc. At these shapes skewness is below 1e-30.
    for (x, want) in [
        (0.5f64.next_down(), 0.5),
        (0.5, 0.760_249_938_906_523_3),
        (0.5f64.next_up(), 0.983_052_573_237_655_4),
    ] {
        assert_close(Real::beta_reg(x, a, b), want, 8e-16, "beta transition");
    }
    assert_close(
        Real::beta_reg(0.5, b, a),
        0.239_750_061_093_476_7,
        8e-16,
        "beta transition with reversed shapes",
    );
}

#[test]
fn beta_reg_resolves_the_transition_at_an_extreme_shape_ratio() {
    let a = 2.0f64.powi(104);
    let b = 3.0 * a * a;
    let x = (1.0 / a) / 3.0;
    // Binary64 rounds 1/3 down to (1 − 2^-54)/3. Thus b·x is
    // a − sqrt(a)/4 before rounding, and the beta variance gives
    // z = −1/4 to working precision. Φ(−1/4) is the normal limit.
    assert_close(
        Real::beta_reg(x, a, b),
        0.401_293_674_317_076_3,
        8e-16,
        "beta transition with widely separated shapes",
    );

    let a = 2.0f64.powi(200);
    let b = 3.0 * a * a;
    let x = (1.0 / a) / 3.0;
    assert_eq!(Real::beta_reg(x, a, b), 0.0);
    assert_eq!(Real::beta_reg(x.next_up(), a, b), 1.0);
}

#[test]
fn beta_reg_is_a_probability_and_monotone_at_every_large_shape() {
    // The invariant the overflow violated, stated before any question of accuracy — the same
    // shape of test as the gamma peer's `both_tails_stay_probabilities_at_every_shape`. Swept
    // over six decades of shape and three ratios, several deviations either side of the mean.
    for exponent in [7.0f64, 9.0, 20.0, 60.0, 150.0, 300.0] {
        for &ratio in &[1.0f64, 1e3, 1e-3] {
            let a = 10f64.powf(exponent);
            let b = a * ratio;
            if !(b.is_finite() && b.min(a) >= BETA_EDGEWORTH_MIN) {
                continue;
            }
            let n = a + b;
            let mean = a / n;
            let sd = (a / n).sqrt() * (b / n).sqrt() / (n + 1.0).sqrt();
            // Whether `f64` can even represent a point one deviation from the mean. `σ` falls
            // like `1/√a` while the spacing at the mean is fixed, so past `a ≈ 1e31` (where
            // `σ` drops under `ulp(½) = 1.1e-16`) the whole distribution lives inside one
            // representable step and the CDF is a *step function* in this format. That is the
            // format speaking, not the kernel, but it changes what the sweep below can claim —
            // and asserting the strong version anyway is how this test first failed.
            let spacing = Real::next_after(mean, f64::INFINITY) - mean;
            let resolvable = sd > spacing;
            let mut previous = -1.0f64;
            for step in -60..=60 {
                let x = mean + f64::from(step) * 0.1 * sd;
                if !(0.0..=1.0).contains(&x) {
                    continue;
                }
                let value = Real::beta_reg(x, a, b);
                assert!(
                    (0.0..=1.0).contains(&value),
                    "I_{x}(1e{exponent}, ratio {ratio:e}) = {value} is not a probability"
                );
                assert!(
                    value >= previous,
                    "I_x(1e{exponent}, ratio {ratio:e}) decreased at x = {x}: \
                     {previous} then {value}"
                );
                previous = value;
            }
            if resolvable {
                // The sweep has to actually cross the distribution, or it proves nothing: a
                // constant zero would satisfy every assertion above.
                assert!(
                    previous > 0.999,
                    "the sweep at 1e{exponent}, ratio {ratio:e} never reached the upper tail"
                );
            } else {
                // Unresolvable, so step in *ulps* instead and require the transition to be
                // sited at the mean rather than merely to exist somewhere.
                assert_exact(
                    Real::beta_reg(Real::next_after(mean, f64::NEG_INFINITY), a, b),
                    0.0,
                    &format!("one ulp below the mean at 1e{exponent}, ratio {ratio:e}"),
                );
                assert_exact(
                    Real::beta_reg(Real::next_after(mean, f64::INFINITY), a, b),
                    1.0,
                    &format!("one ulp above the mean at 1e{exponent}, ratio {ratio:e}"),
                );
            }
        }
    }
}
