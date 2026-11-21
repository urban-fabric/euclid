//! Unit tests for the regularized incomplete gamma pair.
//!
//! Beside [`super`] rather than with `Γ`'s own tests, because what they reach is this module's:
//! the [`IncompleteGammaMethod`] table and the small-shape logarithm. The method table is the
//! part no value assertion covers — every route is accurate in its own region, so one region
//! over still agrees to several digits and only the decision separates them.

use super::*;

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
}

#[test]
fn small_shape_log_normalizer_matches_mpfr_at_the_cutoff() {
    // From `extreme_shapes`, a local generator over MPFR at 256 bits. See CONTRIBUTING.md,
    // "Reference values are pasted in": the generator is not committed, so the precision is
    // recorded here instead of a path.
    let expected = -0.576_393_598_283_369_6;
    assert!((ln_gamma1p_over_a(SMALL_SHAPE) - expected).abs() <= f64::EPSILON / 2.0);
}

#[test]
fn regularized_incomplete_gamma_bounds_and_complement() {
    for &(a, x) in &[(1.0, 0.5), (2.5, 3.0), (5.0, 2.0), (10.0, 12.0)] {
        let p = gamma_lr(a, x);
        let q = gamma_ur(a, x);
        assert!((0.0..=1.0).contains(&p), "P out of range: {p}");
        close(p + q, 1.0, 1e-10);
    }
    close(gamma_lr(1.0, 2.0), 1.0 - (-2.0f64).exp(), 1e-10);
}

/// The eight incomplete-gamma routes, at and either side of every boundary.
///
/// Both tails dispatch on this one decision, so a misroute moves *two* public functions at once
/// — and every arm returns a probability in `[0, 1]`, so a wrong one still looks plausible. The
/// table is the only thing that separates them.
#[test]
fn the_incomplete_gamma_methods_partition_the_quadrant() {
    use IncompleteGammaMethod::{
        AtInfinity, AtZero, Edgeworth, Invalid, LowerSeries, SmallShapeFraction, SmallShapeSeries,
        UpperFraction,
    };
    for (a, x, want) in [
        // Rejected inputs: a non-positive or non-finite shape, a negative argument, any NaN.
        (0.0, 1.0, Invalid),
        (-1.0, 1.0, Invalid),
        (f64::INFINITY, 1.0, Invalid),
        (f64::NAN, 1.0, Invalid),
        (1.0, -0.125, Invalid),
        (1.0, f64::NAN, Invalid),
        // The two limits, which both tails return as exact constants.
        (2.0, 0.0, AtZero),
        (2.0, -0.0, AtZero),
        (2.0, f64::INFINITY, AtInfinity),
        // The Edgeworth handover, closed on the left.
        (GAMMA_EDGEWORTH_MIN, 1.0, Edgeworth),
        (GAMMA_EDGEWORTH_MIN.next_down(), 1.0, LowerSeries),
        (1e30, 1.0, Edgeworth),
        // The small shape, also closed, and it still picks a tail because the series is an
        // expansion in `x` valid only below `a + 1`.
        (SMALL_SHAPE, 0.5, SmallShapeSeries),
        (SMALL_SHAPE, 2.0, SmallShapeFraction),
        (SMALL_SHAPE.next_up(), 0.5, LowerSeries),
        (SMALL_SHAPE.next_up(), 2.0, UpperFraction),
        (1e-30, 0.5, SmallShapeSeries),
        // And the tail split at `x = a + 1`, which belongs to the fraction.
        (2.0, 1.0, LowerSeries),
        (2.0, 3.0_f64.next_down(), LowerSeries),
        (2.0, 3.0, UpperFraction),
        (2.0, 5.0, UpperFraction),
    ] {
        assert_eq!(
            IncompleteGammaMethod::for_arguments(a, x),
            want,
            "at a={a:e}, x={x:e}"
        );
    }

    // The two tails agree about the route, which is what lets each be a bare `match`: every
    // argument gives a complementary pair, whichever arm produced it.
    for (a, x) in [
        (SMALL_SHAPE, 0.5),
        (SMALL_SHAPE, 4.0),
        (2.0, 1.0),
        (2.0, 5.0),
        (1e4, 1e4),
        (GAMMA_EDGEWORTH_MIN, 3e7),
    ] {
        let (lower, upper) = (gamma_lr(a, x), gamma_ur(a, x));
        assert!((lower + upper - 1.0).abs() <= 1e-12, "at a={a:e}, x={x:e}");
        assert!((0.0..=1.0).contains(&lower) && (0.0..=1.0).contains(&upper));
    }
    // The limits are exact in both tails, not merely close.
    assert_eq!((gamma_lr(2.0, 0.0), gamma_ur(2.0, 0.0)), (0.0, 1.0));
    assert_eq!(
        (gamma_lr(2.0, f64::INFINITY), gamma_ur(2.0, f64::INFINITY)),
        (1.0, 0.0)
    );
    assert!(gamma_lr(0.0, 1.0).is_nan() && gamma_ur(0.0, 1.0).is_nan());
}

/// The incomplete gamma at integer shape against the Poisson sum, which needs no reference.
///
/// For integer `a`, `Q(a,x) = e^{−x}·Σ_{k<a} x^k/k!` exactly. Every term of that sum is
/// positive, so it carries only its own `a` roundings — a few ulp — which makes it an honest
/// yardstick for a routine whose error this test exists to bound.
///
/// # The comparison is made on `Q`, and that is not a detail
///
/// The first version of this test compared `P` against `1 − poisson_upper` and failed at
/// `P(20, 11.06)` by `43` ulp. **The reference was the wrong side.** `Q` there is `0.99`, so
/// its few-ulp error is `2e-15` absolute, and against a `P` of `9.8e-3` that is over a thousand
/// ulp — the standing trap that a reference may not be formed by a subtraction the kernel
/// exists to avoid. The Poisson sum computes `Q` *directly* and keeps its relative accuracy at
/// any `x`, so `Q` is the quantity to compare and no subtraction appears on either side.
///
/// # Why this is the test the prefactor needed
///
/// It is sharpest exactly where [`log_scaled_gamma`](crate::math::gamma::log_scaled_gamma) was
/// weakest. At `a = 100`, `x = 120` the old form summed `a·ln x`, `x` and `lnΓ(a)` — `479`,
/// `120` and `359` — into an exponent of order one, and `exp` turned the surviving absolute
/// error into hundreds of ulp of relative error. The rearranged form's terms there are `−1.76`,
/// `1.38` and `8e-4`.
#[test]
fn the_incomplete_gamma_matches_the_poisson_sum_at_integer_shape() {
    /// `e^{−x}·Σ_{k<a} x^k/k!`, accumulated as positive terms so nothing cancels.
    fn poisson_upper(shape: u32, x: f64) -> f64 {
        let mut term = (-x).exp();
        let mut sum = term;
        for k in 1..shape {
            term *= x / f64::from(k);
            sum += term;
        }
        sum
    }

    let mut worst = (0.0_f64, 0.0_f64, 0.0_f64);
    let mut checked = 0usize;
    for shape in [20_u32, 25, 30, 40, 50, 60, 80, 100] {
        let a = f64::from(shape);
        // The upper arm over `x ∈ (a, 1.5a]`, which is exactly `0 < u ≤ ½` — the rearranged
        // prefactor's whole band, and the side on which `Q` is the directly computed tail.
        for step in 1..=8 {
            let x = a + f64::from(step) * a * 0.05;
            let want = poisson_upper(shape, x);
            let got = gamma_ur(a, x);
            let ulps = (got - want).abs() / crate::Real::ulp(want);
            if ulps > worst.0 {
                worst = (ulps, a, x);
            }
            assert!(
                ulps <= 40.0,
                "Q({a}, {x}) = {got:e}, Poisson gives {want:e}, {ulps} ulp apart"
            );
            checked += 1;
        }
    }
    assert!(
        checked >= 60,
        "only {checked} arguments reached the assertion"
    );
    assert!(
        worst.0 > 0.0,
        "no shape produced a measurable difference; the comparison is vacuous"
    );
}
