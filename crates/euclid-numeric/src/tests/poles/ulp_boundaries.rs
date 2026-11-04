//! Boundaries probed one representable step at a time.
//!
//! Part of the [poles suite](super). Where the other files in this suite sample *regions* that
//! are hard, this one attacks the exact arguments at which behaviour changes, and the single
//! representable neighbours on either side. Those are the arguments that a sampled sweep steps
//! over, and they are where sign errors, wrong branches and `0/0` live.
//!
//! Three kinds of boundary are covered:
//!
//! - **Domain edges.** The first representable value inside a domain, the last one outside, and
//!   the signed zeros between them.
//! - **Poles.** One ulp either side, where the divergence must have the right sign and the
//!   right magnitude rather than merely being large.
//! - **Exact zeros.** One ulp either side of a zero the implementation returns *exactly*, where
//!   the value must become small-but-nonzero with the correct sign.

use crate::Real;
use euclid_utils::assert_close;

/// The next representable value above `x`.
fn up(x: f64) -> f64 {
    Real::next_after(x, f64::INFINITY)
}

/// The next representable value below `x`.
fn down(x: f64) -> f64 {
    Real::next_after(x, f64::NEG_INFINITY)
}

// --- domain edges, one step at a time -----------------------------------

#[test]
fn the_first_representable_argument_inside_each_domain_is_accepted() {
    // `Y`, `K`, `Ci` and `E₁` are real only for `x > 0`. The first argument inside that domain
    // is the smallest subnormal, and it must be *evaluated*, not rejected: all four have
    // finite values there (`K₁` and `Y₁` excepted, whose `1/x` genuinely overflows).
    let first = up(0.0);
    assert_eq!(
        first, 5e-324,
        "the first step above zero is the smallest subnormal"
    );

    assert!(Real::bessel_y0(first).is_finite() && Real::bessel_y0(first) < -400.0);
    assert!(Real::bessel_k0(first).is_finite() && Real::bessel_k0(first) > 400.0);
    assert!(Real::cos_int(first).is_finite() && Real::cos_int(first) < -400.0);
    assert!(Real::exp_int_e1(first).is_finite() && Real::exp_int_e1(first) > 400.0);
    // And the last argument *outside* it is rejected, one step the other way.
    let last_outside = down(0.0);
    assert_eq!(last_outside, -5e-324);
    assert!(Real::bessel_y0(last_outside).is_nan());
    assert!(Real::bessel_k0(last_outside).is_nan());
    assert!(Real::cos_int(last_outside).is_nan());
    assert!(Real::exp_int_e1(last_outside).is_nan());
}

#[test]
fn negative_zero_is_treated_as_the_boundary_rather_than_as_negative() {
    // `-0.0 < 0.0` is false, so a negative zero lands on the pole branch rather than the
    // out-of-domain one. That is the right convention — it is a zero, not a negative number —
    // but it is worth pinning, because the alternative reading would return `NaN` for an
    // argument that merely arrived with a sign bit set.
    assert_eq!(Real::bessel_y0(-0.0f64), f64::NEG_INFINITY);
    assert_eq!(Real::bessel_k0(-0.0f64), f64::INFINITY);
    assert_eq!(Real::cos_int(-0.0f64), f64::NEG_INFINITY);
    assert_eq!(Real::exp_int_e1(-0.0f64), f64::INFINITY);
    // The odd functions instead carry the sign of the zero through.
    assert!(Real::sin_int(-0.0f64).is_sign_negative());
    assert!(Real::bessel_j1(-0.0f64).is_sign_negative());
    assert_eq!(Real::bessel_j0(-0.0f64), 1.0, "J0 is even, so -0 gives 1");
    assert_eq!(Real::lambert_w(-0.0f64), 0.0);
    assert!(
        Real::lambert_w(-0.0f64).is_sign_negative(),
        "W is odd at the origin"
    );
}

#[test]
fn the_lambert_branch_point_is_resolved_to_the_last_representable_step() {
    // Below `−1/e` there is no real `W`; at it, `W = −1`; above it, `W > −1`. All three must
    // hold at one-ulp resolution, and the principal branch must never dip below `−1` — a
    // Newton step from a poor guess is exactly how that happens.
    let branch = crate::constants::f64::NEG_INV_E;
    assert!(
        Real::lambert_w(down(branch)).is_nan(),
        "one step below the branch point there is no real value"
    );
    assert_close(
        Real::lambert_w(branch),
        -1.0,
        1e-8,
        "at the branch point W = -1",
    );
    for _ in 0..1 {
        let inside = up(branch);
        let w = Real::lambert_w(inside);
        assert!(w >= -1.0 - 1e-8, "W({inside}) = {w} fell below -1");
        assert!(w.is_finite());
    }
    // Stepping away from the branch point, W must increase monotonically and stay above -1.
    let mut previous = f64::NEG_INFINITY;
    let mut x = branch;
    for _ in 0..200 {
        x = up(x);
        let w = Real::lambert_w(x);
        assert!(w >= -1.0 - 1e-8, "W({x}) = {w} below -1");
        assert!(w >= previous - 1e-8, "W not monotone at {x}");
        previous = w;
    }
}

// --- poles, one step at a time ------------------------------------------

#[test]
fn zeta_diverges_with_the_right_sign_one_step_either_side_of_its_pole() {
    // This is the test that caught the `1 − 2^{1−s}` cancellation. Just *below* the pole the
    // divisor of the alternating series is `−δ ln 2`; computed as the literal subtraction it
    // rounds to zero at `δ = 1e-16`, which returned `+∞` where the true value is `−9.0e15` —
    // both the sign and the magnitude wrong, at an argument a sweep would never land on.
    assert_eq!(Real::zeta(1.0f64), f64::INFINITY, "the pole itself");

    for &delta in &[1e-8f64, 1e-12, 1e-14, 1e-15, 1e-16] {
        let below = 1.0 - delta;
        let above = 1.0 + delta;
        // The two sides are not symmetric: the gap between representable values is 1.1e-16
        // just below 1 and 2.2e-16 just above it, because 1 is the bottom of its binade. So
        // `1.0 - 1e-16` is a distinct number while `1.0 + 1e-16` rounds back to exactly 1,
        // where the comparison below would be inf/inf. Skip any side that is not actually
        // off the pole.
        let (test_below, test_above) = (below != 1.0, above != 1.0);
        // zeta ~ 1/(s-1) + gamma on both sides, so the signs are opposite and the magnitudes
        // are both about 1/delta. The reference is formed from the *stored* s rather than
        // from delta: `1.0 - 1e-12` is 9.9998e-13 away from 1, not 1e-12, and at this
        // magnification that 2e-5 discrepancy would swamp the comparison.
        let expected_below = 1.0 / (below - 1.0) + euclid_utils::EULER_GAMMA;
        let expected_above = 1.0 / (above - 1.0) + euclid_utils::EULER_GAMMA;
        if test_below {
            assert!(
                Real::zeta(below) < 0.0,
                "zeta({below}) = {} must be negative below the pole",
                Real::zeta(below)
            );
            assert_close(
                Real::zeta(below) / expected_below,
                1.0,
                1e-9,
                &format!("zeta(1 - {delta:e}) ~ -1/delta"),
            );
        }
        if test_above {
            assert!(
                Real::zeta(above) > 0.0,
                "zeta({above}) must be positive above the pole"
            );
            assert_close(
                Real::zeta(above) / expected_above,
                1.0,
                1e-9,
                &format!("zeta(1 + {delta:e}) ~ 1/delta"),
            );
        }
    }
    // And at literally one representable step, where the only requirement left is that the
    // sign and the sheer magnitude are right.
    assert!(Real::zeta(down(1.0f64)) < -1e15, "one ulp below the pole");
    assert!(Real::zeta(up(1.0f64)) > 1e15, "one ulp above the pole");
}

#[test]
fn the_origin_poles_are_approached_monotonically_from_the_first_step_up() {
    // Walking up from the smallest subnormal, `K₀` must decrease and `Y₀` increase without a
    // single reversal — the log singularity is where a series whose leading term was formed by
    // a cancelling subtraction would wobble.
    let mut x = up(0.0);
    let mut previous_k = f64::INFINITY;
    let mut previous_y = f64::NEG_INFINITY;
    for _ in 0..400 {
        let (k, y) = (Real::bessel_k0(x), Real::bessel_y0(x));
        assert!(k.is_finite() && y.is_finite(), "non-finite at x = {x:e}");
        assert!(k <= previous_k, "K0 rose at x = {x:e}");
        assert!(y >= previous_y, "Y0 fell at x = {x:e}");
        previous_k = k;
        previous_y = y;
        // Stride by many ulps so the walk covers decades rather than a single binade.
        for _ in 0..1000 {
            x = up(x);
        }
    }
}

// --- exact zeros, one step at a time ------------------------------------

#[test]
fn zeta_leaves_its_trivial_zeros_with_the_right_sign() {
    // `zeta` returns the trivial zeros as *exact* zeros rather than letting `sin(πs/2)`
    // cancel. One step away it must be small but nonzero, and the sign must alternate between
    // consecutive zeros — which is the part an exact-return shortcut could get wrong.
    for k in 1..8 {
        let s = -2.0 * f64::from(k);
        assert_eq!(Real::zeta(s), 0.0, "zeta({s}) must be exactly zero");
        let (lo, hi) = (Real::zeta(s - 1e-6), Real::zeta(s + 1e-6));
        assert!(
            lo != 0.0 && hi != 0.0,
            "zeta is exactly zero either side of {s}, so the zero is not isolated"
        );
        assert!(
            lo * hi < 0.0,
            "zeta does not cross zero at s = {s}: {lo:e} then {hi:e}"
        );
    }
    // Negative *odd* integers are not zeros, and must not be swallowed by the same branch.
    for k in 0..6 {
        let s = -1.0 - 2.0 * f64::from(k);
        assert!(
            Real::zeta(s) != 0.0,
            "zeta({s}) is a Bernoulli value, not a zero"
        );
    }
    // s = 0 is likewise not a trivial zero.
    assert_close(Real::zeta(0.0f64), -0.5, 1e-15, "zeta(0) = -1/2");
}

#[test]
fn the_oscillatory_zeros_are_crossed_within_a_few_ulps() {
    // At a zero the returned value is dominated by how far the nearest `f64` sits from the
    // true zero, so the meaningful statement is not "small" but "crosses": stepping a handful
    // of ulps either way must change the sign. A branch that returned a constant near a zero,
    // or one whose two regimes disagreed there, would fail this while passing an absolute
    // smallness check.
    /// A named function paired with one of its zeros.
    type ZeroCase = (&'static str, fn(f64) -> f64, f64);
    let zeros: &[ZeroCase] = &[
        ("J0", Real::bessel_j0, 2.404_825_557_695_773),
        ("J0", Real::bessel_j0, 8.653_727_912_911_012),
        ("J1", Real::bessel_j1, 3.831_705_970_207_512_4),
        ("Y0", Real::bessel_y0, 0.893_576_966_279_167_5),
        ("Y1", Real::bessel_y1, 2.197_141_326_031_017),
        ("Ci", Real::cos_int, 0.616_505_485_620_716_2),
        ("Ei", Real::exp_int_ei, 0.372_507_410_781_366_8),
    ];
    for &(name, f, zero) in zeros {
        // Step out until the sign is unambiguous; a few hundred ulps is far less than any
        // feature of these functions, and vastly less than the gap to the next zero.
        let mut lo = zero;
        let mut hi = zero;
        for _ in 0..256 {
            lo = down(lo);
            hi = up(hi);
        }
        assert!(
            f(lo) * f(hi) < 0.0,
            "{name} does not change sign across its zero at {zero}: {:e} then {:e}",
            f(lo),
            f(hi)
        );
        // And the value at the zero is tiny in absolute terms, which is all that is available.
        assert!(
            f(zero).abs() < 1e-14,
            "{name}({zero}) = {:e} should be at the noise floor",
            f(zero)
        );
    }
}
