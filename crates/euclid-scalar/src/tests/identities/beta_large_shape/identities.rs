//! Identities and joins inside the Edgeworth branch.
//!
//! Part of the [`beta_large_shape`](super) suite. The reflection identity
//! `I_x(a, b) = 1 − I_{1−x}(b, a)` constrains the branch against itself at a different argument,
//! which is the only instrument available where no reference reaches.

use super::*;

#[test]
fn the_reflection_identity_holds_inside_the_edgeworth_branch() {
    // `I_x(a,b) = 1 − I_{1−x}(b,a)` is exact, and the Edgeworth form satisfies it
    // analytically: swapping the shapes negates `z` and `γ₁` together, so the correction
    // changes sign with the tail it corrects. That makes this a test of the *implementation* —
    // in particular that the skewness keeps its sign — rather than of the approximation.
    //
    // Measured worst 7.0e-12, at (1e7, 1e9) where the mean sits closest to an endpoint.
    for &(a, b) in &[
        (1e7f64, 1e7f64),
        (1e7, 1e8),
        (1e8, 1e7),
        (1e7, 1e9),
        (2e7, 5e7),
    ] {
        let n = a + b;
        let sd = (a / n).sqrt() * (b / n).sqrt() / (n + 1.0).sqrt();
        for step in -40..=40 {
            let x = a / n + f64::from(step) * 0.1 * sd;
            if !(0.0..=1.0).contains(&x) {
                continue;
            }
            assert_close(
                Real::beta_reg(x, a, b),
                1.0 - Real::beta_reg(1.0 - x, b, a),
                5e-11,
                &format!("reflection at x = {x}, ({a:e}, {b:e})"),
            );
        }
    }
}

#[test]
fn the_shape_crossover_leaves_no_step() {
    // The continued fraction and the Edgeworth form share no code and no derivation — a
    // term-by-term Lentz recurrence against a normal CDF with a skewness correction — so
    // agreement at the join is the evidence that the join is in the right place. Checked at
    // three ratios, since the two routes have quite different dependence on it.
    //
    // Measured worst step 9.0e-9, which is both sides carrying their own ~6e-9: the fraction's
    // prefactor cancellation below, the expansion's truncation above. The kernel's threshold
    // derivation sets those equal here, and this is that prediction holding.
    for &z in &[-3.0f64, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0] {
        for &ratio in &[1.0f64, 10.0, 100.0] {
            let at = |a: f64| {
                let b = a * ratio;
                let n = a + b;
                let sd = (a / n).sqrt() * (b / n).sqrt() / (n + 1.0).sqrt();
                Real::beta_reg(a / n + z * sd, a, b)
            };
            let below = at(Real::next_after(BETA_EDGEWORTH_MIN, f64::NEG_INFINITY));
            let above = at(Real::next_after(BETA_EDGEWORTH_MIN, f64::INFINITY));
            assert_close(
                above,
                below,
                2e-8,
                &format!("the shape crossover steps at z = {z}, ratio {ratio}"),
            );
        }
    }
}

#[test]
fn a_degenerate_shape_ratio_still_gives_a_certainty() {
    // `f64::MAX` against `1e7` keeps `a + b` *finite* — adding `1e7` to `MAX` rounds back to
    // `MAX` — so this is not the overflowing-sum case but the underflowing-variance one: the
    // variance is `3e-610` and gone, while `σ` is `1.8e-305` and fine. The whole distribution
    // then sits inside one representable step, so every `x` is a certainty, and the question
    // is only whether it is the *right* certainty rather than a NaN.
    //
    // Which certainty is decided by the mean, not by the midpoint of `[0, 1]`, and the mean is
    // `5.6e-302` for `(1e7, MAX)` — not zero. `MIN_POSITIVE` is six decades *below* that, so
    // `I = 0` there is correct and an earlier version of this test was simply wrong to want 1.
    for &(a, b) in &[(f64::MAX, 1e7f64), (1e7, f64::MAX)] {
        let mean = a / (a + b);
        for &x in &[
            f64::MIN_POSITIVE,
            1e-300,
            1e-6,
            0.1,
            0.5,
            0.9,
            1.0 - f64::EPSILON,
        ] {
            let want = if x < mean {
                0.0
            } else if x > mean {
                1.0
            } else {
                0.5
            };
            assert_exact(
                Real::beta_reg(x, a, b),
                want,
                &format!("I_{x}({a:e}, {b:e}) against a point mass at {mean:e}"),
            );
        }
    }
    // The endpoints keep their exact values regardless of shape.
    for &(a, b) in &[(f64::MAX, 1e7f64), (1e7, f64::MAX), (f64::MAX, f64::MAX)] {
        assert_exact(Real::beta_reg(0.0, a, b), 0.0, "I_0 = 0");
        assert_exact(Real::beta_reg(1.0, a, b), 1.0, "I_1 = 1");
    }
}
