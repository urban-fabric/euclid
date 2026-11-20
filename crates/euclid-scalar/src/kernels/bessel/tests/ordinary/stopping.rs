//! Term counts and iteration caps — what a stopping rule changes that a value does not.
//!
//! Every tolerance in this family sits below `f64`'s resolution, so admitting or refusing a term
//! at the margin moves no result and mutation testing duly finds each comparison unkillable. The
//! count is the deterministic observable that separates them, which is why these kernels return
//! [`Converged`](crate::math::pair::Converged) and why this file exists.
//!
//! One of these counts found a real defect rather than just an unkillable mutant; see
//! [`the_y_series_stop_where_their_tolerance_says`].

use crate::kernels::bessel::ordinary_second::{Y_SERIES_MAX, y0_series, y1_series};

/// The `Y` series' **term counts**, which are the only observable their stopping rules have.
///
/// `TERM_EPS` is `1e-19`, below `f64`'s resolution, so a term admitted or refused at the margin
/// moves no value — mutation testing finds every comparison in `t < sum·ε` unkillable, including
/// the degenerate `==` that stops the break firing at all and runs the loop to `MAX_TERMS`. The
/// count separates them and nothing else does.
///
/// # The count found a real defect, not just an unkillable mutant
///
/// `y0_series` ran **999 of its 1,000 iterations** for every argument below about `3e-162`.
/// Below there `x²/4` underflows to zero, so `t` and `sum` are both exactly zero and the
/// relative test `0 < 0·ε` is false — the break could never fire. The returned value was right
/// the whole time, which is why nothing else noticed; only the cost was wrong. It is one
/// iteration now, and `y0_series`' exit carries an explicit zero-term case.
///
/// Counts are pinned rather than bounded: a change is exactly what a regression in a stopping
/// rule or in `TERM_EPS` looks like.
#[test]
fn the_y_series_stop_where_their_tolerance_says() {
    use crate::kernels::bessel::ordinary_second::{MAX_TERMS, y0_series_terms, y1_series_terms};

    for (x, y0_want, y1_want) in [
        // The underflow case that used to spin to the cap.
        (1e-308_f64, 1, 3),
        (f64::MIN_POSITIVE, 1, 3),
        (0.01, 5, 5),
        (0.1, 7, 7),
        (0.5, 9, 10),
        (1.0, 11, 12),
        (2.0, 14, 14),
        (Y_SERIES_MAX, 18, 18),
    ] {
        let y0 = y0_series_terms(x);
        assert_eq!(y0.terms, y0_want, "Y₀ at x = {x:e}");
        assert!(y0.terms < MAX_TERMS, "Y₀ at x = {x:e} reached the cap");
        let y1 = y1_series_terms(x);
        assert_eq!(y1.terms, y1_want, "Y₁ at x = {x:e}");
        assert!(y1.terms < MAX_TERMS, "Y₁ at x = {x:e} reached the cap");
    }

    // Monotone in the argument over the band where the series actually sums something, which is
    // what makes the break a convergence test rather than a fixed schedule.
    assert!(y0_series_terms(Y_SERIES_MAX).terms > y0_series_terms(0.01).terms);
    assert!(y1_series_terms(Y_SERIES_MAX).terms > y1_series_terms(0.01).terms);

    // `y1_series`' break carries `&& k > 1`, so it always enters at least two terms however
    // negligible the first is.
    assert!(
        y1_series_terms(f64::MIN_POSITIVE).terms >= 2,
        "the k > 1 guard should force a second term"
    );

    // The counted variants return exactly what the plain ones do.
    for i in 0..=20 {
        let x = 0.2 + f64::from(i) * 0.19;
        assert_eq!(y0_series_terms(x).value, y0_series(x));
        assert_eq!(y1_series_terms(x).value, y1_series(x));
    }
}

/// Stopping early at a vanished term changes the **cost** and not the answer.
///
/// The zero-term exit added to `y0_series` skips iterations that were summing exact zeros, so
/// the value it returns is the same to the bit. Asserted against the pole's leading behaviour,
/// which is the only thing `Y₀` has down there: `Y₀(x) → (2/π)(ln(x/2) + γ)` as `x → 0`.
#[test]
fn the_vanished_term_exit_does_not_move_the_value() {
    use core::f64::consts::PI;

    for &x in &[1e-308_f64, f64::MIN_POSITIVE, 1e-200, 1e-170, 3e-162] {
        let got = y0_series(x);
        // Below the underflow the series contributes exactly nothing, so `Y₀` is its leading
        // term — and that is a closed form, independent of the summation entirely.
        let leading = 2.0 / PI * ((x / 2.0).ln() + crate::constants::f64::EULER_GAMMA);
        assert!(
            (got - leading).abs() <= 1e-14 * leading.abs(),
            "Y₀({x:e}) = {got:e} against its leading term {leading:e}"
        );
        assert!(got.is_finite() && got < 0.0, "Y₀({x:e}) = {got:e}");
    }
    // And it is still monotone increasing towards the origin's pole.
    assert!(y0_series(1e-300) < y0_series(1e-200));
    assert!(y0_series(1e-200) < y0_series(1e-100));
}

/// The Hankel expansion's **term count**, which is all its tolerance exit changes.
///
/// `hankel` itself cannot see it: the growth check stops the divergent expansion at the same
/// `P` and `Q` whatever the tolerance does, so mutation testing finds every spelling of
/// `magnitude < TERM_EPS` unkillable through the public value. `hankel_sums` returns the count,
/// which is why it is a separate function.
#[test]
fn the_hankel_expansion_truncates_sooner_as_the_argument_grows() {
    use crate::kernels::bessel::asymptotic::{MAX_TERMS as ASYMPTOTIC_MAX, hankel_sums};

    // Further out, fewer terms: `a_k/x^k` reaches the floor earlier, so the tolerance exit wins
    // and wins sooner. That monotone relation is what the second exit exists to produce.
    let near = hankel_sums(0.0, 18.0);
    let far = hankel_sums(0.0, 1.0e6);
    assert!(
        far.terms < near.terms,
        "x = 1e6 should truncate sooner than x = 18: {} vs {}",
        far.terms,
        near.terms
    );
    for &x in &[18.0_f64, 50.0, 1.0e3, 1.0e6] {
        let counted = hankel_sums(0.0, x);
        assert!(
            counted.terms < ASYMPTOTIC_MAX,
            "x = {x:e} took {} terms",
            counted.terms
        );
        // `P → 1` and `Q → 0` as the argument grows, which is the expansion's leading behaviour
        // and an independent check that the sums are the right way round.
        assert!(
            (counted.value.first_kind - 1.0).abs() < 1e-2,
            "P({x:e}) = {}",
            counted.value.first_kind
        );
        assert!(counted.value.second_kind.abs() < 1e-2);
    }
}

/// The march's **step count**, which is what makes its cap a backstop rather than a schedule.
///
/// `MARCH_MAX_STEPS` exists because three mutations of the march's advance made the suite *hang*
/// rather than fail — a timeout says nothing about which change caused it, and in a library a
/// non-terminating loop is worse than a wrong number. The cap converts those into bounded wrong
/// answers; this asserts that no real argument comes near it, which is the other half of the
/// bargain.
#[test]
fn the_march_reaches_every_target_well_inside_its_cap() {
    use crate::kernels::bessel::ordinary_second::{MARCH_MAX_STEPS, y_march_steps};

    let mut worst = 0;
    // The whole band the march serves: from the series cutoff to the Hankel handover.
    for i in 0..=280 {
        let target = Y_SERIES_MAX + f64::from(i) * 0.05;
        let marched = y_march_steps(target);
        assert!(
            marched.terms < MARCH_MAX_STEPS,
            "march to {target} took {} steps against a cap of {MARCH_MAX_STEPS}",
            marched.terms
        );
        assert!(marched.value.value.is_finite());
        worst = worst.max(marched.terms);
    }
    // Measured: **seven**. The constant's own estimate is `log(18/4)/log(1.25) ≈ 6.7`, so seven
    // is that rounded up — the final step is short, being clamped to `target - x`. Pinned
    // because a change here is what a regression in `MARCH_RATIO` looks like.
    assert_eq!(worst, 7, "worst-case march length moved to {worst}");
    // Monotone in the target: further to go, more steps.
    assert!(y_march_steps(18.0).terms > y_march_steps(5.0).terms);
}
