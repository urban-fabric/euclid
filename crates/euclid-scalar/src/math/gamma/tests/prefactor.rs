//! The log prefactor `a·ln x − x − lnΓ(a)` and the three arrangements that evaluate it.
//!
//! Split from the parent so each file stays inside the crate's 400-line budget. The sub-topic is
//! a real one: everything here is about [`ScaledGammaForm`] — that the two rearrangements are the
//! same quantity as the sum they replace, and that each argument reaches the arm its own note
//! claims. Nothing here evaluates `Γ` or `lnΓ` as such.

use super::*;

#[test]
fn log_scale_matches_the_prefactor_at_integer_shape() {
    // Γ(2) = 1 and 1.5 * 4² = 24.
    let actual = log_scaled_gamma(2.0_f64, 4.0, 1.5_f64.ln());
    assert!((actual - (24.0_f64.ln() - 4.0)).abs() <= 2e-15);
}

/// The rearranged prefactor is the same quantity as the direct one, where both are sound.
///
/// [`log_scaled_gamma`] replaces `a·ln x − x − lnΓ(a)` with `a·[ln(1+u) − u] + ½ln(a/2π) − μ(a)`,
/// which is an identity only if Stirling's series is substituted correctly. An algebra slip
/// there does not produce nonsense — it produces a smooth, plausible, wrong function — so it
/// needs pinning against the form it replaced.
///
/// Asserted just above [`STIRLING_MIN`] and at small `|u|`, which is the one corner where both
/// forms are trustworthy: `lnΓ(a)` is still only `39`, so the direct form has lost little, while
/// the rearranged form is in its own best region. Further up the direct form is the wrong one
/// and agreement would prove nothing — the same reason `erf_inv`'s two numerators are compared
/// at the threshold rather than in the tail.
#[test]
fn the_rearranged_prefactor_agrees_with_the_direct_one_where_both_are_sound() {
    for a in [20.0_f64, 21.0, 24.0] {
        for u in [-0.05_f64, -0.01, 0.0, 0.01, 0.05] {
            let x = a * (1.0 + u);
            let rearranged = log_scaled_gamma(a, x, 0.0);
            let direct = -x + a * x.ln() - <f64 as crate::Real>::ln_gamma(a);
            assert!(
                (rearranged - direct).abs() <= 2e-13,
                "at a={a}, x={x} the rearranged prefactor is {rearranged} \
                 and the direct one {direct}"
            );
        }
    }
    // Below `STIRLING_MIN` the direct form is what runs, so the two are the same expression and
    // must agree exactly — this is what pins that branch to the right side of its own test.
    for (a, x) in [(10.0_f64, 11.0_f64), (19.0, 2.0), (3.0, 40.0)] {
        assert_eq!(
            log_scaled_gamma(a, x, 0.25),
            0.25 - x + a * x.ln() - <f64 as crate::Real>::ln_gamma(a),
            "at a={a}, x={x}"
        );
    }
}

/// The `t = x/a` rearrangement is the same quantity as the direct form, where both are sound.
///
/// [`ScaledGammaForm::Ratio`] is reached outside the `ln1p` band, so the comparison in
/// [`the_rearranged_prefactor_agrees_with_the_direct_one_where_both_are_sound`] never touches
/// it — and it is a *different* algebraic rearrangement, `a·[ln t + 1 − t]` rather than
/// `a·[ln(1+u) − u]`, so it needs its own pinning. A slip in it produces a smooth and plausible
/// wrong function, exactly as a slip in the other one does.
///
/// Sound for both means: `|u|` just outside the band, where `Ratio` runs, at a shape low enough
/// that `lnΓ(a)` is only `39`–`145` and so the direct form has lost little. The arm is asserted
/// alongside the value, because if the router sent these to `Direct` the comparison would be an
/// expression against itself and would pass having checked nothing.
#[test]
fn the_ratio_rearrangement_agrees_with_the_direct_form_where_both_are_sound() {
    for a in [20.0_f64, 24.0, 50.0] {
        for factor in [0.4_f64, 0.45, 1.6, 1.8] {
            let x = a * factor;
            assert_eq!(
                ScaledGammaForm::for_arguments(a, x),
                ScaledGammaForm::Ratio,
                "at a={a}, x={x} the comparison below would be vacuous"
            );
            let rearranged = log_scaled_gamma(a, x, 0.0);
            let direct = -x + a * x.ln() - <f64 as crate::Real>::ln_gamma(a);
            assert!(
                (rearranged - direct).abs() <= 2e-13,
                "at a={a}, x={x} the ratio form is {rearranged} and the direct one {direct}"
            );
        }
    }
}

/// Each form is reached where its own note says it is.
///
/// The outside-the-band choice is a *measurement* — whichever arrangement builds the exponent
/// from smaller terms — so unlike a magnitude ladder it cannot be read off the arguments. These
/// are the four points the enum's documentation argues from, which is what makes this a check on
/// the prose as much as on the code.
#[test]
fn the_prefactor_form_is_chosen_by_which_arrangement_has_less_to_lose() {
    use ScaledGammaForm::{Centred, Direct, Ratio};
    for (a, x, want) in [
        // Below `STIRLING_MIN` there is nothing large to cancel.
        (10.0_f64, 11.0_f64, Direct),
        // `|u| = 0.4` is inside the band, where `u` is exact and the series applies.
        (1000.0, 600.0, Centred),
        // The direct form would build a `−605` exponent out of terms of `3074`.
        (445.0, 1000.0, Ratio),
        // Outside the band but small `x`: `a·ln x` is `276` against the ratio form's `336`.
        (20.0, 1e-6, Direct),
    ] {
        assert_eq!(
            ScaledGammaForm::for_arguments(a, x),
            want,
            "at a={a}, x={x}"
        );
    }
}

/// The `Ratio` arm beats the direct sum across the band it was measured on.
///
/// Reference-free and the counterpart to the agreement tests above: those pin the *algebra*,
/// this pins the *choice*. Where the exponent is `O(1)`-ish but `lnΓ(a)` is in the thousands,
/// the direct sum's largest term exceeds the answer by orders of magnitude while the
/// rearrangement's does not — so the predictor must prefer the rearrangement, and by a wide
/// margin rather than a marginal one.
///
/// These are three of the largest improvements the differential sweep measured — `9703 → 503`,
/// `32941 → 472` and `25849 → 522` ulp. The sweep's single worst point, `a = 2215.1`,
/// `x = 1107.5` at `29603 → 322`, is deliberately *not* used here: its `u` is `−0.50002`, a
/// fifth of a thousandth outside the band, so it would read as an assertion about `Ratio` while
/// resting on which side of `½` a rounding fell. The assertion is on the predictor rather than
/// on ulp because the reference it would need is a 400-bit incomplete gamma, which this crate
/// does not carry.
#[test]
fn the_ratio_arm_is_preferred_by_a_wide_margin_where_the_sweep_found_its_worst_points() {
    for (a, x) in [(1419.9_f64, 480.32_f64), (1807.8, 3175.2), (2320.2, 4288.8)] {
        let direct = largest_term(&direct_terms(a, x));
        let ratio = largest_term(&ratio_terms(a, x));
        assert_eq!(ScaledGammaForm::for_arguments(a, x), ScaledGammaForm::Ratio);
        assert!(
            ratio * 3.0 < direct,
            "at a={a}, x={x} the ratio form's largest term is {ratio} against the direct \
             form's {direct}, which is not the wide margin the sweep measured"
        );
    }
}
