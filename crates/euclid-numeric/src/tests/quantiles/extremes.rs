//! Quantiles at extreme shape parameters, and against the shapes where they are elementary.
//!
//! [`super`] exercises both inverses over ordinary parameters. This file attacks the
//! small-shape regime, where the root is pushed to the bottom of the exponent range, and where
//! both inverses were badly broken:
//!
//! - `gamma_lr_inv(p, 1e-3)` returned `6.2e-64` for *every* `p` from `1e-12` to `0.5`. Its
//!   bracket expansion was capped at 200 octaves, so the search gave up at `2^-200` while the
//!   root — `0.5^1000 ≈ 5.2e-302` for the median — lay a hundred octaves further down.
//! - `beta_reg_inv(p, 1e-3, b)` returned `3.1e-61` for every `p`, by the same arithmetic: it
//!   bisects `[0, 1]`, and 200 halvings reach exactly `2^-200`.
//!
//! Both were *constant functions of their own argument* over the whole regime, which is the
//! shape of defect this file is built to catch: the first test asserts the quantile actually
//! varies with `p`, before asking anything about its value.
//!
//! # The closed-form shapes
//!
//! Four parameter choices collapse the incomplete functions to elementary ones:
//!
//! ```text
//! P(1, x)   = 1 − e^{−x}            I_x(a, 1) = x^a
//! P(½, x)   = erf(√x)               I_x(1, b) = 1 − (1 − x)^b
//! ```
//!
//! so the quantiles there have *exact* inverses — `−ln(1 − p)`, `erf_inv(p)²`, `p^{1/a}`,
//! `1 − (1 − p)^{1/b}`. These are the strongest references available for a root-find, because
//! they involve no reference table and no quadrature, and they tie the inverses to `ln`, `erf`
//! and `powf` rather than to the forward functions they invert.

use crate::Real;
use euclid_utils::{assert_close, assert_nondecreasing};

/// `true` when no representable neighbour of `x` reproduces `p` better than `x` itself does.
///
/// The honest success criterion for an inverse whose forward function is steep. At
/// `a = b = 0.01` and `p = 0.6` the root sits at `1 − 2e-10`, where the density is `2e7` and so
/// a single ulp of `x` moves the CDF by `2.2e-9`. No tolerance tighter than that is reachable
/// by *any* implementation; insisting on one would be a statement about `f64`'s spacing rather
/// than about the code. Asking instead for the best representable answer is scale-free, and it
/// is strictly stronger than a loosened tolerance wherever the problem is well conditioned.
fn is_best_representable(forward: impl Fn(f64) -> f64, x: f64, p: f64) -> bool {
    let here = (forward(x) - p).abs();
    let above = (forward(Real::next_after(x, f64::INFINITY)) - p).abs();
    let below = (forward(Real::next_after(x, f64::NEG_INFINITY)) - p).abs();
    here <= above && here <= below
}

#[test]
fn the_quantiles_are_not_constant_functions_of_their_argument() {
    // The invariant the small-shape bug violated, stated before any question of accuracy:
    // a quantile must move when its probability moves, and must move the right way.
    for &a in &[1e-3f64, 1e-2, 0.1, 0.5, 1.0, 10.0, 1e3] {
        let probabilities: Vec<f64> = (1..20).map(|i| f64::from(i) * 0.05).collect();
        let quantiles: Vec<f64> = probabilities
            .iter()
            .map(|&p| Real::gamma_lr_inv(p, a))
            .collect();
        assert_nondecreasing(
            quantiles.clone(),
            0.0,
            &format!("gamma_lr_inv must increase with p at a = {a}"),
        );
        // Distinct probabilities must give distinct quantiles — but only where the format can
        // tell them apart. Two filters are needed for that, and both are about `f64` rather
        // than about the function:
        //
        // - roots below `MIN_POSITIVE` are subnormal, where the spacing is *absolute*, so
        //   several consecutive `p` genuinely round to the same `x` (at `a = 1e-3` the root is
        //   `p^1000`, and everything below `p ≈ 0.48` collapses onto `5e-324` or zero);
        // - roots at zero have underflowed entirely.
        //
        // Above that floor the quantile must be strictly increasing, which is what the old
        // constant-valued inverse failed.
        let positive: Vec<f64> = quantiles
            .into_iter()
            .filter(|&x| x > f64::MIN_POSITIVE)
            .collect();
        for pair in positive.windows(2) {
            assert!(
                pair[1] > pair[0],
                "gamma_lr_inv at a = {a} repeats {:e} across different p",
                pair[0]
            );
        }
    }
    for &(a, b) in &[(1e-3f64, 1.0f64), (1e-2, 0.5), (0.1, 2.0), (2.0, 3.0)] {
        let quantiles: Vec<f64> = (1..20)
            .map(|i| Real::beta_reg_inv(f64::from(i) * 0.05, a, b))
            .collect();
        assert_nondecreasing(
            quantiles.clone(),
            0.0,
            &format!("beta_reg_inv must increase with p at ({a}, {b})"),
        );
        let positive: Vec<f64> = quantiles
            .into_iter()
            .filter(|&x| x > f64::MIN_POSITIVE && x < 1.0 - f64::EPSILON)
            .collect();
        for pair in positive.windows(2) {
            assert!(
                pair[1] > pair[0],
                "beta_reg_inv at ({a}, {b}) repeats {:e} across different p",
                pair[0]
            );
        }
    }
}

#[test]
fn the_beta_quantile_matches_its_exact_closed_forms() {
    // I_x(a, 1) = x^a, so the quantile is exactly p^{1/a}; and I_x(1, b) = 1 − (1 − x)^b, so
    // it is exactly 1 − (1 − p)^{1/b}.
    //
    // The tolerance has to carry the *conditioning of the reference*: raising p to the power
    // 1/a amplifies a relative error in p by 1/a, which is a factor of 1000 at a = 1e-3.
    for &a in &[1e-3f64, 1e-2, 0.1, 0.5, 1.0, 3.0, 50.0] {
        let slack = (20.0 * f64::EPSILON / a).max(1e-12);
        for i in 1..20 {
            let p = f64::from(i) * 0.05;
            let exact = p.powf(1.0 / a);
            if exact == 0.0 || !exact.is_finite() {
                continue; // the root is outside the format, not wrong
            }
            assert_close(
                Real::beta_reg_inv(p, a, 1.0) / exact,
                1.0,
                slack,
                &format!("beta_reg_inv({p}, {a}, 1) = p^(1/a)"),
            );
            // And the forward direction, which is exact: I_x(a, 1) = x^a.
            assert_close(
                Real::beta_reg(exact, a, 1.0),
                p,
                1e-13,
                &format!("beta_reg(x, {a}, 1) = x^a"),
            );
        }
    }
    for &b in &[1e-2f64, 0.5, 1.0, 3.0, 50.0] {
        for i in 1..20 {
            let p = f64::from(i) * 0.05;
            let exact = 1.0 - (1.0 - p).powf(1.0 / b);
            if exact <= 0.0 || exact >= 1.0 {
                continue;
            }
            assert_close(
                Real::beta_reg_inv(p, 1.0, b) / exact,
                1.0,
                (20.0 * f64::EPSILON / b).max(1e-11),
                &format!("beta_reg_inv({p}, 1, {b}) = 1 - (1-p)^(1/b)"),
            );
        }
    }
}

#[test]
fn the_gamma_quantile_matches_its_exact_closed_forms() {
    // P(1, x) = 1 − e^{−x}, so the quantile is −ln(1 − p); and P(½, x) = erf(√x), so it is
    // erf_inv(p)². Both tie the root-find to functions it shares no code with.
    for i in 1..200 {
        let p = f64::from(i) * 0.005;
        assert_close(
            Real::gamma_lr_inv(p, 1.0),
            -(1.0 - p).ln(),
            1e-12,
            &format!("gamma_lr_inv({p}, 1) = -ln(1-p)"),
        );
        let root = Real::erf_inv(p);
        assert_close(
            Real::gamma_lr_inv(p, 0.5),
            root * root,
            1e-10,
            &format!("gamma_lr_inv({p}, 1/2) = erf_inv(p)^2"),
        );
    }
}

#[test]
fn the_gamma_quantile_matches_its_small_shape_asymptote() {
    // For `a → 0` the gamma CDF is `P(a, x) ≈ x^a / Γ(1+a)` while `x` is small, so the
    // quantile is `(p·Γ(1+a))^{1/a}`. That reference is only valid where the root is small,
    // which for these shapes means the lower part of the probability range — and it is exactly
    // the region the bug occupied.
    for &a in &[1e-3f64, 1e-2, 0.05] {
        let slack = (40.0 * f64::EPSILON / a).max(1e-10);
        for &p in &[0.3f64, 0.5, 0.7, 0.8] {
            let asymptote = (p * Real::gamma(1.0 + a)).powf(1.0 / a);
            if asymptote == 0.0 || asymptote > 1e-3 {
                continue; // outside the format, or outside the asymptote's own validity
            }
            // The asymptote drops a term of relative size O(x), so its own accuracy is the
            // binding constraint once x is not tiny: at a = 0.01, p = 0.8 the root is 1.2e-10
            // and the reference is good to just that.
            assert_close(
                Real::gamma_lr_inv(p, a) / asymptote,
                1.0,
                slack.max(10.0 * asymptote),
                &format!("gamma_lr_inv({p}, {a}) small-shape asymptote"),
            );
        }
    }
}

#[test]
fn both_quantiles_round_trip_wherever_the_root_is_representable() {
    // The general statement, over shapes spanning nine orders of magnitude. Where the true
    // root falls outside the format the inverse saturates — to `0` for the gamma, to the last
    // `f64` below 1 for the beta — and saturation is the right answer rather than a failure,
    // so those cases are identified and skipped rather than loosened away.
    for &a in &[1e-3f64, 1e-2, 0.1, 0.5, 1.0, 10.0, 1e3, 1e5] {
        for i in 1..40 {
            let p = f64::from(i) * 0.025;
            let x = Real::gamma_lr_inv(p, a);
            assert!(x >= 0.0 && !x.is_nan(), "gamma_lr_inv({p}, {a}) = {x}");
            if x == 0.0 || !x.is_finite() {
                continue;
            }
            // Either the round trip closes to 1e-9, or `x` is already the best `f64` there
            // is — at a = 1e5 the CDF's slope turns `x`'s last bits into ~1e-11 of
            // probability, which is the format speaking rather than the code.
            let err = (Real::gamma_lr(x, a) - p).abs();
            assert!(
                err < 1e-9 || is_best_representable(|v| Real::gamma_lr(v, a), x, p),
                "gamma round trip at p = {p}, a = {a}: off by {err:e} and not optimal"
            );
        }
    }
    for &a in &[1e-2f64, 0.1, 0.5, 1.0, 10.0, 1e3] {
        for &b in &[1e-2f64, 0.1, 0.5, 1.0, 10.0, 1e3] {
            for i in 1..20 {
                let p = f64::from(i) * 0.05;
                let x = Real::beta_reg_inv(p, a, b);
                assert!((0.0..=1.0).contains(&x), "beta_reg_inv({p},{a},{b}) = {x}");
                // Saturation at either end means no representable `x` reproduces `p`; the
                // distribution is simply too steep there.
                if x == 0.0 || x >= 1.0 - f64::EPSILON {
                    continue;
                }
                let err = (Real::beta_reg(x, a, b) - p).abs();
                assert!(
                    err < 1e-10 || is_best_representable(|v| Real::beta_reg(v, a, b), x, p),
                    "beta round trip at p = {p}, a = {a}, b = {b}: \
                     off by {err:e} and not optimal"
                );
            }
        }
    }
}
