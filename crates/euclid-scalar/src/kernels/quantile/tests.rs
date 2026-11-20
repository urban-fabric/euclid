//! Unit tests for the shared solver machinery.
//!
//! A child module so implementation plus tests stay inside the crate's 400-line budget, and
//! because what needs testing here is not the root. The solver is a **safeguarded** Newton: it
//! falls back to bisection whenever the density misbehaves, so a wrong derivative costs
//! iterations rather than accuracy. `residual_tolerance` and `best_of` are the same shape of
//! thing — each is a pure function, so each is asserted as a value.
//!
//! The two quantiles keep their own tests beside their modules, in [`super::gamma`] and
//! [`super::beta`].

use super::*;

#[test]
fn quantile_endpoints_and_domain_errors() {
    assert_eq!(gamma_lr_inv(0.0, 2.0), 0.0);
    assert!(gamma_lr_inv(1.0, 2.0).is_infinite());
    assert!(gamma_lr_inv(1.5, 2.0).is_nan());
    assert!(gamma_lr_inv(0.5, -1.0).is_nan());
    assert_eq!(beta_reg_inv(0.0, 2.0, 3.0), 0.0);
    assert_eq!(beta_reg_inv(1.0, 2.0, 3.0), 1.0);
    assert!(beta_reg_inv(-0.1, 2.0, 3.0).is_nan());
    assert!(beta_reg_inv(0.5, 0.0, 3.0).is_nan());
}

/// The stopping rule, which scales with the shape and is capped.
///
/// Every operator in it survived: the solver converges to the same root at any tolerance this
/// returns, so only the returned number itself distinguishes them.
#[test]
fn the_residual_tolerance_scales_with_the_shape_and_is_capped() {
    const EPS: f64 = f64::EPSILON;
    // `ε(1 + 2a)`, so the floor at a zero shape is one ulp of `p` and it grows linearly.
    assert_eq!(residual_tolerance(0.0), EPS);
    assert_eq!(residual_tolerance(0.5), 2.0 * EPS);
    assert_eq!(residual_tolerance(2.0), 5.0 * EPS);
    assert_eq!(residual_tolerance(100.0), 201.0 * EPS);
    assert_eq!(residual_tolerance(1000.0), 2001.0 * EPS);
    // A negative or `NaN` shape clamps to the floor rather than inverting the tolerance.
    assert_eq!(residual_tolerance(-5.0), EPS);
    assert_eq!(residual_tolerance(f64::NAN), EPS);
    // The cap binds above `a = 2097151.5`, which is where an uncapped shape term would start
    // asking for less accuracy than the format can deliver — the bug the cap exists for.
    assert!(residual_tolerance(2.0e6) < RESIDUAL_TOLERANCE_CAP);
    assert_eq!(residual_tolerance(2.1e6), RESIDUAL_TOLERANCE_CAP);
    assert_eq!(residual_tolerance(1e7), RESIDUAL_TOLERANCE_CAP);
    // And an infinite shape takes the cap through the non-finite arm rather than returning one.
    assert!(!(EPS * (1.0 + 2.0 * f64::INFINITY)).is_finite());
    assert_eq!(residual_tolerance(f64::INFINITY), RESIDUAL_TOLERANCE_CAP);
    // Monotone in the shape, which is the property the measurement table asserts.
    let ladder: Vec<f64> = [0.0, 0.5, 2.0, 1e2, 1e3, 1e6]
        .iter()
        .map(|&a| residual_tolerance(a))
        .collect();
    assert!(ladder.windows(2).all(|w| w[0] < w[1]), "{ladder:?}");
}

/// Which of the three candidates the solver hands back, and the sign of the residual it ranks by.
#[test]
fn best_of_ranks_the_three_candidates_by_absolute_residual() {
    let identity = |x: f64| x;
    // The midpoint wins when it is the closest, which is the common case at convergence.
    assert_eq!(best_of(&identity, 0.5, 0.4, 0.6, None, None), 0.5);
    // An endpoint wins when it is closer than the midpoint. This is the case that pins the
    // *sign*: ranking by `cdf + p` instead of `cdf − p` would leave the midpoint in front.
    assert_eq!(best_of(&identity, 0.3, 0.2, 1.0, None, None), 0.2);
    assert_eq!(best_of(&identity, 0.9, 0.0, 0.8, None, None), 0.8);
    // A remembered residual is used in place of a fresh evaluation, and must rank identically.
    assert_eq!(
        best_of(&identity, 0.3, 0.2, 1.0, Some(-0.1), Some(0.7)),
        0.2
    );
    // A `None` on the side that would win still evaluates, so the answer does not depend on
    // whether the loop happened to move that side.
    assert_eq!(best_of(&identity, 0.3, 0.2, 1.0, None, Some(0.7)), 0.2);
    // Ties keep the midpoint: the comparison is strict, so neither endpoint displaces an equal
    // candidate and the returned root is the interior one.
    assert_eq!(best_of(&identity, 0.5, 0.0, 1.0, None, None), 0.5);
}
