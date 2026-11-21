//! Numerical checks for the relation module.

use super::*;
use crate::relation::forward_stable_sum;

/// An accurate kernel is forward stable; a deliberately perturbed one is not.
#[test]
fn forward_stable_accepts_an_accurate_kernel_and_rejects_a_sloppy_one() {
    // `exp` against itself is exact, so it is trivially within its own bound.
    forward_stable("exp", 1.5, f64::exp, f64::exp, 8.0).expect("exp agrees with itself");

    // A kernel off by a part in ten thousand is nowhere near forward stable at a benign point.
    forward_stable(
        "sloppy-exp",
        1.5,
        |x| f64::exp(x) * (1.0 + 1e-4),
        f64::exp,
        8.0,
    )
    .expect_err("a relative error of 1e-4 is not forward stable");
}

/// The failure message has to distinguish a bad implementation from a hard problem, since that
/// is the whole reason to use a conditioning-based bound rather than a literal.
#[test]
fn forward_stable_names_the_conditioning_in_its_message() {
    let failure = forward_stable("sloppy", 1.5, |x| f64::exp(x) * (1.0 + 1e-4), f64::exp, 8.0)
        .expect_err("not stable");
    assert!(failure.message.contains("condition"), "{}", failure.message);
    assert!(
        failure.message.contains("unstable kernel"),
        "at a benign point the message should blame the implementation: {}",
        failure.message
    );
}

/// Where the bound is undefined the relation abstains rather than failing. `x = 0` has no
/// relative condition number, and reporting a failure there would be reporting one for a
/// question that was never asked.
#[test]
fn forward_stable_abstains_where_the_bound_is_undefined() {
    forward_stable("at-zero", 0.0, f64::exp, f64::exp, 8.0).expect("x = 0 abstains");
    forward_stable("non-finite", f64::INFINITY, f64::exp, f64::exp, 8.0)
        .expect("a non-finite input abstains");
}

/// The margin is the score a budget baseline records, so it has to be a ratio around 1 rather
/// than an absolute error: 1.0 means "exactly at the admissible bound".
#[test]
fn the_margin_is_a_ratio_against_the_admissible_bound() {
    let exact = forward_stable_margin(1.5, f64::exp, f64::exp, 8.0).expect("defined at 1.5");
    assert!(exact < 1.0, "an exact kernel is inside its bound: {exact}");

    let sloppy = forward_stable_margin(1.5, |x| f64::exp(x) * (1.0 + 1e-4), f64::exp, 8.0)
        .expect("defined at 1.5");
    assert!(sloppy > 1.0, "a sloppy kernel is over its bound: {sloppy}");
    // And the ratio scales with the error, which is what makes it climbable.
    let sloppier = forward_stable_margin(1.5, |x| f64::exp(x) * (1.0 + 1e-3), f64::exp, 8.0)
        .expect("defined at 1.5");
    assert!(sloppier > sloppy, "{sloppier} should exceed {sloppy}");
}

#[test]
fn the_margin_is_undefined_where_the_reference_vanishes() {
    assert_eq!(forward_stable_margin(1.0, |_| 0.0, |_| 0.0, 8.0), None);
    assert_eq!(forward_stable_margin(0.0, f64::exp, f64::exp, 8.0), None);
}

/// A correct narrow kernel agrees with the wide one to narrow precision; one using the wrong
/// coefficients does not.
#[test]
fn precision_agrees_accepts_a_matching_narrow_kernel() {
    precision_agrees("exp", 1.25, f64::exp, f32::exp, 2).expect("f32 exp matches f64 exp");
    precision_agrees("sqrt", 2.0, f64::sqrt, f32::sqrt, 2).expect("f32 sqrt matches f64 sqrt");
}

#[test]
fn precision_agrees_rejects_a_narrow_kernel_that_drifted() {
    precision_agrees("drifted", 1.25, f64::exp, |x| f32::exp(x) * 1.001, 2)
        .expect_err("a part in a thousand is far more than 2 f32 ulp");
}

/// The comparison is in `f32` ulps against the *rounded* wide result. Comparing in `f64` terms
/// would reject every correct narrow kernel, since narrowing alone costs about 2^-24.
#[test]
fn precision_agrees_does_not_charge_the_narrow_kernel_for_narrowing() {
    // `ln` at an awkward argument: the f32 answer cannot be closer to the f64 one than one f32
    // ulp, and the relation must accept that rather than treating it as disagreement.
    for x in [0.1_f64, 0.7, 1.9, 12.75, 1e6] {
        precision_agrees("ln", x, f64::ln, f32::ln, 2)
            .unwrap_or_else(|e| panic!("f32 ln should agree at {x}: {}", e.message));
    }
}

/// Both sides agreeing on a non-finite answer is agreement, not a disagreement that happens to
/// involve infinity.
#[test]
fn precision_agrees_treats_matching_non_finite_results_as_agreement() {
    precision_agrees("overflow", 1e30, |x| x * x, |x| x * x, 2)
        .expect("both overflow to +inf, which is agreement");
    precision_agrees("nan", -1.0, f64::sqrt, f32::sqrt, 2).expect("both are NaN");
}

/// The margin is the relative error **divided by** the admissible bound, and the arithmetic has
/// to be that and not merely monotone in it.
///
/// [`the_margin_is_a_ratio_against_the_admissible_bound`] checks which side of `1.0` the margin
/// falls on and that it grows with the error. Mutation testing showed that is not enough: turning
/// the division in `relative` into a multiplication or a remainder leaves the margin on the same
/// side of one and still growing, because `exp(1.5)` is a number near four and scaling the error
/// by it changes nothing qualitative. Both mutants survived.
///
/// So this computes the bound independently and asserts the ratio exactly. The margin is the
/// score a [`budget`](crate::budget) baseline records, so a constant factor in it is a silently
/// wrong number in a committed file.
#[test]
fn the_margin_is_exactly_the_relative_error_over_the_bound() {
    let x = 1.5_f64;
    let error = 1e-4;
    let sloppy = move |v: f64| f64::exp(v) * (1.0 + error);

    let bound = crate::numeric::forward_error_bound(sloppy, x, 8.0).expect("bound is defined");
    let margin = forward_stable_margin(x, sloppy, f64::exp, 8.0).expect("margin is defined");

    // `(exp(x) * (1 + error) - exp(x)) / exp(x)` is exactly `error`.
    let expected = error / bound;
    assert!(
        (margin - expected).abs() <= expected * 1e-9,
        "margin was {margin:e}, expected {expected:e} (relative error {error:e} over bound \
         {bound:e})"
    );
}

/// A reference that vanishes or runs away has no relative error to measure, even where the bound
/// itself is perfectly well defined.
///
/// [`the_margin_is_undefined_where_the_reference_vanishes`] looks like it covers this and does
/// not: it uses a *constant* kernel, whose condition number is zero, so the `None` it observes
/// comes from the bound being zero rather than from the guard on the reference. Mutation testing
/// weakened that guard from `||` to `&&` — making it unsatisfiable, since zero is finite — and
/// nothing failed.
///
/// A linear kernel has a condition number of one, so the bound is positive and the guard is the
/// only thing standing between a zero reference and a division by it.
#[test]
fn the_margin_is_undefined_against_a_zero_or_runaway_reference() {
    let identity = |v: f64| v;
    assert!(
        crate::numeric::forward_error_bound(identity, 1.0, 8.0).is_some_and(|b| b > 0.0),
        "precondition: the bound must be positive, or this tests the wrong guard"
    );

    assert_eq!(
        forward_stable_margin(1.0, identity, |_| 0.0, 8.0),
        None,
        "a zero reference has no relative error"
    );
    assert_eq!(
        forward_stable_margin(1.0, identity, |_| f64::INFINITY, 8.0),
        None,
        "a non-finite reference has no relative error"
    );
}

/// The failure report must state the error it actually measured.
///
/// [`forward_stable_names_the_conditioning_in_its_message`] checks the prose and not the numbers,
/// and the numbers are what a reader acts on. Mutation testing found four ways to corrupt the
/// relative error on the report while leaving the verdict correct — the failure is decided from
/// the margin, so a wrong number here is a wrong number in a message nobody can cross-check.
///
/// The inputs are chosen so the right answer is exactly `1e0` while every mutation lands
/// somewhere visibly different: a remainder gives `0e0`, an addition `3e0`, a multiplication
/// `9e0`, a division `6.66e-1`.
///
/// `x = 3` rather than `x = 1` for exactly that reason. At `x = 1` the reference is `1`, and
/// multiplying by it is the same as dividing by it — so the multiplication mutant produced the
/// right answer by coincidence and survived the first version of this test.
#[test]
fn the_failure_report_states_the_relative_error_it_measured() {
    let failure = forward_stable("doubling", 3.0, |v: f64| v * 2.0, |v| v, 8.0)
        .expect_err("a kernel returning twice the answer is not forward stable");

    assert!(
        failure.message.contains("relative:   1e0"),
        "a kernel off by a factor of two has a relative error of exactly 1; the report said:\n{}",
        failure.message
    );
}

/// A narrow kernel must not be allowed to invent a finite answer where the wide one has none.
///
/// Mutation testing relaxed `rounded.is_nan() && computed.is_nan()` to `||`, which accepts
/// exactly that: the wide reference returns NaN, the narrow kernel returns a number, and the
/// relation calls it agreement. That is the most dangerous direction for this check to fail,
/// because a narrow kernel returning plausible garbage outside its domain is a real bug class.
#[test]
fn precision_agrees_refuses_a_narrow_kernel_that_invents_a_finite_answer() {
    precision_agrees("invented", 1.0, |_| f64::NAN, |_| 1.0_f32, 2)
        .expect_err("a finite narrow result cannot agree with a NaN wide one");
    precision_agrees("invented", 1.0, |_| 1.0_f64, |_| f32::NAN, 2)
        .expect_err("a NaN narrow result cannot agree with a finite wide one");

    // Both absent in the same way is agreement, which is the case the guard exists to allow.
    precision_agrees("both nan", 1.0, |_| f64::NAN, |_| f32::NAN, 2).expect("two NaNs agree");
    precision_agrees(
        "both infinite",
        1.0,
        |_| f64::INFINITY,
        |_| f32::INFINITY,
        2,
    )
    .expect("two infinities of the same sign agree");
}

/// A degenerate safety factor abstains rather than reporting a NaN margin.
///
/// `forward_error_bound` floors itself at `safety * UNIT_ROUNDOFF`, so the bound is positive for
/// every sane call and the `bound > 0.0` guard looks like dead weight — mutation testing relaxed
/// it to `>=` and nothing failed. It is not dead: `safety = 0` drives the bound to zero, and
/// without the guard the margin becomes `0.0 / 0.0`.
///
/// That matters more than it looks. The margin is the score a [`budget`](crate::budget) records,
/// and the budget machinery silently *ignores* non-finite scores — so a NaN here would not fail
/// loudly, it would quietly stop recording a baseline and leave the property protected by
/// nothing at all. Abstaining is the honest answer.
#[test]
fn a_degenerate_safety_factor_abstains_rather_than_scoring_nan() {
    assert_eq!(
        forward_stable_margin(1.5, f64::exp, f64::exp, 0.0),
        None,
        "a zero safety factor leaves no admissible bound to measure against"
    );
    // And the relation built on it abstains too, rather than reporting a failure it cannot
    // describe.
    forward_stable("zero-safety", 1.5, f64::exp, f64::exp, 0.0)
        .expect("a zero safety factor must abstain, not fail");
}

/// [`forward_stable_sum`]'s bound is `safety·(κ + Σ|tᵢ|/|Σtᵢ|)·u`, each piece pinned.
///
/// Mutation testing asked for this: eleven mutants inside the function survived, every one of
/// them in the bound's own arithmetic or its guard. A relation that decides whether other
/// properties pass has to be tested at least as carefully as they are, and nothing here was.
#[test]
fn forward_stable_sum_charges_for_the_reference_cancellation() {
    // A reference assembled with no cancellation: the terms sum to their own magnitude, so the
    // ratio is 1 and the bound is `forward_stable`'s. An `exp` that agrees passes.
    forward_stable_sum("exact", 1.5, f64::exp, &[f64::exp(1.5)], 8.0)
        .expect("a single-term reference cannot cancel");

    // The cancellation term is what admits this. `1e5 - 99999.0` leaves `1.0` from terms of
    // `1e5`, a ratio of about `2e5`, so a kernel off by `1e-12` relative is inside the bound —
    // and would be far outside `forward_stable`'s, since `κ(|_| 1.0)` is zero.
    let terms = [1e5, -99_999.0];
    forward_stable_sum("cancelling", 1.0, |_| 1.0 + 1e-12, &terms, 1.0)
        .expect("the reference's own cancellation is charged for");

    // But it is not a blank cheque: ten thousand times further out still fails, and the message
    // says which term is binding rather than blaming the kernel.
    let failure = forward_stable_sum("cancelling", 1.0, |_| 1.0 + 1e-8, &terms, 1.0)
        .expect_err("an error far past the cancellation is still the kernel's");
    assert!(
        failure.message.contains("cancellation"),
        "the bound should report the cancellation term: {}",
        failure.message
    );
    assert!(
        failure.message.contains("orientation"),
        "a cancellation-dominated failure should suggest reorienting: {}",
        failure.message
    );

    // Guards abstain rather than fail: a reference summing to zero has no relative error, and a
    // non-finite computed value is not a disagreement to measure.
    forward_stable_sum("zero sum", 1.0, |_| 1.0, &[1.0, -1.0], 1.0).expect("abstains at zero");
    forward_stable_sum("nan", 1.0, |_| f64::NAN, &[1.0], 1.0).expect("abstains on non-finite");
    forward_stable_sum("inf term", 1.0, |_| 1.0, &[f64::INFINITY], 1.0)
        .expect("abstains on a non-finite reference");
}
