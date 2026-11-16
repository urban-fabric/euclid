//! Each relation must accept a law that holds and reject one that does not.
//!
//! Both halves matter equally. A relation that returned `Ok` unconditionally would pass every
//! positive test in this file, so every relation below is also handed a function that violates
//! it, and required to complain.

use super::{
    Absolute, Compare, Exact, Relative, Ulps, agree, associative, bounded, commutative,
    distributes, even, f32_ulps_between, fixed_point, forward_stable, forward_stable_margin,
    idempotent, involution, monotone, odd, order_independent, precision_agrees, preserves,
    round_trip,
};

// ===== The comparators ====================================================

#[test]
fn exact_is_equality_and_describes_itself() {
    assert!(Exact.same(&3, &3));
    assert!(!Exact.same(&3, &4));
    assert!(Compare::<i32>::describe(&Exact).contains("exactly"));
}

/// The comparator that exists because `==` is wrong for a computed float: a value one ulp away
/// must be accepted at `Ulps(1)` and refused at `Ulps(0)`.
#[test]
fn ulps_accepts_a_neighbour_and_refuses_an_identical_demand() {
    let one = 1.0_f64;
    let next = f64::from_bits(one.to_bits() + 1);
    assert!(Ulps(1).same(&one, &next));
    assert!(!Ulps(0).same(&one, &next));
}

#[test]
fn relative_and_absolute_differ_on_scale() {
    // Two values 10% apart at a tiny scale: absolute accepts, relative does not.
    assert!(Absolute(1.0).same(&1e-9, &1.1e-9));
    assert!(!Relative(0.01).same(&1e-9, &1.1e-9));
}

#[test]
fn every_comparator_refuses_nan() {
    assert!(!Ulps(u64::MAX).same(&f64::NAN, &f64::NAN));
    assert!(!Relative(1.0).same(&f64::NAN, &f64::NAN));
    assert!(!Absolute(f64::INFINITY).same(&f64::NAN, &f64::NAN));
}

// ===== The relations ======================================================

#[test]
fn involution_accepts_negation_and_rejects_increment() {
    assert!(involution("negate", 7i64, |v| -v, &Exact).is_ok());
    assert!(involution("increment", 7i64, |v| v + 1, &Exact).is_err());
}

/// The case a one-sided test would miss: the identity function *is* an involution, so a
/// relation that only checked `f(x) != x` would be wrong to reject it.
#[test]
fn involution_accepts_the_identity() {
    assert!(involution("identity", 7i64, |v| v, &Exact).is_ok());
}

#[test]
fn involution_works_over_floats_with_a_tolerance() {
    // Two divisions do not round-trip exactly, which is why the tolerance is the point.
    assert!(involution("reciprocal", 3.0_f64, |v| 1.0 / v, &Ulps(2)).is_ok());
    assert!(involution("halve", 3.0_f64, |v| v / 2.0, &Ulps(2)).is_err());
}

#[test]
fn idempotent_accepts_sorting_and_rejects_rotation() {
    let sort = |mut xs: Vec<i32>| {
        xs.sort_unstable();
        xs
    };
    assert!(idempotent("sort", vec![3, 1, 2], sort, &Exact).is_ok());

    let rotate = |mut xs: Vec<i32>| {
        xs.rotate_left(1);
        xs
    };
    assert!(idempotent("rotate", vec![1, 2, 3], rotate, &Exact).is_err());
}

#[test]
fn round_trip_accepts_a_lossless_pair_and_rejects_a_lossy_one() {
    assert!(
        round_trip(
            "to string",
            42i64,
            |v| v.to_string(),
            |s: String| s.parse().unwrap_or(0),
            &Exact
        )
        .is_ok()
    );
    // Truncating to i32 loses the high bits, so the trip does not return.
    assert!(
        round_trip(
            "narrow",
            i64::from(i32::MAX) + 1,
            |v| v as i32,
            i64::from,
            &Exact
        )
        .is_err()
    );
}

#[test]
fn commutative_accepts_addition_and_rejects_subtraction() {
    assert!(commutative("add", 3i64, 5, |a, b| a + b, &Exact).is_ok());
    assert!(commutative("subtract", 3i64, 5, |a, b| a - b, &Exact).is_err());
}

#[test]
fn associative_accepts_addition_and_rejects_subtraction() {
    assert!(associative("add", 1i64, 2, 3, |a, b| a + b, &Exact).is_ok());
    assert!(associative("subtract", 1i64, 2, 3, |a, b| a - b, &Exact).is_err());
}

/// Float addition is famously *not* associative, and the relation must say so rather than
/// quietly accepting it. This is the measurement that turns a law check into a conditioning
/// observation.
#[test]
fn associative_catches_float_addition_at_a_strict_tolerance() {
    let add = |a: f64, b: f64| a + b;
    // A large value and two small ones: the bracketing decides whether the small ones survive.
    assert!(
        associative("float add", 1e16, 1.0, 1.0, add, &Ulps(0)).is_err(),
        "float addition is not associative and the relation must detect it"
    );
    // The same operands are associative for well-scaled input.
    assert!(associative("float add", 1.0, 2.0, 4.0, add, &Ulps(0)).is_ok());
}

#[test]
fn distributes_accepts_arithmetic_and_rejects_a_swap() {
    assert!(
        distributes(
            "times over plus",
            2i64,
            3,
            4,
            |a, b| a * b,
            |a, b| a + b,
            &Exact
        )
        .is_ok()
    );
    // Addition does not distribute over multiplication.
    assert!(
        distributes(
            "plus over times",
            2i64,
            3,
            4,
            |a, b| a + b,
            |a, b| a * b,
            &Exact
        )
        .is_err()
    );
}

#[test]
fn agree_accepts_two_spellings_and_rejects_a_divergence() {
    assert!(agree("double", 5i64, |v| v * 2, |v| v + v, &Exact).is_ok());
    assert!(agree("double", 5i64, |v| v * 2, |v| v + 1, &Exact).is_err());
}

#[test]
fn monotone_accepts_an_increasing_map_and_rejects_negation() {
    assert!(monotone("double", 2i64, 7, |v| v * 2).is_ok());
    assert!(
        monotone("double", 7i64, 2, |v| v * 2).is_ok(),
        "order of arguments must not matter"
    );
    assert!(monotone("negate", 2i64, 7, |v| -v).is_err());
}

#[test]
fn monotone_accepts_a_constant_since_it_is_non_decreasing() {
    assert!(monotone("constant", 2i64, 7, |_| 0i64).is_ok());
}

#[test]
fn odd_and_even_accept_their_own_functions_and_reject_the_other() {
    assert!(odd("sin", 0.7, f64::sin, &Ulps(1)).is_ok());
    assert!(even("cos", 0.7, f64::cos, &Ulps(1)).is_ok());

    assert!(odd("cos", 0.7, f64::cos, &Ulps(1)).is_err());
    assert!(even("sin", 0.7, f64::sin, &Ulps(1)).is_err());
}

#[test]
fn bounded_accepts_an_in_range_result_and_rejects_an_escape() {
    assert!(bounded("clamp", 5.0_f64, |v| v.clamp(0.0, 1.0), 0.0, 1.0).is_ok());
    assert!(bounded("identity", 5.0_f64, |v| v, 0.0, 1.0).is_err());
}

#[test]
fn preserves_accepts_a_permutation_and_rejects_a_truncation() {
    let reverse = |mut xs: Vec<i32>| {
        xs.reverse();
        xs
    };
    assert!(preserves("reverse", vec![1, 2, 3], reverse, |xs| xs.len()).is_ok());

    let drop_last = |mut xs: Vec<i32>| {
        xs.pop();
        xs
    };
    assert!(preserves("drop", vec![1, 2, 3], drop_last, |xs| xs.len()).is_err());
}

/// `preserves` is the relation for "changes the value but not this property of it", so it must
/// accept a function that genuinely changes the value — a version that secretly required
/// equality would be useless.
#[test]
fn preserves_accepts_a_function_that_changes_the_value() {
    let reverse = |mut xs: Vec<i32>| {
        xs.reverse();
        xs
    };
    let before = vec![1, 2, 3];
    let after = reverse(before.clone());
    assert_ne!(before, after, "precondition: the value must change");

    let multiset = |xs: &Vec<i32>| {
        let mut sorted = xs.clone();
        sorted.sort_unstable();
        sorted
    };
    assert!(preserves("reverse", before, reverse, multiset).is_ok());
}

#[test]
fn fixed_point_accepts_one_and_rejects_a_non_fixed_value() {
    assert!(fixed_point("square", 1i64, |v| v * v, &Exact).is_ok());
    assert!(fixed_point("square", 3i64, |v| v * v, &Exact).is_err());
}

/// Every failure must name the relation it came from, or a suite with a dozen relation calls
/// reports a failure nobody can locate.
#[test]
fn a_failure_names_the_relation_and_the_comparison() {
    let failure =
        involution("my-relation", 7i64, |v| v + 1, &Exact).expect_err("this is not an involution");
    assert!(
        failure.message.contains("my-relation"),
        "{}",
        failure.message
    );
    assert!(
        failure.message.contains("f(f(x)) = x"),
        "{}",
        failure.message
    );
    assert!(
        failure.message.contains("exactly equal"),
        "{}",
        failure.message
    );
}

// ===== order_independent ==================================================

/// A sum over well-scaled values is order-independent to a few ulps, and a reduction that
/// genuinely depends on order is not.
#[test]
fn order_independent_accepts_a_sum_and_rejects_an_order_dependent_fold() {
    let xs = vec![1.0_f64, 2.0, 3.0, 4.0];
    let order = vec![3usize, 1, 0, 2];

    order_independent("sum", &xs, &order, |v| v.iter().sum::<f64>(), &Ulps(2))
        .expect("a sum of well-scaled values is order independent");

    // The classic bug this exists to catch: an accumulator seeded from the first element instead
    // of the additive identity, which double-counts whatever happens to be first.
    order_independent(
        "seeded-fold",
        &xs,
        &order,
        |v| v.first().copied().unwrap_or(0.0) + v.iter().sum::<f64>(),
        &Ulps(2),
    )
    .expect_err("a fold seeded from the first element depends on order");
}

/// `min` and `max` reassociate exactly, so `Exact` is the right bound and must be honoured
/// rather than quietly tolerated.
#[test]
fn order_independent_holds_exactly_for_a_selection_reduction() {
    let xs = vec![5.0_f64, -2.0, 9.0, 0.5];
    let order = vec![2usize, 0, 3, 1];
    order_independent(
        "max",
        &xs,
        &order,
        |v| v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        &Exact,
    )
    .expect("max is exactly order independent");
}

/// A malformed order is the test's bug, not the reduction's, and the message has to say so —
/// otherwise a wrong permutation in a suite reads as a failure of the code under test.
#[test]
fn order_independent_rejects_a_malformed_order_as_the_callers_fault() {
    let xs = vec![1.0_f64, 2.0];
    for bad in [vec![0usize], vec![0, 0, 1], vec![0, 5]] {
        let failure = order_independent("r", &xs, &bad, |v| v.iter().sum::<f64>(), &Exact)
            .expect_err("a malformed order must be refused");
        assert!(
            failure.message.contains("bug in the test"),
            "{}",
            failure.message
        );
    }
}

// ===== forward_stable =====================================================

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

// ===== precision_agrees ===================================================

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

// ===== The comparators, pinned in both directions =========================

/// Every comparator must say **yes** to something.
///
/// The suite above checks each one's refusals thoroughly and its acceptances barely, and mutation
/// testing found the hole: `Relative::same` could be replaced with a function returning `false`
/// unconditionally and the whole workspace stayed green. A comparator that never agrees turns
/// every relation built on it into a vacuous pass-by-failure — or rather, it would make them all
/// fail, which is worse, and nothing was exercising the agreeing path to notice.
#[test]
fn every_comparator_accepts_a_value_that_matches() {
    assert!(Exact.same(&3_u32, &3_u32));
    assert!(Ulps(0).same(&1.0_f64, &1.0_f64));
    assert!(Relative(1e-9).same(&1.0_f64, &1.0_f64));
    assert!(Relative(0.01).same(&1.0_f64, &1.005_f64));
    assert!(Absolute(1e-9).same(&1.0_f64, &1.0_f64));
    assert!(Absolute(0.5).same(&1.0_f64, &1.25_f64));
}

/// A comparator's description reaches the reader on every failure, so it is part of the
/// deliverable rather than decoration.
///
/// Mutation testing replaced three of these four with `String::new()` and with `"xyzzy"`, and
/// nothing failed: only `Exact` had its description checked. A failure report that cannot say
/// *what* notion of sameness was being applied is the report that sends someone to read the
/// source of the relation they just used.
#[test]
fn every_comparator_describes_the_notion_it_applies() {
    let described = [
        <Exact as Compare<f64>>::describe(&Exact),
        <Ulps as Compare<f64>>::describe(&Ulps(3)),
        <Relative as Compare<f64>>::describe(&Relative(1e-9)),
        <Absolute as Compare<f64>>::describe(&Absolute(0.5)),
    ];
    for description in &described {
        assert!(
            !description.is_empty(),
            "a comparator described itself as nothing"
        );
    }
    // Each must name its own tolerance, or two failures at different tolerances read identically.
    assert!(
        <Ulps as Compare<f64>>::describe(&Ulps(3)).contains('3'),
        "Ulps did not name its tolerance: {}",
        described[1]
    );
    assert!(
        <Relative as Compare<f64>>::describe(&Relative(1e-9)).contains("e-9"),
        "Relative did not name its tolerance: {}",
        described[2]
    );
    // Scientific notation, like `Relative` — `0.5` reaches the reader as `5e-1`.
    assert!(
        <Absolute as Compare<f64>>::describe(&Absolute(0.5)).contains("5e-1"),
        "Absolute did not name its tolerance: {}",
        described[3]
    );
    // And they must be distinguishable from one another.
    let distinct: std::collections::BTreeSet<&String> = described.iter().collect();
    assert_eq!(
        distinct.len(),
        described.len(),
        "two comparators describe themselves identically: {described:?}"
    );
}

// ===== f32 ulp distance ===================================================

/// The `f32` ulp metric, pinned to exact counts.
///
/// `precision_agrees` is built entirely on this function and was the only thing exercising it —
/// through a tolerance comparison, which cannot distinguish a gap of 1 from a gap of 1,065,353,216
/// when the allowance is generous. Mutation testing left six survivors here, including flipping
/// the sign test that makes the mapping monotone across zero.
///
/// The constants are the point. `0.0` to `1.0` is `0x3F800000` steps because that *is* the bit
/// pattern of `1.0f32`, and a pair straddling zero is twice that — the property that a
/// sign-magnitude format does not give you for free and that this function exists to supply.
#[test]
fn f32_ulps_between_counts_representable_steps() {
    assert_eq!(f32_ulps_between(1.0, 1.0), Some(0));
    assert_eq!(f32_ulps_between(-1.0, -1.0), Some(0));

    let next = f32::from_bits(1.0_f32.to_bits() + 1);
    assert_eq!(f32_ulps_between(1.0, next), Some(1));
    assert_eq!(f32_ulps_between(next, 1.0), Some(1), "must be symmetric");

    // Signed zero is zero steps apart, not 2^31.
    assert_eq!(f32_ulps_between(-0.0, 0.0), Some(0));

    // Exact distances, which a tolerance comparison cannot pin.
    assert_eq!(f32_ulps_between(0.0, 1.0), Some(1_065_353_216));
    assert_eq!(f32_ulps_between(-1.0, 1.0), Some(2_130_706_432));

    // Non-finite on *either* side is unmeasurable, not large.
    assert_eq!(f32_ulps_between(f32::NAN, 1.0), None);
    assert_eq!(f32_ulps_between(1.0, f32::NAN), None);
    assert_eq!(f32_ulps_between(f32::INFINITY, 1.0), None);
    assert_eq!(f32_ulps_between(1.0, f32::NEG_INFINITY), None);
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
