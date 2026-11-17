//! Metamorphic checks for the relation module.

use super::*;

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

#[test]
fn insensitive_to_accepts_a_genuine_independence_and_rejects_a_dependence() {
    // Periodicity: angle reduction must ignore whole revolutions, exactly.
    insensitive_to(
        "reduction is periodic",
        90.0_f64,
        |x| x + 360.0,
        |x| x.rem_euclid(360.0),
        &Exact,
    )
    .expect("reduction ignores a whole turn");

    // Scale invariance: a ratio must ignore a common positive factor. Exact here because the
    // factor is a power of two, which is the only way a float ratio is exactly invariant — a
    // reminder that this relation is usually about exactness.
    insensitive_to(
        "a ratio ignores a common scale",
        (3.0_f64, 4.0),
        |(a, b)| (a * 4.0, b * 4.0),
        |(a, b)| a / b,
        &Exact,
    )
    .expect("a ratio ignores a power-of-two scale");

    // And it must notice a function that does depend on what was perturbed, or it asserts
    // nothing: `x * 2` is not periodic in 360.
    insensitive_to(
        "doubling is not periodic",
        90.0_f64,
        |x| x + 360.0,
        |x| x * 2.0,
        &Exact,
    )
    .expect_err("doubling depends on the perturbation and must be rejected");
}

/// The failure must name the relation and both values, so a reader knows what was ignored and
/// what came out instead.
#[test]
fn insensitive_to_reports_the_relation_and_both_values() {
    let failure = insensitive_to("identity", 1.0_f64, |x| x + 1.0, |x| x, &Exact)
        .expect_err("the identity depends on its argument");
    assert!(
        failure.message.contains("f(perturb(x)) = f(x)"),
        "must name the relation: {}",
        failure.message
    );
    assert!(
        failure.message.contains('2') && failure.message.contains('1'),
        "must report both values: {}",
        failure.message
    );
}
