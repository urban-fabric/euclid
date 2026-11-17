use super::*;

fn rejects(body: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)).is_err()
}

/// The assertion helpers have to **fire**, which is not what the tests above establish.
///
/// Every other test here calls `assert_close` and `no_step_across` on input that satisfies
/// them, and mutation testing showed what that misses: `no_step_across` could be replaced
/// with an empty body and `assert_close`'s bound could be made additive instead of
/// multiplicative, with the whole workspace suite green.
///
/// These are *assertion helpers* in a dev-only crate, so a vacuous one is a silent hole
/// under every suite that uses it — which is six crates. The same argument as the
/// conformance suites in `euclid-algebra`: a check nobody has seen fail is a check nobody
/// has tested.
#[test]
fn the_assertion_helpers_reject_what_they_are_meant_to_reject() {
    // --- assert_close ---
    assert!(!rejects(|| assert_close(1.0, 1.0, 1e-12, "equal")));
    assert!(
        rejects(|| assert_close(2.0, 1.0, 1e-12, "far apart")),
        "assert_close must reject a value twice the target"
    );

    // The bound is `tol * max(|want|, 1)` — *relative* away from zero. An additive bound
    // would accept this, because `1e-12 + 1e9` is enormous.
    assert!(
        rejects(|| assert_close(1e9 * (1.0 + 1e-6), 1e9, 1e-12, "relative")),
        "assert_close's tolerance must scale with the target, not be added to it"
    );
    // And it really does scale: the same relative error passes at a looser tolerance.
    assert!(!rejects(|| assert_close(
        1e9 * (1.0 + 1e-13),
        1e9,
        1e-12,
        "inside"
    )));

    // --- assert_exact ---
    assert!(!rejects(|| assert_exact(1.0, 1.0, "identical")));
    assert!(rejects(|| assert_exact(
        1.0 + f64::EPSILON,
        1.0,
        "one ulp apart"
    )));
    assert!(rejects(|| assert_exact(-0.0, 0.0, "different zero signs")));

    // --- no_step_across ---
    // A smooth function has no step at any threshold.
    assert!(!rejects(|| no_step_across(
        "exp",
        f64::exp,
        1.0,
        f64::exp(1.0),
        1e-12
    )));
    // A function that jumps by one at the threshold does.
    assert!(
        rejects(|| no_step_across(
            "step",
            |v| if v >= 1.0 { 1.0 } else { 0.0 },
            1.0,
            0.0,
            1e-12
        )),
        "no_step_across must detect a unit jump at the threshold it is given"
    );
}

#[test]
fn f32_tolerance_scales_with_the_reference_and_the_ulp_budget() {
    let near_zero = 2.0 * f32::EPSILON;
    assert_close_f32(near_zero, 0.0, 3.0, "within the absolute budget");
    assert!(rejects(|| assert_close_f32(
        near_zero,
        0.0,
        1.0,
        "outside the absolute budget"
    )));

    let large = 2.0f32.powi(24);
    assert_close_f32(large.next_up(), f64::from(large), 1.0, "one ulp");
    assert!(rejects(|| assert_close_f32(
        large.next_up().next_up(),
        f64::from(large),
        1.0,
        "two ulps"
    )));
}

#[test]
fn seam_checks_allow_the_slope_and_rounding_budgets() {
    no_step_across("linear through zero", |x| 2.0 * (x - 1.0), 1.0, 2.0, 0.0);
    assert!(rejects(|| no_step_across(
        "jump exceeds the slope allowance",
        |x| if x > 1.0 { 1.0 } else { 0.0 },
        1.0,
        2.0,
        0.0,
    )));
    no_step_across(
        "jump fits in the relative allowance",
        |x| if x > 1.0 { 2.0625 } else { 2.0 },
        1.0,
        2.0,
        0.05,
    );
    no_step_across(
        "rounding floor",
        |x| if x > 1.0 { f64::MIN_POSITIVE } else { 0.0 },
        1.0,
        0.0,
        0.0,
    );
}

#[test]
fn mixed_criterion_behaves_absolutely_near_zero_and_relatively_away() {
    // Near zero: the bound is `tol`, so two roundings of zero compare equal.
    assert!(is_close(1e-18, 0.0, 1e-12));
    assert!(!is_close(1e-6, 0.0, 1e-12));
    // Away from zero: the bound scales, so 1e-12 means 1e-12 *relative*.
    assert!(is_close(1e9 * (1.0 + 1e-13), 1e9, 1e-12));
    assert!(!is_close(1e9 * (1.0 + 1e-10), 1e9, 1e-12));
}

#[test]
fn derivative_recovers_known_derivatives() {
    for x in sweep(0.5, 4.0, 15) {
        assert_close(derivative(f64::sin, x), x.cos(), 1e-9, "sin' = cos");
        assert_close(derivative(f64::exp, x), x.exp(), 1e-9, "exp' = exp");
        assert_close(derivative(f64::ln, x), 1.0 / x, 1e-9, "ln' = 1/x");
    }
}

#[test]
fn integrate_recovers_known_integrals() {
    // int_0^pi sin = 2
    assert_close(
        integrate(f64::sin, 0.0, core::f64::consts::PI, 2000),
        2.0,
        1e-12,
        "int sin",
    );
    // int_0^1 x^2 = 1/3 (Simpson is exact for cubics)
    assert_close(
        integrate(|x| x * x, 0.0, 1.0, 10),
        1.0 / 3.0,
        1e-15,
        "int x^2",
    );
}

#[test]
fn integration_rounds_the_panel_request_to_an_even_count() {
    let points = std::cell::RefCell::new(Vec::new());
    for (requested, actual) in [(0, 2), (1, 2), (4, 4), (5, 4)] {
        assert_close(
            integrate(
                |x| {
                    points.borrow_mut().push(x);
                    x.powi(3)
                },
                0.0,
                1.0,
                requested,
            ),
            0.25,
            0.0,
            "Simpson integrates a cubic",
        );
        assert_eq!(points.take().len(), actual + 1);
    }
}

#[test]
fn sweeps_hit_their_endpoints() {
    let s: Vec<f64> = sweep(-1.0, 1.0, 5).collect();
    assert_eq!(s.len(), 5);
    assert_exact(s[0], -1.0, "sweep lo");
    assert_exact(s[4], 1.0, "sweep hi");
    let l: Vec<f64> = log_sweep(1e-3, 1e3, 7).collect();
    assert_close(l[0], 1e-3, 1e-12, "log_sweep lo");
    assert_close(l[6], 1e3, 1e-12, "log_sweep hi");
    for n in [0, 1, 2, 3, 8] {
        let points: Vec<_> = sweep(-0.0, 0.0, n).collect();
        assert_eq!(points.len(), n.max(2));
        assert_eq!(points[0].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(points[n.max(2) - 1].to_bits(), 0.0_f64.to_bits());
    }
}

#[test]
fn two_pass_variance_survives_a_large_offset() {
    // The case that breaks `E[x^2] - E[x]^2`: a small spread on a huge mean. The naive
    // form loses every digit here; the two-pass form does not.
    let xs: Vec<f64> = (0..5).map(|i| 1e9 + f64::from(i)).collect();
    let naive = {
        let m = mean(&xs);
        xs.iter().map(|x| x * x).sum::<f64>() / xs.len() as f64 - m * m
    };
    assert_close(variance(&xs), 2.0, 1e-9, "two-pass variance");
    assert_exact(std_dev(&[3.0, 7.0]), 2.0, "population standard deviation");
    assert!(
        rel(naive, 2.0) > 1e-3,
        "the naive form is expected to be badly wrong here, got {naive}"
    );
}

#[test]
fn shape_assertions_accept_and_reject() {
    assert_nondecreasing(sweep(0.0, 1.0, 10), 0.0, "sweep rises");
    assert_nonincreasing(sweep(1.0, 0.0, 10), 0.0, "reversed sweep falls");
    assert_probability(sweep(0.0, 1.0, 10), "unit sweep");
    assert_all_finite(sweep(-5.0, 5.0, 10), "finite sweep");

    assert_nondecreasing([1.0, 0.875], 0.125, "allowed decrease");
    assert_nonincreasing([1.0, 1.125], 0.125, "allowed increase");
    assert!(rejects(|| assert_nondecreasing(
        [1.0, 0.75],
        0.125,
        "too much decrease"
    )));
    assert!(rejects(|| assert_nonincreasing(
        [1.0, 1.25],
        0.125,
        "too much increase"
    )));
    for value in [-2.0, 2.0, f64::NAN] {
        assert!(rejects(|| assert_in_range(
            [value],
            -1.0,
            1.0,
            "outside range"
        )));
    }
    for value in [-0.125, 1.125, f64::NAN] {
        assert!(rejects(|| assert_probability(
            [value],
            "invalid probability"
        )));
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(rejects(|| assert_all_finite([value], "non-finite value")));
    }
}
