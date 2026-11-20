//! The form boundaries, asserted as decisions.
//!
//! Split from the accuracy tests because these reach `IForm` and `LnI0Form` rather than any
//! value. Both tables route between approximations that agree to several digits across their
//! join — which is what makes a join a free choice — so no accuracy comparison can see a moved
//! boundary, and only the decision can be pinned.

use crate::kernels::bessel::asymptotic::i_scaled;
use crate::kernels::bessel::modified_first::{I1_NO_LOG_MAX, *};

#[test]
fn the_stirling_threshold_leaves_no_step_at_ulp_resolution() {
    // `stirling_remainder` switches from `ln_gamma` to its asymptotic series at a peak index
    // of 20. The peak index is `round(x/2)`, so the switch happens as `x` crosses 39 — an
    // internal boundary with no outward sign, and one no reference table would ever probe.
    // Measured disagreement across the switch is 1.5e-14 relative, and it is the *lower*
    // side that carries it: below the threshold `mu(k)` is formed as a difference of two
    // O(k ln k) quantities, costing about `eps*42/0.0042 = 1.5e-14`, while above it the
    // asymptotic series is exact to rounding. That is the documented 1.7e-14 worst case for
    // this family, and it is bounded here rather than merely asserted elsewhere.
    use euclid_utils::no_step_across;
    let threshold = 39.0;
    no_step_across(
        "I0e",
        i0e,
        threshold,
        i0e(threshold) - i1e(threshold),
        1e-13,
    );
    no_step_across("I1e", i1e, threshold, i1e(threshold), 1e-13);
    no_step_across("ln_i0", ln_i0, threshold, i1_over_i0(threshold), 1e-13);
    no_step_across("i1_over_i0", i1_over_i0, threshold, 1.0, 1e-13);
    // The peak index also steps by one every time x crosses an odd integer, since it is a
    // rounding of x/2. Those are threshold crossings too, and there are many of them.
    for odd in [5.0f64, 11.0, 21.0, 41.0, 101.0] {
        no_step_across("I0e at a peak step", i0e, odd, i0e(odd) - i1e(odd), 1e-13);
        no_step_across("ln_i0 at a peak step", ln_i0, odd, i1_over_i0(odd), 1e-13);
    }
}

#[test]
fn the_series_and_the_asymptotic_branch_agree_where_they_meet() {
    // The two routes share no code — a peak-factored power series against a divergent
    // asymptotic expansion — so agreement across the threshold is real evidence. Probed at
    // the two representable values straddling it, where no derivative slack is needed.
    use euclid_utils::no_step_across;
    let x = I_ASYMPTOTIC_MIN;
    no_step_across("i0e", i0e, x, i0e(x) - i1e(x), 1e-14);
    no_step_across("i1e", i1e, x, i1e(x), 1e-14);
    no_step_across("ln_i0", ln_i0, x, i1_over_i0(x), 1e-15);
    no_step_across("i1_over_i0", i1_over_i0, x, 1.0, 1e-14);
    // And across a whole band below the threshold, where the series is still the one in use
    // and must already be converging on the expansion.
    for i in 0..40 {
        let x = 3e3 + f64::from(i) * 175.0;
        let expected = i_scaled(0.0, x);
        assert!(
            ((i0e(x) - expected) / expected).abs() < 5e-15,
            "series i0e({x:e}) vs the expansion"
        );
    }
}

/// The regime split is one decision, so it is pinned in one place.
///
/// **`cargo mutants` asked for this test specifically.** `replace < with <=` on `i1e`'s
/// `ax < I1_NO_LOG_MAX` survived the whole suite, and it is not an equivalent mutant — it moves
/// `|x| = 1` from the peak-factored form to the no-logarithm one. No value assertion can see it,
/// because a crossover is *sited* where its two forms agree. The routing is the thing with
/// content, so it is asserted directly.
///
/// Both boundaries are checked at the constant and one ulp either side, since a `<` drifting to
/// `<=` is exactly a one-ulp change.
#[test]
fn the_i_form_boundaries_sit_where_the_constants_say() {
    use crate::kernels::bessel::modified_first::IForm;

    assert_eq!(IForm::for_magnitude(0.0), IForm::NoLog);
    assert_eq!(IForm::for_magnitude(5e-324), IForm::NoLog);
    assert_eq!(
        IForm::for_magnitude(f64::from_bits(I1_NO_LOG_MAX.to_bits() - 1)),
        IForm::NoLog,
        "one ulp below I1_NO_LOG_MAX must still skip the logarithm"
    );
    assert_eq!(
        IForm::for_magnitude(I1_NO_LOG_MAX),
        IForm::PeakFactored,
        "I1_NO_LOG_MAX itself is peak-factored, as k_peak = round(0.5) = 1 already made it"
    );
    // An interior point of the peak-factored band. It was `700.0` while `I_ASYMPTOTIC_MIN` was
    // `1e4`; that argument is on the expansion now, so this picks one that is still comfortably
    // inside the series' range rather than one that merely used to be.
    assert_eq!(IForm::for_magnitude(10.0), IForm::PeakFactored);
    assert_eq!(IForm::for_magnitude(700.0), IForm::Asymptotic);
    assert_eq!(
        IForm::for_magnitude(f64::from_bits(I_ASYMPTOTIC_MIN.to_bits() - 1)),
        IForm::PeakFactored,
        "one ulp below I_ASYMPTOTIC_MIN must still use the series"
    );
    assert_eq!(
        IForm::for_magnitude(I_ASYMPTOTIC_MIN),
        IForm::Asymptotic,
        "I_ASYMPTOTIC_MIN itself takes the expansion"
    );
    assert_eq!(IForm::for_magnitude(f64::MAX), IForm::Asymptotic);
    assert_eq!(IForm::for_magnitude(f64::INFINITY), IForm::Asymptotic);

    // The unification the enum rests on, and the only step of the refactor that was not a
    // definitional rewrite: `NoLog` must coincide with `I₀`'s peak index falling below one,
    // which is what `ln_i0_minus_x` used to test separately.
    let mut ax = 1e-8f64;
    while ax < 4.0 {
        let k_peak = (ax * 0.5).round().max(0.0);
        assert_eq!(
            IForm::for_magnitude(ax) == IForm::NoLog,
            k_peak < 1.0,
            "NoLog must coincide with k_peak < 1 at {ax:e}"
        );
        ax *= 1.0005;
    }
}

/// The `ln I₀` crossover, which is this format's own measurement.
///
/// The series the low arm sums, and its term count, moved to [`crate::math::bessel`] — generic,
/// and tested there for both formats at once. What stays here is the threshold: it is a separate
/// measurement in each precision, and the comparison that applies it survived the suite because
/// the crossover is *sited* where the two forms agree, which is what makes it a good crossover.
#[test]
fn the_log_i0_crossover_is_half_open() {
    use LnI0Form::{Series, Shifted};
    assert_eq!(
        LnI0Form::for_magnitude(LN_I0_SERIES_MAX.next_down()),
        Series
    );
    assert_eq!(LnI0Form::for_magnitude(LN_I0_SERIES_MAX), Shifted);
    assert_eq!(LnI0Form::for_magnitude(LN_I0_SERIES_MAX.next_up()), Shifted);
    assert_eq!(LnI0Form::for_magnitude(0.0), Series);
    assert_eq!(LnI0Form::for_magnitude(1e300), Shifted);
}
