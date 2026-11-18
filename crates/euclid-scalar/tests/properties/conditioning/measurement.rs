//! Measurement properties.

use super::*;

// ===== Conditioning as a measurement ======================================

/// The conditioning curve: naive summation's error as a function of the condition number, with
/// compensated summation's error alongside it.
///
/// Not a property but a *measurement*, and the clearest statement of what this suite is for. A
/// fixed-point accuracy test can say "the sum was right at these terms"; only a sweep can say
/// "the naive algorithm loses every digit past this condition number, and the compensated one
/// does not".
///
/// The terms are `[κ/2, 1, -κ/2]`, whose exact sum is `1` and whose condition number is about
/// `κ`. The interesting threshold is `2^53`: below it the `1` still fits in the running total's
/// ulp and the naive fold is fine, above it the `1` is rounded away entirely.
///
/// A previous version of this test tried to show `exp(a+b) = exp(a)·exp(b)` degrading with
/// magnitude. It does not, and the measurement said so — `exp`'s error is *relative*, so the
/// product compounds it by a constant factor rather than a growing one, and the worst relative
/// error sat at `2.2e-16` across every decade. Keeping the honest negative result here because
/// the wrong intuition is a natural one: not every identity is ill-conditioned at scale, and
/// the way to find out is to measure.
#[test]
fn naive_summation_degrades_with_condition_number_and_compensated_does_not() {
    let mut rows = Vec::new();

    for exponent in [0i32, 4, 8, 12, 15, 16, 18, 20] {
        let half = 10f64.powi(exponent);
        let terms = [half, 1.0, -half];
        let exact = 1.0;

        let condition = tolerance::condition_number(&terms);
        let naive = terms.iter().fold(0.0_f64, |acc, t| acc + t);
        let compensated = compensated::sum(&terms);

        let naive_error = (naive - exact).abs() / exact;
        let compensated_error = (compensated - exact).abs() / exact;
        rows.push((exponent, condition, naive_error, compensated_error));
    }

    eprintln!("summation error against condition number:");
    eprintln!("    terms         condition      naive error   compensated error");
    for (exponent, condition, naive, compensated) in &rows {
        eprintln!(
            "    [1e{exponent:<2}, 1, -1e{exponent:<2}]  {condition:>9.2e}  {naive:>13.2e}  {compensated:>17.2e}"
        );
    }

    // Compensated summation is exact at every condition number tested. This is the claim worth
    // making, and it is much stronger than "accurate to a tolerance".
    for (exponent, condition, _, compensated_error) in &rows {
        assert_eq!(
            *compensated_error, 0.0,
            "compensated summation lost accuracy at 1e{exponent} (condition {condition:e})"
        );
    }

    // And the naive fold really does fail, which is what makes the comparison meaningful. Past
    // 2^53 the small term is below the running total's ulp and is discarded outright.
    let worst_naive = rows
        .iter()
        .map(|(_, _, naive, _)| *naive)
        .fold(0.0_f64, f64::max);
    assert!(
        worst_naive >= 1.0,
        "the naive fold never lost the small term, so this sweep is not exercising \
         cancellation at all; worst relative error was {worst_naive:e}"
    );

    // The degradation is monotone in the condition number rather than noisy: error never
    // improves as conditioning worsens.
    for window in rows.windows(2) {
        let (before, after) = (&window[0], &window[1]);
        assert!(
            after.2 >= before.2,
            "naive error fell from {:e} to {:e} as the condition number rose from {:e} to {:e}",
            before.2,
            after.2,
            before.1,
            after.1
        );
    }
}
