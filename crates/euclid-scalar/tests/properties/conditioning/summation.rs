//! Summation properties.

use super::*;

// ===== Compensated summation ==============================================

property! {
    #![config(
        settings("compensated_sum")
            .require("ill conditioned", 5.0)
            .require("naive loses 6+ digits", 2.0)
    )]

    /// Compensated summation must survive cancellation that destroys the naive fold.
    ///
    /// The terms are built so their true total is known by construction: a large value, its
    /// negation, and a handful of small ones sum to exactly the small ones. The naive
    /// left-to-right fold cannot see them, because each is below the running total's ulp.
    ///
    /// The coverage requirements are what make this a search rather than an anecdote.
    fn compensated_summation_beats_naive_under_cancellation(
        big in generate::float::magnitude(20, 200),
        smalls in generate::vec(generate::float::in_range(-4.0, 4.0), 1, 6),
    ) {
        let mut terms = vec![big];
        terms.extend_from_slice(&smalls);
        terms.push(-big);

        let exact: f64 = smalls.iter().sum();
        let condition = tolerance::condition_number(&terms);

        classify!("ill conditioned", condition > 1e12);
        classify!("benign", condition < 1e3);

        let naive = terms.iter().fold(0.0_f64, |acc, t| acc + t);
        let compensated = compensated::sum(&terms);

        let naive_loss = digits_destroyed(naive, exact);
        classify!("naive loses 6+ digits", naive_loss >= 6.0);
        target(naive_loss);

        check_relative!(
            compensated,
            exact,
            1e-12,
            "compensated summation failed at condition number {condition:e}"
        );
    }
}

property! {
    // A block of its own, because the requirements name *this* property's labels. A shared
    // block-level config applies its requirements to every property in the block, which fails
    // confusingly when two properties classify differently — the first draft of this file hit
    // exactly that, and the coverage report said so.
    #![config(settings("compensated_dot").require("naive loses 4+ digits", 2.0))]

    /// The compensated dot product must be exact for inputs whose cancellation is known by
    /// construction, where the naive form accumulates the loss term by term.
    fn compensated_dot_survives_cancellation(
        scale in generate::float::magnitude(20, 150),
        tail in generate::float::in_range(-2.0, 2.0),
    ) {
        // Order matters, and the first draft got it wrong: `[scale, -scale, tail]` cancels
        // *first*, so the naive fold loses nothing and the coverage requirement failed. The
        // small term has to sit between the large ones to be rounded away.
        let a = [scale, tail, -scale];
        let b = [1.0, 1.0, 1.0];

        let naive: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let compensated = compensated::dot(&a, &b);

        let loss = digits_destroyed(naive, tail);
        classify!("naive loses 4+ digits", loss >= 4.0);
        target(loss);

        check_relative!(compensated, tail, 1e-12, "compensated dot failed");
    }
}
