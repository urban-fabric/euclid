//! Telling the search what to maximise.
//!
//! By default a property search is uniform: it draws cases from the generator's distribution
//! and hopes one of them is bad. For a question like "how accurate is this kernel" that is a
//! weak strategy, because the worst input is usually a thin region — a near-cancellation, an
//! argument just inside a branch boundary — and uniform draws find it by luck or not at all.
//!
//! [`target`] replaces luck with a climb. The property computes how bad the current case is and
//! reports it; the runner remembers the highest-scoring tape and spends the second half of its
//! budget perturbing it. The search stops asking "does this hold on random input" and starts
//! asking "how badly can this be made to fail".
//!
//! This is Hypothesis's `target()`, and for numerics it is the single most useful feature in
//! the framework:
//!
//! ```
//! use euclid_stability::{check_ulps, generate, property, target, tolerance};
//!
//! property! {
//!     fn exp_ln_round_trips(x in generate::float::positive()) {
//!         let returned = x.ln().exp();
//!
//!         // Steer the search towards the worst round-trip error it can find, rather than
//!         // settling for whatever 256 uniform draws happened to contain.
//!         if let Some(error) = tolerance::ulps_between(returned, x) {
//!             #[allow(clippy::cast_precision_loss)]
//!             target(error as f64);
//!         }
//!
//!         check_ulps!(returned, x, 4);
//!     }
//! }
//! ```
//!
//! # What makes a good score
//!
//! It has to be **continuous in the input**, or there is nothing to climb. "ULP error" works
//! because a slightly worse input scores slightly higher. "Did it fail" does not work — it is
//! a step function, and the climber gets no gradient from it.
//!
//! Higher must mean worse. Report error, not accuracy.
//!
//! A non-finite score is ignored rather than treated as infinitely good, since `NaN` usually
//! means the case was degenerate rather than maximally interesting.
//!
//! # It never weakens a property
//!
//! Targeting changes which cases are *tried*, never which ones pass. A property with a target
//! is as strict as the same property without one; it just gets shown harder input. So a passing
//! targeted property is a stronger result than a passing uniform one, and there is no reason
//! not to add a score wherever a meaningful one exists.

use std::cell::Cell;

thread_local! {
    /// The score reported by the case currently running, if any.
    static SCORE: Cell<Option<f64>> = const { Cell::new(None) };
}

/// Report how interesting the current case is, so the search can look for worse.
///
/// Higher is worse. Called more than once in a case, the largest value wins — so a property
/// checking several things can report each one's error without tracking the maximum itself.
pub fn target(score: f64) {
    SCORE.with(|current| {
        let best = match current.get() {
            Some(previous) if previous >= score => previous,
            _ => score,
        };
        current.set(Some(best));
    });
}

/// The score the case just reported.
pub(crate) fn reported() -> Option<f64> {
    SCORE.with(Cell::get)
}

/// Forget the previous case's score.
pub(crate) fn clear() {
    SCORE.with(|current| current.set(None));
}

#[cfg(test)]
mod tests {
    use super::{clear, reported, target};

    #[test]
    fn a_score_is_remembered_until_cleared() {
        clear();
        assert_eq!(reported(), None);
        target(3.5);
        assert_eq!(reported(), Some(3.5));
        clear();
        assert_eq!(reported(), None);
    }

    /// Several reports in one case keep the worst, so a property can score each of its checks
    /// without maintaining a running maximum.
    #[test]
    fn the_worst_score_in_a_case_wins() {
        clear();
        target(1.0);
        target(9.0);
        target(4.0);
        assert_eq!(reported(), Some(9.0));
    }

    #[test]
    fn a_negative_score_is_still_a_score() {
        clear();
        target(-5.0);
        assert_eq!(reported(), Some(-5.0));
        target(-1.0);
        assert_eq!(reported(), Some(-1.0));
    }
}
