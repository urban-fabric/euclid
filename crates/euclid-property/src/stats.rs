//! Case classification, and coverage you can require.
//!
//! # The problem this solves
//!
//! A property that ran 256 cases and passed tells you almost nothing on its own, because it
//! does not tell you *what* it ran. A generator can be degenerate in ways nothing complains
//! about: a float generator whose 256 draws were all normal numbers around `1e200` has tested
//! none of the behaviour near zero, and an accuracy property over it is green and vacuous.
//!
//! ScalaCheck's `classify`, jqwik's statistics and Hypothesis's `event()` all address the
//! reporting half: label each case, print the distribution, let a human notice. That is useful
//! and it is not enough, because nobody reads a passing test's output.
//!
//! So the labels here can be **required**. [`Config::require`](crate::Config::require) names a
//! label and a minimum share of cases, and the run *fails* if the generator did not produce
//! them. "At least 5% of cases must be subnormal" turns an assumption about the generator into
//! an assertion about it, checked on every run, and that is the difference between a suite that
//! looks thorough and one that is.
//!
//! # Usage
//!
//! Label from inside the property body, after you know which case you got:
//!
//! ```
//! use euclid_property::{Config, check, classify, generate, property};
//!
//! property! {
//!     #![config(Config::default().require("subnormal", 1.0).require("normal", 50.0))]
//!     fn classifying_floats(x in generate::float::finite()) {
//!         if x == 0.0 {
//!             classify!("zero");
//!         } else if x.abs() < f64::MIN_POSITIVE {
//!             classify!("subnormal");
//!         } else {
//!             classify!("normal");
//!         }
//!     }
//! }
//! ```
//!
//! Set `EUCLID_PROPERTY_STATS=1` to print the distribution for every property, required or not.
//!
//! # A label is not a branch
//!
//! Labels are for describing the *input*, not for recording which path the code took. A label
//! that says "the fast path ran" is a coverage measurement wearing the wrong hat — it will
//! silently start passing when the code changes. Say what the case *was*.

use std::cell::RefCell;
use std::collections::BTreeMap;

thread_local! {
    /// Labels applied during the current property run, with counts.
    static LABELS: RefCell<BTreeMap<&'static str, usize>> = const { RefCell::new(BTreeMap::new()) };
    /// Cases that have been *counted*, which is the denominator for a coverage share. Not the
    /// configured case count: rejected cases never ran the body and must not dilute coverage.
    static COUNTED: RefCell<usize> = const { RefCell::new(0) };
}

/// Tag the case currently running. Prefer the [`classify!`](crate::classify) macro.
pub fn label(name: &'static str) {
    LABELS.with(|labels| {
        *labels.borrow_mut().entry(name).or_insert(0) += 1;
    });
}

/// Note that one case ran to completion, labelled or not.
pub(crate) fn count_case() {
    COUNTED.with(|counted| *counted.borrow_mut() += 1);
}

/// Clear the collector. Called at the start of each property, since the thread is reused across
/// properties within a test binary and counts must not leak between them.
pub(crate) fn reset() {
    LABELS.with(|labels| labels.borrow_mut().clear());
    COUNTED.with(|counted| *counted.borrow_mut() = 0);
}

/// The labels applied so far, with counts.
#[must_use]
pub fn collected() -> BTreeMap<&'static str, usize> {
    LABELS.with(|labels| labels.borrow().clone())
}

/// How many cases ran the property body.
#[must_use]
pub fn cases_counted() -> usize {
    COUNTED.with(|counted| *counted.borrow())
}

/// A label's share of the cases that ran, as a percentage.
#[must_use]
pub fn share(name: &str) -> f64 {
    let total = cases_counted();
    if total == 0 {
        return 0.0;
    }
    let hits = collected().get(name).copied().unwrap_or(0);
    #[allow(clippy::cast_precision_loss)]
    {
        (hits as f64) * 100.0 / (total as f64)
    }
}

/// The distribution as a human-readable block, for a report or a stats dump.
#[must_use]
pub fn report() -> String {
    let labels = collected();
    let total = cases_counted();
    if labels.is_empty() {
        return format!("{total} cases, no labels applied");
    }

    let mut lines = vec![format!("{total} cases:")];
    let width = labels.keys().map(|k| k.len()).max().unwrap_or(0);
    for (name, hits) in &labels {
        #[allow(clippy::cast_precision_loss)]
        let percent = if total == 0 {
            0.0
        } else {
            (*hits as f64) * 100.0 / (total as f64)
        };
        lines.push(format!("    {name:<width$}  {hits:>6}  {percent:>6.2}%"));
    }
    lines.join("\n")
}

/// Which required labels were not met, as `(label, required, actual)`.
pub(crate) fn unmet(requirements: &[(String, f64)]) -> Vec<(String, f64, f64)> {
    requirements
        .iter()
        .filter_map(|(name, minimum)| {
            let actual = share(name);
            if actual < *minimum {
                Some((name.clone(), *minimum, actual))
            } else {
                None
            }
        })
        .collect()
}

/// Tag the case currently running with a label, for the distribution and for coverage
/// requirements. See [`crate::stats`].
#[macro_export]
macro_rules! classify {
    ($name:expr) => {
        $crate::stats::label($name)
    };
    ($name:expr, $condition:expr) => {
        if $condition {
            $crate::stats::label($name);
        }
    };
}

#[cfg(test)]
mod tests;
