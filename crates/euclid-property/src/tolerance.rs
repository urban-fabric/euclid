//! Comparing floats, and saying which comparison you meant.
//!
//! `==` is the wrong check for a computed float and `abs(a - b) < 1e-9` is the wrong check for
//! a float of unknown magnitude — it passes trivially at `1e-300` and fails always at `1e300`.
//! Neither is a tolerance so much as a way of not choosing one.
//!
//! The three that are defensible, and when:
//!
//! | comparison | use when |
//! |---|---|
//! | [`within_ulps`] | the result should be *as accurate as the format allows* — an identity, a round trip, a kernel's own contract |
//! | [`within_relative`] | the result is accurate to a documented number of digits, regardless of scale |
//! | [`within_absolute`] | the quantity has a known scale and a meaningful zero — a residual, a probability, an angle |
//!
//! Prefer [`within_ulps`] where it applies. It is scale-free, it is the unit the hardware and
//! the literature both use, and "within two ulps" is a claim about the implementation rather
//! than a number someone picked until the test passed.
//!
//! # Zero and sign
//!
//! Every function here treats `+0.0` and `-0.0` as equal, because they compare equal under `==`
//! and no numerical contract distinguishes them as *results*. ULP distance across zero is
//! defined by the ordered-integer mapping below, so `-0.0` and the smallest positive subnormal
//! are one ulp apart, which is the useful answer.
//!
//! # NaN
//!
//! `NaN` is never close to anything, *including another `NaN`*, and these return `false` rather
//! than panicking. A property that wants "both are NaN" should say so with
//! [`f64::is_nan`] — conflating "equally undefined" with "equally accurate" is how a NaN-
//! producing bug passes a tolerance check.

/// Map a `f64` onto an `i64` whose integer ordering matches float ordering.
///
/// The standard trick: floats are already lexicographically ordered within a sign, so flipping
/// the negatives into a descending integer range makes the whole domain monotone. The distance
/// between two mapped values is then the number of representable floats between them, which is
/// what "ulps apart" means.
fn ordered(value: f64) -> i64 {
    let bits = value.to_bits() as i64;
    if bits < 0 {
        // Negative floats: sign bit set. Reflect them below zero in the integer ordering.
        i64::MIN - bits
    } else {
        bits
    }
}

/// How many representable `f64`s lie between `a` and `b`, or `None` if either is `NaN`.
///
/// Zero means bit-identical (or `±0.0`). One means adjacent. The count saturates rather than
/// overflowing for values at opposite ends of the range.
#[must_use]
pub fn ulps_between(a: f64, b: f64) -> Option<u64> {
    if a.is_nan() || b.is_nan() {
        return None;
    }
    if a == b {
        // Catches ±0.0, whose bit patterns differ.
        return Some(0);
    }
    if a.is_infinite() || b.is_infinite() {
        // One is infinite and they are not equal, so they are not comparably close.
        return None;
    }
    let distance = ordered(a).saturating_sub(ordered(b));
    Some(distance.unsigned_abs())
}

/// Whether `a` and `b` are within `ulps` representable steps of each other.
///
/// The scale-free accuracy check, and the one to reach for by default. `within_ulps(a, b, 0)` is
/// exact-equality-modulo-signed-zero; `2` is a reasonable bar for a composed identity; a kernel
/// claiming to be correctly rounded should meet `1`.
///
/// Returns `false` if either value is `NaN`, or if one is infinite and they are not equal.
#[must_use]
pub fn within_ulps(a: f64, b: f64, ulps: u64) -> bool {
    ulps_between(a, b).is_some_and(|distance| distance <= ulps)
}

/// Whether `|a - b| / max(|a|, |b|)` is at most `tolerance`.
///
/// For a result documented to a number of digits rather than to the format's limit — a series
/// truncated at a fixed term count, a quadrature rule, an iterative solve with a stopping
/// criterion. `1e-12` is about eleven digits.
///
/// Both exactly zero compares equal. One zero and one not compares `false` for any tolerance
/// below one, which is correct: there is no relative accuracy to speak of against zero, and
/// silently passing that case is how an implementation that returns zero everywhere survives
/// its own test suite. Use [`within_absolute`] when zero is a legitimate expected value.
#[must_use]
pub fn within_relative(a: f64, b: f64, tolerance: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    if a == b {
        return true;
    }
    let scale = a.abs().max(b.abs());
    if scale == 0.0 || !scale.is_finite() {
        return false;
    }
    (a - b).abs() / scale <= tolerance
}

/// Whether `|a - b|` is at most `tolerance`.
///
/// Only defensible when the quantity has a known scale — a residual that should be near zero, a
/// probability in `[0, 1]`, an angle in radians. On a value of unknown magnitude this is the
/// comparison that passes for the wrong reason.
#[must_use]
pub fn within_absolute(a: f64, b: f64, tolerance: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    if a == b {
        return true;
    }
    (a - b).abs() <= tolerance
}

/// The condition number of `a + b` — `(|a| + |b|) / |a + b|`.
///
/// How much a summation amplifies round-off, and therefore how many digits a naive sum is
/// entitled to lose. A value near one is benign; `1e15` means a `f64` sum has no correct digits
/// left and a compensated sum is not optional.
///
/// Useful for *stating* a property rather than asserting one: "compensated summation beats
/// naive summation by a margin that grows with the condition number" is a claim with teeth,
/// where "compensated summation is accurate" is not.
///
/// Infinite when the terms cancel exactly, which is the honest answer — the relative error of a
/// zero result is undefined.
///
/// # The denominator is summed compensated, and it has to be
///
/// This is the mistake a first version made, and it is a nice one: a function for measuring
/// ill-conditioning computed its own denominator with a naive fold, so on exactly the inputs it
/// exists to describe it was wrong. For `[1e16, 1, -1e16]` the true sum is `1` and the naive
/// fold returns `0`, so the reported condition number was `inf` rather than `2e16` — the
/// function diagnosed catastrophic cancellation by suffering from it.
///
/// Kahan–Babuška–Neumaier below, which is the same algorithm euclid-numeric's `compensated::sum`
/// uses. It is inlined rather than borrowed because this crate is dev-only and must not depend
/// on the crates whose tests depend on it.
#[must_use]
pub fn condition_number(terms: &[f64]) -> f64 {
    let magnitude: f64 = compensated_sum(&terms.iter().map(|t| t.abs()).collect::<Vec<_>>());
    let total = compensated_sum(terms);
    if total == 0.0 {
        return f64::INFINITY;
    }
    magnitude / total.abs()
}

/// Kahan–Babuška–Neumaier summation: the correction is accumulated separately rather than
/// folded back, which is what makes it correct when a later term is larger than the running
/// total — the case plain Kahan gets wrong.
fn compensated_sum(terms: &[f64]) -> f64 {
    let mut total = 0.0_f64;
    let mut correction = 0.0_f64;
    for &term in terms {
        let candidate = total + term;
        correction += if total.abs() >= term.abs() {
            (total - candidate) + term
        } else {
            (term - candidate) + total
        };
        total = candidate;
    }
    total + correction
}

/// Fail the property unless `left` and `right` are within `ulps` representable steps.
#[macro_export]
macro_rules! check_ulps {
    ($left:expr, $right:expr, $ulps:expr) => {
        $crate::check_ulps!($left, $right, $ulps, "")
    };
    ($left:expr, $right:expr, $ulps:expr, $($arg:tt)+) => {{
        let left: f64 = $left;
        let right: f64 = $right;
        let ulps: u64 = $ulps;
        if !$crate::tolerance::within_ulps(left, right, ulps) {
            let context = format!($($arg)+);
            let separator = if context.is_empty() { "" } else { ": " };
            let distance = match $crate::tolerance::ulps_between(left, right) {
                Some(d) => format!("{d} ulps apart"),
                None => "not comparable (NaN or infinity)".to_string(),
            };
            return Err($crate::Failure::new(
                format!(
                    "{}{}{} and {} differ by more than {} ulp\n    \
                     left:  {:?}\n    right: {:?}\n    {}",
                    context, separator,
                    stringify!($left), stringify!($right), ulps,
                    left, right, distance,
                ),
                concat!(file!(), ":", line!()),
            ));
        }
    }};
}

/// Fail the property unless `left` and `right` agree to a relative `tolerance`.
#[macro_export]
macro_rules! check_relative {
    ($left:expr, $right:expr, $tolerance:expr) => {
        $crate::check_relative!($left, $right, $tolerance, "")
    };
    ($left:expr, $right:expr, $tolerance:expr, $($arg:tt)+) => {{
        let left: f64 = $left;
        let right: f64 = $right;
        let tolerance: f64 = $tolerance;
        if !$crate::tolerance::within_relative(left, right, tolerance) {
            let context = format!($($arg)+);
            let separator = if context.is_empty() { "" } else { ": " };
            let scale = left.abs().max(right.abs());
            let relative = if scale > 0.0 { (left - right).abs() / scale } else { f64::NAN };
            return Err($crate::Failure::new(
                format!(
                    "{}{}{} and {} differ by more than {:e} relative\n    \
                     left:     {:?}\n    right:    {:?}\n    relative: {:e}",
                    context, separator,
                    stringify!($left), stringify!($right), tolerance,
                    left, right, relative,
                ),
                concat!(file!(), ":", line!()),
            ));
        }
    }};
}

#[cfg(test)]
mod tests;
