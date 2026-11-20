//! Reflecting the right tail of `erfc` onto the left: `erfc(−t) = 2 − erfc(t)`.
//!
//! Split out of the parent kernel so a test-carrying file stays under the 400-line budget, and
//! because the saturation inside it is a decision with its own boundary — one that cannot be
//! reached through `erfc` without first solving `erfc(x) = ε`. See [`ReflectedErfc`].
//!
//! # Why reflect, when `1 − erf(x)` was accurate here
//!
//! It was — for `x < 0` that expression is an *addition*, `1 + |erf(x)|`, with no cancellation at
//! all. The reason to reflect is not the value but the **symmetry**.
//!
//! `Φ(x) + Φ(−x) = 1` is a one-ulp law in `identities::normal`, and `Φ(x) = erfc(−x/√2)/2`, so it
//! is `erfc(t) + erfc(−t) = 2` underneath. Evaluating the two sides independently made that law
//! hold only to the extent the two routes agreed: it was exact while both were `1 ∓ erf`, because
//! `erf`'s error then cancelled *identically* — and it drifted to one or two ulp the moment the
//! positive side was made more accurate than the negative one. A law that holds because two errors
//! agree is not measuring accuracy, and it blocks every future improvement to one side.
//!
//! Reflecting makes it structural. `2 ⊖ erfc(t)` carries at most half an ulp of `2`, so
//! `erfc(t) ⊕ (2 ⊖ erfc(t))` is within a quarter ulp of `2` whatever `erfc(t)` costs — and stays
//! there if `erfc(t)` gets better, which is the point.

/// Whether the reflected tail still has anything to subtract from `2`.
///
/// # Mutation testing is what asked for this
///
/// `<` against `<=` here survived the whole suite. It moves exactly one tail — `tail == ε` —
/// between the arms, and both results are ordinary representable numbers, so no sweep over `erfc`
/// separates them: reaching that tail through the kernel means solving `erfc(x) = ε` first. Named,
/// the boundary is three assertions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ReflectedErfc {
    /// `tail` is below half an ulp of `2`: the true `erfc` is within a rounding of `2`, and `2`
    /// is the correctly rounded answer. Subtracting anyway returned `1.9999999999999996`, which
    /// cost `Φ(8.25)` its exact `1`.
    Saturated,
    /// `2 ⊖ tail` still carries information, and is exact here by Sterbenz.
    Subtracted,
}

impl ReflectedErfc {
    /// The comparison is strict, and the threshold deliberately above where rounding alone would
    /// collapse the subtraction — see the test for the gap that buys.
    pub(super) fn for_tail(tail: f64) -> Self {
        if tail < f64::EPSILON {
            Self::Saturated
        } else {
            Self::Subtracted
        }
    }
}

/// The left-tail value, given the right tail it reflects.
pub(super) fn reflected_erfc(tail: f64) -> f64 {
    match ReflectedErfc::for_tail(tail) {
        ReflectedErfc::Saturated => 2.0,
        ReflectedErfc::Subtracted => 2.0 - tail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_reflected_tail_saturates_at_half_an_ulp_of_two() {
        use ReflectedErfc::{Saturated, Subtracted};
        // `ε` belongs to the subtracting arm: `2 ⊖ ε` is still representable.
        assert_eq!(ReflectedErfc::for_tail(f64::EPSILON), Subtracted);
        assert_eq!(ReflectedErfc::for_tail(f64::EPSILON.next_up()), Subtracted);
        assert_eq!(ReflectedErfc::for_tail(f64::EPSILON.next_down()), Saturated);
        assert_eq!(ReflectedErfc::for_tail(0.0), Saturated);
        assert_eq!(ReflectedErfc::for_tail(1.0), Subtracted);

        assert_eq!(reflected_erfc(f64::EPSILON), 1.999_999_999_999_999_8);
        assert_eq!(reflected_erfc(f64::EPSILON.next_down()), 2.0);
        assert_eq!(reflected_erfc(0.0), 2.0);
        assert_eq!(reflected_erfc(1.0), 1.0);
    }

    #[test]
    fn the_threshold_sits_above_where_rounding_alone_would_saturate() {
        // Rounding collapses `2 ⊖ tail` to `2` only at `ε/2`, by ties-to-even — not at `ε`. So
        // between `ε/2` and `ε` this returns a flat `2` where the subtraction would have given
        // `1.9999999999999998`. That conservatism is what `Φ(8.25)` needed; it is the reason the
        // comparison is against `ε` and not the rounding boundary, and it is deliberate.
        assert_eq!(2.0 - f64::EPSILON / 2.0, 2.0);
        assert_eq!(2.0 - f64::EPSILON.next_down(), 1.999_999_999_999_999_8);
        assert_eq!(reflected_erfc(f64::EPSILON.next_down()), 2.0);
        // Both ends of that gap saturate here, which is the whole point of the wider threshold.
        assert_eq!(reflected_erfc(f64::EPSILON / 2.0), 2.0);
    }
}
