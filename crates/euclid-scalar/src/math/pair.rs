//! A function and its derivative, carried together.
//!
//! The kernels that solve a second-order equation — Airy, the Bessel marches — produce both at
//! once and pass them around as a unit. They used to do that as an anonymous `(T, T)`, which
//! reads fine where it is destructured and badly where it is not: `pair(x, Half::First).0` and
//! `.1` say nothing about which of the two is the value and which the derivative, and the two
//! have the same type, so transposing them is a silent error rather than one the compiler sees.
//!
//! Naming the fields costs nothing at runtime — a two-field struct of the carrier type lowers to
//! the same registers the tuple did — and makes the transposition impossible to write by
//! accident.

use crate::HostReal;

/// A value and its first derivative at the same argument.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Slope<T> {
    /// The function's value.
    pub(crate) value: T,
    /// Its first derivative.
    pub(crate) derivative: T,
}

impl<T: HostReal> Slope<T> {
    /// Carry a value and its derivative together.
    pub(crate) fn new(value: T, derivative: T) -> Self {
        Self { value, derivative }
    }

    /// Scale both components by the same factor.
    ///
    /// The common shape when an equation is solved in a scaled coordinate and recovered: the
    /// factor applies to the derivative exactly as it does to the value, because the scaling is
    /// in the dependent variable rather than the independent one.
    pub(crate) fn scaled_by(self, factor: T) -> Self {
        Self::new(self.value * factor, self.derivative * factor)
    }

    /// The pair as a tuple, for the call sites that destructure it immediately.
    pub(crate) fn into_parts(self) -> (T, T) {
        (self.value, self.derivative)
    }
}

/// The two independent solutions of a second-order equation, at one argument.
///
/// `J` and `Y` for Bessel, `Ai` and `Bi` for Airy. Like [`Slope`], the two have the same type
/// and no intrinsic order, so an anonymous pair makes transposing them a silent error — and
/// unlike `Slope`, the call sites here mostly want *one* of the two, which is where `.0` and
/// `.1` are at their least readable: `hankel(0.0, x).1` says nothing, `hankel(0.0, x).second_kind`
/// says which function is being asked for.
///
/// It composes with `Slope`: a pair of solutions each carrying its derivative — the Airy
/// quadruple — is a `Solutions<Slope<T>>` rather than a fourth distinct shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Solutions<T> {
    /// The first-kind solution, regular at the origin: `J` for Bessel, `Ai` for Airy.
    pub(crate) first_kind: T,
    /// The second-kind solution: `Y` for Bessel, `Bi` for Airy.
    pub(crate) second_kind: T,
}

impl<T> Solutions<T> {
    /// Carry both solutions together.
    pub(crate) fn new(first_kind: T, second_kind: T) -> Self {
        Self {
            first_kind,
            second_kind,
        }
    }
}

/// A series' value together with the number of terms it took to reach it.
///
/// # Why a count travels with a value
///
/// A truncated series' **cost** is invisible to every accuracy assertion in this tree: the
/// stopping rule decides how many terms are summed, and because the tolerances here sit below
/// the format's resolution, stopping one term early or late moves no result. Mutation testing
/// duly finds those comparisons unkillable — widening `t < sum·ε` to `<=`, or to `==` so the
/// break never fires at all, leaves every value identical and silently runs the loop to its cap.
///
/// The term count is the deterministic observable that *does* separate them, so the kernels that
/// have a stopping rule worth protecting return it. The public entry point discards it; nothing
/// branches on it. `Slope` exists for the same reason one step up — a quantity that was being
/// passed around anonymously and so could not be asserted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Converged<T> {
    /// The summed value.
    pub(crate) value: T,
    /// Terms the loop entered before its stopping rule fired, or the cap if it never did.
    pub(crate) terms: u32,
}

impl<T> Converged<T> {
    /// Carry a value and the term count that produced it.
    ///
    /// Unbounded in `T`: a counted result is sometimes a *pair* of sums rather than one number —
    /// the Hankel expansion accumulates `P` and `Q` together — so this wraps whatever the loop
    /// accumulated rather than requiring it to be a scalar.
    pub(crate) fn new(value: T, terms: u32) -> Self {
        Self { value, terms }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fields_survive_a_round_trip_and_a_scaling() {
        let slope = Slope::new(2.0_f64, -3.0);
        assert_eq!(slope.value, 2.0);
        assert_eq!(slope.derivative, -3.0);
        assert_eq!(slope.into_parts(), (2.0, -3.0));
        // Scaling applies to both components, and to neither in the other's place — which is the
        // transposition the named fields exist to prevent.
        let scaled = slope.scaled_by(0.5);
        assert_eq!(scaled.value, 1.0);
        assert_eq!(scaled.derivative, -1.5);
        assert_eq!(
            Slope::new(1.0_f64, 2.0).scaled_by(0.0).into_parts(),
            (0.0, 0.0)
        );
        // Copy, so a pair can be read twice without a move.
        let twice = (slope.value, slope.value);
        assert_eq!(twice, (2.0, 2.0));
        assert_eq!(
            Slope::new(1.0_f32, 2.0).scaled_by(2.0).into_parts(),
            (2.0, 4.0)
        );
    }
}
