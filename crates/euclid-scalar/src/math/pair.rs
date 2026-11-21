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

impl<T> Slope<T> {
    /// Carry a value and its derivative together.
    pub(crate) fn new(value: T, derivative: T) -> Self {
        Self { value, derivative }
    }

    /// The pair as a tuple, for the assertions that want both components on one line.
    ///
    /// **Test-only, and that is the point.** This existed for the kernels that destructured a
    /// pair the moment they received one; there are none left — every route through Airy and the
    /// Bessel marches now carries `Slope` and `Solutions` end to end, so the last `.0`/`.1` in
    /// the library is gone. Keeping the method compiled under `cfg(test)` means a kernel cannot
    /// quietly reintroduce one without the reviewer seeing an attribute change.
    #[cfg(test)]
    pub(crate) fn into_parts(self) -> (T, T) {
        (self.value, self.derivative)
    }
}

impl<T: Copy> Slope<T> {
    /// Both components the same, for the saturating limits where a pair is wholly `NaN` or zero.
    pub(crate) fn uniform(both: T) -> Self {
        Self::new(both, both)
    }
}

impl<T: HostReal> Slope<T> {
    /// Scale both components by the same factor.
    ///
    /// The common shape when an equation is solved in a scaled coordinate and recovered: the
    /// factor applies to the derivative exactly as it does to the value, because the scaling is
    /// in the dependent variable rather than the independent one.
    pub(crate) fn scaled_by(self, factor: T) -> Self {
        Self::new(self.value * factor, self.derivative * factor)
    }
}

/// Differentiation is linear, so a sum of two solutions carries the sum of their derivatives.
///
/// That is what makes the operator worth having rather than two componentwise expressions: the
/// Airy pair is a fixed linear combination of the two Maclaurin solutions, `Ai = c₁f − c₂g` and
/// `Bi = √3(c₁f + c₂g)`, with the *same* combination applied to the derivatives. Written out by
/// component that is four expressions in which `value` and `derivative` can be interchanged by
/// accident; written as `cf - cg` it is the formula from the paper and the transposition cannot
/// be spelled.
impl<T: HostReal> core::ops::Add for Slope<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new(self.value + other.value, self.derivative + other.derivative)
    }
}

/// The subtraction half of [`Add`](core::ops::Add)'s argument — and the one that cancels, which
/// is why `Ai` leaves the series so much earlier than `Bi`.
impl<T: HostReal> core::ops::Sub for Slope<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::new(self.value - other.value, self.derivative - other.derivative)
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

/// Which of the two solutions a caller asked for.
///
/// The selector that makes [`Solutions`] worth having at an entry point: `Ai` and `Ai′` are the
/// same traversal of the axis as `Bi` and `Bi′`, differing only in which half of the result is
/// read. Naming the half lets one private function serve all eight public Airy entry points,
/// and — because it is matched rather than compared — a third kind of solution would be a
/// compile error at every site that has to choose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// The solution regular at the origin: `J` for Bessel, `Ai` for Airy.
    First,
    /// The second solution: `Y` for Bessel, `Bi` for Airy.
    Second,
}

impl<T> Solutions<T> {
    /// Carry both solutions together.
    pub(crate) fn new(first_kind: T, second_kind: T) -> Self {
        Self {
            first_kind,
            second_kind,
        }
    }

    /// The half the caller asked for.
    pub(crate) fn of_kind(self, kind: Kind) -> T {
        match kind {
            Kind::First => self.first_kind,
            Kind::Second => self.second_kind,
        }
    }
}

impl<T: Copy> Solutions<T> {
    /// Both solutions the same, for the saturating limits where all four values coincide.
    pub(crate) fn uniform(both: T) -> Self {
        Self::new(both, both)
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
        // Uniform is the limit case, where there is nothing to transpose.
        assert!(Slope::uniform(f64::NAN).value.is_nan());
        assert_eq!(Slope::uniform(0.0_f32).into_parts(), (0.0, 0.0));
    }

    #[test]
    fn the_operators_combine_both_components_at_once() {
        // Componentwise, and in the same order — which is the whole content of `Ai = c₁f − c₂g`
        // holding for the derivative as well as for the value.
        let f = Slope::new(5.0_f64, 7.0);
        let g = Slope::new(2.0_f64, 3.0);
        assert_eq!((f + g).into_parts(), (7.0, 10.0));
        assert_eq!((f - g).into_parts(), (3.0, 4.0));
        // Not commutative in the way a transposition would make it: `f - g` is not `g - f`, so a
        // swapped operand is a sign error rather than a silent equality.
        assert_eq!((g - f).into_parts(), (-3.0, -4.0));
        // The scaling distributes over the sum, which is why `combine` may factor `√3` out.
        let scaled = (f + g).scaled_by(3.0);
        assert_eq!(scaled.into_parts(), (21.0, 30.0));
        assert_eq!(
            (f.scaled_by(2.0) - g.scaled_by(2.0)).into_parts(),
            (6.0, 8.0)
        );
        assert_eq!(
            (Slope::new(1.0_f32, 2.0) + Slope::new(0.5, 0.25)).into_parts(),
            (1.5, 2.25)
        );
    }

    #[test]
    fn selecting_a_kind_reads_the_half_it_names() {
        let pair = Solutions::new(Slope::new(1.0_f64, 2.0), Slope::new(3.0, 4.0));
        assert_eq!(pair.of_kind(Kind::First).into_parts(), (1.0, 2.0));
        assert_eq!(pair.of_kind(Kind::Second).into_parts(), (3.0, 4.0));
        // The four values of an Airy quadruple, reached by name rather than by position — which
        // is the arrangement this type exists to replace.
        assert_eq!(pair.first_kind.value, 1.0);
        assert_eq!(pair.first_kind.derivative, 2.0);
        assert_eq!(pair.second_kind.value, 3.0);
        assert_eq!(pair.second_kind.derivative, 4.0);
        // Uniform at both levels: the `NaN` and zero quadruples the kernels return off-domain.
        let zeros = Solutions::uniform(Slope::uniform(0.0_f64));
        assert_eq!(zeros.first_kind.into_parts(), (0.0, 0.0));
        assert_eq!(zeros.second_kind.into_parts(), (0.0, 0.0));
        // A counted pair of sums, which is the other composition in use.
        let counted = Converged::new(Solutions::new(1.0_f64, 2.0), 7);
        assert_eq!(counted.terms, 7);
        assert_eq!(counted.value.of_kind(Kind::Second), 2.0);
    }
}
