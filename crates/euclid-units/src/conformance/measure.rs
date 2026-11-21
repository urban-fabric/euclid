//! The [`Measure`] contract: the additive core, the operator bridge, and scaling by a bare
//! representation.
//!
//! These are the laws every unit obeys regardless of what it measures — that `add` then `sub`
//! returns you where you started, that the operator forms agree with the method forms, and that
//! multiplying by a raw `Repr` distributes. A carrier that fails one of these is wrong about
//! arithmetic rather than about units.

use super::*;

/// The raw round-trip: `from_raw` and `raw` must be inverses.
///
/// The most basic thing a unit carrier can get wrong, and the one a mistaken field index or a
/// stray unit conversion inside an accessor breaks immediately. Everything else in this crate
/// is built on it, so it is checked separately from the tiers above.
pub fn measure_laws<Q>(xs: &[Q], what: &str)
where
    Q: Measure + PartialEq + core::fmt::Debug,
    Q::Repr: PartialEq + core::fmt::Debug,
{
    for x in xs {
        assert_eq!(
            &Q::from_raw(x.raw()),
            x,
            "{what}: from_raw(x.raw()) must be x"
        );
        let v = x.raw();
        assert_eq!(
            Q::from_raw(v.clone()).raw(),
            v,
            "{what}: raw(from_raw(v)) must be v"
        );
    }
}

/// [`measure_laws`] over a carrier's own [`ConformanceSample`].
pub fn measure_laws_for<Q>(what: &str)
where
    Q: Measure + ConformanceSample + PartialEq + core::fmt::Debug,
    Q::Repr: PartialEq + core::fmt::Debug,
{
    measure_laws(&Q::sample(), what);
}

/// The **operator bridge**: a carrier's `+`, `-`, `+=`, `-=`, `-` and `sum` must agree with the
/// trait methods they abbreviate.
///
/// # Why this is a separate suite
///
/// [`LinearQuantity`] supplies `add`, `sub` and `neg` as **default bodies**, so one body serves
/// every carrier in every consuming crate -- which is exactly what [`quantity_laws`] checks, and
/// checking it once is enough. The operators are the other half, and they are not: each carrier
/// writes `impl Add for Length`, `impl AddAssign for Length`, `impl Neg for Length` out by hand,
/// so there are as many implementations as there are units and nothing tied them to the trait.
///
/// Mutation testing measured the gap in `euclid-quantity`: roughly 300 surviving mutants across
/// `Add`, `Sub`, `Neg`, `AddAssign` and `SubAssign`, including every unit's `+=` being
/// replaceable with a body that does nothing. The shared default was well covered and its
/// sixty-one hand-written twins were not.
///
/// The operator traits are not bounds on [`LinearQuantity`] -- a carrier may host the tier
/// without overloading anything -- so this cannot live inside `quantity_laws` and takes its own
/// bounds instead. A carrier that *does* overload should run it.
///
/// `sum` is included because `xs.iter().sum()` is the one of these a caller is most likely to
/// reach for and the least likely to have written carefully: it must agree with folding `+` from
/// zero.
pub fn operator_bridge_laws<Q>(xs: &[Q], what: &str)
where
    Q: LinearQuantity
        + Copy
        + PartialEq
        + core::fmt::Debug
        + core::ops::Add<Output = Q>
        + core::ops::Sub<Output = Q>
        + core::ops::Neg<Output = Q>
        + core::ops::AddAssign
        + core::ops::SubAssign
        + core::iter::Sum<Q>,
{
    for &a in xs {
        assert_eq!(LinearQuantity::neg(a), -a, "{what}: neg is unary -");

        for &b in xs {
            assert_eq!(LinearQuantity::add(a, b), a + b, "{what}: add is +");
            assert_eq!(LinearQuantity::sub(a, b), a - b, "{what}: sub is -");

            let mut assigned = a;
            assigned += b;
            assert_eq!(assigned, a + b, "{what}: a += b agrees with a + b");

            let mut assigned = a;
            assigned -= b;
            assert_eq!(assigned, a - b, "{what}: a -= b agrees with a - b");
        }
    }

    // `sum` against an explicit fold, including the empty case, which must be zero rather than
    // a panic or a first element.
    let folded = xs
        .iter()
        .copied()
        .fold(Q::from_raw(<Q::Repr as Element>::zero()), |acc, x| acc + x);
    let summed: Q = xs.iter().copied().sum();
    assert_eq!(summed, folded, "{what}: sum agrees with a fold of +");

    let empty: Q = core::iter::empty().sum();
    assert_eq!(
        empty,
        Q::from_raw(<Q::Repr as Element>::zero()),
        "{what}: the empty sum is zero"
    );
}

/// [`operator_bridge_laws`] over a carrier's own [`ConformanceSample`].
pub fn operator_bridge_laws_for<Q>(what: &str)
where
    Q: LinearQuantity
        + ConformanceSample
        + Copy
        + PartialEq
        + core::fmt::Debug
        + core::ops::Add<Output = Q>
        + core::ops::Sub<Output = Q>
        + core::ops::Neg<Output = Q>
        + core::ops::AddAssign
        + core::ops::SubAssign
        + core::iter::Sum<Q>,
{
    operator_bridge_laws(&Q::sample(), what);
}

/// **Scalar scaling**: `a * k`, `k * a` and `a / k` must agree with the trait operations they
/// abbreviate.
///
/// The third face of the same gap [`operator_bridge_laws`] closes, and the largest one in
/// `euclid-quantity`: 118 surviving mutants across `Mul<Repr>`, `Div<Repr>` and the commuted
/// `Mul<Q> for Repr`. Every one of those impls is a one-line delegation to a shared helper, the
/// helper is covered, and the sixty-odd delegations were not.
///
/// # Multiplication and division are *not* symmetric here, and the law says so
///
/// `a * k` is exactly [`LinearQuantity::scale`], so that one is an equality.
///
/// `a / k` is **not** `scale(1/k)`, and the difference is deliberate rather than incidental: the
/// reciprocal rounds once before multiplying, so `a.scale(1.0 / k)` and `a / k` disagree in the
/// last bits. A carrier must divide the raw value directly. Asserting the wrong one of these
/// would pin a subtly less accurate implementation, so the law is stated against
/// [`Measure::from_raw`] of the divided raw value — which is the definition — rather than
/// against `scale`.
///
/// That also makes this law the thing standing between the current implementation and an
/// "obvious simplification" to `scale(1.0 / k)`.
pub fn scaling_laws<Q>(xs: &[Q], factors: &[Q::Repr], what: &str)
where
    Q: LinearQuantity
        + Copy
        + PartialEq
        + core::fmt::Debug
        + core::ops::Mul<Q::Repr, Output = Q>
        + core::ops::Div<Q::Repr, Output = Q>,
    Q::Repr: Copy + PartialEq + core::ops::Mul<Q, Output = Q>,
{
    let zero = <Q::Repr as Element>::zero();
    let one = <Q::Repr as Element>::one();

    for &a in xs {
        assert_eq!(a * one, a, "{what}: a * 1 = a");
        assert_eq!(a / one, a, "{what}: a / 1 = a");
        assert_eq!(
            a * zero,
            Q::from_raw(zero),
            "{what}: a * 0 is the zero quantity"
        );

        for &k in factors {
            assert_eq!(
                a * k,
                LinearQuantity::scale(a, k),
                "{what}: a * k is scale(a, k)"
            );
            assert_eq!(k * a, a * k, "{what}: scalar multiplication commutes");

            if k != zero {
                assert_eq!(
                    a / k,
                    Q::from_raw(a.raw() / k),
                    "{what}: a / k divides the raw value directly, not via a reciprocal"
                );
            }
        }
    }
}

/// [`scaling_laws`] over a carrier's own [`ConformanceSample`], with a fixed factor set.
///
/// The factors are the ones that matter for a scaling law rather than a wide sweep: the
/// identities, a sign change, a magnifier and a divisor whose reciprocal is inexact — `3`, whose
/// reciprocal is not representable, is what separates `a / k` from `a.scale(1/k)`.
pub fn scaling_laws_for<Q>(what: &str)
where
    Q: LinearQuantity
        + ConformanceSample
        + Copy
        + PartialEq
        + core::fmt::Debug
        + core::ops::Mul<Q::Repr, Output = Q>
        + core::ops::Div<Q::Repr, Output = Q>,
    Q::Repr: Copy + PartialEq + Real + core::ops::Mul<Q, Output = Q>,
{
    let one = <Q::Repr as Element>::one();
    let two = one + one;
    let three = two + one;
    let factors = [
        one,
        <Q::Repr as Element>::zero(),
        two,
        three,
        <Q::Repr as Element>::zero() - one,
        one / two,
    ];
    scaling_laws(&Q::sample(), &factors, what);
}
