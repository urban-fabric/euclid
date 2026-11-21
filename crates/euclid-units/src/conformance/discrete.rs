//! The [`DiscreteQuantity`] and [`Quantity`] contracts.
//!
//! The smallest two surfaces in the family, kept together because a discrete quantity is a
//! quantity with a step: the laws here are mostly about that step being respected by rounding
//! and by conversion.

use super::*;

/// The counting laws for an integer-backed [`DiscreteQuantity`].
///
/// - `add` is commutative and associative
/// - adding the carrier's zero is the identity
/// - `saturating_sub(a, a)` is zero, and subtracting zero is the identity
/// - subtraction saturates rather than wrapping: `saturating_sub(zero, a)` is zero
///
/// That last law is the one worth having. The carrier has no negatives, so a naive `-` would
/// wrap to a huge value, and the difference between saturating and wrapping is invisible until
/// a count goes negative in production.
pub fn discrete_laws<Q>(xs: &[Q], what: &str)
where
    Q: DiscreteQuantity + PartialEq + core::fmt::Debug + Clone,
    Q::Repr: euclid_scalar::Semiring,
{
    let zero = Q::from_raw(<Q::Repr as Element>::zero());

    for a in xs {
        assert_eq!(a.clone().add(zero.clone()), *a, "{what}: a + 0 must be a");
        assert_eq!(
            a.clone().saturating_sub(zero.clone()),
            *a,
            "{what}: a − 0 must be a"
        );
        assert_eq!(
            a.clone().saturating_sub(a.clone()),
            zero,
            "{what}: a − a must be 0"
        );
        assert_eq!(
            zero.clone().saturating_sub(a.clone()),
            zero,
            "{what}: 0 − a must saturate to 0, not wrap"
        );

        for b in xs {
            assert_eq!(
                a.clone().add(b.clone()),
                b.clone().add(a.clone()),
                "{what}: addition must be commutative"
            );
            for c in xs {
                assert_eq!(
                    a.clone().add(b.clone()).add(c.clone()),
                    a.clone().add(b.clone().add(c.clone())),
                    "{what}: addition must be associative"
                );
            }
        }
    }
}

/// [`discrete_laws`] over a carrier's own [`ConformanceSample`].
pub fn discrete_laws_for<Q>(what: &str)
where
    Q: DiscreteQuantity + ConformanceSample + PartialEq + core::fmt::Debug + Clone,
    Q::Repr: euclid_scalar::Semiring,
{
    discrete_laws(&Q::sample(), what);
}

/// Every law that applies to a `Real`-backed quantity, in one call.
///
/// Convenience for the common case: a carrier that is a [`Quantity`] and a [`LinearQuantity`]
/// passes through [`measure_laws`], [`scalar_laws`] and [`linear_laws`]. An absolute scalar such
/// as a temperature is not linear, so it calls the first two itself.
pub fn quantity_laws<Q>(xs: &[Q], tolerance: &Q, what: &str)
where
    Q: Quantity + LinearQuantity + PartialEq + core::fmt::Debug,
    Q::Repr: PartialEq + core::fmt::Debug + Real,
    Mask<Q>: PartialEq + core::fmt::Debug,
{
    measure_laws(xs, what);
    scalar_laws(xs, tolerance, what);
    linear_laws(xs, tolerance, what);
}

/// [`quantity_laws`] over a carrier's own [`ConformanceSample`].
pub fn quantity_laws_for<Q>(what: &str)
where
    Q: Quantity + LinearQuantity + ConformanceSample + PartialEq + core::fmt::Debug,
    Q::Repr: PartialEq + core::fmt::Debug + Real,
    Mask<Q>: PartialEq + core::fmt::Debug,
{
    quantity_laws(&Q::sample(), &Q::tolerance(), what);
}
