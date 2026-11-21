//! The tiers that add inverses: ring, field, and Euclidean domain.
//!
//! Each step here buys an inverse and pays for it in generality — additive for a ring,
//! multiplicative for a field, division-with-remainder for a Euclidean domain. The laws are
//! correspondingly about what each new inverse must agree with, since an inverse that does not
//! round-trip is the failure they exist to catch.

use super::*;

/// The ring axioms a semiring lacks — additive inverses — plus the **operator bridge**.
///
/// The bridge is the law specific to this crate's design: a numeric carrier joins the
/// method-based trunk by making [`AdditiveMagma::add`] *be* `+`. If the two ever disagree, a
/// formula written against the trunk and one written with operators compute different things on
/// the same type, which is precisely the failure the library exists to prevent.
#[allow(
    clippy::eq_op,
    reason = "`a - a` is the law under test, not an accidental self-comparison"
)]
pub fn ring_laws<T>(xs: &[T], what: &str)
where
    T: Ring + core::fmt::Debug + Copy,
{
    for &a in xs {
        assert_eq!(a + (-a), T::ZERO, "{what}: a + (−a) = 0");
        assert_eq!(-(-a), a, "{what}: −(−a) = a");
        assert_eq!(a - a, T::ZERO, "{what}: a − a = 0");

        for &b in xs {
            assert_eq!(AdditiveMagma::add(a, b), a + b, "{what}: ⊕ is +");
            assert_eq!(MultiplicativeMagma::mul(a, b), a * b, "{what}: ⊗ is *");
            assert_eq!(a - b, a + (-b), "{what}: a − b = a + (−b)");
            assert_eq!(-(a * b), (-a) * b, "{what}: −(ab) = (−a)b");

            // The **compound-assignment bridge**, which is the operator bridge's other half and
            // was missing until mutation testing noticed. `Ring` requires `AddAssign`,
            // `SubAssign` and `MulAssign`, and nothing checked that any of them agrees with the
            // binary operator it abbreviates — so `a += b` could be written to do nothing at all
            // and every carrier in the fleet still conformed.
            //
            // It is a separate impl block from `Add`, which is exactly why it drifts: the usual
            // spelling is `*self = *self + rhs`, and the usual mistake is a typo in the operator
            // or an operand order that only matters for a non-commutative carrier.
            let mut assigned = a;
            assigned += b;
            assert_eq!(assigned, a + b, "{what}: a += b agrees with a + b");

            let mut assigned = a;
            assigned -= b;
            assert_eq!(assigned, a - b, "{what}: a −= b agrees with a − b");

            let mut assigned = a;
            assigned *= b;
            assert_eq!(assigned, a * b, "{what}: a ×= b agrees with a × b");
        }
    }
}

/// [`ring_laws`] over a carrier's own [`ConformanceSample`].
pub fn ring_laws_for<T>(what: &str)
where
    T: Ring + ConformanceSample + core::fmt::Debug + Copy,
{
    ring_laws(&T::sample(), what);
}

/// The field axiom: every nonzero element is invertible, and division is multiplication by the
/// inverse.
///
/// [`Field`] exposes invertibility through `Div` rather than an `inverse` method, so the inverse
/// is obtained as `ONE / a` — which is also what makes the division law non-trivial. A field is
/// an integral domain, so the absence of zero divisors in the sample is checked too.
#[allow(
    clippy::eq_op,
    reason = "`a / a` is the law under test, not an accidental self-comparison"
)]
pub fn field_laws<T>(xs: &[T], what: &str)
where
    T: Field + core::fmt::Debug + Copy,
{
    for &a in xs {
        if a == T::ZERO {
            continue; // not invertible, and division by it is undefined
        }
        let inv = T::ONE / a;
        assert_eq!(a * inv, T::ONE, "{what}: a · a⁻¹ = 1");
        assert_eq!(a / a, T::ONE, "{what}: a / a = 1");

        for &b in xs {
            assert_eq!(b / a, b * inv, "{what}: b / a = b · a⁻¹");

            // The division half of the compound-assignment bridge; see `ring_laws` for why this
            // whole family went unchecked. `Field` requires `DivAssign`, and `/=` could be
            // written to do nothing.
            let mut assigned = b;
            assigned /= a;
            assert_eq!(assigned, b / a, "{what}: b /= a agrees with b / a");
        }
    }
    for &a in xs {
        for &b in xs {
            if a != T::ZERO && b != T::ZERO {
                assert_ne!(a * b, T::ZERO, "{what}: no zero divisors");
            }
        }
    }
}

/// [`field_laws`] over a carrier's own [`ConformanceSample`].
pub fn field_laws_for<T>(what: &str)
where
    T: Field + ConformanceSample + core::fmt::Debug + Copy,
{
    field_laws(&T::sample(), what);
}

/// `div_rem` must reconstruct its input: `a = (a / b) · b + (a mod b)`.
///
/// This is the defining property of a Euclidean domain and the one a caller actually relies on.
/// It holds regardless of how a carrier rounds the quotient, which is why it is worth more than
/// asserting any particular quotient — two carriers may round differently and both be correct.
pub fn euclidean_laws<T>(xs: &[T], what: &str)
where
    T: EuclideanDomain + core::fmt::Debug + Copy,
{
    for &a in xs {
        for &b in xs {
            if b == T::ZERO {
                continue;
            }
            let (q, r) = a.div_rem(b);
            assert_eq!(q * b + r, a, "{what}: a = (a/b)·b + (a mod b)");
        }
    }
}

/// [`euclidean_laws`] over a carrier's own [`ConformanceSample`].
pub fn euclidean_laws_for<T>(what: &str)
where
    T: EuclideanDomain + ConformanceSample + core::fmt::Debug + Copy,
{
    euclidean_laws(&T::sample(), what);
}
