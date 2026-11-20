//! The [`ScalarQuantity`] and [`LinearQuantity`] contracts.
//!
//! The predicate-and-rounding surface, and the linear structure underneath it. Mask-valued
//! throughout: these compare masks against [`Element::one`]/[`Element::zero`] rather than
//! branching on them, which is what lets an expression or tensor carrier be checked by the same
//! laws as an `f64` one.

use super::*;

/// The scalar helpers: selection, rounding and the predicates.
///
/// - `min`/`max` are commutative and idempotent, and each returns one of its arguments
/// - `clamp(x, lo, hi) == max(lo, min(x, hi))` — the definition
/// - `floor(x) ⪯ x ⪯ ceil(x)`, stated through `min`/`max` so no ordering bound is needed
/// - `floor`, `ceil`, `round` and `trunc` are idempotent
/// - `is_finite` holds for a finite sample, and `is_nan`/`is_infinite` do not
/// - `approx_eq` is reflexive and symmetric
pub fn scalar_laws<Q>(xs: &[Q], tolerance: &Q, what: &str)
where
    Q: ScalarQuantity + PartialEq + core::fmt::Debug,
    Mask<Q>: PartialEq + core::fmt::Debug,
{
    let t = <Mask<Q> as Element>::one();
    let f = <Mask<Q> as Element>::zero();

    for a in xs {
        // idempotence
        assert_eq!(a.clone().min(a.clone()), *a, "{what}: min(a, a) must be a");
        assert_eq!(a.clone().max(a.clone()), *a, "{what}: max(a, a) must be a");

        // rounding is idempotent — a rounded value is already round
        for (name, rounded) in [
            ("floor", a.clone().floor()),
            ("ceil", a.clone().ceil()),
            ("round", a.clone().round()),
            ("trunc", a.clone().trunc()),
        ] {
            let twice = match name {
                "floor" => rounded.clone().floor(),
                "ceil" => rounded.clone().ceil(),
                "round" => rounded.clone().round(),
                _ => rounded.clone().trunc(),
            };
            assert_eq!(twice, rounded, "{what}: {name} must be idempotent");
        }

        // floor(x) ⪯ x ⪯ ceil(x), without needing an order on Q
        assert_eq!(
            a.clone().floor().min(a.clone()),
            a.clone().floor(),
            "{what}: floor(x) must not exceed x"
        );
        assert_eq!(
            a.clone().ceil().max(a.clone()),
            a.clone().ceil(),
            "{what}: ceil(x) must not be below x"
        );

        // predicates, as masks
        assert_eq!(a.is_finite(), t, "{what}: a finite sample must be finite");
        assert_eq!(a.is_nan(), f, "{what}: a finite sample must not be NaN");
        assert_eq!(
            a.is_infinite(),
            f,
            "{what}: a finite sample must not be infinite"
        );

        // approximate equality is reflexive
        assert_eq!(
            a.approx_eq(a, tolerance),
            t,
            "{what}: approx_eq(a, a, tol) must hold"
        );

        for b in xs {
            // commutativity
            assert_eq!(
                a.clone().min(b.clone()),
                b.clone().min(a.clone()),
                "{what}: min must be commutative"
            );
            assert_eq!(
                a.clone().max(b.clone()),
                b.clone().max(a.clone()),
                "{what}: max must be commutative"
            );
            // selection returns one of its arguments
            let m = a.clone().min(b.clone());
            assert!(
                m == *a || m == *b,
                "{what}: min must return one of its arguments, got {m:?}"
            );
            // and it is symmetric
            assert_eq!(
                a.approx_eq(b, tolerance),
                b.approx_eq(a, tolerance),
                "{what}: approx_eq must be symmetric"
            );

            for c in xs {
                // clamp is defined by min and max
                assert_eq!(
                    a.clone()
                        .clamp(b.clone().min(c.clone()), b.clone().max(c.clone())),
                    a.clone()
                        .min(b.clone().max(c.clone()))
                        .max(b.clone().min(c.clone())),
                    "{what}: clamp(x, lo, hi) must equal max(lo, min(x, hi))"
                );
            }
        }
    }
}

/// [`scalar_laws`] over a carrier's own [`ConformanceSample`].
pub fn scalar_laws_for<Q>(what: &str)
where
    Q: ScalarQuantity + ConformanceSample + PartialEq + core::fmt::Debug,
    Mask<Q>: PartialEq + core::fmt::Debug,
{
    scalar_laws(&Q::sample(), &Q::tolerance(), what);
}

/// The displacement laws: a [`LinearQuantity`] behaves like a vector.
///
/// Exact, for any carrier:
///
/// - `neg` is an involution, and `abs(neg(a)) == abs(a)`
/// - `a.sub(b) == a.add(b.neg())` — the same operation, so the same rounding
/// - `add` is commutative (floating-point addition is commutative; it is *associativity* that
///   is not)
/// - `lerp(a, b, 0) == a` and `lerp(a, b, 1) == b` — the weights are exactly 1 and 0
/// - `scaled(1) == a` and `scale(one) == a`
///
/// Up to `tolerance`, because they are not exact when `Repr` is not:
///
/// - `a.add(b).sub(b) ≈ a` — addition and subtraction are inverses
/// - `add` is associative
///
/// Those two are the strongest laws here and also the two an inexact carrier cannot satisfy
/// exactly. `(-9.8 + 100.0) - 100.0` is `-9.799999999999997` in `f64`, which is not a contract
/// violation but a property of the representation — so they are checked through
/// [`ScalarQuantity::approx_eq`] with the tolerance you supply, and an exact carrier can pass a
/// tolerance of zero to get the strict law back.
pub fn linear_laws<Q>(xs: &[Q], tolerance: &Q, what: &str)
where
    Q: LinearQuantity + PartialEq + core::fmt::Debug,
    Mask<Q>: PartialEq + core::fmt::Debug,
{
    let t = <Mask<Q> as Element>::one();
    let one = <Q::Repr as Element>::one();
    let zero = <Q::Repr as Element>::zero();

    for a in xs {
        assert_eq!(a.clone().neg().neg(), *a, "{what}: −(−a) must be a");
        assert_eq!(
            a.clone().neg().abs(),
            a.clone().abs(),
            "{what}: |−a| must equal |a|"
        );
        assert_eq!(
            a.clone().scaled(1.0),
            *a,
            "{what}: scaling by the literal 1 must be the identity"
        );
        assert_eq!(
            a.clone().scale(one.clone()),
            *a,
            "{what}: scaling by ONE must be the identity"
        );

        for b in xs {
            // the inverse law, up to tolerance — see the note on `tolerance` above
            let round_trip = a.clone().add(b.clone()).sub(b.clone());
            assert_eq!(
                round_trip.approx_eq(a, tolerance),
                t,
                "{what}: (a + b) − b must be a, within tolerance (got {round_trip:?}, want {a:?})"
            );
            // commutativity
            assert_eq!(
                a.clone().add(b.clone()),
                b.clone().add(a.clone()),
                "{what}: a + b must equal b + a"
            );
            // subtraction is addition of the negation
            assert_eq!(
                a.clone().sub(b.clone()),
                a.clone().add(b.clone().neg()),
                "{what}: a − b must equal a + (−b)"
            );
            // interpolation hits its endpoints
            assert_eq!(
                a.clone().lerp(b.clone(), zero.clone()),
                *a,
                "{what}: lerp(a, b, 0) must be a"
            );
            assert_eq!(
                a.clone().lerp(b.clone(), one.clone()),
                *b,
                "{what}: lerp(a, b, 1) must be b"
            );

            for c in xs {
                let left = a.clone().add(b.clone()).add(c.clone());
                let right = a.clone().add(b.clone().add(c.clone()));
                assert_eq!(
                    left.approx_eq(&right, tolerance),
                    t,
                    "{what}: addition must be associative, within tolerance \
                     (got {left:?} vs {right:?})"
                );
            }
        }
    }
}
