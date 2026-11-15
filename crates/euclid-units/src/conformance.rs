//! Conformance suites — the quantity contracts as functions you can check a carrier against.
//!
//! This crate ships no concrete unit, so every one of these traits is implemented by someone
//! else. A carrier that wraps the wrong field, loses the sign in `neg`, or whose `clamp`
//! disagrees with `min`/`max` compiles and type-checks exactly like a correct one, and then
//! quietly computes different answers from every other carrier in the fleet.
//!
//! # Mask-valued, not `bool`-valued
//!
//! [`ScalarQuantity`]'s predicates return a [`Mask`], not a host `bool`, so these suites compare
//! masks against [`Element::one`]/[`Element::zero`] rather than branching on them. That is what
//! lets the same laws check an expression or tensor carrier as well as an `f64` one — nothing
//! here needs a host boolean, and a deferred carrier is not excluded.
//!
//! # Laws are relations, not reference values
//!
//! Nothing below asserts a computed number. `clamp(x, lo, hi) == max(lo, min(x, hi))` is the
//! *definition* of clamping, and `a.add(b).sub(b) == a` is what makes a quantity linear — both
//! hold for any correct carrier regardless of representation, and neither can be mistyped into
//! passing. Where a law would need an ordering (`floor(x) <= x`) it is restated through
//! `min`/`max`, which the contract already provides, so no extra bound is needed.
//!
//! # Availability
//!
//! Gated behind the `conformance` feature, so a consumer that only writes formulas pays nothing
//! for it, and always compiled under `cfg(test)` so this crate's own suites exercise the same
//! code a carrier author calls.
//!
//! These docs are the module's own (`//!`), with no outer `///` on the `mod` declaration: the
//! two forms are concatenated, and an outer comment resolves intra-doc links in the *parent's*
//! scope, so a link to a function in here would silently fail from there.
//!
//! # Samples
//!
//! You supply them, and they should be **finite** — the predicate laws assert finiteness
//! holds, which is a property of your sample rather than of the contract. Four or five values
//! including a zero and a negative is plenty; the three-operand laws run over every triple.

use euclid_numeric::{Element, Real};

use crate::{DiscreteQuantity, LinearQuantity, Mask, Measure, Quantity, ScalarQuantity};

/// A carrier's own curated sample, declared once.
///
/// euclid's answer to proptest's `Arbitrary`, with the generator replaced by a fixed list. A
/// carrier says which values stress *it*, and every suite in this module then draws from the
/// same place rather than each call site inventing a sample and getting it slightly wrong.
///
/// Fixed rather than generated on purpose: a generator buys input breadth at the cost of
/// reproducibility, turning a failure into a seed to recover rather than a value in the source.
/// Keep samples small — the multi-operand laws run over every pair or triple, so `n` values
/// cost `n²` or `n³` cases per law.
pub trait ConformanceSample: Sized {
    /// The values this carrier should be checked over. Must be **finite** — the predicate laws
    /// assert finiteness holds.
    fn sample() -> Vec<Self>;

    /// The tolerance for the two inexact linear laws, as a quantity in this unit.
    ///
    /// The add/sub inverse and associativity are not exact when `Repr` is not — subtracting
    /// `100.0` from `-9.8 + 100.0` gives `-9.799999999999997`, not `-9.8`. An exact carrier can
    /// return its zero to recover the strict reading.
    fn tolerance() -> Self;
}

/// [`measure_laws`] over a carrier's own [`ConformanceSample`].
pub fn measure_laws_for<Q>(what: &str)
where
    Q: Measure + ConformanceSample + PartialEq + core::fmt::Debug,
    Q::Repr: PartialEq + core::fmt::Debug,
{
    measure_laws(&Q::sample(), what);
}

/// [`scalar_laws`] over a carrier's own [`ConformanceSample`].
pub fn scalar_laws_for<Q>(what: &str)
where
    Q: ScalarQuantity + ConformanceSample + PartialEq + core::fmt::Debug,
    Mask<Q>: PartialEq + core::fmt::Debug,
{
    scalar_laws(&Q::sample(), &Q::tolerance(), what);
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

/// [`discrete_laws`] over a carrier's own [`ConformanceSample`].
pub fn discrete_laws_for<Q>(what: &str)
where
    Q: DiscreteQuantity + ConformanceSample + PartialEq + core::fmt::Debug + Clone,
    Q::Repr: euclid_numeric::Semiring,
{
    discrete_laws(&Q::sample(), what);
}

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
    Q::Repr: euclid_numeric::Semiring,
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

/// The same laws over **generated** carriers, for a carrier author who wants the search rather
/// than a fixed five-value sample.
///
/// # What generation reaches that a fixed sample does not
///
/// The tier laws themselves are the kind a fixed sample serves well — they either hold
/// structurally or fail on the first awkward triple. The reason to generate anyway is the
/// **default method bodies**.
///
/// [`ScalarQuantity`] and [`LinearQuantity`] supply `min`, `max`, `clamp`, `floor`, `ceil`,
/// `round`, `trunc`, `approx_eq`, `add`, `sub`, `neg` and `scale` as defaults, so one body serves
/// every carrier in every consuming crate — each one a single point of failure. Those bodies
/// branch on *orderings* and *signs*, which is a combinatorial space rather than a structural
/// one:
///
/// - `clamp(x, lo, hi)` has six orderings of its three arguments, and five fixed values in one
///   fixed order reach a handful of them;
/// - `floor`, `ceil` and `trunc` agree on positives and differ on negatives, so a mostly-positive
///   sample would pass a `trunc` written as `floor`;
/// - `round` has to choose a direction at exactly `±0.5`.
///
/// The generator draws signs and half-integers deliberately for those three reasons, and the
/// coverage floors require that mixed-sign samples actually appeared rather than hoping they did.
///
/// # No new trait method was needed
///
/// The carrier is built through [`Measure::from_raw`], which the contract already requires. The
/// `Repr = f64` bound is the one restriction: `Repr` is any [`euclid_numeric::Element`],
/// and euclid cannot generate an arbitrary one — an integer-backed carrier should run the fixed
/// suite, or pass its own values to the slice-taking laws above.
#[cfg(any(feature = "conformance", test))]
pub mod generated {
    use euclid_numeric::Real;
    use euclid_stability::{Config, Gen, check, classify, generate};

    use super::{Mask, linear_laws, measure_laws, quantity_laws, scalar_laws};
    use crate::{LinearQuantity, Measure, Quantity, ScalarQuantity};

    /// A raw carrier value, weighted toward what the default bodies branch on.
    ///
    /// The half-integers and the small negatives are not decoration: they are the inputs that
    /// separate `round` from a truncation and `trunc` from a `floor`.
    fn raw() -> Gen<f64> {
        generate::weighted(vec![
            (2, generate::just(0.0)),
            (2, generate::just(0.5)),
            (2, generate::just(-0.5)),
            (2, generate::just(-1.5)),
            (2, generate::just(1.0)),
            (12, generate::float::magnitude(-10, 20)),
        ])
    }

    /// Three is the minimum that exercises a three-operand law like `clamp`; six matches what a
    /// fixed sample typically uses, and the laws are `O(n³)`, so wider buys repetition.
    fn carriers<Q: Measure<Repr = f64> + 'static>() -> Gen<Vec<Q>> {
        generate::vec(raw().map(Q::from_raw), 3, 6)
    }

    fn signs(xs: &[f64]) -> &'static str {
        match (xs.iter().any(|v| *v < 0.0), xs.iter().any(|v| *v > 0.0)) {
            (true, true) => "mixed signs",
            (true, false) => "all non-positive",
            (false, true) => "all non-negative",
            (false, false) => "all zero",
        }
    }

    fn settings() -> Config {
        Config::default()
            .cases(400)
            .require("mixed signs", 1.0)
            .require("all non-negative", 1.0)
    }

    /// [`measure_laws`] and [`scalar_laws`] over generated carriers.
    ///
    /// The pair a carrier that stops short of linearity can run — an affine scale such as a
    /// temperature, where addition is not defined and [`LinearQuantity`] is deliberately
    /// withheld.
    pub fn measure_and_scalar_laws_generated<Q>(tolerance: &Q, what: &str)
    where
        Q: Quantity + ScalarQuantity + Measure<Repr = f64> + PartialEq + core::fmt::Debug + 'static,
        Q::Repr: PartialEq + core::fmt::Debug + Real,
        Mask<Q>: PartialEq + core::fmt::Debug,
    {
        check(what, &settings(), &carriers::<Q>(), |xs| {
            classify!(signs(&xs.iter().map(Measure::raw).collect::<Vec<_>>()));
            measure_laws(&xs, what);
            scalar_laws(&xs, tolerance, what);
            Ok(())
        });
    }

    /// [`quantity_laws`] — the aggregate — over generated carriers, for a linear carrier.
    pub fn quantity_laws_generated<Q>(tolerance: &Q, what: &str)
    where
        Q: Quantity + LinearQuantity + Measure<Repr = f64> + PartialEq + core::fmt::Debug + 'static,
        Q::Repr: PartialEq + core::fmt::Debug + Real,
        Mask<Q>: PartialEq + core::fmt::Debug,
    {
        check(what, &settings(), &carriers::<Q>(), |xs| {
            classify!(signs(&xs.iter().map(Measure::raw).collect::<Vec<_>>()));
            quantity_laws(&xs, tolerance, what);
            Ok(())
        });
    }

    /// [`linear_laws`] alone, for a carrier whose scalar half is already covered.
    pub fn linear_laws_generated<Q>(tolerance: &Q, what: &str)
    where
        Q: Quantity + LinearQuantity + Measure<Repr = f64> + PartialEq + core::fmt::Debug + 'static,
        Q::Repr: PartialEq + core::fmt::Debug + Real,
        Mask<Q>: PartialEq + core::fmt::Debug,
    {
        check(what, &settings(), &carriers::<Q>(), |xs| {
            classify!(signs(&xs.iter().map(Measure::raw).collect::<Vec<_>>()));
            linear_laws(&xs, tolerance, what);
            Ok(())
        });
    }
}
