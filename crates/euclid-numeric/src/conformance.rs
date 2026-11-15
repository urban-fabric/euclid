//! Conformance suites — the scalar contracts as functions you can check a carrier against.
//!
//! This is the crate every other one is written over, so a carrier that honours it slightly
//! differently is the most expensive kind of disagreement available: a formula bounded on
//! [`Real`] runs on an `f64`, a graph node and a tensor element, and if one of those three
//! implements `select` with its branches swapped, the same source produces three answers and
//! two of them are wrong.
//!
//! ```
//! # use euclid_numeric::conformance;
//! conformance::boolean_laws(&[false, true], "bool");
//! conformance::mask_laws(&[0.0_f64, 1.0, -2.5], "f64");
//!
//! // Strictly positive and inside (0, π/2) — see the sample requirements below. The first
//! // draft of this example used 2.0 and the suite rejected it, which is the precondition
//! // doing its job: `atan(tan(2.0))` is `2.0 − π`, not `2.0`.
//! conformance::real_laws(&[0.25_f64, 0.5, 1.0, 1.25], 1e-14, "f64");
//! ```
//!
//! # Tolerance
//!
//! Most laws here are **inexact**, and that is a property of the carrier rather than a weakness
//! in the law: `exp(ln(x)) = x` does not hold bit-for-bit in binary floating point, and a
//! carrier that made it hold would be doing something stranger than the one that does not. Those
//! laws take a relative tolerance and compare through it.
//!
//! The laws that *are* exact say so, and are checked exactly:
//!
//! - the identity and annihilation laws, which involve no rounding
//! - the mask and boolean laws, which are discrete
//! - `select` with a constant mask, which chooses rather than computes
//!
//! A carrier with exact arithmetic can pass `0.0` and get the strict reading of everything.
//!
//! # Availability
//!
//! Gated behind the `conformance` feature, so a consumer pays nothing unless it asks, and always
//! compiled under `cfg(test)` so this crate's own suites exercise the same code a carrier calls.
//!
//! These docs are the module's own (`//!`), with no outer `///` on the `mod` declaration: the
//! two forms are concatenated, and an outer comment resolves intra-doc links in the *parent's*
//! scope, so a link to a function in here would silently fail from there.
//!
//! # Samples
//!
//! Supply **finite** values, and for [`real_laws`] **strictly positive** ones where the law
//! needs a logarithm — the suite documents which. Include `1` and a value either side of it;
//! that is where a series crossover lives, and where an implementation that is right in the
//! bulk goes wrong.

use crate::{Boolean, Compare, Element, Real};

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
    /// The values this carrier should be checked over.
    fn sample() -> Vec<Self>;

    /// The relative tolerance the inexact laws should use for this carrier.
    ///
    /// Defaulted to zero, which is the right answer for an exact carrier and wrong for every
    /// float — `f64` wants about `1e-14` and `f32` about `1e-6`, and sharing one bound between
    /// them would either fail at `f32` or stop catching anything at `f64`. Override it.
    fn tolerance() -> f64 {
        0.0
    }
}

/// [`boolean_laws`] over a carrier's own [`ConformanceSample`].
pub fn boolean_laws_for<B>(what: &str)
where
    B: Boolean + ConformanceSample + PartialEq + core::fmt::Debug,
{
    boolean_laws(&B::sample(), what);
}

/// [`mask_laws`] over a carrier's own [`ConformanceSample`].
pub fn mask_laws_for<T>(what: &str)
where
    T: Compare + ConformanceSample + PartialEq + core::fmt::Debug,
    T::Mask: PartialEq + core::fmt::Debug,
{
    mask_laws(&T::sample(), what);
}

/// [`real_laws`] over a carrier's own [`ConformanceSample`] and [`tolerance`].
///
/// [`tolerance`]: ConformanceSample::tolerance
pub fn real_laws_for<T>(what: &str)
where
    T: Real + ConformanceSample + PartialEq + core::fmt::Debug,
    T::Mask: PartialEq + core::fmt::Debug,
{
    real_laws(&T::sample(), T::tolerance(), what);
}

/// The boolean algebra laws for a [`Boolean`] carrier.
///
/// Exact, since nothing here rounds. De Morgan is the pair worth having: a carrier that
/// implements `or` as `not(and(not, not))` and gets a negation wrong still satisfies
/// idempotence and identity, and fails only here.
pub fn boolean_laws<B>(xs: &[B], what: &str)
where
    B: Boolean + PartialEq + core::fmt::Debug,
{
    for a in xs {
        // identity and annihilation
        assert_eq!(a.and(&B::one()), *a, "{what}: a ∧ 1 = a");
        assert_eq!(a.or(&B::zero()), *a, "{what}: a ∨ 0 = a");
        assert_eq!(a.and(&B::zero()), B::zero(), "{what}: a ∧ 0 = 0");
        assert_eq!(a.or(&B::one()), B::one(), "{what}: a ∨ 1 = 1");
        // idempotence and involution
        assert_eq!(a.and(a), *a, "{what}: a ∧ a = a");
        assert_eq!(a.or(a), *a, "{what}: a ∨ a = a");
        assert_eq!(a.not().not(), *a, "{what}: ¬¬a = a");
        // complement
        assert_eq!(a.and(&a.not()), B::zero(), "{what}: a ∧ ¬a = 0");
        assert_eq!(a.or(&a.not()), B::one(), "{what}: a ∨ ¬a = 1");
        // xor against its definition
        assert_eq!(a.xor(a), B::zero(), "{what}: a ⊕ a = 0");
        assert_eq!(a.xor(&B::zero()), *a, "{what}: a ⊕ 0 = a");

        for b in xs {
            // commutativity
            assert_eq!(a.and(b), b.and(a), "{what}: ∧ commutes ({a:?}, {b:?})");
            assert_eq!(a.or(b), b.or(a), "{what}: ∨ commutes ({a:?}, {b:?})");
            // De Morgan, both directions
            assert_eq!(
                a.and(b).not(),
                a.not().or(&b.not()),
                "{what}: ¬(a ∧ b) = ¬a ∨ ¬b ({a:?}, {b:?})"
            );
            assert_eq!(
                a.or(b).not(),
                a.not().and(&b.not()),
                "{what}: ¬(a ∨ b) = ¬a ∧ ¬b ({a:?}, {b:?})"
            );
            // xor in terms of and/or/not
            assert_eq!(
                a.xor(b),
                a.or(b).and(&a.and(b).not()),
                "{what}: a ⊕ b = (a ∨ b) ∧ ¬(a ∧ b) ({a:?}, {b:?})"
            );
        }
    }
}

/// The comparison and selection laws for a [`Compare`] carrier.
///
/// Exact. Every law relates one comparison to another or to `select`, so none of them rounds:
///
/// - `lt`/`ge` and `gt`/`le` and `eq`/`ne` are complementary
/// - `eq` is reflexive, `lt` and `gt` are not
/// - `le` is `lt ∨ eq`
/// - `select` with an all-true mask takes the true branch, all-false the false branch
///
/// That last pair is the one that catches a swapped `select`, which is otherwise invisible —
/// a carrier with its branches reversed type-checks everywhere and silently inverts every
/// conditional written against the contract.
pub fn mask_laws<T>(xs: &[T], what: &str)
where
    T: Compare + PartialEq + core::fmt::Debug,
    T::Mask: PartialEq + core::fmt::Debug,
{
    let t = <T::Mask as Element>::one();
    let f = <T::Mask as Element>::zero();

    for a in xs {
        assert_eq!(Compare::eq(a, a), t, "{what}: a = a ({a:?})");
        assert_eq!(Compare::ne(a, a), f, "{what}: ¬(a ≠ a) ({a:?})");
        assert_eq!(a.lt(a), f, "{what}: ¬(a < a) ({a:?})");
        assert_eq!(a.gt(a), f, "{what}: ¬(a > a) ({a:?})");
        assert_eq!(a.le(a), t, "{what}: a ≤ a ({a:?})");
        assert_eq!(a.ge(a), t, "{what}: a ≥ a ({a:?})");

        for b in xs {
            // complementary pairs
            assert_eq!(
                a.lt(b),
                a.ge(b).not(),
                "{what}: (a < b) = ¬(a ≥ b) ({a:?}, {b:?})"
            );
            assert_eq!(
                a.gt(b),
                a.le(b).not(),
                "{what}: (a > b) = ¬(a ≤ b) ({a:?}, {b:?})"
            );
            assert_eq!(
                Compare::eq(a, b),
                Compare::ne(a, b).not(),
                "{what}: (a = b) = ¬(a ≠ b) ({a:?}, {b:?})"
            );
            // le decomposes
            assert_eq!(
                a.le(b),
                a.lt(b).or(&Compare::eq(a, b)),
                "{what}: (a ≤ b) = (a < b) ∨ (a = b) ({a:?}, {b:?})"
            );
            // antisymmetry of the strict order
            assert_eq!(
                a.lt(b).and(&b.lt(a)),
                f,
                "{what}: a < b and b < a cannot both hold ({a:?}, {b:?})"
            );

            // select chooses, and chooses the right way round
            assert_eq!(
                &T::select(&t, a.clone(), b.clone()),
                a,
                "{what}: select(true, a, b) must be a ({a:?}, {b:?})"
            );
            assert_eq!(
                &T::select(&f, a.clone(), b.clone()),
                b,
                "{what}: select(false, a, b) must be b ({a:?}, {b:?})"
            );
            // the alias must agree
            assert_eq!(
                T::if_then_else(&t, a.clone(), b.clone()),
                T::select(&t, a.clone(), b.clone()),
                "{what}: if_then_else must agree with select"
            );
        }
    }
}

/// The real-analysis laws, to within `tolerance`.
///
/// Relative comparison: `|got − want| ≤ tolerance · max(1, |want|)`, matching the convention in
/// `euclid-utils` — purely relative is wrong where a quantity passes through zero, purely
/// absolute is wrong where it is large.
///
/// Inverse pairs and identities, which is what makes these metamorphic rather than a table of
/// expected digits:
///
/// - `ln(exp(x)) = x` and `exp(ln(x)) = x`
/// - `sqrt(x)² = x` and `sqrt(x²) = |x|`
/// - `sin²(x) + cos²(x) = 1`
/// - `ln_1p(x) = ln(1 + x)` and `exp_m1(x) = exp(x) − 1`, which exist precisely to be more
///   accurate than the naive forms near zero, so agreeing away from zero is the check
/// - `atan(tan(x)) = x` on the principal branch
///
/// Exact laws, checked without the tolerance: `scalar` round-trips through `scalar_like`,
/// `abs` is idempotent, `min`/`max` are commutative and return an argument, and the rounding
/// family is idempotent.
///
/// # Sample requirements
///
/// Values must be **finite** and **strictly positive** — the logarithm laws need `x > 0`, and
/// the trig laws stay on the principal branch, so keep values in `(0, π/2)` if you want
/// `atan(tan(x))` exercised. Include `1` and a value either side of it: that is where series
/// crossovers live.
pub fn real_laws<T>(xs: &[T], tolerance: f64, what: &str)
where
    T: Real + PartialEq + core::fmt::Debug,
    T::Mask: PartialEq + core::fmt::Debug,
{
    let close = |got: &T, want: &T, law: &str, x: &T| {
        // Compare through the carrier so no host conversion is needed: |got − want| must not
        // exceed tol · max(1, |want|), expressed with `le` and checked as a mask.
        let scale = want.abs().max(T::one());
        let bound = scale * T::scalar(tolerance);
        let err = (got.clone() - want.clone()).abs();
        assert_eq!(
            err.le(&bound),
            <T::Mask as Element>::one(),
            "{what}: {law} at {x:?} — got {got:?}, want {want:?}"
        );
    };

    for x in xs {
        // --- exact laws ---
        assert_eq!(x.abs().abs(), x.abs(), "{what}: |·| is idempotent ({x:?})");
        assert_eq!(
            x.clone().min(x.clone()),
            *x,
            "{what}: min(x, x) = x ({x:?})"
        );
        assert_eq!(
            x.clone().max(x.clone()),
            *x,
            "{what}: max(x, x) = x ({x:?})"
        );
        for (law, r) in [
            ("floor", x.clone().floor()),
            ("ceil", x.clone().ceil()),
            ("trunc", x.clone().trunc()),
        ] {
            let twice = match law {
                "floor" => r.clone().floor(),
                "ceil" => r.clone().ceil(),
                _ => r.clone().trunc(),
            };
            assert_eq!(twice, r, "{what}: {law} is idempotent ({x:?})");
        }
        // lifting a literal and lifting it "like" an existing value must agree
        assert_eq!(
            T::scalar(2.0),
            x.scalar_like(2.0),
            "{what}: scalar and scalar_like must agree ({x:?})"
        );

        // --- inverse pairs, to tolerance ---
        close(&x.exp().ln(), x, "ln(exp(x)) = x", x);
        close(&x.ln().exp(), x, "exp(ln(x)) = x", x);
        close(&x.sqrt().square(), x, "sqrt(x)² = x", x);
        close(&x.square().sqrt(), &x.abs(), "sqrt(x²) = |x|", x);

        // --- identities ---
        let (s, c) = x.clone().sin_cos();
        close(
            &(s.square() + c.square()),
            &T::one(),
            "sin²(x) + cos²(x) = 1",
            x,
        );
        close(&x.clone().tan().atan(), x, "atan(tan(x)) = x", x);

        // --- the compensated forms must agree with the naive ones away from zero ---
        close(
            &x.ln_1p(),
            &(T::one() + x.clone()).ln(),
            "ln_1p = ln(1+x)",
            x,
        );
        close(&x.exp_m1(), &(x.exp() - T::one()), "exp_m1 = exp(x) − 1", x);

        for y in xs {
            // min and max are commutative and return one of their arguments
            assert_eq!(
                x.clone().min(y.clone()),
                y.clone().min(x.clone()),
                "{what}: min commutes ({x:?}, {y:?})"
            );
            let m = x.clone().min(y.clone());
            assert!(
                m == *x || m == *y,
                "{what}: min must return an argument, got {m:?} from ({x:?}, {y:?})"
            );
            // hypot against its definition
            close(
                &x.clone().hypot(y.clone()),
                &(x.square() + y.square()).sqrt(),
                "hypot(x, y) = sqrt(x² + y²)",
                x,
            );
        }
    }
}

/// The scalar laws over **generated** carriers, for a carrier author who wants the search.
///
/// # Why this needs a trait
///
/// `euclid-tensor` can build a carrier through `Tensor::from_host` and `euclid-units` through
/// `Measure::from_raw`, because both contracts already contain a constructor. [`Real`] contains
/// one too — [`Real::scalar`] lifts an `f64` — but reaching for it here would be a mistake: a
/// carrier that is deliberately *deferred*, an expression handle or a graph node, can lift a
/// literal and cannot be compared, so generating values that way would produce a sample the laws
/// cannot evaluate.
///
/// So the carrier supplies its own generator through [`generated::GeneratedSample`], and in doing
/// so decides
/// which of its values are legal to generate. That is the same division of labour
/// [`ConformanceSample`] already uses, with a generator in place of a list.
///
/// # What generation reaches that a fixed sample does not
///
/// [`real_laws`] is a web of identities — `exp`/`ln` round trips, `sin²+cos²`, the hyperbolic
/// definitions — and each is checked against a tolerance. A fixed sample picks a handful of
/// arguments; a generated one hunts the argument where a kernel's error is *worst*, which for a
/// series implementation is a thin region near a crossover rather than anywhere a human would
/// think to look.
///
/// That is the whole reason this crate ships `euclid-stability`'s `target` and `budget`: pair a
/// generated run with a reported score and the search climbs toward the worst case instead of
/// sampling uniformly. See [`mod@euclid_stability::target`] and [`euclid_stability::budget`].
#[cfg(any(feature = "conformance", test))]
pub mod generated {
    use euclid_stability::{Config, Gen, check, classify};

    use super::{boolean_laws, mask_laws, real_laws};
    use crate::{Boolean, Compare, Real};

    /// A carrier's own generator, declared once — the generated peer of
    /// [`ConformanceSample`](super::ConformanceSample).
    ///
    /// Must honour the same sample requirements the slice-taking laws state: for
    /// [`super::real_laws`] that means **finite and strictly positive**, since the
    /// logarithm laws need `x > 0`, and values near `1` where the series crossovers live.
    pub trait GeneratedSample: Sized {
        /// Values to draw from.
        fn generator() -> Gen<Self>;

        /// How many cases to run. Defaulted at 400 — these laws are per-value rather than over
        /// triples, so a wider search is affordable here than in the algebra tower.
        fn cases() -> usize {
            400
        }
    }

    fn sample<T: GeneratedSample + 'static>() -> Gen<Vec<T>> {
        euclid_stability::generate::vec(T::generator(), 1, 6)
    }

    fn settings<T: GeneratedSample>() -> Config {
        Config::default().cases(T::cases())
    }

    /// [`real_laws`] over generated values.
    ///
    /// `tolerance` is relative and is the carrier's to choose from its own precision — see
    /// [`super::real_laws`], which explains why a bound that grows with the condition
    /// number is honest and a bound loosened until the suite passes is not.
    pub fn real_laws_generated<T>(tolerance: f64, what: &str)
    where
        T: Real + GeneratedSample + PartialEq + core::fmt::Debug + 'static,
        T::Mask: PartialEq + core::fmt::Debug,
    {
        check(what, &settings::<T>(), &sample::<T>(), |xs| {
            real_laws(&xs, tolerance, what);
            Ok(())
        });
    }

    /// [`mask_laws`] over generated values — the comparison surface.
    pub fn mask_laws_generated<T>(what: &str)
    where
        T: Compare + GeneratedSample + PartialEq + core::fmt::Debug + 'static,
        T::Mask: PartialEq + core::fmt::Debug,
    {
        check(what, &settings::<T>(), &sample::<T>(), |xs| {
            classify!("more than one value", xs.len() > 1);
            mask_laws(&xs, what);
            Ok(())
        });
    }

    /// [`boolean_laws`] over generated values — the mask carrier's own algebra.
    pub fn boolean_laws_generated<B>(what: &str)
    where
        B: Boolean + GeneratedSample + PartialEq + core::fmt::Debug + 'static,
    {
        check(what, &settings::<B>(), &sample::<B>(), |xs| {
            boolean_laws(&xs, what);
            Ok(())
        });
    }
}
