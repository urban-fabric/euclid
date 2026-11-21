//! The tiers over a separate scalar: left semimodules, and the by-reference magma surface.
//!
//! A semimodule's elements and its scalars are *different types*, which is where these laws stop
//! being about one carrier and start being about an action of one on another. `ref_magma_laws`
//! is here because it asks the same question for a carrier too large to copy — that the
//! by-reference forms agree with the by-value ones.

use super::*;

/// The six semimodule axioms: a scalar semiring acting on a value.
///
/// ```text
///   ONE · m = m                  (s ⊗ t) · m = s · (t · m)
///   ZERO · m = 0_M               s · 0_M     = 0_M
///   (s ⊕ t) · m = s·m ⊞ t·m      s · (m ⊞ n) = s·m ⊞ s·n
/// ```
///
/// Pass the same slice twice to check a semiring as a semimodule over itself, which the blanket
/// impl in this crate provides. Pass different ones to check a value type acted on by a
/// different scalar algebra.
pub fn left_semimodule_laws<S, M>(scalars: &[S], modules: &[M], what: &str)
where
    S: Semiring + core::fmt::Debug + Copy,
    M: LeftSemimodule<S> + core::fmt::Debug + Copy,
{
    for &m in modules {
        assert_eq!(m.scale(S::ONE), m, "{what}: 1 · m = m");
        assert_eq!(m.scale(S::ZERO), M::ZERO, "{what}: 0 · m = 0_M");
    }
    for &s in scalars {
        assert_eq!(M::ZERO.scale(s), M::ZERO, "{what}: s · 0_M = 0_M");
        for &m in modules {
            for &n in modules {
                assert_eq!(
                    m.add(n).scale(s),
                    m.scale(s).add(n.scale(s)),
                    "{what}: s · (m ⊞ n) = s·m ⊞ s·n"
                );
            }
            for &t in scalars {
                assert_eq!(
                    m.scale(s.add(t)),
                    m.scale(s).add(m.scale(t)),
                    "{what}: (s ⊕ t) · m = s·m ⊞ t·m"
                );
                assert_eq!(
                    m.scale(s.mul(t)),
                    m.scale(t).scale(s),
                    "{what}: (s ⊗ t) · m = s · (t · m)"
                );
            }
        }
    }
}

/// The by-reference magma operations agree with their by-value counterparts.
///
/// [`RefAdditiveMagma::add_ref`] and [`RefMultiplicativeMagma::mul_ref`] exist for carriers that
/// are not [`Copy`], and their documentation states the law directly: "Must agree with
/// [`AdditiveMagma::add`] where both exist: `a.add_ref(&b) == a.add(b)` for a `Copy` carrier."
///
/// Nothing checked it. Mutation testing found every `add_ref` and `mul_ref` in the crate
/// replaceable with `Default::default()` — returning zero for every input — and the whole
/// workspace suite stayed green, across `i64`, `i128`, `f64`, `f32` and `GaussianRational`.
///
/// The reason the gap is structural rather than careless: a by-reference operation is written so
/// that a generic algorithm can avoid a clone, so it is reached only from code paths that are
/// generic over the *ref* traits — and the fleet's own suites all instantiate at `Copy` carriers,
/// which take the by-value path. The duplicate implementation is never the one that runs here,
/// and an implementation nothing runs is an implementation nothing checks.
pub fn ref_magma_laws<T>(xs: &[T], what: &str)
where
    T: AdditiveMagma
        + MultiplicativeMagma
        + RefAdditiveMagma
        + RefMultiplicativeMagma
        + core::fmt::Debug
        + Copy,
{
    for &a in xs {
        for &b in xs {
            assert_eq!(
                a.add_ref(&b),
                AdditiveMagma::add(a, b),
                "{what}: a.add_ref(&b) must agree with a ⊕ b"
            );
            assert_eq!(
                a.mul_ref(&b),
                MultiplicativeMagma::mul(a, b),
                "{what}: a.mul_ref(&b) must agree with a ⊗ b"
            );
        }
    }
}

/// [`ref_magma_laws`] over a carrier's own [`ConformanceSample`].
pub fn ref_magma_laws_for<T>(what: &str)
where
    T: AdditiveMagma
        + MultiplicativeMagma
        + RefAdditiveMagma
        + RefMultiplicativeMagma
        + ConformanceSample
        + core::fmt::Debug
        + Copy,
{
    ref_magma_laws(&T::sample(), what);
}
