//! The complex/quaternion number traits: [`LazyComplex`], [`LazyQuaternion`],
//! and the concrete [`Complex`] number.

use super::{NormedDivisionAlgebra, Real};

/// A complex-valued numeric scalar: a [`NormedDivisionAlgebra`] that additionally exposes its
/// imaginary part. Implemented by genuinely-complex types — `Complex{32,64}` and a
/// complex-dtype `Tensor` — **not** by the real scalars (`f64`/`f32` are [`LazyReal`](super::LazyReal)s).
///
/// The real field ([`RealPart`](NormedDivisionAlgebra::RealPart)), `conjugate`, `magnitude`, and
/// `norm_sqr` all come from [`NormedDivisionAlgebra`]; this trait adds only the **imaginary**
/// component, with `re`/`conj` as the idiomatic complex spellings of
/// [`real_part`](NormedDivisionAlgebra::real_part)/[`conjugate`](NormedDivisionAlgebra::conjugate).
///
/// Construction from explicit `(re, im)` parts is **not** here; it lives on the
/// concrete complex constructors (e.g. `Complex64::new`, `Tensor::complex_from_parts`).
pub trait LazyComplex: NormedDivisionAlgebra {
    /// Imaginary part.
    fn im(&self) -> Self::RealPart;

    /// Real part — the complex spelling of [`NormedDivisionAlgebra::real_part`].
    fn re(&self) -> Self::RealPart {
        self.real_part()
    }
    /// Complex conjugate — the complex spelling of [`NormedDivisionAlgebra::conjugate`].
    fn conj(&self) -> Self {
        self.conjugate()
    }
}

/// A **quaternion** numeric scalar (4D): a [`NormedDivisionAlgebra`] that exposes its
/// three imaginary components `i`, `j`, `k` (`q = w + x·i + y·j + z·k`). The real part
/// `w`, conjugate, magnitude, and `norm_sqr` come from [`NormedDivisionAlgebra`]; this
/// trait adds only the imaginary basis. Implemented by `Quaternion{32,64}`.
///
/// ℍ is **non-commutative** (`i·j = k` but `j·i = −k`), so it is a *division ring*, not a
/// [`Field`](crate::algebra::Field) — it carries the numeric/division-algebra structure
/// here, but not the commutative-field algebra (and so not, yet, the matrix kernels).
pub trait LazyQuaternion: NormedDivisionAlgebra {
    /// The `i` (first imaginary) component.
    fn i(&self) -> Self::RealPart;
    /// The `j` (second imaginary) component.
    fn j(&self) -> Self::RealPart;
    /// The `k` (third imaginary) component.
    fn k(&self) -> Self::RealPart;
}

/// A **concrete complex number** — the complex peer of [`Real`]. It fuses the
/// algebraically-closed field structure ([`crate::algebra::AlgebraicallyClosedField`])
/// with [`LazyComplex`] analysis (`conj`/`re`/`im`/`magnitude`), `Copy`, a concrete
/// [`Real`] component, and the complex transcendentals. Implemented by
/// `jacobi::Complex{32,64}`.
pub trait Complex:
    crate::algebra::AlgebraicallyClosedField + LazyComplex<RealPart: Real> + Copy
{
    /// Lift a real literal into the complex plane (`x + 0i`).
    fn from_f64(x: f64) -> Self;

    /// The phase angle `arg(z) ∈ (−π, π]`.
    fn argument(self) -> Self::RealPart;
    /// The principal square root.
    fn sqrt(self) -> Self;
    /// `e^z`.
    fn exp(self) -> Self;
    /// The principal natural logarithm.
    fn ln(self) -> Self;
    /// `z^p` for a real exponent.
    fn powf(self, p: f64) -> Self;
    /// The reciprocal `1/z`.
    fn recip(self) -> Self;
}
