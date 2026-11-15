//! The concrete complex number [`Complex`] — the complex peer of [`Real`].

use crate::Real;

/// A **concrete complex number** — the complex peer of [`Real`]. It fuses the
/// algebraically-closed field structure ([`euclid_algebra::AlgebraicallyClosedField`])
/// with [`euclid_numeric::Complex`] analysis
/// (`conj`/`re`/`im`/`magnitude`), `Copy`, a concrete [`Real`] component, and the
/// complex transcendentals. Implemented by `jacobi::Complex{32,64}`.
pub trait Complex:
    euclid_algebra::AlgebraicallyClosedField + euclid_numeric::Complex<RealPart: Real> + Copy
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
