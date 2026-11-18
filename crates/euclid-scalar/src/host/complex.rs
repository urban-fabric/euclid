//! The concrete complex number [`HostComplex`] — the complex peer of [`HostReal`].

use crate::HostReal;

/// A **concrete complex number** — the complex peer of [`HostReal`]. It fuses the
/// algebraically-closed field structure ([`euclid_algebra::AlgebraicallyClosedField`])
/// with [`crate::Complex`] analysis
/// (`conj`/`re`/`im`/`magnitude`), `Copy`, a concrete [`HostReal`] component, and the
/// complex transcendentals. Implemented by `jacobi::Complex{32,64}`.
pub trait HostComplex:
    euclid_algebra::AlgebraicallyClosedField + crate::Complex<RealPart: HostReal> + Copy
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
