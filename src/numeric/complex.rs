//! The complex/quaternion lazy traits: [`LazyComplex`] and [`LazyQuaternion`]. The concrete
//! `Complex` number (the `Copy` tier) lives in the `napier` crate.

use super::NormedDivisionAlgebra;

/// A complex-valued numeric scalar: a [`NormedDivisionAlgebra`] that adds the imaginary
/// part. Implemented by genuinely-complex types (`Complex{32,64}`, a complex `Tensor`), not
/// the real scalars. The real field, `conjugate`, `magnitude`, and `norm_sqr` come from
/// [`NormedDivisionAlgebra`]; `from_parts`, `re`, and `conj` are the idiomatic complex
/// spellings.
pub trait LazyComplex: NormedDivisionAlgebra {
    /// Construct `re + i·im` from real and imaginary parts.
    fn from_parts(re: Self::RealPart, im: Self::RealPart) -> Self;

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

/// A **quaternion** numeric scalar (4D): a [`NormedDivisionAlgebra`] adding the three
/// imaginary components of `q = w + x·i + y·j + z·k`. Implemented by `Quaternion{32,64}`.
/// ℍ is non-commutative (`i·j = k`, `j·i = −k`), so a division ring rather than a `Field` —
/// it carries the division-algebra structure here, but not the matrix kernels.
pub trait LazyQuaternion: NormedDivisionAlgebra {
    /// The `i` (first imaginary) component.
    fn i(&self) -> Self::RealPart;
    /// The `j` (second imaginary) component.
    fn j(&self) -> Self::RealPart;
    /// The `k` (third imaginary) component.
    fn k(&self) -> Self::RealPart;
}
