//! The boolean **mask** tier: [`LazyBool`] — the elementwise Boolean algebra a comparison
//! produces and `select` consumes — and [`LazyCompare`] — the comparison + `select`
//! surface on the ordered numeric carriers.
//!
//! Booleans are a **Boolean algebra / semiring**, *not* a ring: `∨` is idempotent
//! (`a ∨ a = a`) with no additive inverse. So this tier sits **beside** the
//! [`Arithmetic`](super::Arithmetic)/[`LazyRing`](super::LazyRing) tower, never within it
//! — the concrete boolean semiring belongs to the abstract-algebra trunk (now in `napier`). It is
//! the lazy tier's resolution of "a comparison cannot return a scalar `bool` on a tensor
//! backend": a comparison returns the abstract [`Mask`](LazyCompare::Mask) — `bool` for an
//! eager scalar, a `pred`-dtype tensor (the JAX mask) for a tensor backend — so a masked /
//! piecewise / truncated formula written once runs on every backend.

/// The elementwise Boolean algebra of masks — `∧`/`∨`/`⊕`/`¬`. `bool` eagerly; a
/// `pred`-dtype tensor on a tensor backend. **Not** a ring (no additive inverse), so it
/// does not implement [`LazyRing`](super::LazyRing); logical ops, not `+ - *`.
pub trait LazyBool: Clone {
    /// Logical AND (`∧`).
    fn and(&self, other: &Self) -> Self;
    /// Logical OR (`∨`).
    fn or(&self, other: &Self) -> Self;
    /// Logical XOR (`⊕`).
    fn xor(&self, other: &Self) -> Self;
    /// Logical NOT (`¬`).
    fn not(&self) -> Self;
}

impl LazyBool for bool {
    #[inline]
    fn and(&self, other: &Self) -> Self {
        *self && *other
    }
    #[inline]
    fn or(&self, other: &Self) -> Self {
        *self || *other
    }
    #[inline]
    fn xor(&self, other: &Self) -> Self {
        *self ^ *other
    }
    #[inline]
    fn not(&self) -> Self {
        !*self
    }
}

/// Elementwise comparison to a [`LazyBool`] mask, plus the branchless
/// [`select`](LazyCompare::select) the mask drives. Implemented by the ordered carriers —
/// `f64`/`f32`, `i64`/`i128`, and a backend tensor — so a piecewise / truncated / masked
/// kernel is write-once over every backend.
///
/// The methods share names with `PartialOrd`/`PartialEq` (`lt`/`eq`/…) but return the
/// abstract [`Mask`](Self::Mask), not a `bool`; in a generic kernel (`T: LazyCompare`)
/// they resolve unambiguously, and on a concrete eager scalar one disambiguates with
/// `LazyCompare::lt(&a, &b)`.
pub trait LazyCompare: Clone {
    /// The boolean mask a comparison yields — `bool` eagerly, a `pred` tensor on a backend.
    type Mask: LazyBool;

    /// `self < other`.
    fn lt(&self, other: &Self) -> Self::Mask;
    /// `self <= other`.
    fn le(&self, other: &Self) -> Self::Mask;
    /// `self > other`.
    fn gt(&self, other: &Self) -> Self::Mask;
    /// `self >= other`.
    fn ge(&self, other: &Self) -> Self::Mask;
    /// `self == other`.
    fn eq(&self, other: &Self) -> Self::Mask;
    /// `self != other`.
    fn ne(&self, other: &Self) -> Self::Mask;

    /// Branchless `where`: `mask ? on_true : on_false`, elementwise (XLA `Select`). Lets a
    /// piecewise kernel choose between branches without a `bool`-returning comparison the
    /// lazy tier cannot express.
    fn select(mask: &Self::Mask, on_true: Self, on_false: Self) -> Self;
}

macro_rules! eager_compare {
    ($($t:ty),*) => {$(
        impl LazyCompare for $t {
            type Mask = bool;
            #[inline]
            fn lt(&self, other: &Self) -> bool { *self < *other }
            #[inline]
            fn le(&self, other: &Self) -> bool { *self <= *other }
            #[inline]
            fn gt(&self, other: &Self) -> bool { *self > *other }
            #[inline]
            fn ge(&self, other: &Self) -> bool { *self >= *other }
            #[inline]
            fn eq(&self, other: &Self) -> bool { *self == *other }
            #[inline]
            fn ne(&self, other: &Self) -> bool { *self != *other }
            #[inline]
            fn select(mask: &bool, on_true: Self, on_false: Self) -> Self {
                if *mask { on_true } else { on_false }
            }
        }
    )*};
}

// The ordered carriers: the real and integer scalars. (Complex is unordered, so it carries
// no `lt`/`gt`; only equality would apply, and that is left out until a consumer needs it.)
eager_compare!(f64, f32, i64, i128);
