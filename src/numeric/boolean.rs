//! The boolean **mask** tier: [`LazyBool`], the Boolean algebra a comparison produces and
//! `select` consumes, and [`LazyCompare`], the comparison + `select` surface on the ordered
//! carriers.
//!
//! Booleans are a Boolean algebra, not a ring (`∨` is idempotent, with no additive inverse),
//! so this tier sits beside the [`LazyRing`](super::LazyRing) tower. It resolves "a
//! comparison cannot return a scalar `bool` on a tensor backend": a comparison yields the
//! abstract [`Mask`](LazyCompare::Mask) — `bool` eagerly, a `pred`-dtype tensor on a backend
//! — so a masked/piecewise formula is write-once.

/// The elementwise Boolean algebra of masks — `∧`/`∨`/`⊕`/`¬`. `bool` eagerly, a
/// `pred`-dtype tensor on a backend. Not a ring, so it is not a [`LazyRing`](super::LazyRing).
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
/// [`select`](LazyCompare::select) it drives. Implemented by the ordered carriers
/// (`f64`/`f32`, `i64`/`i128`, `u32`/`u64`/`u128`, a backend tensor).
///
/// The methods share names with `PartialOrd`/`PartialEq` but return the [`Mask`](Self::Mask),
/// not a `bool`; on a concrete scalar, disambiguate with `LazyCompare::lt(&a, &b)`.
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

    /// Branchless `where`: `mask ? on_true : on_false`, elementwise (XLA `Select`).
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

// The ordered carriers. (Complex is unordered, so it carries no comparison here.)
eager_compare!(f64, f32, i64, i128, u32, u64, u128);
