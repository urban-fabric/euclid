//! Boolean masks and mask-producing comparisons.
//!
//! [`Boolean`] is the Boolean algebra for masks. [`Compare`] is the ordered comparison surface
//! that produces those masks and consumes them with `select`. A comparison returns
//! [`Compare::Mask`] — `bool` for host scalars, a predicate value for expression/tensor
//! carriers — never a scalar `bool`.

/// The elementwise Boolean algebra of masks — `∧`/`∨`/`⊕`/`¬`. `bool` for host values,
/// carrier masks for expression/tensor values.
///
/// A boolean is an [`Element`](super::Element) (its `0`/`1` are `false`/`true`), so boolean
/// tensors can be built with `zeros`. It is not a [`Ring`](super::Ring): `∨` has no additive
/// inverse.
pub trait Boolean: super::Element {
    /// Logical AND (`∧`).
    fn and(&self, other: &Self) -> Self;
    /// Logical OR (`∨`).
    fn or(&self, other: &Self) -> Self;
    /// Logical XOR (`⊕`).
    fn xor(&self, other: &Self) -> Self;
    /// Logical NOT (`¬`).
    fn not(&self) -> Self;
}

/// Elementwise comparison to a [`Boolean`] mask, plus the branchless
/// [`select`](Compare::select)/[`if_then_else`](Compare::if_then_else) it drives. Implemented
/// by ordered carriers (`f64`/`f32`, `i32`/`i64`/`i128`, `u32`/`u64`/`u128`, and ordered
/// expression/tensor values); complex stays unordered and carries no comparison.
///
/// [`Real`](super::Real) requires this, so `if`/`then`/`else` over real numbers needs no
/// separate `+ Compare` bound.
///
/// The methods share names with `PartialOrd`/`PartialEq` but return the [`Mask`](Self::Mask),
/// not a `bool`; on a concrete scalar, disambiguate with `Compare::lt(&a, &b)`.
pub trait Compare: Clone {
    /// The boolean mask a comparison yields — `bool` for host values or a carrier mask.
    type Mask: Boolean;

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

    /// Branchless `where`: `mask ? on_true : on_false`, elementwise for array carriers.
    fn select(mask: &Self::Mask, on_true: Self, on_false: Self) -> Self;

    /// An alias for [`select`](Self::select), named for ordinary `if`/`then`/`else`. Pick
    /// whichever reads better at the call site; a carrier implements only `select`.
    #[inline]
    fn if_then_else(mask: &Self::Mask, on_true: Self, on_false: Self) -> Self {
        Self::select(mask, on_true, on_false)
    }

    /// Repeatedly advance a loop-carried state `C` by `body` while `cond` holds — the loop
    /// counterpart of [`select`](Self::select).
    ///
    /// `cond` maps the carried state to the "keep iterating" [`Mask`](Self::Mask); `body`
    /// advances it. `C` may be a single value or a bundle such as a `(lo, hi)` bracket. Host
    /// carriers can run an ordinary host loop; expression/tensor carriers use their native
    /// control-flow representation.
    fn while_loop<C, P, B>(init: C, cond: P, body: B) -> C
    where
        P: Fn(&C) -> Self::Mask,
        B: Fn(C) -> C;
}
