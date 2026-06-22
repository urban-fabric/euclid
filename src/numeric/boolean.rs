//! Boolean masks and mask-producing comparisons.
//!
//! [`Boolean`] is the Boolean algebra for masks. [`Compare`] is the ordered comparison
//! surface that produces those masks and consumes them with `select`.
//!
//! Booleans are not rings: `or` is idempotent and there is no additive inverse. The mask
//! tier therefore sits beside [`Ring`](super::Ring). A comparison returns
//! [`Compare::Mask`] (`bool` for host scalars, a predicate value for expression/tensor carriers),
//! not a scalar `bool`. The concrete `bool`/`f*`/`i*`/`u*` impls live in
//! [`impls::boolean_impl`](super::impls).

/// The elementwise Boolean algebra of masks — `∧`/`∨`/`⊕`/`¬`. `bool` for host values,
/// carrier masks for expression/tensor values. A boolean is a tensor element, so it is a
/// [`Element`](super::Element) (its `0`/`1` are `false`/`true`) — which also lets a
/// boolean [`Tensor`](crate::tensor::Tensor) be built with `zeros`. It is not a ring,
/// though (`∨` has no additive inverse), so it is not a [`Ring`](super::Ring).
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
/// [`select`](Compare::select)/[`if_then_else`](Compare::if_then_else) it drives.
/// Implemented by ordered carriers (`f64`/`f32`, `i32`/`i64`/`i128`,
/// `u32`/`u64`/`u128`, and ordered expression/tensor values).
///
/// The ordered analysis surface folds this in directly: [`Real`](super::Real)`:
/// Compare`, so `if`/`then`/`else` over real numbers is reachable through the
/// [`Real`](super::Real) bound alone, with no separate `+ Compare`. (Complex
/// stays unordered, so it carries no comparison.)
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

    /// Branchless conditional `mask ? on_true : on_false`, named for ordinary
    /// `if`/`then`/`else` control flow. An alias for [`select`](Self::select) — both lower
    /// to the same elementwise `Select`; pick whichever reads better at the call site.
    /// Provided, so a carrier only implements [`select`](Self::select).
    #[inline]
    fn if_then_else(mask: &Self::Mask, on_true: Self, on_false: Self) -> Self {
        Self::select(mask, on_true, on_false)
    }

    /// Iterative control flow: repeatedly advance a loop-carried state `C` by `body`
    /// while `cond` holds — the scalar analog of
    /// [`TensorControlFlow::while_loop`](crate::tensor::TensorControlFlow::while_loop),
    /// and the counterpart of [`select`](Self::select) for loops rather than branches.
    ///
    /// `cond` maps the carried state to the [`Mask`](Self::Mask) "keep iterating"
    /// predicate; `body` advances it. The state `C` may be a single value or a bundle such
    /// as a `(lo, hi)` bracket. Host carriers whose [`Mask`](Self::Mask) is a host `bool`
    /// can run an ordinary host loop; expression/tensor carriers implement this with their native
    /// control-flow representation.
    fn while_loop<C, P, B>(init: C, cond: P, body: B) -> C
    where
        P: Fn(&C) -> Self::Mask,
        B: Fn(C) -> C;
}
