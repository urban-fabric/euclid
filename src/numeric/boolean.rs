//! Boolean masks and lazy comparisons.
//!
//! [`LazyBool`] is the Boolean algebra for masks. [`LazyCompare`] is the ordered comparison
//! surface that produces those masks and consumes them with `select`.
//!
//! Booleans are not rings: `or` is idempotent and there is no additive inverse. The mask
//! tier therefore sits beside [`LazyRing`](super::LazyRing). A comparison returns
//! [`LazyCompare::Mask`] (`bool` for eager scalars, a predicate value for lazy carriers),
//! not a scalar `bool`.

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
/// [`select`](LazyCompare::select)/[`if_then_else`](LazyCompare::if_then_else) it drives.
/// Implemented by ordered carriers (`f64`/`f32`, `i32`/`i64`/`i128`,
/// `u32`/`u64`/`u128`, and ordered lazy values).
///
/// The ordered analysis surface folds this in directly: [`LazyReal`](super::LazyReal)`:
/// LazyCompare`, so `if`/`then`/`else` over real numbers is reachable through the
/// [`LazyReal`](super::LazyReal) bound alone, with no separate `+ LazyCompare`. (Complex
/// stays unordered, so it carries no comparison.)
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
    /// [`LazyTensorControlFlow::while_loop`](crate::array::LazyTensorControlFlow::while_loop),
    /// and the counterpart of [`select`](Self::select) for loops rather than branches.
    ///
    /// `cond` maps the carried state to the [`Mask`](Self::Mask) "keep iterating"
    /// predicate; `body` advances it. The state `C` may be a single value or a bundle such
    /// as a `(lo, hi)` bracket. Eager carriers whose [`Mask`](Self::Mask) is a host `bool`
    /// can run an ordinary host loop; lazy carriers implement this with their native
    /// control-flow representation.
    fn while_loop<C, P, B>(init: C, cond: P, body: B) -> C
    where
        P: Fn(&C) -> Self::Mask,
        B: Fn(C) -> C;
}

fn host_while_loop<C, P, B>(init: C, cond: P, body: B) -> C
where
    P: Fn(&C) -> bool,
    B: Fn(C) -> C,
{
    let mut state = init;
    while cond(&state) {
        state = body(state);
    }
    state
}

// The ordered carriers. (Complex is unordered, so it carries no comparison here.)
impl LazyCompare for f64 {
    type Mask = bool;

    #[inline]
    fn lt(&self, other: &Self) -> bool {
        *self < *other
    }
    #[inline]
    fn le(&self, other: &Self) -> bool {
        *self <= *other
    }
    #[inline]
    fn gt(&self, other: &Self) -> bool {
        *self > *other
    }
    #[inline]
    fn ge(&self, other: &Self) -> bool {
        *self >= *other
    }
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        *self == *other
    }
    #[inline]
    fn ne(&self, other: &Self) -> bool {
        *self != *other
    }
    #[inline]
    fn select(mask: &bool, on_true: Self, on_false: Self) -> Self {
        if *mask { on_true } else { on_false }
    }
    #[inline]
    fn while_loop<C, P, B>(init: C, cond: P, body: B) -> C
    where
        P: Fn(&C) -> bool,
        B: Fn(C) -> C,
    {
        host_while_loop(init, cond, body)
    }
}

impl LazyCompare for f32 {
    type Mask = bool;

    #[inline]
    fn lt(&self, other: &Self) -> bool {
        *self < *other
    }
    #[inline]
    fn le(&self, other: &Self) -> bool {
        *self <= *other
    }
    #[inline]
    fn gt(&self, other: &Self) -> bool {
        *self > *other
    }
    #[inline]
    fn ge(&self, other: &Self) -> bool {
        *self >= *other
    }
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        *self == *other
    }
    #[inline]
    fn ne(&self, other: &Self) -> bool {
        *self != *other
    }
    #[inline]
    fn select(mask: &bool, on_true: Self, on_false: Self) -> Self {
        if *mask { on_true } else { on_false }
    }
    #[inline]
    fn while_loop<C, P, B>(init: C, cond: P, body: B) -> C
    where
        P: Fn(&C) -> bool,
        B: Fn(C) -> C,
    {
        host_while_loop(init, cond, body)
    }
}

impl LazyCompare for i32 {
    type Mask = bool;

    #[inline]
    fn lt(&self, other: &Self) -> bool {
        *self < *other
    }
    #[inline]
    fn le(&self, other: &Self) -> bool {
        *self <= *other
    }
    #[inline]
    fn gt(&self, other: &Self) -> bool {
        *self > *other
    }
    #[inline]
    fn ge(&self, other: &Self) -> bool {
        *self >= *other
    }
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        *self == *other
    }
    #[inline]
    fn ne(&self, other: &Self) -> bool {
        *self != *other
    }
    #[inline]
    fn select(mask: &bool, on_true: Self, on_false: Self) -> Self {
        if *mask { on_true } else { on_false }
    }
    #[inline]
    fn while_loop<C, P, B>(init: C, cond: P, body: B) -> C
    where
        P: Fn(&C) -> bool,
        B: Fn(C) -> C,
    {
        host_while_loop(init, cond, body)
    }
}

impl LazyCompare for i64 {
    type Mask = bool;

    #[inline]
    fn lt(&self, other: &Self) -> bool {
        *self < *other
    }
    #[inline]
    fn le(&self, other: &Self) -> bool {
        *self <= *other
    }
    #[inline]
    fn gt(&self, other: &Self) -> bool {
        *self > *other
    }
    #[inline]
    fn ge(&self, other: &Self) -> bool {
        *self >= *other
    }
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        *self == *other
    }
    #[inline]
    fn ne(&self, other: &Self) -> bool {
        *self != *other
    }
    #[inline]
    fn select(mask: &bool, on_true: Self, on_false: Self) -> Self {
        if *mask { on_true } else { on_false }
    }
    #[inline]
    fn while_loop<C, P, B>(init: C, cond: P, body: B) -> C
    where
        P: Fn(&C) -> bool,
        B: Fn(C) -> C,
    {
        host_while_loop(init, cond, body)
    }
}

impl LazyCompare for i128 {
    type Mask = bool;

    #[inline]
    fn lt(&self, other: &Self) -> bool {
        *self < *other
    }
    #[inline]
    fn le(&self, other: &Self) -> bool {
        *self <= *other
    }
    #[inline]
    fn gt(&self, other: &Self) -> bool {
        *self > *other
    }
    #[inline]
    fn ge(&self, other: &Self) -> bool {
        *self >= *other
    }
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        *self == *other
    }
    #[inline]
    fn ne(&self, other: &Self) -> bool {
        *self != *other
    }
    #[inline]
    fn select(mask: &bool, on_true: Self, on_false: Self) -> Self {
        if *mask { on_true } else { on_false }
    }
    #[inline]
    fn while_loop<C, P, B>(init: C, cond: P, body: B) -> C
    where
        P: Fn(&C) -> bool,
        B: Fn(C) -> C,
    {
        host_while_loop(init, cond, body)
    }
}

impl LazyCompare for u32 {
    type Mask = bool;

    #[inline]
    fn lt(&self, other: &Self) -> bool {
        *self < *other
    }
    #[inline]
    fn le(&self, other: &Self) -> bool {
        *self <= *other
    }
    #[inline]
    fn gt(&self, other: &Self) -> bool {
        *self > *other
    }
    #[inline]
    fn ge(&self, other: &Self) -> bool {
        *self >= *other
    }
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        *self == *other
    }
    #[inline]
    fn ne(&self, other: &Self) -> bool {
        *self != *other
    }
    #[inline]
    fn select(mask: &bool, on_true: Self, on_false: Self) -> Self {
        if *mask { on_true } else { on_false }
    }
    #[inline]
    fn while_loop<C, P, B>(init: C, cond: P, body: B) -> C
    where
        P: Fn(&C) -> bool,
        B: Fn(C) -> C,
    {
        host_while_loop(init, cond, body)
    }
}

impl LazyCompare for u64 {
    type Mask = bool;

    #[inline]
    fn lt(&self, other: &Self) -> bool {
        *self < *other
    }
    #[inline]
    fn le(&self, other: &Self) -> bool {
        *self <= *other
    }
    #[inline]
    fn gt(&self, other: &Self) -> bool {
        *self > *other
    }
    #[inline]
    fn ge(&self, other: &Self) -> bool {
        *self >= *other
    }
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        *self == *other
    }
    #[inline]
    fn ne(&self, other: &Self) -> bool {
        *self != *other
    }
    #[inline]
    fn select(mask: &bool, on_true: Self, on_false: Self) -> Self {
        if *mask { on_true } else { on_false }
    }
    #[inline]
    fn while_loop<C, P, B>(init: C, cond: P, body: B) -> C
    where
        P: Fn(&C) -> bool,
        B: Fn(C) -> C,
    {
        host_while_loop(init, cond, body)
    }
}

impl LazyCompare for u128 {
    type Mask = bool;

    #[inline]
    fn lt(&self, other: &Self) -> bool {
        *self < *other
    }
    #[inline]
    fn le(&self, other: &Self) -> bool {
        *self <= *other
    }
    #[inline]
    fn gt(&self, other: &Self) -> bool {
        *self > *other
    }
    #[inline]
    fn ge(&self, other: &Self) -> bool {
        *self >= *other
    }
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        *self == *other
    }
    #[inline]
    fn ne(&self, other: &Self) -> bool {
        *self != *other
    }
    #[inline]
    fn select(mask: &bool, on_true: Self, on_false: Self) -> Self {
        if *mask { on_true } else { on_false }
    }
    #[inline]
    fn while_loop<C, P, B>(init: C, cond: P, body: B) -> C
    where
        P: Fn(&C) -> bool,
        B: Fn(C) -> C,
    {
        host_while_loop(init, cond, body)
    }
}
