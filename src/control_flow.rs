use crate::numeric::Compare;

/// Explicit scalar graph-control-flow contract.
///
/// Array carriers expose rank-specific control-flow traits in [`crate::tensor`].
/// This trait is the scalar counterpart for carriers that can represent graph
/// conditionals and loops over scalar values.
pub trait ControlFlow: Compare {
    /// Graph-level conditional over scalar values.
    fn cond<T, F>(
        pred: &Self::Mask,
        true_arg: Self,
        true_branch: T,
        false_arg: Self,
        false_branch: F,
    ) -> Self
    where
        T: FnOnce(Self) -> Self,
        F: FnOnce(Self) -> Self;

    /// Graph-level while loop over scalar state.
    fn while_loop<C, B>(init: Self, cond: C, body: B) -> Self
    where
        C: Fn(&Self) -> Self::Mask,
        B: Fn(Self) -> Self;
}
