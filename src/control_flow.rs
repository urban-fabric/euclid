use crate::numeric::Compare;

/// Explicit graph-control-flow contract.
///
/// Carriers that can represent graph conditionals and loops implement this trait for their
/// scalar, expression, or tensor values.
pub trait ControlFlow: Compare {
    /// Graph-level conditional over carrier values.
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

    /// Graph-level while loop over same-carrier state.
    fn while_loop<C, B>(init: Self, cond: C, body: B) -> Self
    where
        C: Fn(&Self) -> Self::Mask,
        B: Fn(Self) -> Self;
}
