//! Generic structured control-flow contracts.
//!
//! This module is intentionally smaller than a pytree or tensor runtime. It lets a math package
//! state that a backend can branch or loop over a particular state type, where that state may be
//! a tensor, tuple, or user struct chosen by the backend implementation.

use crate::{array::TensorControlFlow, numeric::Compare};

/// Backend support for branch/loop control flow over `State`.
///
/// `State` is a type parameter rather than an associated tensor type so packages can ask for
/// control flow over the exact loop state they carry. A backend may implement this for a single
/// tensor handle, for tuples of handles, or for domain structs without euclid needing to know how
/// those states are flattened internally.
pub trait ControlFlow<State> {
    /// Predicate carrier used by this backend for the state.
    type Predicate: Clone;

    /// Graph/eager conditional over a structured state value.
    fn cond<T, F>(
        pred: &Self::Predicate,
        true_state: State,
        true_branch: T,
        false_state: State,
        false_branch: F,
    ) -> State
    where
        T: FnOnce(State) -> State,
        F: FnOnce(State) -> State;

    /// Graph/eager while loop over a structured state value.
    fn while_loop<C, B>(init: State, cond: C, body: B) -> State
    where
        C: Fn(&State) -> Self::Predicate,
        B: Fn(State) -> State;
}

impl<T> ControlFlow<T> for T
where
    T: TensorControlFlow,
{
    type Predicate = <T as Compare>::Mask;

    fn cond<TrueBranch, FalseBranch>(
        pred: &Self::Predicate,
        true_state: T,
        true_branch: TrueBranch,
        false_state: T,
        false_branch: FalseBranch,
    ) -> T
    where
        TrueBranch: FnOnce(T) -> T,
        FalseBranch: FnOnce(T) -> T,
    {
        <T as TensorControlFlow>::cond(pred, true_state, true_branch, false_state, false_branch)
    }

    fn while_loop<Cond, Body>(init: T, cond: Cond, body: Body) -> T
    where
        Cond: Fn(&T) -> Self::Predicate,
        Body: Fn(T) -> T,
    {
        <T as TensorControlFlow>::while_loop(init, cond, body)
    }
}

#[cfg(test)]
mod tests {
    use super::ControlFlow;

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct SolverState {
        step: i32,
        total: i32,
    }

    struct HostControlFlow;

    impl<State> ControlFlow<State> for HostControlFlow {
        type Predicate = bool;

        fn cond<T, F>(
            pred: &Self::Predicate,
            true_state: State,
            true_branch: T,
            false_state: State,
            false_branch: F,
        ) -> State
        where
            T: FnOnce(State) -> State,
            F: FnOnce(State) -> State,
        {
            if *pred {
                true_branch(true_state)
            } else {
                false_branch(false_state)
            }
        }

        fn while_loop<C, B>(init: State, cond: C, body: B) -> State
        where
            C: Fn(&State) -> Self::Predicate,
            B: Fn(State) -> State,
        {
            let mut state = init;
            while cond(&state) {
                state = body(state);
            }
            state
        }
    }

    #[test]
    fn control_flow_is_generic_over_tuple_state() {
        let out = <HostControlFlow as ControlFlow<(i32, i32)>>::while_loop(
            (0, 0),
            |(step, _)| *step < 4,
            |(step, total)| (step + 1, total + step),
        );
        assert_eq!(out, (4, 6));
    }

    #[test]
    fn control_flow_is_generic_over_struct_state() {
        let out = <HostControlFlow as ControlFlow<SolverState>>::cond(
            &true,
            SolverState { step: 1, total: 2 },
            |mut state| {
                state.total += 10;
                state
            },
            SolverState { step: 0, total: 0 },
            |state| state,
        );
        assert_eq!(out, SolverState { step: 1, total: 12 });
    }
}
