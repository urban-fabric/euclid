//! **Does the f32 carrier compute the same function as the f64 one?**
//!
//! Agreement between the two precisions, measured in f32 ulps against the f64 result correctly
//! rounded — which is the best the narrow kernel could possibly return, and therefore the only
//! non-arbitrary bound available. `euclid_stability::relation::precision_agrees` is the
//! instrument; its docstring states the risk this whole module exists for, that an f32 kernel
//! "is written second, exercised less, and its constants are usually the f64 coefficients
//! rounded — which is the right starting point and not always the right answer, because a
//! polynomial fitted for f64 can be the wrong polynomial at f32".
//!
//! Split in two because the two halves make *opposite* claims, and a reader needs to know which
//! one a kernel is making:
//!
//! | file | claim |
//! |---|---|
//! | [`delegating`] | this kernel computes in f64 and rounds once, so it is **bit-exact** |
//! | [`parameterised`] | this kernel is separately implemented, so it has a **measured budget** |
//!
//! A kernel moving between the two files is a real change in its character, and the split makes
//! that visible in the diff rather than as an edited integer.
//!
//! # These are budgets, not ideals
//!
//! Every number is measured, in the same spirit as the monotonicity counts in
//! `tests/stability.rs`: the honest treatment is to measure and pin rather than to assert an
//! ideal nobody promised. Today's behaviour passes; a kernel that drifts fails. Never loosen one
//! without understanding why the error grew.
//!
//! # Why the loose eight-way check was not enough
//!
//! `f32_and_f64_agree_to_single_precision` in [`delegating`] compares eight functions at a
//! relative tolerance of `1e-5`. f32 carries about `1.2e-7` of resolution, so that bound is
//! roughly eighty times looser than the format — wide enough for a mutated polynomial
//! coefficient to sit inside unnoticed. Mutation testing measured the consequence: around 480
//! surviving mutants in the f32 tree, and `<impl Real for f32>::asin` could return `0.0` for
//! every input.
//!
//! # What is still not covered here
//!
//! Five kernels are module-private `fn`s rather than `pub(crate)`, so this tree cannot see them
//! at all: `gamma_pdf`, `beta_pdf`, `gamma_seed`/`gamma_start`, `hurwitz_zeta` and
//! `gamma_edgeworth`. Per the house rules in [`crate::tests`] those belong in a `#[cfg(test)]`
//! child of the module that defines them, and that is where their cross-precision pass lives.

mod delegating;
mod parameterised;
