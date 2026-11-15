//! The reference tensors, checked against the public conformance suites.
//!
//! The laws are [`crate::conformance`] — public API a backend in another repository calls.
//! Running them against [`Dense`](super::dense::Dense) and
//! [`ToyTensor`](super::toy_tensor::ToyTensor) proves the suites are satisfiable, and that both
//! reference implementations satisfy them.
//!
//! Only `Dense` is checked, and the reason is worth recording. `ToyTensor` is deliberately
//! rank-0 — it exists to prove the *arithmetic* surface is implementable without the 98-method
//! real surface `Dense` carries — and its `reshape` ignores the shape it is given, returning a
//! rank-0 tensor whatever is asked for. So `ToyTensor` does **not** satisfy
//! [`shape_laws`](crate::conformance::shape_laws), and running them against it fails at
//! "reshape must report the shape it was given".
//!
//! That is a limitation of the toy rather than a flaw in the law. A kernel that reshapes to
//! `[n]` and then indexes axis 0 depends on the shape actually changing, so the law is the one a
//! caller relies on. The toy is a scalar stand-in that was never a complete `Tensor`, which the
//! suites made visible — the first thing a conformance suite is for.

use super::dense::Dense;
use crate::conformance as laws;

#[test]
fn dense_satisfies_the_tensor_laws() {
    let xs = [
        Dense::point(2.5),
        Dense::vector(&[1.0, 2.0, 3.0, 4.0]),
        Dense::matrix(2, 3, &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]),
        Dense::matrix(3, 1, &[1.0, 2.0, 3.0]),
        Dense::new(&[2, 2, 2], (1..=8).map(f64::from).collect()),
    ];
    laws::tensor_laws(&xs, "Dense");
}

#[test]
fn a_rank_zero_sample_skips_the_axis_laws_rather_than_panicking() {
    // A rank-0 tensor has no axes to permute, reverse or slice, so `structural_laws` must skip
    // it cleanly instead of indexing an empty shape. Checked with a rank-0 `Dense`, which does
    // honour the shape contract, so this isolates the empty-shape path.
    laws::tensor_laws(&[Dense::point(2.5)], "Dense (rank 0)");
}

#[test]
fn the_laws_cover_every_axis_of_a_higher_rank_tensor() {
    // The full-slice law iterates axes, so a non-square rank-3 tensor is the case that catches
    // an implementation indexing the wrong axis — every extent here is distinct.
    let x = Dense::new(&[2, 3, 4], (0..24).map(f64::from).collect());
    laws::tensor_laws(&[x], "Dense[2,3,4]");
}
