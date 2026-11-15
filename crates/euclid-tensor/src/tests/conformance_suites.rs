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
use crate::Tensor;
use crate::conformance::{self as laws, ConformanceSample};

impl ConformanceSample for Dense {
    /// Ranks 0 through 3, with every extent of the rank-3 case distinct. The rank-3 entry is
    /// load-bearing: the 3-cycle transpose law only applies at rank ≥ 3, and it is the only
    /// law that can tell a permutation from its inverse.
    fn sample() -> Vec<Self> {
        vec![
            Dense::point(2.5),
            Dense::vector(&[1.0, 2.0, 3.0, 4.0]),
            Dense::matrix(2, 3, &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]),
            Dense::matrix(3, 1, &[1.0, 2.0, 3.0]),
            Dense::new(&[2, 3, 4], (0..24).map(f64::from).collect()),
        ]
    }
}

#[test]
fn dense_satisfies_the_tensor_laws() {
    laws::tensor_laws_for::<Dense>("Dense");
}

#[test]
fn a_rank_zero_sample_skips_the_axis_laws_rather_than_panicking() {
    // A rank-0 tensor has no axes to permute, reverse or slice, so `structural_laws` must skip
    // it cleanly instead of indexing an empty shape. Checked with a rank-0 `Dense`, which does
    // honour the shape contract, so this isolates the empty-shape path.
    laws::tensor_laws(&[Dense::point(2.5)], "Dense (rank 0)");
}

#[test]
fn the_sample_covers_the_ranks_the_laws_need() {
    // The declared sample has to reach rank 3, or the 3-cycle transpose law never runs and a
    // backend applying a permutation backwards goes undetected. Asserted rather than assumed.
    let ranks: Vec<usize> = <Dense as ConformanceSample>::sample()
        .iter()
        .map(Tensor::rank)
        .collect();
    assert!(ranks.contains(&0), "a rank-0 case exercises the skip path");
    assert!(ranks.contains(&2), "rank 2 exercises the swap laws");
    assert!(
        ranks.iter().any(|&r| r >= 3),
        "rank 3+ is required for the 3-cycle law: {ranks:?}"
    );
}
