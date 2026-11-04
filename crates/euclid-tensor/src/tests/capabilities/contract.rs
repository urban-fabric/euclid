//! `TensorContract`: general contraction and Einstein summation.

use super::super::dense::Dense;
use crate::{Tensor, TensorContract};
use euclid_utils::is_close;

/// Hand-computed expected values in this suite are exact decimals, so the only error is
/// floating-point rounding through a few operations.
const TOL: f64 = 1e-9;

/// [`is_close`] at this suite's tolerance.
fn close(a: f64, b: f64) -> bool {
    is_close(a, b, TOL)
}

/// Elementwise [`close`] over a tensor's values against an expected slice.
fn all_close(got: &Dense, want: &[f64]) -> bool {
    got.values().len() == want.len() && got.values().iter().zip(want).all(|(&g, &w)| close(g, w))
}

// --- TensorContract -----------------------------------------------------

#[test]
fn dot_general_reproduces_matmul() {
    let a = Dense::matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Dense::matrix(3, 2, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let via_dot = a.dot_general(&b, &[1], &[0], &[], &[]);
    let via_matmul = a.clone().matmul(b);
    assert_eq!(via_dot.dims(), &[2, 2]);
    assert_eq!(via_dot, via_matmul);
    // [[58,64],[139,154]]
    assert!(all_close(&via_dot, &[58.0, 64.0, 139.0, 154.0]));
}

#[test]
fn dot_general_batches_over_the_named_axes() {
    // Two independent 2x2 matmuls stacked on a leading batch axis.
    let a = Dense::new(&[2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 1.0]);
    let b = Dense::new(&[2, 2, 2], vec![1.0, 0.0, 0.0, 1.0, 5.0, 6.0, 7.0, 8.0]);
    let got = a.dot_general(&b, &[2], &[1], &[0], &[0]);
    assert_eq!(got.dims(), &[2, 2, 2]);
    // batch 0: A·I = A ; batch 1: I·B = B
    assert!(all_close(&got, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]));
}

#[test]
fn dot_general_contracts_vectors_to_a_scalar() {
    let a = Dense::vector(&[1.0, 2.0, 3.0]);
    let b = Dense::vector(&[4.0, 5.0, 6.0]);
    let got = a.dot_general(&b, &[0], &[0], &[], &[]);
    assert!(got.dims().is_empty());
    assert!(close(got.item(), 32.0));
}

#[test]
fn einsum1_takes_a_diagonal() {
    let m = Dense::matrix(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    assert!(all_close(&m.einsum1("ii->i"), &[1.0, 5.0, 9.0]));
}

#[test]
fn einsum1_transposes_and_sums() {
    let m = Dense::matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let t = m.einsum1("ij->ji");
    assert_eq!(t.dims(), &[3, 2]);
    assert!(all_close(&t, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]));
    // Dropping a letter sums over it.
    assert!(all_close(&m.einsum1("ij->j"), &[5.0, 7.0, 9.0]));
    assert!(close(m.einsum1("ij->").item(), 21.0));
}

#[test]
fn einsum2_spells_matmul_and_inner_product() {
    let a = Dense::matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Dense::matrix(3, 2, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    assert_eq!(a.einsum2(&b, "ij,jk->ik"), a.clone().matmul(b));

    let u = Dense::vector(&[1.0, 2.0, 3.0]);
    let v = Dense::vector(&[4.0, 5.0, 6.0]);
    assert!(close(u.einsum2(&v, "i,i->").item(), 32.0));

    // Outer product keeps both letters.
    let outer = u.einsum2(&v, "i,j->ij");
    assert_eq!(outer.dims(), &[3, 3]);
    assert!(all_close(
        &outer,
        &[4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 12.0, 15.0, 18.0]
    ));
}
