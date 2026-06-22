use crate::array::{Matrix, Tensor, Vector};

use super::ToyTensor;

#[test]
fn vector_and_matrix_like_constructors_use_shape_context() {
    let vector = ToyTensor(3.0);
    assert_eq!(<ToyTensor as Vector>::zeros(4), ToyTensor(0.0));
    assert_eq!(
        <ToyTensor as Vector>::full_like(&vector, 7.0),
        ToyTensor(7.0)
    );
    assert_eq!(<ToyTensor as Vector>::zeros_like(&vector), ToyTensor(0.0));

    let matrix = ToyTensor(5.0);
    assert_eq!(<ToyTensor as Matrix>::zeros(2, 3), ToyTensor(0.0));
    assert_eq!(
        <ToyTensor as Matrix>::full_like(&matrix, 9.0),
        ToyTensor(9.0)
    );
    assert_eq!(<ToyTensor as Matrix>::zeros_like(&matrix), ToyTensor(0.0));
}

#[test]
fn tensor_constructors_structural_ops_and_reductions_are_available() {
    let tensor = ToyTensor(4.0);
    assert_eq!(
        <ToyTensor as Tensor>::full(&[2, 2], ToyTensor(8.0)),
        ToyTensor(8.0)
    );
    assert_eq!(
        <ToyTensor as Tensor>::full_like(&tensor, ToyTensor(8.0)),
        ToyTensor(8.0)
    );
    assert_eq!(tensor.slice_in_dim(0, 1, 1, 0), ToyTensor(4.0));
    assert_eq!(
        ToyTensor::concatenate(&[ToyTensor(1.0), ToyTensor(2.0)], 0),
        ToyTensor(1.0)
    );
    assert_eq!(
        ToyTensor::stack(&[ToyTensor(1.0), ToyTensor(2.0)], 0),
        ToyTensor(1.0)
    );
    assert_eq!(tensor.pad(ToyTensor(0.0), &[1], &[1], &[0]), ToyTensor(4.0));
    assert_eq!(tensor.reverse(&[0]), ToyTensor(4.0));
    assert_eq!(tensor.prod(&[]), ToyTensor(4.0));
}
