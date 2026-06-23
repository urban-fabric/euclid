use crate::tensor::Tensor;

use super::ToyTensor;

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
