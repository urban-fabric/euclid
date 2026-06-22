use super::ToyTensor;
use crate::array::{
    Matrix, MatrixControlFlow, Tensor, TensorControlFlow, Vector, VectorControlFlow,
};
use crate::numeric::Compare;

#[test]
fn lazy_vector_control_flow_selects_values() {
    let mask = ToyTensor(1.0).lt(&ToyTensor(2.0));
    assert_eq!(
        <ToyTensor as Vector>::where_cond(&mask, ToyTensor(10.0), ToyTensor(20.0)),
        ToyTensor(10.0)
    );
    assert_eq!(
        <ToyTensor as Vector>::if_then_else(&false, ToyTensor(10.0), ToyTensor(20.0)),
        ToyTensor(20.0)
    );
    assert_eq!(
        <ToyTensor as VectorControlFlow>::cond(
            &false,
            ToyTensor(1.0),
            |x| ToyTensor(x.0 + 10.0),
            ToyTensor(2.0),
            |x| ToyTensor(x.0 + 100.0),
        ),
        ToyTensor(102.0)
    );
}

#[test]
fn lazy_vector_while_loop_is_implemented_by_the_carrier() {
    assert_eq!(
        <ToyTensor as VectorControlFlow>::while_loop(
            ToyTensor(0.0),
            |x| x.lt(&ToyTensor(3.0)),
            |x| ToyTensor(x.0 + 1.0),
        ),
        ToyTensor(3.0)
    );
}

#[test]
fn lazy_matrix_control_flow_selects_values() {
    let mask = ToyTensor(1.0).lt(&ToyTensor(2.0));
    assert_eq!(
        <ToyTensor as Matrix>::where_cond(&mask, ToyTensor(10.0), ToyTensor(20.0)),
        ToyTensor(10.0)
    );
    assert_eq!(
        <ToyTensor as Matrix>::if_then_else(&false, ToyTensor(10.0), ToyTensor(20.0)),
        ToyTensor(20.0)
    );
    assert_eq!(
        <ToyTensor as MatrixControlFlow>::cond(
            &false,
            ToyTensor(1.0),
            |x| ToyTensor(x.0 + 10.0),
            ToyTensor(2.0),
            |x| ToyTensor(x.0 + 100.0),
        ),
        ToyTensor(102.0)
    );
}

#[test]
fn lazy_matrix_while_loop_is_implemented_by_the_carrier() {
    assert_eq!(
        <ToyTensor as MatrixControlFlow>::while_loop(
            ToyTensor(1.0),
            |x| x.lt(&ToyTensor(8.0)),
            |x| ToyTensor(x.0 * 2.0),
        ),
        ToyTensor(8.0)
    );
}

#[test]
fn lazy_tensor_control_flow_selects_values() {
    let mask = ToyTensor(1.0).lt(&ToyTensor(2.0));
    assert_eq!(
        <ToyTensor as Tensor>::where_cond(&mask, ToyTensor(10.0), ToyTensor(20.0)),
        ToyTensor(10.0)
    );
    assert_eq!(
        <ToyTensor as Tensor>::if_then_else(&false, ToyTensor(10.0), ToyTensor(20.0)),
        ToyTensor(20.0)
    );
    assert_eq!(
        <ToyTensor as TensorControlFlow>::cond(
            &false,
            ToyTensor(1.0),
            |x| ToyTensor(x.0 + 10.0),
            ToyTensor(2.0),
            |x| ToyTensor(x.0 + 100.0),
        ),
        ToyTensor(102.0)
    );
}

#[test]
fn lazy_tensor_while_loop_is_implemented_by_the_carrier() {
    assert_eq!(
        <ToyTensor as TensorControlFlow>::while_loop(
            ToyTensor(0.0),
            |x| x.lt(&ToyTensor(4.0)),
            |x| ToyTensor(x.0 + 1.0),
        ),
        ToyTensor(4.0)
    );
}
