use super::ToyTensor;
use crate::array::{
    LazyMatrix, LazyMatrixControlFlow, LazyTensor, LazyTensorControlFlow, LazyVector,
    LazyVectorControlFlow,
};
use crate::numeric::LazyCompare;

#[test]
fn lazy_vector_control_flow_selects_values() {
    let mask = ToyTensor(1.0).lt(&ToyTensor(2.0));
    assert_eq!(
        <ToyTensor as LazyVector>::where_cond(&mask, ToyTensor(10.0), ToyTensor(20.0)),
        ToyTensor(10.0)
    );
    assert_eq!(
        <ToyTensor as LazyVector>::if_then_else(&false, ToyTensor(10.0), ToyTensor(20.0)),
        ToyTensor(20.0)
    );
    assert_eq!(
        <ToyTensor as LazyVectorControlFlow>::cond(
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
        <ToyTensor as LazyVectorControlFlow>::while_loop(
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
        <ToyTensor as LazyMatrix>::where_cond(&mask, ToyTensor(10.0), ToyTensor(20.0)),
        ToyTensor(10.0)
    );
    assert_eq!(
        <ToyTensor as LazyMatrix>::if_then_else(&false, ToyTensor(10.0), ToyTensor(20.0)),
        ToyTensor(20.0)
    );
    assert_eq!(
        <ToyTensor as LazyMatrixControlFlow>::cond(
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
        <ToyTensor as LazyMatrixControlFlow>::while_loop(
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
        <ToyTensor as LazyTensor>::where_cond(&mask, ToyTensor(10.0), ToyTensor(20.0)),
        ToyTensor(10.0)
    );
    assert_eq!(
        <ToyTensor as LazyTensor>::if_then_else(&false, ToyTensor(10.0), ToyTensor(20.0)),
        ToyTensor(20.0)
    );
    assert_eq!(
        <ToyTensor as LazyTensorControlFlow>::cond(
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
        <ToyTensor as LazyTensorControlFlow>::while_loop(
            ToyTensor(0.0),
            |x| x.lt(&ToyTensor(4.0)),
            |x| ToyTensor(x.0 + 1.0),
        ),
        ToyTensor(4.0)
    );
}
