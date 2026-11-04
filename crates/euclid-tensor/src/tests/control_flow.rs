use super::ToyTensor;
use crate::Tensor;
use euclid_numeric::Compare;
use euclid_numeric::ControlFlow;

#[test]
fn lazy_tensor_control_flow_selects_values() {
    let mask = ToyTensor(1.0).lt(&ToyTensor(2.0));
    assert_eq!(
        <ToyTensor as Tensor>::if_then_else(&mask, ToyTensor(10.0), ToyTensor(20.0)),
        ToyTensor(10.0)
    );
    assert_eq!(
        <ToyTensor as Tensor>::if_then_else(&false, ToyTensor(10.0), ToyTensor(20.0)),
        ToyTensor(20.0)
    );
    assert_eq!(
        <ToyTensor as ControlFlow>::cond(
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
        <ToyTensor as ControlFlow>::while_loop(
            ToyTensor(0.0),
            |x| x.lt(&ToyTensor(4.0)),
            |x| ToyTensor(x.0 + 1.0),
        ),
        ToyTensor(4.0)
    );
}
