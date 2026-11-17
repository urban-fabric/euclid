//! Exercise the public conformance helper on an eager rank-zero carrier.

#![cfg(feature = "conformance")]

use euclid_matrix::conformance::assert_close;
use euclid_scalar::Real;
use euclid_tensor::Tensor;

// Reuse the scalar law suite's generated f64 delegation.
mod traits {
    pub(super) use euclid_scalar::{Arithmetic, Compare, Element, NormedDivisionAlgebra};
}
#[path = "../../euclid-scalar/src/tests/vacuity/carrier.rs"]
mod carrier;

use carrier::Faulty;
type Scalar = Faulty<0>;

macro_rules! unused_tensor_methods {
    ($(fn $name:ident($($params:tt)*) -> Self;)*) => {
        $(
            fn $name($($params)*) -> Self {
                unimplemented!(concat!("rank-zero residual fixture has no ", stringify!($name)));
            }
        )*
    };
}

impl Tensor for Scalar {
    type Index = usize;

    fn shape(&self) -> Vec<usize> {
        vec![]
    }
    fn broadcast_in_dim(&self, shape: &[usize], axes: &[usize]) -> Self {
        assert!(shape.is_empty() && axes.is_empty());
        *self
    }
    fn max(self, axes: &[usize]) -> Self {
        assert!(axes.is_empty());
        self
    }
    fn sum(self, axes: &[usize]) -> Self {
        assert!(axes.is_empty());
        self
    }

    unused_tensor_methods! {
        fn reshape(self, _: &[usize]) -> Self;
        fn full(_: &[usize], _: Self) -> Self;
        fn from_host(_: &[f64], _: &[usize]) -> Self;
        fn transpose_axes(&self, _: &[usize]) -> Self;
        fn squeeze_axis(&self, _: isize) -> Self;
        fn slice_in_dim(&self, _: usize, _: usize, _: usize, _: usize) -> Self;
        fn concatenate(_: &[Self], _: usize) -> Self;
        fn stack(_: &[Self], _: usize) -> Self;
        fn pad(&self, _: Self, _: &[usize], _: &[usize], _: &[usize]) -> Self;
        fn reverse(&self, _: &[usize]) -> Self;
        fn gather(&self, _: &usize, _: &[usize], _: &[usize], _: &[usize], _: Option<usize>,
            _: &[usize]) -> Self;
        fn take(&self, _: &usize, _: usize) -> Self;
        fn dynamic_slice(&self, _: &[usize], _: &[usize]) -> Self;
        fn dynamic_update_slice(&self, _: &Self, _: &[usize]) -> Self;
        fn scatter_update_1d(&self, _: &usize, _: &Self) -> Self;
        fn scatter_add_1d(&self, _: &usize, _: &Self) -> Self;
        fn scatter_update(&self, _: &usize, _: &Self, _: &[usize], _: &[usize], _: &[usize],
            _: Option<usize>, _: bool, _: bool) -> Self;
        fn scatter_add(&self, _: &usize, _: &Self, _: &[usize], _: &[usize], _: &[usize],
            _: Option<usize>, _: bool, _: bool) -> Self;
        fn prod(self, _: &[usize]) -> Self;
        fn min(self, _: &[usize]) -> Self;
        fn mean(self, _: &[usize]) -> Self;
        fn any(self, _: &[usize]) -> Self;
        fn all(self, _: &[usize]) -> Self;
        fn matmul(self, _: Self) -> Self;
    }
    fn if_then_else(_: &<Self as euclid_scalar::Compare>::Mask, _: Self, _: Self) -> Self
    where
        Self: euclid_scalar::Compare,
    {
        unimplemented!("rank-zero residual fixture has no tensor selection")
    }
}

fn accepts(got: f64, want: f64, tolerance: f64) -> bool {
    std::panic::catch_unwind(|| {
        assert_close(
            &Faulty::<0>(got),
            &Faulty::<0>(want),
            tolerance,
            "close",
            "rank zero",
        );
    })
    .is_ok()
}

#[test]
fn residual_checks_use_absolute_and_relative_bounds() {
    assert!(accepts(0.125, 0.0, 0.125));
    assert!(!accepts(0.25, 0.0, 0.125));
    assert!(accepts(18.0, 16.0, 0.125));
    assert!(!accepts(20.0, 16.0, 0.125));
}

#[test]
fn residual_checks_cover_overflow_and_underflow() {
    assert!(!accepts(0.0, 1e200, 1e-8));
    assert!(!accepts(1e-300, 0.0, 1e-301));
    assert!(accepts(1e-300, 0.0, 2e-300));
    assert!(accepts(f64::MAX, f64::MAX, 0.0));
    assert!(!accepts(f64::from_bits(1), 0.0, 0.0));
    assert!(accepts(f64::MAX, -f64::MAX, 2.0));
    assert!(!accepts(f64::MAX, -f64::MAX, 1.5));
}

#[test]
fn residual_checks_reject_nonfinite_values_and_invalid_tolerances() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(!accepts(value, 0.0, 1.0));
        assert!(!accepts(0.0, value, 1.0));
        assert!(!accepts(0.0, 0.0, value));
        assert!(!accepts(value, value, 0.0));
    }
    assert!(!accepts(0.0, 0.0, -1.0));
}
