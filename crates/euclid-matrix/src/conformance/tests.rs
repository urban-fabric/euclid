use super::scaled_residual_within;

#[test]
fn frobenius_bound_uses_both_accumulated_norms() {
    // Four equal residual magnitudes of 0.5 give error 1. Nine reference
    // magnitudes of 2 give norm 6. The expected relative error is 1/6.
    assert!(scaled_residual_within::<f64>(
        (0.5, 2.0),
        (2.0, 3.0),
        1.0,
        0.18
    ));
    assert!(!scaled_residual_within::<f64>(
        (0.5, 2.0),
        (2.0, 3.0),
        1.0,
        0.15
    ));
}
