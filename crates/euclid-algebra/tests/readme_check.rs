use euclid_algebra::Semiring;

/// One routine, any semiring: ordinary arithmetic, min-plus, or boolean.
fn fold<T: Semiring>(xs: &[T]) -> T {
    xs.iter().copied().fold(T::ZERO, |acc, x| acc.add(x))
}

#[test]
fn readme_example_compiles() {
    // Ordinary arithmetic: `add` is `+`, `ZERO` is `0.0`.
    assert_eq!(fold(&[1.0_f64, 2.0, 3.0]), 6.0);
    assert_eq!(fold::<f64>(&[]), 0.0);
    assert_eq!(fold(&[1.0_f32, 2.0, 3.0]), 6.0);
}
