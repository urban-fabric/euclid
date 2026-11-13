use napier::Real;

fn circle_area<T: Real>(r: T) -> T {
    T::PI * r * r
}

#[test]
fn readme_example_compiles() {
    let a = circle_area(2.0_f64);
    let b = circle_area(2.0_f32);
    assert!((a - 12.566370614359172).abs() < 1e-12);
    assert!((b - 12.566371_f32).abs() < 1e-4);
}
