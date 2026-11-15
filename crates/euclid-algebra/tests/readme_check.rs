//! The README's example, compiled.
//!
//! A crate whose deliverable is its trait surface has a README whose example *is* part of that
//! surface, and a README example that no longer compiles is worse than none — it documents a
//! signature the crate no longer has. Keeping it here as a test means a rename breaks the build
//! rather than quietly invalidating the front page.

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
