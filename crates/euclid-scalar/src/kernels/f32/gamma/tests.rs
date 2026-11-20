//! Unit tests for the single-precision `Γ` and `lnΓ`.
//!
//! A child module so implementation plus tests stay inside the crate's 400-line budget. The
//! incomplete pair and the polygammas keep their tests beside their own modules, in
//! [`super::incomplete`] and [`super::psi`], because what those assert is private to them.
//!
//! The `f64` kernel is the reference throughout: it is four orders more accurate than `f32` can
//! represent, so it serves as truth for a native single-precision route.

use super::*;

/// Cross-check the native `f32` result against the `f64` reference, narrowed — the guard
/// against the two implementations silently drifting apart.
fn agrees(a: f32, b: f64, tol: f32) {
    let b = b as f32;
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
}

#[test]
fn gamma_matches_f64_reference() {
    for &x in &[0.3, 0.7, 1.5, 3.2, 5.0, 10.0] {
        agrees(gamma(x), crate::kernels::gamma::gamma(x as f64), 1e-5);
        agrees(ln_gamma(x), crate::kernels::gamma::ln_gamma(x as f64), 1e-5);
    }
}

#[test]
fn the_log_gamma_method_prefers_the_reduction_on_the_positive_side() {
    use LnGammaMethod::{Lanczos, Reduced, Reflected};
    // The sign test is on the *sign bit*, so a negative zero reflects rather than reducing.
    assert_eq!(LnGammaMethod::for_argument(0.0), Reduced);
    assert_eq!(LnGammaMethod::for_argument(1.0), Reduced);
    assert_eq!(
        LnGammaMethod::for_argument(LN_GAMMA_RECURRENCE_MAX.next_down()),
        Reduced
    );
    assert_eq!(
        LnGammaMethod::for_argument(LN_GAMMA_RECURRENCE_MAX),
        Lanczos
    );
    assert_eq!(LnGammaMethod::for_argument(1e30), Lanczos);
    assert_eq!(LnGammaMethod::for_argument(-0.0), Reflected);
    assert_eq!(LnGammaMethod::for_argument(-1.5), Reflected);
    assert_eq!(LnGammaMethod::for_argument(f32::NEG_INFINITY), Reflected);
}
