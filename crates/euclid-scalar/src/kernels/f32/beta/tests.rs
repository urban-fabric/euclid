//! Unit tests for the single-precision `B(a, b)` and `lnB(a, b)`.
//!
//! A child module so implementation plus tests stay inside the crate's 400-line budget. The
//! incomplete beta keeps its tests beside its own module, in [`super::incomplete`], because what
//! they reach — the Edgeworth components, the continued fraction, the method table — is private
//! to it.

#![allow(clippy::panic)]

use super::{LN_BETA_STIRLING_MIN, LnBetaForm, beta, ln_beta};

fn close(a: f32, b: f32, tol: f32) {
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
}

fn agrees(a: f32, b: f64, tol: f32) {
    let b = b as f32;
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
}

#[test]
fn beta_matches_f64_reference() {
    for &(a, b) in &[(1.0, 1.0), (2.0, 3.0), (0.5, 0.5), (5.0, 2.0)] {
        agrees(
            beta(a, b),
            crate::kernels::beta::beta(a as f64, b as f64),
            1e-5,
        );
    }
    close(beta(1.0, 1.0), 1.0, 1e-6);
    close(beta(2.0, 3.0), 1.0 / 12.0, 1e-6);
}

#[test]
fn the_log_beta_form_switches_on_the_smaller_shape() {
    use LnBetaForm::{CombinedStirling, RisingFactorial};
    // `ln_beta` sorts its arguments, so `base ≥ increment` always holds at this call.
    assert_eq!(
        LnBetaForm::for_arguments(2048.0, LN_BETA_STIRLING_MIN),
        CombinedStirling
    );
    assert_eq!(
        LnBetaForm::for_arguments(2048.0, LN_BETA_STIRLING_MIN.next_down()),
        RisingFactorial
    );
    assert_eq!(LnBetaForm::for_arguments(2048.0, 1.0), RisingFactorial);
    // A non-finite base keeps the rising-factorial arm, which propagates rather than forming
    // `ln_1p` of a ratio with an infinity in it.
    assert_eq!(
        LnBetaForm::for_arguments(f32::INFINITY, 4096.0),
        RisingFactorial
    );

    // Both arms against the `f64` kernel. `B(a,b)` is tiny at these shapes — `ln B(2048, 2048)`
    // is about `−2840` — so this is the Stirling arm doing the work the fraction could not.
    for &(a, b) in &[
        (2048.0_f32, 2048.0_f32),
        (4096.0, 1024.0),
        (1e6, 1e6),
        (1e30, 1e30),
    ] {
        assert_eq!(
            LnBetaForm::for_arguments(a.max(b), a.min(b)),
            CombinedStirling
        );
        agrees(
            ln_beta(a, b),
            crate::kernels::beta::ln_beta(a as f64, b as f64),
            1e-5,
        );
    }
    // And the rising-factorial arm, below the switch.
    for &(a, b) in &[
        (2.0_f32, 3.0_f32),
        (0.5, 0.5),
        (100.0, 4.0),
        (1023.0, 1023.0),
    ] {
        agrees(
            ln_beta(a, b),
            crate::kernels::beta::ln_beta(a as f64, b as f64),
            1e-4,
        );
    }
}
