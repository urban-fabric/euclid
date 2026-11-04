//! Identities of the Bessel family: `J₀`, `J₁`, `Y₀`, `Y₁`, `I₀`, `I₁`, `K₀`, `K₁`.
//!
//! The family is unusually well served by identity testing, because its members are tied
//! together by relations that hold for *no* incorrect implementation:
//!
//! - **The Wronskians** pin each kind against the other kind of the same equation. Because the
//!   two solutions are independent, a Wronskian is a single scalar function of `x` that both
//!   must conspire to produce — `J₀Y₁ − J₁Y₀ = −2/(πx)` and `I₀K₁ + I₁K₀ = 1/x`.
//! - **The differential equation** each pair satisfies constrains every one of them alone.
//! - **The recurrences** relate neighbouring orders, and the derivative relations relate a
//!   function to its neighbour rather than to itself.
//!
//! These matter more here than for most families because the implementations switch method
//! across the line (series, quadrature, outward march, asymptotic expansion), so an identity
//! swept across `x` crosses every branch and every join.

use crate::Real;
use euclid_utils::{assert_close, integrate, sweep};

// --- Wronskians ---------------------------------------------------------

#[test]
fn the_oscillatory_wronskian_holds_across_every_regime() {
    // J0(x)Y1(x) - J1(x)Y0(x) = -2/(pi x).
    for x in sweep(0.05, 60.0, 1200) {
        let residual = Real::bessel_j0(x) * Real::bessel_y1(x)
            - Real::bessel_j1(x) * Real::bessel_y0(x)
            + 2.0 / (core::f64::consts::PI * x);
        // Judged against the Wronskian's own magnitude, 2/(pi x).
        let scale = 2.0 / (core::f64::consts::PI * x);
        assert!(
            residual.abs() <= 1e-13 * scale.max(1e-3),
            "J/Y Wronskian broken at x = {x}: residual {residual:e}"
        );
    }
}

#[test]
fn the_modified_wronskian_holds_from_subnormal_to_enormous() {
    // I0(x)K1(x) + I1(x)K0(x) = 1/x, written in the scaled forms so the e^{-x} and e^{+x}
    // factors cancel pairwise. Unscaled, this identity could not be checked past x = 710 at
    // all: one factor overflows while the other underflows.
    for &x in &[
        1e-300f64, 1e-100, 1e-20, 1e-8, 0.01, 0.5, 1.0, 2.0, 5.0, 50.0, 500.0, 1e4, 1e6,
    ] {
        let residual =
            Real::bessel_i0e(x) * Real::bessel_k1e(x) + Real::bessel_i1e(x) * Real::bessel_k0e(x);
        assert_close(
            residual * x,
            1.0,
            1e-13,
            &format!("I/K Wronskian at x = {x}"),
        );
    }
}

// --- the differential equations -----------------------------------------

#[test]
fn each_pair_satisfies_its_own_differential_equation() {
    // J and Y solve x^2 y'' + x y' + x^2 y = 0; I and K solve x^2 y'' + x y' - x^2 y = 0.
    // Second differences amplify the function's own error by 1/h^2, so h cannot be tiny: at
    // h = 1e-4 x the truncation is O(h^2) and the amplified rounding O(eps/h^2) land together
    // around 1e-8.
    for x in sweep(0.5, 25.0, 60) {
        let h = 1e-4 * x;
        for (name, f, sign) in [
            ("J0", Real::bessel_j0 as fn(f64) -> f64, 1.0),
            ("Y0", Real::bessel_y0 as fn(f64) -> f64, 1.0),
            ("K0", Real::bessel_k0 as fn(f64) -> f64, -1.0),
        ] {
            let second = (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h);
            let first = (f(x + h) - f(x - h)) / (2.0 * h);
            let residual = x * x * second + x * first + sign * x * x * f(x);
            // Scaled by how large the individual terms are, since K0 decays exponentially
            // while J0 and Y0 stay O(1).
            let scale = (x * x * f(x).abs()).max(1e-13);
            assert!(
                residual.abs() <= 1e-6 * scale,
                "{name} breaks its equation at x = {x}: residual {residual:e}"
            );
        }
    }
}

// --- derivative relations -----------------------------------------------

#[test]
fn the_first_order_is_the_negated_derivative_of_the_zeroth() {
    // J0' = -J1, Y0' = -Y1, K0' = -K1, and I0' = +I1 (the modified family does not alternate).
    for x in sweep(0.2, 30.0, 150) {
        let h = 1e-5 * x;
        let d_j0 = (Real::bessel_j0(x + h) - Real::bessel_j0(x - h)) / (2.0 * h);
        assert_close(d_j0, -Real::bessel_j1(x), 1e-7, "J0' = -J1");
        let d_y0 = (Real::bessel_y0(x + h) - Real::bessel_y0(x - h)) / (2.0 * h);
        assert_close(d_y0, -Real::bessel_y1(x), 1e-7, "Y0' = -Y1");
    }
    // K and I decay/grow exponentially, so compare the *scaled* forms' logs instead, which
    // stay O(1): d/dx ln K0 = -K1/K0.
    for x in sweep(0.5, 20.0, 80) {
        let h = 1e-5 * x;
        let d_ln_k0 = (Real::bessel_k0(x + h).ln() - Real::bessel_k0(x - h).ln()) / (2.0 * h);
        assert_close(
            d_ln_k0,
            -Real::bessel_k1(x) / Real::bessel_k0(x),
            1e-7,
            "d ln K0 = -K1/K0",
        );
        let d_ln_i0 = (Real::ln_bessel_i0(x + h) - Real::ln_bessel_i0(x - h)) / (2.0 * h);
        assert_close(d_ln_i0, Real::bessel_i1_over_i0(x), 1e-8, "d ln I0 = I1/I0");
    }
}

#[test]
fn the_x_times_first_order_derivative_relations_hold() {
    // (x J1)' = x J0 and (x Y1)' = x Y0; for the modified pair, (x K1)' = -x K0.
    for x in sweep(0.3, 25.0, 120) {
        let h = 1e-5 * x;
        let d_xj1 =
            ((x + h) * Real::bessel_j1(x + h) - (x - h) * Real::bessel_j1(x - h)) / (2.0 * h);
        assert_close(d_xj1, x * Real::bessel_j0(x), 1e-6, "(x J1)' = x J0");
        let d_xy1 =
            ((x + h) * Real::bessel_y1(x + h) - (x - h) * Real::bessel_y1(x - h)) / (2.0 * h);
        assert_close(d_xy1, x * Real::bessel_y0(x), 1e-6, "(x Y1)' = x Y0");
    }
}

// --- recurrences --------------------------------------------------------

#[test]
fn the_order_recurrence_generates_a_consistent_second_order() {
    // 2n/x · f_n = f_{n-1} + f_{n+1}, so f_2 = 2 f_1 / x - f_0. The second order is not on the
    // surface, but it must satisfy its *own* differential equation, which closes the loop
    // without needing a reference value for it:
    //     x^2 f'' + x f' + (x^2 - 4) f = 0.
    let j2 = |x: f64| 2.0 * Real::bessel_j1(x) / x - Real::bessel_j0(x);
    let y2 = |x: f64| 2.0 * Real::bessel_y1(x) / x - Real::bessel_y0(x);
    for x in sweep(1.0, 20.0, 40) {
        let h = 1e-4 * x;
        for (name, f) in [
            ("J2", &j2 as &dyn Fn(f64) -> f64),
            ("Y2", &y2 as &dyn Fn(f64) -> f64),
        ] {
            let second = (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h);
            let first = (f(x + h) - f(x - h)) / (2.0 * h);
            let residual = x * x * second + x * first + (x * x - 4.0) * f(x);
            assert!(
                residual.abs() <= 1e-6 * (x * x).max(1.0),
                "{name} from the recurrence breaks its equation at x = {x}: {residual:e}"
            );
        }
    }
    // The modified family's recurrence carries the opposite sign:
    // I_{n-1} - I_{n+1} = 2n I_n / x, and K_{n+1} - K_{n-1} = 2n K_n / x.
    for x in sweep(0.5, 15.0, 40) {
        let k2 = 2.0 * Real::bessel_k1(x) / x + Real::bessel_k0(x);
        // K is increasing in its order, so K2 > K1 > K0 > 0.
        assert!(
            k2 > Real::bessel_k1(x) && Real::bessel_k1(x) > Real::bessel_k0(x),
            "K must increase with order at x = {x}"
        );
    }
}

// --- integral representations -------------------------------------------

#[test]
fn i0_matches_its_integral_representation() {
    // I0(x) = (1/pi) int_0^pi e^{x cos t} dt. Smooth and periodic, so Simpson converges fast;
    // this is an entirely different route from the peak-factored series under test.
    for x in sweep(0.25, 8.0, 20) {
        let quadrature = integrate(|t| (x * t.cos()).exp(), 0.0, core::f64::consts::PI, 4000)
            / core::f64::consts::PI;
        assert_close(
            Real::bessel_i0(x),
            quadrature,
            1e-10,
            &format!("I0({x}) by quadrature"),
        );
    }
}

#[test]
fn k0_matches_its_integral_representation() {
    // K0(x) = int_0^inf e^{-x cosh t} dt, truncated where the integrand has died. The
    // implementation uses a trapezoidal rule on this same integral, so Simpson here is a
    // check on the *node placement and truncation*, not on the representation.
    for x in sweep(0.5, 6.0, 12) {
        let upper = (1.0 + 60.0 / x).acosh();
        let quadrature = integrate(|t| (-x * t.cosh()).exp(), 0.0, upper, 20_000);
        assert_close(
            Real::bessel_k0(x),
            quadrature,
            1e-9 * Real::bessel_k0(x),
            &format!("K0({x}) by quadrature"),
        );
    }
}

#[test]
fn j0_matches_its_laplace_transform() {
    // int_0^inf e^{-t} J0(t) dt = 1/sqrt(2). The tail past t = 40 contributes ~e^{-40}, well
    // below the tolerance. This ties J0 across its series/quadrature/asymptotic switch into a
    // single exactly-known number.
    let transform = integrate(|t| (-t).exp() * Real::bessel_j0(t), 0.0, 40.0, 40_000);
    assert_close(
        transform,
        1.0 / 2f64.sqrt(),
        1e-9,
        "int e^-t J0(t) dt = 1/sqrt(2)",
    );
}

#[test]
fn the_j1_antiderivative_identity_holds() {
    // int_0^x t J0(t) dt = x J1(x), which follows from (x J1)' = x J0 and so tests the pair
    // jointly over a whole interval rather than pointwise.
    for x in sweep(1.0, 20.0, 15) {
        let quadrature = integrate(|t| t * Real::bessel_j0(t), 0.0, x, 20_000);
        assert_close(
            quadrature,
            x * Real::bessel_j1(x),
            1e-8 * x,
            &format!("int_0^{x} t J0 = x J1(x)"),
        );
    }
}

// --- asymptotic envelopes -----------------------------------------------

#[test]
fn the_oscillatory_pair_lies_on_its_nicholson_envelope() {
    // J0(x)^2 + Y0(x)^2 -> 2/(pi x). The two are 90 degrees out of phase, so their squares
    // sum to a smooth envelope even though each oscillates. A phase error in either -- the
    // most likely defect in an asymptotic expansion -- breaks this immediately, while a
    // pointwise reference check at a handful of arguments could miss it.
    for x in sweep(50.0, 500.0, 200) {
        let modulus = Real::bessel_j0(x).powi(2) + Real::bessel_y0(x).powi(2);
        let envelope = 2.0 / (core::f64::consts::PI * x);
        // The next correction is 1/(8x^2) relative, i.e. 5e-6 at x = 50.
        assert_close(
            modulus / envelope,
            1.0,
            1e-4,
            &format!("Nicholson envelope at x = {x}"),
        );
    }
}

#[test]
fn the_modified_pair_meets_its_product_asymptote() {
    // I0e(x) K0e(x) -> 1/(2x), the modified counterpart of the envelope above. Checkable at
    // any argument because both factors are scaled.
    for x in [10.0f64, 100.0, 1e3, 1e5, 1e7] {
        let product = Real::bessel_i0e(x) * Real::bessel_k0e(x);
        assert_close(
            product * 2.0 * x,
            1.0,
            1e-2 / x.sqrt(),
            &format!("I0e K0e -> 1/2x at x = {x}"),
        );
    }
}
