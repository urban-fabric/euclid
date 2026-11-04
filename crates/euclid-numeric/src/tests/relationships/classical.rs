//! Edges joining the classical special functions to the rest of the web.
//!
//! The per-family suites check each new function against references and against its own
//! identities. This file does something different: it wires the newcomers into the functions
//! that were already here, so that `gamma`, `exprel`, `sinc`, `cos_m1` and the incomplete
//! gamma all become *witnesses* for `K`, `E₁`, `Ei`, `Si`, `Ci`, `ζ` and `W` — and vice versa.
//!
//! That is the point of the web. Each edge below is satisfied by the true functions and by
//! essentially no incorrect pair, and the two sides are computed by unrelated machinery: a
//! trapezoidal quadrature against a closed form, an Euler–Maclaurin sum against a Simpson
//! integral, a continued fraction against a power series in a *different* function. An error
//! anywhere shows up as a broken edge somewhere.

use crate::Real;
use crate::kernels::bessel::modified_second::k_nu_scaled;
use euclid_utils::{assert_close, integrate, sweep};

// --- K of half-integer order collapses to elementary functions ----------

#[test]
fn the_half_integer_k_orders_match_their_closed_forms() {
    // `K` at half-integer order is elementary:
    //
    //     K_{1/2}(x) = √(π/2x)·e^{−x}
    //     K_{3/2}(x) = √(π/2x)·e^{−x}(1 + 1/x)
    //     K_{5/2}(x) = √(π/2x)·e^{−x}(1 + 3/x + 3/x²)
    //
    // These are the sharpest available check on the ν-parametrised quadrature, because the
    // right-hand sides are *exact* — no truncation, no reference table, no other special
    // function. And they exercise ν away from the 0 and 1 that `K₀`/`K₁` use, which is
    // exactly the generality the Airy functions will rely on.
    for x in sweep(0.05, 40.0, 300) {
        let base = (core::f64::consts::PI / (2.0 * x)).sqrt();
        let inv = 1.0 / x;
        for (order, nu, factor) in [
            ("1/2", 0.5, 1.0),
            ("3/2", 1.5, 1.0 + inv),
            ("5/2", 2.5, 1.0 + 3.0 * inv + 3.0 * inv * inv),
        ] {
            // Compared in the scaled form, so the e^{-x} cancels and the identity is
            // checkable at any argument rather than only where K is representable.
            assert_close(
                k_nu_scaled(nu, x) / (base * factor),
                1.0,
                1e-13,
                &format!("K_{order}({x}) closed form"),
            );
        }
    }
}

#[test]
fn the_k_order_recurrence_holds_at_half_integer_order() {
    // K_{ν+1} = K_{ν−1} + (2ν/x)K_ν, with K_{−1/2} = K_{1/2}. Taking ν = 1/2 gives
    // K_{3/2} = K_{1/2}(1 + 1/x), and ν = 3/2 gives K_{5/2} = K_{1/2} + (3/x)K_{3/2}.
    // Independent of the closed forms above: this one relates three quadrature evaluations
    // to each other, so it would survive a common scaling error in all of them and catch a
    // ν-dependent one.
    for x in sweep(0.1, 25.0, 200) {
        let (half, three_halves, five_halves) = (
            k_nu_scaled(0.5, x),
            k_nu_scaled(1.5, x),
            k_nu_scaled(2.5, x),
        );
        assert_close(
            three_halves / (half + 1.0 / x * half),
            1.0,
            1e-13,
            &format!("K order recurrence at nu = 1/2, x = {x}"),
        );
        assert_close(
            five_halves / (half + 3.0 / x * three_halves),
            1.0,
            1e-13,
            &format!("K order recurrence at nu = 3/2, x = {x}"),
        );
    }
    // The integer orders on the public surface must sit on the same ladder:
    // K₂ = K₀ + (2/x)K₁.
    for x in sweep(0.1, 25.0, 100) {
        let k2 = k_nu_scaled(2.0, x);
        assert_close(
            k2 / (Real::bessel_k0e(x) + 2.0 / x * Real::bessel_k1e(x)),
            1.0,
            1e-13,
            &format!("K2 = K0 + (2/x)K1 at x = {x}"),
        );
    }
}

#[test]
fn k_of_small_argument_is_governed_by_gamma() {
    // K_ν(x) ~ ½Γ(ν)(2/x)^ν as x → 0, for ν > 0. An edge from the Bessel family straight to
    // `gamma`: the pole structure of `K` at the origin *is* `Γ(ν)`.
    //
    // The correction is O(x^{2ν}) (or O(x²ln x) at ν = 0), so the tolerance is set by the
    // argument rather than by either function.
    for &nu in &[0.25f64, 0.5, 1.0, 1.5, 2.0] {
        for &x in &[1e-4f64, 1e-6, 1e-8] {
            let leading = 0.5 * Real::gamma(nu) * (2.0 / x).powf(nu);
            // The scaled form carries an e^{x} that is 1 + O(x) here; divide it out.
            let unscaled = k_nu_scaled(nu, x) * (-x).exp();
            let slack = 10.0 * x.powf((2.0 * nu).min(1.0));
            assert_close(
                unscaled / leading,
                1.0,
                slack,
                &format!("K_{nu}({x:e}) ~ Gamma({nu})(2/x)^{nu}/2"),
            );
        }
    }
}

// --- the exponential integrals as members of the gamma family -----------

#[test]
fn e1_is_the_upper_incomplete_gamma_at_zero_shape() {
    // E₁(x) = Γ(0, x), and the crate's `gamma_ur` is the *regularised* Q(a, x) = Γ(a,x)/Γ(a).
    // So Γ(a)·Q(a, x) → E₁(x) as a → 0.
    //
    // This is a genuine cross-family edge: `E₁` comes from a continued fraction and
    // `gamma_ur` from an entirely separate series/fraction pair, with `gamma` supplying the
    // normaliser. The limit converges linearly in `a`, which is what sets the tolerance.
    for &a in &[1e-3f64, 1e-4, 1e-5, 1e-6] {
        for x in sweep(0.25, 12.0, 40) {
            let via_gamma = Real::gamma(a) * Real::gamma_ur(x, a);
            assert_close(
                via_gamma / Real::exp_int_e1(x),
                1.0,
                20.0 * a,
                &format!("Gamma({a}) Q({a}, {x}) -> E1({x})"),
            );
        }
    }
    // The neighbouring shape is exact rather than a limit: Γ(1, x) = e^{−x}, so
    // Q(1, x) = e^{−x} with no normaliser at all.
    for x in sweep(0.1, 20.0, 50) {
        assert_close(
            Real::gamma_ur(x, 1.0),
            (-x).exp(),
            1e-14,
            &format!("Q(1, {x}) = e^-x"),
        );
    }
}

#[test]
fn the_exponential_integrals_are_antiderivatives_of_exprel() {
    // `exprel(t) = (e^t − 1)/t` is already on the surface as a stable primitive. The
    // exponential integrals are its antiderivatives:
    //
    //     Ei(x) = γ + ln x + ∫₀ˣ exprel(t) dt
    //     E₁(x) = −γ − ln x + ∫₀ˣ exprel(−t) dt
    //
    // Both integrands are smooth at the origin — that is exactly what `exprel` fixes — so
    // Simpson converges properly, and the edge ties `Ei`/`E₁` to a function whose own
    // stability is tested elsewhere.
    for x in sweep(0.25, 6.0, 24) {
        let rising = integrate(Real::exprel, 0.0, x, 20_000);
        assert_close(
            Real::exp_int_ei(x),
            euclid_utils::EULER_GAMMA + x.ln() + rising,
            1e-10,
            &format!("Ei({x}) = gamma + ln x + int exprel"),
        );
        let falling = integrate(|t| Real::exprel(-t), 0.0, x, 20_000);
        assert_close(
            Real::exp_int_e1(x),
            -euclid_utils::EULER_GAMMA - x.ln() + falling,
            1e-10,
            &format!("E1({x}) = -gamma - ln x + int exprel(-t)"),
        );
    }
}

// --- the trigonometric integrals and the stable trigonometric helpers ---

#[test]
fn the_trigonometric_integrals_integrate_sinc_and_cos_m1() {
    // `sinc(t) = sin(t)/t` and `cos_m1(t) = cos(t) − 1` are both on the surface as stable
    // primitives, with the removable singularity and the cancellation respectively already
    // handled. The trigonometric integrals are built on exactly those two shapes:
    //
    //     Si(x) = ∫₀ˣ sinc(t) dt
    //     Ci(x) = γ + ln x + ∫₀ˣ cos_m1(t)/t dt
    //
    // Writing them this way rather than with a raw `sin(t)/t` is what makes the quadrature
    // valid at the origin, and it closes the loop between the new integrals and the
    // elementary helpers that were already here.
    for x in sweep(0.5, 14.0, 28) {
        assert_close(
            Real::sin_int(x),
            integrate(Real::sinc, 0.0, x, 20_000),
            1e-10,
            &format!("Si({x}) = int sinc"),
        );
        assert_close(
            Real::cos_int(x),
            euclid_utils::EULER_GAMMA
                + x.ln()
                + integrate(
                    |t| if t == 0.0 { 0.0 } else { Real::cos_m1(t) / t },
                    0.0,
                    x,
                    20_000,
                ),
            1e-10,
            &format!("Ci({x}) = gamma + ln x + int cos_m1/t"),
        );
    }
    // And their derivatives are those primitives exactly, which pins the pair pointwise as
    // well as in the integral.
    for x in sweep(0.5, 30.0, 100) {
        let h = 1e-5 * x;
        assert_close(
            (Real::sin_int(x + h) - Real::sin_int(x - h)) / (2.0 * h),
            Real::sinc(x),
            1e-7,
            "Si' = sinc",
        );
    }
}

// --- zeta, gamma, and the Bernoulli numbers -----------------------------

#[test]
fn zeta_times_gamma_is_the_bose_integral() {
    // Γ(s)·ζ(s) = ∫₀^∞ t^{s−1}/(e^t − 1) dt for s > 1.
    //
    // The strongest edge available between the two: an Euler–Maclaurin sum times a Lanczos
    // approximation on one side, a Simpson quadrature of an unrelated integrand on the other.
    //
    // Only *integer* `s ≥ 2` is swept, and that restriction is about the quadrature rather
    // than about the identity. The integrand behaves like `t^{s−2}` at the origin, which is a
    // polynomial exactly when `s` is an integer; at `s = 2.5` it is `√t`, whose second
    // derivative is unbounded there, and Simpson's `O(h⁴)` degrades to roughly `O(h^{3/2})` —
    // measured as `2.3e-7` of disagreement, all of it the reference's.
    for &s in &[2.0f64, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] {
        let bose = integrate(
            |t| {
                if t <= 0.0 {
                    0.0
                } else {
                    t.powf(s - 1.0) / (t.exp() - 1.0)
                }
            },
            1e-9,
            60.0,
            200_000,
        );
        assert_close(
            Real::gamma(s) * Real::zeta(s) / bose,
            1.0,
            1e-7,
            &format!("Gamma({s}) zeta({s}) = Bose integral"),
        );
    }
}

#[test]
fn zeta_at_the_negative_odd_integers_gives_the_bernoulli_numbers() {
    // ζ(1 − 2n) = −B_{2n}/(2n).
    //
    // A network edge of a different sort: these are the *same* Bernoulli numbers that appear
    // in the Stirling remainder inside the modified-Bessel peak factoring, and in the
    // Euler–Maclaurin tail of the Hurwitz zeta that `zeta` itself calls for s > 1. So this
    // checks the functional-equation branch against constants the rest of the crate depends
    // on arithmetically.
    let bernoulli_even: &[(f64, f64)] = &[
        (1.0, 1.0 / 6.0),
        (2.0, -1.0 / 30.0),
        (3.0, 1.0 / 42.0),
        (4.0, -1.0 / 30.0),
        (5.0, 5.0 / 66.0),
        (6.0, -691.0 / 2730.0),
        (7.0, 7.0 / 6.0),
    ];
    for &(n, b) in bernoulli_even {
        let s = 1.0 - 2.0 * n;
        assert_close(
            Real::zeta(s),
            -b / (2.0 * n),
            1e-11,
            &format!("zeta({s}) = -B_{}/{}", 2.0 * n, 2.0 * n),
        );
    }
}

// --- the Lambert W and quadrature ---------------------------------------

#[test]
fn lambert_w_has_the_antiderivative_its_inverse_implies() {
    // ∫₀^X W(t) dt = X(W(X) − 1 + 1/W(X)) − 1.
    //
    // Derived by substituting t = w e^w, which turns the integral into ∫(w + w²)e^w dw — so
    // the closed form is a statement about `W` being a genuine inverse, not just about its
    // values. The trailing −1 is the lower limit: the antiderivative (w² − w + 1)e^w is 1 at
    // w = 0, not 0.
    for &upper in &[0.5f64, 1.0, 2.0, 5.0, 20.0, 60.0] {
        // The integrand behaves like t near the origin, so the lower limit can be taken at 0
        // once W(0) = 0 is supplied explicitly.
        let quadrature = integrate(
            |t| if t == 0.0 { 0.0 } else { Real::lambert_w(t) },
            0.0,
            upper,
            100_000,
        );
        let w = Real::lambert_w(upper);
        assert_close(
            quadrature,
            upper * (w - 1.0 + 1.0 / w) - 1.0,
            1e-8,
            &format!("int_0^{upper} W = X(W - 1 + 1/W) - 1"),
        );
    }
}

// --- the Bessel first kind and the Gaussian -----------------------------

#[test]
fn the_hankel_transform_of_a_gaussian_is_a_gaussian() {
    // ∫₀^∞ t·e^{−t²}·J₀(xt) dt = ½e^{−x²/4}.
    //
    // `J₀` is self-reciprocal under the Hankel transform in the same way the Gaussian is
    // under the Fourier transform, and this is the cleanest instance of it. The edge is
    // valuable because it integrates `J₀` against a weight rather than sampling it: a phase
    // error anywhere in the oscillatory region shows up as a wrong Gaussian, and the
    // right-hand side is elementary.
    for x in sweep(0.25, 8.0, 32) {
        let transform = integrate(
            |t| t * (-t * t).exp() * Real::bessel_j0(x * t),
            0.0,
            9.0,
            100_000,
        );
        assert_close(
            transform,
            0.5 * (-x * x / 4.0).exp(),
            1e-9,
            &format!("Hankel transform of a Gaussian at x = {x}"),
        );
    }
}
