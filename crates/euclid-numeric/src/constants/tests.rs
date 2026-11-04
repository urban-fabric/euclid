//! Provenance checks for the constant catalogue.
//!
//! Every constant is verified against an identity rather than against transcribed digits, so
//! a typo in any of them breaks the relation it is defined by. See the [parent module](super)
//! for the full table of which identity covers which constant.

use crate::Real;
use euclid_utils::assert_close;

/// Every constant is checked against an identity, not against transcribed digits. A typo in
/// any of them breaks the relation it is defined by.
const TOL: f64 = 1e-15;

#[test]
fn euler_gamma_is_negative_digamma_at_one() {
    assert_close(
        super::f64::EULER_GAMMA,
        -Real::digamma(1.0f64),
        1e-11,
        "gamma = -psi(1)",
    );
}

#[test]
fn golden_ratio_solves_its_quadratic() {
    let phi = super::f64::GOLDEN_RATIO;
    assert_close(phi * phi, phi + 1.0, TOL, "phi^2 = phi + 1");
    // And equals (1 + sqrt 5)/2.
    assert_close(phi, (1.0 + 5.0f64.sqrt()) / 2.0, TOL, "phi = (1+sqrt5)/2");
    // Its reciprocal is phi - 1.
    assert_close(1.0 / phi, phi - 1.0, TOL, "1/phi = phi - 1");
}

#[test]
fn the_square_roots_square_back() {
    assert_close(
        super::f64::SQRT_PI * super::f64::SQRT_PI,
        core::f64::consts::PI,
        TOL,
        "sqrt_pi^2 = pi",
    );
    assert_close(
        super::f64::SQRT_TAU * super::f64::SQRT_TAU,
        core::f64::consts::TAU,
        TOL,
        "sqrt_tau^2 = tau",
    );
    // sqrt(pi) is Gamma(1/2).
    assert_close(
        super::f64::SQRT_PI,
        Real::gamma(0.5f64),
        1e-14,
        "sqrt_pi = Gamma(1/2)",
    );
}

#[test]
fn the_logarithms_exponentiate_back() {
    assert_close(
        super::f64::LN_PI.exp(),
        core::f64::consts::PI,
        TOL,
        "exp(ln pi) = pi",
    );
    assert_close(
        super::f64::LN_TAU.exp(),
        core::f64::consts::TAU,
        TOL,
        "exp(ln tau) = tau",
    );
    assert_close(
        super::f64::HALF_LN_TAU,
        0.5 * super::f64::LN_TAU,
        TOL,
        "half_ln_tau = ln_tau / 2",
    );
    // And the normal-density use: exp(-half_ln_tau) = 1/sqrt(2 pi).
    assert_close(
        (-super::f64::HALF_LN_TAU).exp(),
        1.0 / super::f64::SQRT_TAU,
        TOL,
        "the normal normalizer",
    );
}

#[test]
fn apery_is_minus_half_the_second_polygamma_at_one() {
    // zeta(3) = -psi''(1)/2
    assert_close(
        super::f64::APERY,
        -Real::polygamma(1.0f64, 2) / 2.0,
        1e-11,
        "zeta(3) = -psi''(1)/2",
    );
}

#[test]
fn catalan_comes_from_trigamma_at_a_quarter() {
    // psi'(1/4) = pi^2 + 8G, so G = (psi'(1/4) - pi^2)/8.
    let pi2 = core::f64::consts::PI * core::f64::consts::PI;
    assert_close(
        super::f64::CATALAN,
        (Real::trigamma(0.25f64) - pi2) / 8.0,
        1e-10,
        "G = (psi'(1/4) - pi^2)/8",
    );
}

#[test]
fn omega_solves_its_defining_equation() {
    let w = super::f64::OMEGA;
    assert_close(w * w.exp(), 1.0, TOL, "Omega e^Omega = 1");
    // Equivalently, Omega = e^{-Omega} = ln(1/Omega).
    assert_close(w, (-w).exp(), TOL, "Omega = e^-Omega");
    assert_close(w, (1.0 / w).ln(), 1e-14, "Omega = ln(1/Omega)");
}

#[test]
fn the_plastic_number_solves_its_cubic() {
    let rho = super::f64::PLASTIC_NUMBER;
    assert_close(rho * rho * rho, rho + 1.0, TOL, "rho^3 = rho + 1");
}

#[test]
fn the_dottie_number_is_the_fixed_point_of_cosine() {
    let d = super::f64::DOTTIE;
    assert_close(d.cos(), d, TOL, "cos(d) = d");
    // And iterating cosine from anywhere converges to it, which is what makes it the
    // *unique* fixed point.
    let mut x = 1.0f64;
    for _ in 0..200 {
        x = x.cos();
    }
    assert_close(x, d, 1e-14, "cosine iteration converges to Dottie");
}

#[test]
fn the_lemniscate_constant_comes_from_gamma_at_a_quarter() {
    // varpi = Gamma(1/4)^2 / (2 sqrt(2 pi))
    let g = Real::gamma(0.25f64);
    assert_close(
        super::f64::LEMNISCATE,
        g * g / (2.0 * super::f64::SQRT_TAU),
        1e-13,
        "varpi = Gamma(1/4)^2 / (2 sqrt(2 pi))",
    );
}

#[test]
fn gauss_constant_is_the_lemniscate_over_pi() {
    assert_close(
        super::f64::GAUSS,
        super::f64::LEMNISCATE / core::f64::consts::PI,
        TOL,
        "G = varpi / pi",
    );
    // It is also 1/AGM(1, sqrt 2) — computed here by the arithmetic-geometric mean, which
    // converges quadratically.
    let (mut a, mut g) = (1.0f64, 2.0f64.sqrt());
    for _ in 0..10 {
        let next_a = 0.5 * (a + g);
        g = (a * g).sqrt();
        a = next_a;
    }
    assert_close(super::f64::GAUSS, 1.0 / a, 1e-15, "G = 1/AGM(1, sqrt 2)");
}

#[test]
fn the_zeta_values_match_their_closed_forms() {
    let pi = core::f64::consts::PI;
    assert_close(super::f64::ZETA_2, pi * pi / 6.0, TOL, "zeta(2) = pi^2/6");
    assert_close(
        super::f64::ZETA_4,
        pi.powi(4) / 90.0,
        TOL,
        "zeta(4) = pi^4/90",
    );
    // zeta(2) is psi'(1), which ties them to the polygamma ladder.
    assert_close(
        super::f64::ZETA_2,
        Real::trigamma(1.0f64),
        1e-11,
        "zeta(2) = psi'(1)",
    );
}

#[test]
fn neg_inv_e_is_where_x_times_exp_x_turns_around() {
    // The defining property, as a product rather than a reciprocal: the minimum of x e^x
    // is at x = -1, where the value is (-1)·e^{-1}. Forming it this way means the identity
    // does not merely restate the digits in a different order.
    let turning_point = -1.0f64;
    assert_close(
        super::f64::NEG_INV_E,
        turning_point * turning_point.exp(),
        TOL,
        "-1/e = (-1) e^-1",
    );
    // That x = -1 really is the turning point: x e^x is above -1/e on both sides of it.
    for &delta in &[-0.5f64, -0.1, -1e-3, 1e-3, 0.1, 0.5] {
        let x = turning_point + delta;
        assert!(
            x * x.exp() > super::f64::NEG_INV_E,
            "x e^x dips below -1/e at x = {x}"
        );
    }
    // And it is the Lambert branch point. Only ~8 digits are available from W here, which
    // is a property of the function rather than of the constant: the square-root behaviour
    // at the branch point halves the recoverable precision.
    assert_close(
        Real::lambert_w(super::f64::NEG_INV_E),
        -1.0,
        1e-8,
        "W(-1/e) = -1",
    );
    assert!(
        Real::lambert_w(super::f64::NEG_INV_E - 1e-12).is_nan(),
        "below the branch point W has no real value"
    );
}

#[test]
fn the_trait_accessors_return_the_module_constants() {
    // The digits live in this module once; the trait must agree with them on both carriers.
    assert_eq!(<f64 as Real>::euler_gamma(), super::f64::EULER_GAMMA);
    assert_eq!(<f64 as Real>::golden_ratio(), super::f64::GOLDEN_RATIO);
    assert_eq!(<f64 as Real>::sqrt_pi(), super::f64::SQRT_PI);
    assert_eq!(<f64 as Real>::sqrt_tau(), super::f64::SQRT_TAU);
    assert_eq!(<f64 as Real>::apery(), super::f64::APERY);
    assert_eq!(<f64 as Real>::catalan(), super::f64::CATALAN);
    assert_eq!(<f64 as Real>::omega(), super::f64::OMEGA);
    assert_eq!(<f64 as Real>::ln_pi(), super::f64::LN_PI);
    assert_eq!(<f64 as Real>::ln_tau(), super::f64::LN_TAU);
    assert_eq!(<f64 as Real>::half_ln_tau(), super::f64::HALF_LN_TAU);

    assert_eq!(<f32 as Real>::euler_gamma(), super::f32::EULER_GAMMA);
    assert_eq!(<f32 as Real>::golden_ratio(), super::f32::GOLDEN_RATIO);
    assert_eq!(<f32 as Real>::sqrt_pi(), super::f32::SQRT_PI);
    assert_eq!(<f32 as Real>::omega(), super::f32::OMEGA);
}

#[test]
fn the_f32_constants_are_the_narrowed_f64_ones() {
    // Each f32 value must be the f64 value rounded, not an independently-typed literal
    // that drifted.
    macro_rules! narrows {
        ($($name:ident),+ $(,)?) => {
            $({
                let wide = super::f64::$name;
                let narrow = super::f32::$name;
                assert_eq!(
                    narrow, wide as f32,
                    concat!(stringify!($name), " f32 is not the narrowed f64")
                );
            })+
        };
    }
    narrows!(
        EULER_GAMMA,
        GOLDEN_RATIO,
        SQRT_PI,
        SQRT_TAU,
        LN_PI,
        LN_TAU,
        HALF_LN_TAU,
        APERY,
        CATALAN,
        OMEGA,
        NEG_INV_E,
        PLASTIC_NUMBER,
        DOTTIE,
        LEMNISCATE,
        GAUSS,
        ZETA_2,
        ZETA_4,
    );
}
