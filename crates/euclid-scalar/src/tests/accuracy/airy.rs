//! Reference-value accuracy for the Airy family, one sample inside every branch.
//!
//! The identity suite ([`crate::tests::identities::airy_family`]) does the heavy lifting here —
//! the Wronskian constrains all four functions at once and needs no transcribed digits at all.
//! What it *cannot* catch is a consistent error shared by all four, since `Ai·Bi′ − Ai′·Bi`
//! would still come out right. That is what this file is for.
//!
//! # Provenance of the values
//!
//! Generated out-of-band at 200+ `Decimal` digits from the Maclaurin series
//! `Ai = c₁f − c₂g`, with the working precision raised to beat that subtraction's cancellation
//! (about `2ζ/ln 10` digits, so 580 of them at `x = 100`).
//!
//! The two large positive samples were additionally **cross-checked against a second,
//! independent reference**: the asymptotic expansion `Σ(−1)^k u_k ζ^{−k}`, whose coefficients
//! are exact rationals and which therefore shares nothing with the series — not even a `Γ`
//! value. They agree to `4e-26` at `x = 12` and `8e-75` at `x = 25`.
//!
//! That cross-check was not a formality. The naive generator — Maclaurin at a fixed 140 digits,
//! which is plenty everywhere else in this crate — returns `Ai(80) = −3.8e-155` where the true
//! value is `+6.4e-209`: not merely inaccurate but the wrong sign and 54 orders of magnitude
//! out, because `Γ(⅓)` was only known to ~140 digits while the cancellation needed 415. A
//! reference is only a reference once something independent has agreed with it.

use crate::Real;
use euclid_utils::assert_close;

/// `(x, Ai, Ai′, Bi, Bi′)`. One point per branch: the negative Hankel tail, the negative march,
/// the negative series, the positive series, the `K` quadrature, and the `I` asymptotic.
const REFERENCE: &[(f64, f64, f64, f64, f64)] = &[
    (
        -25.0,
        0.16352657883042948,
        0.9623788513876974,
        -0.19214681569037803,
        0.8157197157546059,
    ),
    (
        -15.0,
        0.2782174908708289,
        0.272374204308642,
        -0.06912659453101005,
        1.0764297530843747,
    ),
    (
        -9.0,
        -0.022133721547341403,
        -0.9756639809263316,
        0.3249473234552449,
        -0.05740051384366925,
    ),
    (
        -5.0,
        0.35076100902411433,
        0.32719281855444315,
        -0.13836913490160058,
        0.7784117730018992,
    ),
    (
        -3.0,
        -0.37881429367765806,
        0.3145837692165988,
        -0.19828962637492653,
        -0.6756112226852585,
    ),
    (
        -1.0,
        0.5355608832923521,
        -0.01016056711664521,
        0.1039973894969446,
        0.5923756264227924,
    ),
    (
        -0.5,
        0.4757280916105396,
        -0.20408167033954738,
        0.38035265975105387,
        0.5059337136238472,
    ),
    (
        0.5,
        0.23169360648083348,
        -0.2249105326646839,
        0.8542770431031554,
        0.5445725641405923,
    ),
    (
        1.0,
        0.13529241631288141,
        -0.1591474412967932,
        1.2074235949528713,
        0.9324359333927756,
    ),
    (
        2.5,
        0.01572592338047049,
        -0.026250881035903232,
        6.481660738460579,
        9.421423317334302,
    ),
    (
        5.0,
        0.00010834442813607442,
        -0.0002474138908684625,
        657.7920441711711,
        1435.8190802179824,
    ),
    (
        9.0,
        2.47116843087249e-09,
        -7.480641389658946e-09,
        21472868.891435347,
        63807489.78090821,
    ),
    (
        12.0,
        1.3931846888753607e-13,
        -4.854736554985309e-13,
        329807225829.07416,
        1135507502443.3708,
    ),
    (
        25.0,
        8.116026824691387e-38,
        -4.066089337243281e-37,
        3.9220307780413816e+35,
        1.957073508323331e+36,
    ),
];

/// The tolerance `ζ` earns at `x`, where `ζ = ⅔|x|^{3/2}`.
///
/// Both axes carry a `ζ·ε` term, for mirror-image reasons, and both are documented on the
/// trait. On the **positive** axis `ζ` is a magnitude and reconstructing an unscaled value
/// costs a multiplication by `e^{±ζ}`, whose relative error is `ζ·ε`. On the **negative** axis
/// `ζ` is a *phase*, and one ulp of it moves the value by about `ulp(ζ)` times the amplitude.
/// Different mechanisms, same bound, so one function serves both.
///
/// Writing the tolerance this way rather than as a flat worst case keeps the small-`|x|` end of
/// the table as strict as it can be — `1e-15` at `x = ±1` — while admitting the `3.7e-14` that
/// is genuinely unavoidable at `x = −28`.
fn zeta_tolerance(x: f64) -> f64 {
    let zeta = (2.0 / 3.0) * x.abs() * x.abs().sqrt();
    1e-15 + 4.0 * zeta * f64::EPSILON
}

#[test]
fn the_airy_family_matches_its_reference_values() {
    // Nothing in the table is near a zero, deliberately: relative comparisons at a zero are
    // meaningless, and the zeros get their own treatment in the poles suite.
    for &(x, ai, ai_prime, bi, bi_prime) in REFERENCE {
        let tol = zeta_tolerance(x);
        assert_close(Real::airy_ai(x), ai, tol, &format!("Ai({x})"));
        assert_close(Real::airy_ai_prime(x), ai_prime, tol, &format!("Ai'({x})"));
        assert_close(Real::airy_bi(x), bi, tol, &format!("Bi({x})"));
        assert_close(Real::airy_bi_prime(x), bi_prime, tol, &format!("Bi'({x})"));
    }
}

#[test]
fn the_scaled_forms_match_the_same_references() {
    // The same values read through the scaling, which exercises a *different* code path on the
    // positive axis: the scaled forms never form `e^{ζ}` at all, so they are not simply the
    // unscaled ones times a constant. That makes this an independent check rather than a
    // restatement.
    //
    // The tolerance cannot tighten below [`zeta_tolerance`] even though the implementation is
    // more accurate here, because it is now the *expected value* that is scaled: multiplying
    // the reference by `e^{ζ}` in the test costs the same `ζ·ε` the implementation avoided.
    for &(x, ai, ai_prime, bi, bi_prime) in REFERENCE {
        let growth = if x > 0.0 {
            ((2.0 / 3.0) * x * x.sqrt()).exp()
        } else {
            1.0
        };
        let tol = zeta_tolerance(x);
        assert_close(Real::airy_ai_e(x), ai * growth, tol, &format!("Aie({x})"));
        assert_close(
            Real::airy_ai_e_prime(x),
            ai_prime * growth,
            tol,
            &format!("Aie'({x})"),
        );
        assert_close(Real::airy_bi_e(x), bi / growth, tol, &format!("Bie({x})"));
        assert_close(
            Real::airy_bi_e_prime(x),
            bi_prime / growth,
            tol,
            &format!("Bie'({x})"),
        );
    }
}
