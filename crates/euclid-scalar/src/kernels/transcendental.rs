//! Two transcendental functions that belong to no one family: the Lambert `W` and the Riemann
//! zeta.
//!
//! The other classical functions in this tree group naturally — the Bessel family in
//! [`bessel`](super::bessel), the non-elementary antiderivatives in
//! [`integrals`](super::integrals) — and these two do not. `W` is the inverse of `x ↦ x e^x`,
//! the function needed whenever an unknown appears both inside and outside an exponential,
//! which is a shape that recurs across analysis rather than within one field. `ζ` is the
//! central object of analytic number theory and the value of several otherwise-unnameable
//! constants.

use crate::kernels::gamma::ln_gamma;
use crate::kernels::polygamma::hurwitz_zeta;

mod lambert;

pub(crate) use lambert::lambert_w;

/// Terms used by [`zeta_eta`]. The Cohen–Rodriguez Villegas–Zagier acceleration
/// converges like `(3 + √8)^{−n} ≈ 5.83^{−n}`, so 24 terms give `≈ 2e-18`.
const CVZ_TERMS: u32 = 24;

/// `T_n(3)`, the integer Chebyshev normalization for the eta accelerator.
/// Fits in `u64` through `n = 25`.
const fn cvz_normalizer(terms: u32) -> u64 {
    if terms == 0 {
        return 1;
    }
    let (mut previous, mut current) = (1, 3);
    let mut index = 1;
    while index < terms {
        (previous, current) = (current, 6 * current - previous);
        index += 1;
    }
    current
}

/// `ζ(s)` for `-1 < s < 1`, through the accelerated Dirichlet eta series.
///
/// ```text
/// η(s) = Σ_{k≥1} (−1)^{k−1} k^{−s} = (1 − 2^{1−s})·ζ(s)
/// ```
///
/// `η` converges — conditionally — for every `s > 0`, which is what gets inside the critical
/// strip where both of `zeta`'s other routes fail. Conditional convergence is far too slow to
/// use directly (the terms only decay like `k^{−s}`, so `s = 1/2` would need `1e32` of them),
/// so the sum is accelerated by the Cohen–Rodriguez Villegas–Zagier algorithm, which
/// reweights the first `n` terms by coefficients derived from Chebyshev polynomials and turns
/// the conditional convergence into geometric convergence at rate `(3 + √8)^{−n}`.
/// The acceleration also continues eta across zero. Using it just below zero avoids
/// reflection's loss of `s` in `1-s`: for tiny negative `s` that rounds to exactly `1`,
/// spuriously evaluating the Hurwitz-zeta pole even though `ζ(0) = -1/2` is regular.
///
/// # It degrades on the negative side, and only there
///
/// The accelerated sum weights its `k`th term by `(k+1)^{-s}`, so for `s < 0` the *later*
/// coefficients — which are the large ones, growing like `(3+√8)^n` — get amplified, and the
/// cancellation that produces an `O(1)` answer from them gets worse as `s` falls. Measured
/// against a 400-bit reference, worst ulp per band:
///
/// ```text
/// s        [0.01,0.99]  [-0.25,-0.01]  [-0.75,-0.25]  [-0.99,-0.75]
/// eta              7-9             23             54            210   ulp
/// ```
///
/// Positive `s` — the critical strip, which is the reason this route exists — is unaffected.
/// The remedy for the negative side is not a better sum but a shorter range: see
/// [`ZETA_ETA_MIN`].
///
/// # The divisor near the pole
///
/// `1 − 2^{1−s}` vanishes at `s = 1`, which the caller has already handled as the pole — but
/// writing it that way is a cancellation bug just below it. For `s = 1 − δ` the true divisor
/// is `−δ ln 2`, and the literal form computes it as `1 − (1 + δ ln 2)`: by `δ = 1e-15` that
/// has lost 4% of its value, and at `δ = 1e-16` the sum rounds to exactly `1`, so the divisor
/// is `0` and `ζ` comes back `+∞` with the wrong sign where the true value is `−9.0e15`.
///
/// Phrased with `exp_m1` the subtraction never happens — `1 − e^{u} = −expm1(u)` — and the
/// divisor is exact right up to the pole.
fn zeta_eta(s: f64) -> f64 {
    debug_assert!(s > ZETA_ETA_MIN && s < 1.0);
    let n = f64::from(CVZ_TERMS);

    const NORMALIZER: u64 = cvz_normalizer(CVZ_TERMS);
    let d = NORMALIZER as f64;

    let mut b = -1.0f64;
    let mut c = -d;
    let mut total = 0.0f64;
    for k in 0..CVZ_TERMS {
        let kf = f64::from(k);
        c = b - c;
        // The kth term of eta is (k+1)^{-s}.
        total += c * (kf + 1.0).powf(-s);
        b = crate::math::zeta::next_cvz_coefficient(kf, n, b);
    }
    let eta = total / d;
    eta / crate::math::zeta::eta_divisor(s)
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ZetaRoute {
    Value(f64),
    Hurwitz,
    Eta,
    Reflection,
}

/// Below this `s` the reflection replaces the eta acceleration.
///
/// **`−1/8`, raised from `−1`.** The eta route had the whole of `(−1, 0)` because the reflection
/// needs `1 − s` and that loses `s` entirely as `s → 0⁻`. True, and it does not need the whole
/// interval to avoid it: by `s = −1/8` the reflected argument is `9/8`, far enough from `ζ`'s
/// pole at `1` that the rounding of `1 − s` costs about an ulp.
///
/// Measured, worst ulp per band against a 400-bit reference:
///
/// ```text
/// s              [-0.99,-0.9]  [-0.9,-0.75]  [-0.75,-0.5]  [-0.5,-0.25]  [-0.25,-0.125]
/// handover −1             171           172            72            32              17   ulp
/// handover −1/8             8             9             7            11               8
/// ```
///
/// It stops at `−1/8` because that is where the trade turns over, and both sides were measured:
/// at `−1/16` the band below gains nothing, and at `−1/64` it costs — `[-1/16, -1/64]` goes from
/// `19` ulp to `60` as `ζ(1−s)`'s pole starts to tell. What remains on the eta route,
/// `(−1/8, 0)`, measures `18`–`20` ulp and is the accelerator's own cancellation rather than a
/// routing choice.
const ZETA_ETA_MIN: f64 = -0.125;

fn zeta_route(s: f64) -> ZetaRoute {
    if s.is_nan() || s == f64::NEG_INFINITY {
        ZetaRoute::Value(f64::NAN)
    } else if s == f64::INFINITY {
        ZetaRoute::Value(1.0)
    } else if s > 1.0 {
        ZetaRoute::Hurwitz
    } else if s == 1.0 {
        ZetaRoute::Value(f64::INFINITY)
    } else if s == 0.0 {
        ZetaRoute::Value(-0.5)
    } else if s > ZETA_ETA_MIN {
        ZetaRoute::Eta
    } else if s % 2.0 == 0.0 {
        ZetaRoute::Value(0.0) // Negative even integers are trivial zeros.
    } else {
        ZetaRoute::Reflection
    }
}

/// The Riemann zeta function `ζ(s)` for real `s`.
///
/// # Method
///
/// Three regimes, because no single series covers the line:
///
/// For `s > 1` this is the Hurwitz zeta at `a = 1`, so it reuses that Euler–Maclaurin
/// summation directly rather than repeating it.
///
/// For `-1 < s < 1`, including the critical strip, [`zeta_eta`] takes over. Reflection near
/// zero would lose the argument in `1-s`, so eta is used on both sides of zero. It is worth
/// being explicit about why the functional
/// equation does *not* suffice here, since it is the obvious thing to reach for: the equation
/// relates `s` to `1 − s`, and for `s` in `(0, 1)` the reflected point `1 − s` is still in
/// `(0, 1)`. The strip is mapped onto itself, so no amount of reflection escapes it.
///
/// For `s <= -1` the functional equation is used, since `1 − s >= 2`:
///
/// ```text
/// ζ(s) = 2^s π^{s−1} sin(πs/2) Γ(1−s) ζ(1−s)
/// ```
///
/// which maps the whole left half-plane onto `s > 1` where the series converges. The trivial
/// zeros at the negative even integers come out of the `sin(πs/2)` factor, and are returned
/// exactly rather than left to cancellation.
///
/// `s = 1` is the pole; `+∞` is returned, matching the limit from above.
pub(crate) fn zeta(s: f64) -> f64 {
    match zeta_route(s) {
        ZetaRoute::Value(value) => value,
        ZetaRoute::Hurwitz => hurwitz_zeta(s, 1.0),
        ZetaRoute::Eta => zeta_eta(s),
        ZetaRoute::Reflection => zeta_reflection(s),
    }
}

/// `ζ(s) = 2^s π^{s−1} sin(πs/2) Γ(1−s) ζ(1−s)`, for `s < 0`.
///
/// # The product form is more accurate on average and loses the closed forms
///
/// This is the one place in the family where `exp(a sum of logarithms)` survives a challenge, so
/// the measurement is worth keeping. Formed as a product of its five factors — each of which is
/// representable for `s > −170` — the reflection domain improves by about a factor of two:
///
/// ```text
/// s              [-1, -50]   [-50, -170]   [-170, -250]
/// log domain      3.29e-14      1.82e-13       3.41e-13
/// product         1.99e-14      9.49e-14       3.41e-13
/// ```
///
/// And it breaks `zeta_hits_its_closed_forms_exactly`, which requires `ζ(−1) = −1/12` to **zero
/// ulp**. The reason is a cancellation the product form exposes: at `s = −1` the factors are
/// `π^{−2}` and `ζ(2) = π²/6`, and separately rounded those two no longer cancel — one ulp.
///
/// Neither form is principled there; the log one is *lucky*, and the test has enshrined the luck
/// as a contract. That is a defensible contract — `ζ(−1) = −1/12` is an identity a user will
/// check — so the luck is kept until the closed forms are returned from a table of Bernoulli
/// numbers, which is what would let the product form be used everywhere else. Until then the
/// factor of two is not worth a famous identity.
///
/// The `[-170, -250]` column is identical because `Γ(1−s)` has left the format by then and the
/// log domain is the only route either way.
/// Which form of the functional equation is evaluated.
///
/// The two are algebraically identical and fail in opposite directions, which is the only reason
/// there is a choice to name. See [`ZETA_PRODUCT_MAX`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReflectionForm {
    /// `exp` of the sum of the factors' logarithms.
    LogDomain,
    /// The product of the five factors, taken directly.
    Product,
}

impl ReflectionForm {
    fn for_argument(s: f64) -> Self {
        // `Γ(1−s)` leaves the format past `171`, and the product cannot be formed at all there.
        if s > ZETA_PRODUCT_MAX || 1.0 - s > GAMMA_PRODUCT_MAX {
            Self::LogDomain
        } else {
            Self::Product
        }
    }
}

/// Above this `s` the reflection stays in the log domain.
///
/// # The log form's floor is the size of its own exponent
///
/// `ln_magnitude` is dominated by `lnΓ(1−s)`, which is `318` at `s = −92`. A relative error in
/// `lnΓ` is an *absolute* error in the exponent, and `exp` turns that back into a relative error
/// in the answer — so the log form cannot beat about `|lnΓ(1−s)|` ulp however good `lnΓ` gets,
/// and at `3` ulp of `lnΓ` it measures `959`. The product never takes that logarithm.
///
/// # But the product is worse near the top of the domain
///
/// Its five factors are separately rounded, which costs a fixed handful of ulp wherever the
/// exponent is small enough that the log form had little to amplify. Measured, worst ulp per
/// band against a 400-bit reference:
///
/// ```text
/// s          [-2.5,-1.5]  [-3.5,-2.5]  [-6,-4.5]  [-13,-9]  [-40,-15]  [-100,-40]
/// log                 12           31         67        94        316         959   ulp
/// product             21           18         43        41         58         187
/// ```
///
/// They cross at about `s = −2.5`. **The boundary is placed at `−7/2` rather than there**, and
/// the reason is a contract rather than a measurement: `ζ(−1) = −1/12` and `ζ(−3) = 1/120` are
/// pinned to *zero* ulp by `zeta_hits_its_closed_forms_exactly`, and the product form costs `2`
/// and `5` ulp at those two points. At negative odd integers `1 − s` is even, where `ζ` is a
/// rational multiple of a power of `π`, and that power cancels analytically against `π^{s−1}` —
/// a cancellation the log form keeps by adding logarithms and the product form loses by
/// rounding each factor.
///
/// So one band, `[-7/2, -5/2]`, keeps the worse form at `31` ulp against `18`. That is the price
/// of two identities a user will check, and it is the same trade the surrounding note already
/// records. Returning the closed forms from a table of Bernoulli numbers is what would let the
/// product form run everywhere below the crossover; until then the boundary sits below the last
/// pinned value rather than on the measurement.
const ZETA_PRODUCT_MAX: f64 = -3.5;

/// The largest argument `Γ` is finite at, past which only the log form can be formed.
const GAMMA_PRODUCT_MAX: f64 = 171.0;

fn zeta_reflection(s: f64) -> f64 {
    let one_minus_s = 1.0 - s;
    let sin_term = crate::math::gamma::reflection_sine(0.5 * s);
    match ReflectionForm::for_argument(s) {
        // `exp2` and `powf` rather than `exp` of `s·ln 2` and `(s−1)·ln π`: both are correctly
        // rounded to well under an ulp, where routing them through a logarithm would reintroduce
        // exactly the amplification this form exists to avoid — `105` ulp from the `π` factor
        // alone at `s = −92`.
        ReflectionForm::Product => {
            s.exp2()
                * core::f64::consts::PI.powf(s - 1.0)
                * sin_term
                * crate::kernels::gamma::gamma(one_minus_s)
                * hurwitz_zeta(one_minus_s, 1.0)
        }
        // Formed in the log domain where the product could overflow: `Γ(1−s)` grows without
        // bound as `s → −∞`, so the result must be allowed to reach infinity rather than
        // producing `NaN` from `inf · 0` on the way.
        ReflectionForm::LogDomain => {
            let ln_magnitude = s * core::f64::consts::LN_2
                + (s - 1.0) * core::f64::consts::PI.ln()
                + sin_term.abs().ln()
                + ln_gamma(one_minus_s);
            // ζ(1-s) is positive throughout the reflection domain.
            (ln_magnitude.exp() * hurwitz_zeta(one_minus_s, 1.0)).copysign(sin_term)
        }
    }
}

#[cfg(test)]
mod tests;
