//! Single-precision peer of [`super::super::polygamma`]. Same Euler–Maclaurin scheme with a
//! shorter exact head, since `f32` needs far fewer terms to reach its own noise floor.

use super::gamma::{digamma, ln_gamma, trigamma};

/// Terms summed exactly before the Euler–Maclaurin tail takes over.
const EM_SPLIT: u32 = 12;

/// `B_{2j} / (2j)!` for `j = 1..=6`, as exact rationals.
///
/// `B₂ = 1/6`, `B₄ = −1/30`, `B₆ = 1/42`, `B₈ = −1/30`, `B₁₀ = 5/66`, `B₁₂ = −691/2730`.
const EM_COEFFS: [f32; 6] = [
    1.0 / 12.0,                      // (1/6)/2!
    -1.0 / 720.0,                    // (-1/30)/4!
    1.0 / 30_240.0,                  // (1/42)/6!
    -1.0 / 1_209_600.0,              // (-1/30)/8!
    1.0 / 47_900_160.0,              // (5/66)/10!
    -691.0 / 2730.0 / 479_001_600.0, // (-691/2730)/12!
];

/// The Hurwitz zeta `ζ(s, a)` for real `s > 1` and `a > 0`, by Euler–Maclaurin.
///
/// `pub(super)` and `s > 1` rather than private and `s ≥ 2`, because the Riemann zeta is
/// `ζ(s, 1)` and [`super::transcendental::zeta`] reuses this directly rather than carrying a
/// second copy of the same summation. `polygamma` only ever asks for an integer `s ≥ 2`; the
/// scheme itself has no such restriction, and the `f64` peer exposes it on the same grounds.
/// # Why this does not take the `f64` peer's squaring optimisation
///
/// `kernels::polygamma`'s `ZetaPower` forms the head's powers by exponentiation-by-squaring
/// when `s` is an integer, which is 3.19x faster there. **It does not fit in this format, and the
/// measurement is the reason rather than caution:** applied here it put `polygamma(2, 1.76875)`
/// **4 `f32` ulp** from the wide kernel rounded down, against the 3 that
/// `polygamma_agrees_across_precisions_at_every_order` allows.
///
/// The cause is structural, not a tuning accident. Squaring accumulates roughly `log₂ s`
/// roundings where `powf` carries one, and with 24 bits of mantissa that chain is a far larger
/// share of the budget than it is with 53. The same argument that makes the trade worth taking
/// in `f64` — eleven ulp of a 53-bit format, against three times the speed — makes it too
/// expensive in a format whose entire cross-precision allowance is three ulp.
///
/// So `powf` stays, and the asymmetry is deliberate.
pub(super) fn hurwitz_zeta(s: f32, a: f32) -> f32 {
    debug_assert!(s > 1.0);
    debug_assert!(a > 0.0);

    // Exact head.
    let mut total = 0.0f32;
    for k in 0..EM_SPLIT {
        total += (a + (k as f32)).powf(-s);
    }

    let b = a + (EM_SPLIT as f32);
    // Integral term and the half-endpoint.
    total += b.powf(1.0 - s) / (s - 1.0);
    total += 0.5 * b.powf(-s);

    // Bernoulli corrections. The j-th needs the rising factorial (s)_{2j-1}, built
    // incrementally so no factorial is ever formed.
    let mut rising = s; // (s)_1
    let mut power = b.powf(-s - 1.0); // b^{-s-2j+1} at j = 1
    for (j, &c) in EM_COEFFS.iter().enumerate() {
        // `power` is `b^{-s-2j+1}`, which **underflows to zero** for any large `s` — `b` is
        // `a + EM_SPLIT`, so `17^{-s}` is already zero by `s ≈ 250`. Meanwhile `rising` is
        // `(s)_{2j-1}`, which grows like `s^{2j}` and **overflows to infinity**. Their product is
        // then `inf * 0 = NaN`, and the whole sum is poisoned.
        //
        // That is not hypothetical: `zeta(s)` returned `NaN` for every `s` above `1.0548e28`,
        // where `ζ` is exactly `1`. The boundary is where the overflow of `rising` moves from the
        // sixth Bernoulli term to the fifth, so one of the six terms is `inf * 0` before the loop
        // ends. Found by a range sweep asserting that no kernel returns a non-finite value where
        // the true one is finite.
        //
        // Once `power` is zero every later term is zero too — the tail has nothing left to
        // contribute — so stopping here is exact rather than a truncation. For every `s` where
        // `power` is still non-zero the arithmetic is untouched, which is why this changes no
        // value that was previously finite.
        if power == 0.0 {
            break;
        }
        total += c * rising * power;
        // Advance to j+1: (s)_{2j+1} = (s)_{2j-1} · (s+2j-1)(s+2j)
        let m = 2.0 * (j as f32 + 1.0);
        rising *= (s + m - 1.0) * (s + m);
        power /= b * b;
    }
    total
}

/// Which route evaluates `ψ⁽ⁿ⁾` at this order and argument.
///
/// The same six arms the `f64` peer in [`crate::kernels::polygamma`] carries, and kept apart for
/// the same reason: three of them return `NaN`, but one is a propagated `NaN`, one a genuine pole
/// and one a region where the function exists and this implementation declines to evaluate it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PolygammaRoute {
    /// Order `0`: [`digamma`], which carries its own reflection and asymptotic machinery.
    Digamma,
    /// Order `1`: [`trigamma`], likewise.
    Trigamma,
    /// A `NaN` argument at order `≥ 2`, propagated.
    Undefined,
    /// The non-positive integers, which are poles of every order.
    Pole,
    /// The rest of the negative axis. `ψ⁽ⁿ⁾` is real and finite here, but reaching it needs a
    /// reflection formula this implementation does not carry.
    Unsupported,
    /// Order `≥ 2` at positive `x`, as `(−1)^{n+1}·n!·ζ(n+1, x)` through the Hurwitz zeta.
    Hurwitz,
}

impl PolygammaRoute {
    fn for_arguments(order: u32, x: f32) -> Self {
        match order {
            0 => Self::Digamma,
            1 => Self::Trigamma,
            _ if x.is_nan() => Self::Undefined,
            _ if x <= 0.0 && x.fract() == 0.0 => Self::Pole,
            // A sign-bit test rather than `x < 0.0`, for the reason the `f64` peer gives: both
            // signed zeros are integers, so the arm above has already taken them and no
            // comparison here is distinguishable from its widening.
            _ if x.is_sign_negative() => Self::Unsupported,
            _ => Self::Hurwitz,
        }
    }
}

/// `ψ⁽ⁿ⁾(x)`, the `n`-th derivative of `digamma`.
pub(crate) fn polygamma(order: u32, x: f32) -> f32 {
    match PolygammaRoute::for_arguments(order, x) {
        PolygammaRoute::Digamma => digamma(x),
        PolygammaRoute::Trigamma => trigamma(x),
        PolygammaRoute::Undefined | PolygammaRoute::Pole | PolygammaRoute::Unsupported => f32::NAN,
        PolygammaRoute::Hurwitz => {
            let n = order as f32;
            // (-1)^{n+1} n! zeta(n+1, x)
            let sign = if order.is_multiple_of(2) { -1.0 } else { 1.0 };
            // `n!` narrowed from the exact `f64` factorial rather than `exp(lnΓ(n+1))`; see the
            // `f64` peer. Narrowing the exact value is correctly rounded for this format, where
            // `exp` of an `f32` logarithm is not: `ulp(42.3)/2` in `f32` is `1.9e-6`, sixteen
            // times this format's resolution, for a quantity that has an exact answer.
            sign * (crate::kernels::host::factorial::factorial(u64::from(order)) as f32)
                * hurwitz_zeta(n + 1.0, x)
        }
    }
}

/// Which route evaluates `1/Γ` at this argument.
///
/// The same five arms the `f64` peer in [`crate::kernels::polygamma`] carries, for the same
/// reasons: `1/Γ` is entire, so the division is by what stays in range rather than by any
/// singularity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RgammaRoute {
    /// `NaN`, and `−∞` — where `Γ` oscillates without bound, so the reciprocal has no limit.
    Undefined,
    /// `+∞`, where `Γ` diverges and the reciprocal vanishes.
    AtPositiveInfinity,
    /// The non-positive integers. The poles of `Γ` are the zeros of `1/Γ`, and they are returned
    /// exactly rather than left to [`Self::Reflection`]'s `sin(πx)` to round to zero.
    Pole,
    /// `x > 0`, as `e^{−lnΓ(x)}` — not `gamma(x).recip()`, because `Γ` overflows from about
    /// `x = 35` in this format while `1/Γ` stays representable well past that.
    Exponential,
    /// `x < 0` and not an integer, by the reflection formula; see [`rgamma_reflection`].
    Reflection,
}

impl RgammaRoute {
    fn for_argument(x: f32) -> Self {
        if x.is_nan() || x == f32::NEG_INFINITY {
            Self::Undefined
        } else if x == f32::INFINITY {
            Self::AtPositiveInfinity
        } else if x <= 0.0 && x.fract() == 0.0 {
            Self::Pole
        } else if x.is_sign_positive() {
            // A sign-bit test rather than `x > 0.0`, for the reason the `f64` peer gives: the
            // signed zeros are already taken above, so no comparison here is distinguishable
            // from its widening.
            Self::Exponential
        } else {
            Self::Reflection
        }
    }
}

/// The reciprocal gamma `1/Γ(x)`, entire and zero at the non-positive integers.
///
/// # This keeps the log domain where the `f64` peer stopped using it, and that is measured
///
/// [`crate::kernels::polygamma::rgamma`] forms `1/Γ` as a product of representable factors,
/// because `exp` turns its exponent's absolute rounding into relative error. Ported here, the
/// same change **regressed** `x` near 1 and 2 — from one or two ulp to fifteen — and
/// `the_reciprocal_gamma_carries_its_log_gamma_amplification` caught it.
///
/// The reason is the thing that model names. `lnΓ(1) = lnΓ(2) = 0`, so around those points the
/// exponent is near zero and there is *nothing to amplify*: the log route is already correctly
/// rounded there. What replaces it is `1/Γ(x)`, which inherits this format's `Γ` — and that is
/// the part that differs between precisions. `f64`'s `Γ` is good enough near 2 that the quotient
/// still wins (measured 8.7 ulp before, 7.0 after); `f32`'s is not.
///
/// Measured over the bands, the product form is better away from `|lnΓ| ≈ 0` and worse at it:
///
/// ```text
///                  tiny negative   [-20, -0.5]   [0.5, 34]   near 1 and 2
/// log domain            3.89e-6       5.85e-6     8.04e-6      1-2 ulp
/// product form          4.71e-7       3.16e-6     2.31e-6       15 ulp
/// ```
///
/// So a port is not the fix; the fix is choosing by `|lnΓ(x)|`, which is a threshold this format
/// would have to measure for itself. Left as the log domain throughout until that is done, rather
/// than taking an eight-fold gain in the tails at the cost of the two arguments most callers
/// actually pass. The `f64` peer needs no such split because its quotient wins everywhere.
pub(crate) fn rgamma(x: f32) -> f32 {
    match RgammaRoute::for_argument(x) {
        RgammaRoute::Undefined => f32::NAN,
        RgammaRoute::AtPositiveInfinity => 0.0,
        RgammaRoute::Pole => 0.0,
        RgammaRoute::Exponential => (-ln_gamma(x)).exp(),
        RgammaRoute::Reflection => rgamma_reflection(x),
    }
}

/// `1/Γ(x) = Γ(1−x)·sin(πx)/π`, for the non-integer negative axis.
///
/// Through [`reflection_sine`](crate::math::gamma::reflection_sine) for the reason given on the
/// `f64` peer in [`crate::kernels::polygamma`] — rounding `PI * x` loses the distance to the
/// nearest pole, which is what the sine is small by. The hazard is worse here, not better: `f32`
/// has 24 bits of mantissa, so `ulp(PI |x|)` is about `2^29` times larger than in `f64` at the
/// same argument.
fn rgamma_reflection(x: f32) -> f32 {
    let s = crate::math::gamma::reflection_sine(x);
    let magnitude = (ln_gamma(1.0 - x) + s.abs().ln() - core::f32::consts::PI.ln()).exp();
    // A sign-bit test rather than `s < 0.0`, matching the `f64` peer: the sine is never exactly
    // zero here — that would need an integer `x`, which [`RgammaRoute::Pole`] has taken — so the
    // comparison has no reachable boundary and nothing to distinguish `<` from `<=`.
    if s.is_sign_negative() {
        -magnitude
    } else {
        magnitude
    }
}

/// `(eˣ − 1)/x`, continuous at the origin.
/// Below this magnitude the series is the better-conditioned route.
///
/// The series is better conditioned than expm1(x)/x for tiny x, and it is **two terms**,
/// not four. The f64 peer carries `1 + x/2 + x²/6 + x³/24`; narrowing that body to f32 kept
/// arithmetic the format cannot represent.
///
/// Below the crossover `|x| < 1e-5`, `x²/6` is at most `1.7e-11` against a result near one,
/// where f32's resolution is `1.19e-7` — four orders coarser. Dropping both tail terms
/// changed **none of 400,000 sampled results**; they were a dozen mutants no test could ever
/// kill, because they were arithmetic with no observable consequence. Same bargain as
/// `sinc`'s Taylor branch, which this file's cause 7 records.
///
/// What remains earns its place, which is worth measuring rather than assuming: against the
/// truth, `1 + x/2` is correct to `5.96e-8` — half an ulp, correctly rounded — while
/// `exp_m1(x)/x` reaches `1.07e-7`, and the two differ on 106,786 of those 400,000
/// arguments. So the branch is a real improvement and not defensive habit, and
/// `exprel_is_correctly_rounded_below_its_crossover` pins the gap that says so.
const SERIES_MAX: f32 = 1e-5;

/// Shift for the overflow band; see [`ExprelQuotient::Shifted`].
const SHIFT: f32 = 16.0;

/// Which route evaluates `(eˣ − 1)/x` at this magnitude. The `f32` peer of the `f64` kernel's
/// own `ExprelForm`; the term counts differ and the routing does not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExprelForm {
    /// `x = 0`, where the removable singularity is filled with `1`.
    AtOrigin,
    /// Not a number.
    Undefined,
    /// `±∞`, where the quotient diverges or decays to zero.
    Infinite,
    /// `|x| < SERIES_MAX`: **two** terms of the Maclaurin series, not the `f64` peer's three.
    Series,
    /// Everything else: the quotient, in whichever form of [`ExprelQuotient`] fits.
    Quotient,
}

impl ExprelForm {
    /// The origin first, since it is the one argument the quotient cannot take.
    fn for_argument(x: f32) -> Self {
        if x == 0.0 {
            Self::AtOrigin
        } else if x.is_nan() {
            Self::Undefined
        } else if x.is_infinite() {
            Self::Infinite
        } else if x.abs() < SERIES_MAX {
            Self::Series
        } else {
            Self::Quotient
        }
    }
}

/// Which form of the quotient keeps both of its factors inside the format.
///
/// `e^x` has overflowed while `(e^x − 1)/x` has not. `exp_m1` leaves the format at
/// `ln(MAX) = 88.72` and the quotient only at the argument where `e^x/x` does, so
/// every `x` in `(88.72, 93.26)` has an ordinary number for an answer and was reached
/// through an infinite intermediate — this returned `+∞` for all of them.
///
/// Re-formed as `e^{x−16}·(e^16/x)`, which keeps both factors in range. `16` is exactly
/// representable and `x − 16` is exact for every `x` in the band, so the only error is
/// `exp`'s own — unlike recovering through `e^{x − ln x}`, which would carry `ulp(x)` into
/// the exponent and cost several decimal digits. The `−1` is more than thirty orders below
/// `e^x` here, far under the format's resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExprelQuotient {
    /// The numerator fits: `(eˣ − 1)/x` as written.
    Direct,
    /// The numerator has left the format: `e^{x−16}·(e^16/x)` keeps both factors in range.
    Shifted,
}

impl ExprelQuotient {
    /// The numerator is formed once by [`exprel`], and its range *is* the branch.
    fn for_numerator(numerator: f32) -> Self {
        if numerator.is_finite() {
            Self::Direct
        } else {
            Self::Shifted
        }
    }
}

/// `(eˣ − 1)/x`, continuous at the origin.
pub(crate) fn exprel(x: f32) -> f32 {
    match ExprelForm::for_argument(x) {
        ExprelForm::AtOrigin => 1.0,
        ExprelForm::Undefined => f32::NAN,
        // On the sign rather than `x > 0.0`: the comparison is reached only when `x` is already
        // infinite, so `>` and `>=` decide it identically and the operator is unkillable.
        ExprelForm::Infinite => {
            if x.is_sign_positive() {
                f32::INFINITY
            } else {
                0.0
            }
        }
        ExprelForm::Series => 1.0 + x / 2.0,
        ExprelForm::Quotient => {
            let numerator = x.exp_m1();
            match ExprelQuotient::for_numerator(numerator) {
                ExprelQuotient::Direct => numerator / x,
                ExprelQuotient::Shifted => (x - SHIFT).exp() * (SHIFT.exp() / x),
            }
        }
    }
}

#[cfg(test)]
mod tests;
