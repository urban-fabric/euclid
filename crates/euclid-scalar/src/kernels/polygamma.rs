//! The polygamma function `ψ⁽ⁿ⁾(x)` for arbitrary order, and the reciprocal gamma `1/Γ(x)`.
//!
//! # Polygamma by Euler–Maclaurin
//!
//! For `n ≥ 1`, `ψ⁽ⁿ⁾(x) = (−1)^{n+1} n! ζ(n+1, x)`, where `ζ(s, a) = Σ_{k≥0} (a+k)^{−s}` is
//! the Hurwitz zeta. That series converges like `k^{−(n+1)}`, which is far too slow to use
//! directly: reaching double precision for `n = 1` would take about `1e8` terms.
//!
//! Euler–Maclaurin fixes that by summing a handful of terms exactly and replacing the rest
//! with an integral plus a correction series in Bernoulli numbers:
//!
//! ```text
//! ζ(s, a) ≈ Σ_{k<N} (a+k)^{−s}
//!         + (a+N)^{1−s}/(s−1)
//!         + ½ (a+N)^{−s}
//!         + Σ_{j≥1} c_j · (s)_{2j−1} · (a+N)^{−s−2j+1}
//! ```
//!
//! with `c_j = B_{2j}/(2j)!` and `(s)_m` the rising factorial. With `N = 16` and five
//! correction terms this reaches full double precision for every order used in practice, in a
//! fixed amount of work rather than `O(1e8)`.
//!
//! The Bernoulli coefficients are written as exact rationals rather than decimal digits, so
//! there is nothing to transcribe wrongly.

mod exprel;

pub(crate) use exprel::exprel;

use super::gamma::{digamma, gamma, ln_gamma, trigamma};

/// Terms summed exactly before the Euler–Maclaurin tail takes over.
const EM_SPLIT: u32 = 16;

/// `B_{2j} / (2j)!` for `j = 1..=6`, as exact rationals.
///
/// `B₂ = 1/6`, `B₄ = −1/30`, `B₆ = 1/42`, `B₈ = −1/30`, `B₁₀ = 5/66`, `B₁₂ = −691/2730`.
const EM_COEFFS: [f64; 6] = [
    1.0 / 12.0,                      // (1/6)/2!
    -1.0 / 720.0,                    // (-1/30)/4!
    1.0 / 30_240.0,                  // (1/42)/6!
    -1.0 / 1_209_600.0,              // (-1/30)/8!
    1.0 / 47_900_160.0,              // (5/66)/10!
    -691.0 / 2730.0 / 479_001_600.0, // (-691/2730)/12!
];

/// Largest `|s·ln y|` for which a power is formed by squaring rather than by `exp`/`ln`.
///
/// `ln(f64::MAX)` is `709.78`, and squaring forms `y^{|s|}` as an actual product — so the
/// *intermediate* has to stay in range even when the result would not. `700` keeps a margin
/// under that ceiling for the rounding of the logarithm itself.
///
/// Without this bound the scheme is not merely less accurate, it is wrong: at `s = 32`,
/// `a = 1e10` the product `b^{32}` is `1e320`, which overflows to infinity and makes the
/// reciprocal **zero** where the true term is the subnormal `3.23e-312`. `powf` has no such
/// intermediate, computing `exp(−s ln y)` directly, and is why it remains the general route.
const SQUARING_LOG_MAX: f64 = 700.0;

/// How the Euler–Maclaurin scheme forms its nineteen powers.
///
/// # Both arms exist for a measured reason
///
/// `y^{−s}` through `powf` is `exp(−s·ln y)`: two transcendental calls, and this scheme needs
/// nineteen of them, which is most of its cost. When `s` is an **integer** the same value is a
/// product by exponentiation-by-squaring, which measured **3.19× faster** on the head —
/// `120.2 ns` against `37.6 ns`, interleaved in one process, minimum of seven.
///
/// That matters because the caller that reaches here most often never has a non-integer `s` at
/// all: [`polygamma`] passes `s = n + 1` for an integer order. `zeta` is the one that passes
/// arbitrary reals.
///
/// # What the choice costs, stated honestly
///
/// It is **not** bit-identical, and it is not free. Against the six values of `ζ(2k)` with exact
/// closed forms — `π²/6`, `π⁴/90`, `π⁶/945`, `π⁸/9450`, `π¹⁰/93555`, `691π¹²/638512875` — the two
/// arms carry the *same* error to the ulp: `0, 0, 0.98, 1.00, 3.00, 1.00`. So on every case where
/// the truth is known exactly, squaring is neither better nor worse. Elsewhere the two disagree
/// by up to `2.4e-15` relative, about eleven ulp, and there is no reference that says which is
/// closer. Eleven ulp against three times the speed is the trade, taken deliberately and with
/// the accuracy suites as the arbiter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ZetaPower {
    /// An integer `s` whose powers stay in range: form them by squaring.
    Squaring(i32),
    /// A real `s`, or an integer whose product would leave the format. `powf` it is.
    Transcendental,
}

impl ZetaPower {
    /// `a` and `b` are the smallest and largest bases the scheme will raise, so both ends are
    /// checked: a base below one makes a *negative* exponent large, and one above makes the
    /// positive exponent large.
    ///
    /// # Two guards would be one too many
    ///
    /// This also carried `s < 1024`, to stop the integrality test admitting some `s = 1e28`
    /// whose `as i32` would saturate. **Mutation testing showed that guard was unreachable**
    /// and it is gone. `b` is `a + EM_SPLIT` with `a > 0`, so `ln b > ln 16 = 2.7726` and
    /// therefore `reach > 2.7726·s`; passing `reach < 700` already forces `s < 252.5`. The order
    /// bound could never be the comparison that rejected, which makes the cast below safe on the
    /// range guard alone — and makes the deleted line worse than redundant, since a reader would
    /// have trusted it to be doing something.
    fn for_scheme(s: f64, a: f64, b: f64) -> Self {
        let reach = s * a.ln().abs().max(b.ln().abs());
        if s.fract() == 0.0 && reach < SQUARING_LOG_MAX {
            // `reach < 700` with `b > 16` bounds `s` below 253, so this cannot saturate.
            #[allow(clippy::cast_possible_truncation)]
            Self::Squaring(s as i32)
        } else {
            Self::Transcendental
        }
    }

    /// `y` raised to `offset − s`, which is every power this scheme needs: `0` for `y^{−s}`,
    /// `1` for the integral term and `−1` for the first Bernoulli correction.
    fn at(self, y: f64, s: f64, offset: i32) -> f64 {
        match self {
            Self::Squaring(order) => y.powi(offset - order),
            Self::Transcendental => y.powf(f64::from(offset) - s),
        }
    }
}

/// The Hurwitz zeta `ζ(s, a)` for real `s > 1` and `a > 0`, by Euler–Maclaurin.
///
/// `pub(crate)` because the Riemann zeta is `ζ(s, 1)` and reuses this directly rather than
/// reimplementing the same summation.
pub(crate) fn hurwitz_zeta(s: f64, a: f64) -> f64 {
    debug_assert!(s > 1.0);
    debug_assert!(a > 0.0);

    let b = a + f64::from(EM_SPLIT);
    let power_of = ZetaPower::for_scheme(s, a, b);

    // Exact head.
    let mut total = 0.0f64;
    for k in 0..EM_SPLIT {
        total += power_of.at(a + f64::from(k), s, 0);
    }

    // Integral term and the half-endpoint.
    total += power_of.at(b, s, 1) / (s - 1.0);
    total += 0.5 * power_of.at(b, s, 0);

    // Bernoulli corrections. The j-th needs the rising factorial (s)_{2j-1}, built
    // incrementally so no factorial is ever formed.
    let mut rising = s; // (s)_1
    let mut power = power_of.at(b, s, -1); // b^{-s-2j+1} at j = 1
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
        let m = 2.0 * (j as f64 + 1.0);
        rising *= (s + m - 1.0) * (s + m);
        power /= b * b;
    }
    total
}

/// Which route evaluates `ψ⁽ⁿ⁾` at this order and argument.
///
/// Three arms return `NaN` for three different reasons, and they are kept apart because the
/// reasons are not interchangeable: one is a propagated `NaN`, one is a genuine pole of the
/// function, and one is a region where the function is perfectly well defined and *this
/// implementation* declines to evaluate it. Collapsing them to a single `x <= 0.0` guard — which
/// is what the arithmetic allows — would lose exactly the distinction a caller needs.
///
/// The order is tested before the argument, so `ψ` and `ψ₁` keep their own edge behaviour rather
/// than inheriting the general route's.
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
    /// reflection formula this implementation does not carry — no caller has needed one, and a
    /// `NaN` is a better answer than a silently wrong one.
    Unsupported,
    /// Order `≥ 2` at positive `x`, as `(−1)^{n+1}·n!·ζ(n+1, x)` through the Hurwitz zeta.
    Hurwitz,
}

impl PolygammaRoute {
    fn for_arguments(order: u32, x: f64) -> Self {
        match order {
            0 => Self::Digamma,
            1 => Self::Trigamma,
            _ if x.is_nan() => Self::Undefined,
            _ if x <= 0.0 && x.fract() == 0.0 => Self::Pole,
            // A sign-bit test rather than `x < 0.0`: both signed zeros are integers, so the
            // arm above has already taken them and `<` against `<=` cannot be told apart here.
            // Mutation testing found that spelling unkillable; this one has no operator to flip.
            _ if x.is_sign_negative() => Self::Unsupported,
            _ => Self::Hurwitz,
        }
    }
}

/// `ψ⁽ⁿ⁾(x)`, the `n`-th derivative of `digamma`.
pub(crate) fn polygamma(order: u32, x: f64) -> f64 {
    match PolygammaRoute::for_arguments(order, x) {
        PolygammaRoute::Digamma => digamma(x),
        PolygammaRoute::Trigamma => trigamma(x),
        PolygammaRoute::Undefined | PolygammaRoute::Pole | PolygammaRoute::Unsupported => f64::NAN,
        PolygammaRoute::Hurwitz => {
            let n = f64::from(order);
            // (-1)^{n+1} n! zeta(n+1, x)
            let sign = if order.is_multiple_of(2) { -1.0 } else { 1.0 };
            // `n!` from the exact factorial rather than `exp(lnΓ(n+1))`.
            //
            // The same `exp(a_logarithm)` hazard as `rgamma` and the `I` family: the exponent is
            // `lnΓ(n+1)`, whose absolute rounding `exp` turns into relative error — `ulp(42.3)/2
            // = 3.6e-15` at order 20, where `20!` is an *exactly representable* `f64`. There is
            // nothing to approximate.
            //
            // `host::factorial` is exact through `20!` and falls back to `exp(ln_factorial)`
            // above it, which is what this line was doing unconditionally. So the change is a
            // strict improvement: exact where exactness is available, identical where it is not.
            sign * crate::kernels::host::factorial::factorial(u64::from(order))
                * hurwitz_zeta(n + 1.0, x)
        }
    }
}

/// Which route evaluates `1/Γ` at this argument.
///
/// `1/Γ` is *entire* — it has no poles anywhere — so unlike `Γ` itself none of these arms is a
/// singularity being dodged. They divide the line by what stays in range: the positive axis
/// through the log domain, the negative axis through reflection, and the integers on it by the
/// exact zero that reflection would otherwise have to produce from a vanishing sine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RgammaRoute {
    /// `NaN`, and `−∞` — where `Γ` oscillates without bound, so the reciprocal has no limit for
    /// the sign of zero to be taken from, let alone a value.
    Undefined,
    /// `+∞`, where `Γ` diverges and the reciprocal vanishes.
    AtPositiveInfinity,
    /// The non-positive integers. The poles of `Γ` are the zeros of `1/Γ`, and they are returned
    /// exactly rather than left to [`Self::Reflection`]'s `sin(πx)` to round to zero.
    Pole,
    /// `x > 0`, as `e^{−lnΓ(x)}` — not `gamma(x).recip()`, because `Γ` overflows from about
    /// `x = 172` while `1/Γ` stays representable well past that.
    Exponential,
    /// `x < 0` and not an integer, by the reflection formula; see [`rgamma_reflection`].
    Reflection,
}

impl RgammaRoute {
    fn for_argument(x: f64) -> Self {
        if x.is_nan() || x == f64::NEG_INFINITY {
            Self::Undefined
        } else if x == f64::INFINITY {
            Self::AtPositiveInfinity
        } else if x <= 0.0 && x.fract() == 0.0 {
            Self::Pole
        } else if x.is_sign_positive() {
            // A sign-bit test rather than `x > 0.0`: both signed zeros are integers, so the arm
            // above has already taken them and `>` against `>=` cannot be told apart here.
            // Mutation testing found that spelling unkillable; this one has no operator to flip.
            Self::Exponential
        } else {
            Self::Reflection
        }
    }
}

/// The reciprocal gamma `1/Γ(x)`, entire and zero at the non-positive integers.
pub(crate) fn rgamma(x: f64) -> f64 {
    match RgammaRoute::for_argument(x) {
        RgammaRoute::Undefined => f64::NAN,
        RgammaRoute::AtPositiveInfinity => 0.0,
        RgammaRoute::Pole => 0.0,
        RgammaRoute::Exponential => {
            // `1/Γ(x)` directly where `Γ(x)` is representable, and only through the log domain
            // where it is not. Going through the logarithm unconditionally costs `lnΓ(x)·ε` of
            // relative error, because `exp` turns the exponent's absolute rounding into relative
            // error and that exponent is `O(x ln x)`: `ulp(701)/2 = 5.7e-14` at `x = 170`, where
            // the reciprocal of a finite `Γ` is two roundings. See `math::scaling::unscale` for
            // the same argument in the `I` family.
            let value = gamma(x);
            if value.is_finite() {
                value.recip()
            } else {
                (-ln_gamma(x)).exp()
            }
        }
        RgammaRoute::Reflection => rgamma_reflection(x),
    }
}

/// `1/Γ(x) = Γ(1−x)·sin(πx)/π`, for the non-integer negative axis.
///
/// # Why the sine comes from `reflection_sine`
///
/// Through [`reflection_sine`](crate::math::gamma::reflection_sine) rather than `(PI * x).sin()`,
/// for the reason `ln_gamma` gives at length: rounding `PI * x` discards the argument's distance
/// to the nearest pole, and near a pole that distance is the whole magnitude of the sine. The
/// absolute error in the rounded product is `ulp(PI |x|)`, which dividing by a sine of order
/// `PI * offset` turns into a *relative* error that grows with `|x|`.
///
/// Measured as `|rgamma(x) Gamma(x) - 1|` just off the poles, over a sweep of 1e-5 to 2e-3
/// offsets at every pole down to -39:
///
/// ```text
///                    worst over the sweep        at x = -39.00001
/// (PI * x).sin()     3.60e-10 at x = -39.00001   3.60e-10
/// reflection_sine    4.55e-14 at x = -38.00141   9.55e-15
/// ```
///
/// Four orders, and the shape of the error changes as well as its size: under the old form the
/// worst case sat at the largest `|x|` swept, because that is where `ulp(PI |x|)` is largest. It
/// no longer does. `euclid-stability` found this as `the_reciprocal_gamma_is_entire` at
/// `x = -5.000412007793784`, where the metamorphic relation `rgamma(x) Gamma(x) = 1` missed its
/// 1e-12 budget by `1.07e-12`.
fn rgamma_reflection(x: f64) -> f64 {
    let s = crate::math::gamma::reflection_sine(x);
    // `Γ(1−x)·|sin πx|/π` as a product where every factor is representable, and through the log
    // domain only where `Γ(1−x)` is not — which needs `x < −170.6`.
    //
    // The log form was unconditional and it costs the exponent's absolute rounding as relative
    // error, which is large at *both* ends: the exponent is `lnΓ(1−x) + ln|s| − ln π`, so it is
    // `−637` at `x = −2.2e-277` (where `|s| ≈ π|x|` is tiny) and `+362` at `x = −100.5` (where
    // `lnΓ` is large). Measured `5.13e-14` at the former against `ulp(637)/2 = 5.7e-14`
    // predicted — the whole of it.
    let reflected = gamma(1.0 - x);
    let magnitude = if reflected.is_finite() {
        reflected * s.abs() / core::f64::consts::PI
    } else {
        (ln_gamma(1.0 - x) + s.abs().ln() - core::f64::consts::PI.ln()).exp()
    };
    // A sign-bit test rather than `s < 0.0`: the sine is never exactly zero here — that would
    // need an integer `x`, which [`RgammaRoute::Pole`] has already taken — so the two spellings
    // agree on every reachable argument and `<` against `<=` is an unkillable mutant. This one
    // has no operator to flip.
    if s.is_sign_negative() {
        -magnitude
    } else {
        magnitude
    }
}

#[cfg(test)]
mod tests;
