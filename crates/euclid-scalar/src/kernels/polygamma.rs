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

use super::gamma::{digamma, ln_gamma, trigamma};

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

/// The Hurwitz zeta `ζ(s, a)` for real `s > 1` and `a > 0`, by Euler–Maclaurin.
///
/// `pub(crate)` because the Riemann zeta is `ζ(s, 1)` and reuses this directly rather than
/// reimplementing the same summation.
pub(crate) fn hurwitz_zeta(s: f64, a: f64) -> f64 {
    debug_assert!(s > 1.0);
    debug_assert!(a > 0.0);

    // Exact head.
    let mut total = 0.0f64;
    for k in 0..EM_SPLIT {
        total += (a + f64::from(k)).powf(-s);
    }

    let b = a + f64::from(EM_SPLIT);
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
            sign * (ln_gamma(n + 1.0)).exp() * hurwitz_zeta(n + 1.0, x)
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
        RgammaRoute::Exponential => (-ln_gamma(x)).exp(),
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
    let magnitude = (ln_gamma(1.0 - x) + s.abs().ln() - core::f64::consts::PI.ln()).exp();
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
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() <= tol * b.abs().max(1.0),
            "{a:e} vs {b:e} (tol {tol:e})"
        );
    }

    #[test]
    fn polygamma_reduces_to_digamma_and_trigamma() {
        for i in 1..40 {
            let x = f64::from(i) * 0.25;
            close(polygamma(0, x), digamma(x), 1e-14);
            close(polygamma(1, x), trigamma(x), 1e-14);
        }
    }

    #[test]
    fn polygamma_matches_its_known_values_at_one() {
        // psi^(n)(1) = (-1)^{n+1} n! zeta(n+1)
        // zeta(3) = 1.2020569031595942854, zeta(4) = pi^4/90, zeta(5) = 1.0369277551433699263
        let pi4_90 = core::f64::consts::PI.powi(4) / 90.0;
        close(polygamma(2, 1.0), -2.0 * 1.202_056_903_159_594_3, 1e-12);
        close(polygamma(3, 1.0), 6.0 * pi4_90, 1e-12);
        // zeta(5), written to the digits f64 can hold.
        close(polygamma(4, 1.0), -24.0 * 1.036_927_755_143_37, 1e-12);
    }

    #[test]
    fn polygamma_satisfies_its_recurrence() {
        // psi^(n)(x+1) = psi^(n)(x) + (-1)^n n! / x^{n+1}
        for order in 2..7u32 {
            let n = f64::from(order);
            let factorial = (1..=order).map(f64::from).product::<f64>();
            let sign = if order.is_multiple_of(2) { 1.0 } else { -1.0 };
            for i in 1..30 {
                let x = f64::from(i) * 0.5;
                let lhs = polygamma(order, x + 1.0);
                let rhs = polygamma(order, x) + sign * factorial / x.powf(n + 1.0);
                close(lhs, rhs, 1e-10);
            }
        }
    }

    #[test]
    fn polygamma_is_the_derivative_of_the_previous_order() {
        for order in 1..5u32 {
            for i in 2..20 {
                let x = f64::from(i) * 0.5;
                let h = 1e-5 * x;
                let d = (polygamma(order, x + h) - polygamma(order, x - h)) / (2.0 * h);
                close(d, polygamma(order + 1, x), 1e-7);
            }
        }
    }

    #[test]
    fn rgamma_is_zero_at_the_poles_and_inverts_gamma_elsewhere() {
        for n in 0..10 {
            assert_eq!(rgamma(-f64::from(n)), 0.0, "1/Gamma(-{n}) = 0");
        }
        for i in 1..60 {
            let x = f64::from(i) * 0.25;
            close(rgamma(x), (-ln_gamma(x)).exp(), 1e-14);
        }
        // Negative, non-integer arguments through the reflection formula.
        for &x in &[-0.5f64, -1.5, -2.5, -3.5, -0.25, -4.75] {
            let g = super::super::gamma::gamma(x);
            close(rgamma(x), 1.0 / g, 1e-11);
        }
    }

    #[test]
    fn rgamma_is_defined_where_gamma_overflows() {
        // Gamma(175) overflows; 1/Gamma(175) is a representable subnormal-range value.
        assert!(super::super::gamma::gamma(175.0).is_infinite());
        let r = rgamma(175.0);
        assert!(r > 0.0 && r.is_finite(), "rgamma(175) = {r}");
        // 1/Gamma(175) is about 1e-316, which is *subnormal* — below f64::MIN_POSITIVE at
        // 2.2e-308 — so it carries well under 53 bits of mantissa. The value is recoverable,
        // but a log round trip through it cannot be checked at full precision.
        assert!(
            r < f64::MIN_POSITIVE,
            "expected a subnormal result, got {r}"
        );
        close(r.ln(), -ln_gamma(175.0), 1e-10);

        // Where the result is normal, the round trip is tight.
        let r160 = rgamma(160.0);
        assert!(r160 > f64::MIN_POSITIVE);
        close(r160.ln(), -ln_gamma(160.0), 1e-14);
    }

    /// The five routes, asserted as decisions.
    ///
    /// [`RgammaRoute::Pole`] and [`RgammaRoute::AtPositiveInfinity`] return the same `0.0` for
    /// different reasons, so no value assertion can tell them apart — nor tell either from a
    /// reflection that happened to round to zero, which is exactly what the exact-zero arm is
    /// there to prevent.
    #[test]
    fn the_reciprocal_gamma_routes_split_the_line_at_the_integers() {
        use RgammaRoute::{AtPositiveInfinity, Exponential, Pole, Reflection, Undefined};
        for (x, want) in [
            (f64::NAN, Undefined),
            (f64::NEG_INFINITY, Undefined),
            (f64::INFINITY, AtPositiveInfinity),
            // The non-positive integers, and the signed zeros among them.
            (0.0, Pole),
            (-0.0, Pole),
            (-1.0, Pole),
            (-39.0, Pole),
            (-f64::MAX, Pole),
            // Either side of a pole is reflection, however close.
            ((-1.0_f64).next_up(), Reflection),
            ((-1.0_f64).next_down(), Reflection),
            (-0.5, Reflection),
            (-39.000_01, Reflection),
            // The positive axis, including the subnormal floor and the top of the format.
            (f64::MIN_POSITIVE, Exponential),
            (0.5, Exponential),
            (1.0, Exponential),
            (175.0, Exponential),
            (f64::MAX, Exponential),
        ] {
            assert_eq!(RgammaRoute::for_argument(x), want, "at {x:e}");
        }
        // The positive integers are *not* poles: `1/Γ(n)` is `1/(n−1)!`, finite and nonzero.
        for n in 1..6 {
            assert_eq!(RgammaRoute::for_argument(f64::from(n)), Exponential);
            assert!(rgamma(f64::from(n)) > 0.0);
        }
    }

    /// The six polygamma routes, including the three that all return `NaN`.
    ///
    /// The `Pole`/`Unsupported` split is invisible to any value assertion — both answer `NaN` —
    /// but it is the difference between "this function has no value here" and "this
    /// implementation does not reach a value that exists". Order is tested before argument, so
    /// `ψ` and `ψ₁` keep their own edge behaviour: `polygamma(0, -0.5)` is a finite reflection
    /// while `polygamma(2, -0.5)` is not reached at all.
    #[test]
    fn the_polygamma_routes_separate_a_pole_from_an_unimplemented_region() {
        use PolygammaRoute::{Digamma, Hurwitz, Pole, Trigamma, Undefined, Unsupported};

        // The two low orders take their own routes at every argument, `NaN` included.
        for x in [f64::NAN, -3.0, -0.5, 0.0, 1.0, 1e10] {
            assert_eq!(PolygammaRoute::for_arguments(0, x), Digamma, "at {x:e}");
            assert_eq!(PolygammaRoute::for_arguments(1, x), Trigamma, "at {x:e}");
        }
        // And order 0 at a negative non-integer really does return a value, which is what makes
        // the ordering load-bearing rather than cosmetic.
        assert!(polygamma(0, -0.5).is_finite());
        assert!(polygamma(2, -0.5).is_nan());

        for order in [2, 3, 7] {
            assert_eq!(PolygammaRoute::for_arguments(order, f64::NAN), Undefined);
            // Poles at the non-positive integers, both signed zeros among them.
            for x in [0.0, -0.0, -1.0, -2.0, -39.0] {
                assert_eq!(PolygammaRoute::for_arguments(order, x), Pole, "at {x:e}");
            }
            // Real and finite there, but not reachable without a reflection formula.
            for x in [-0.5, -1.5, -10.5, (-1.0_f64).next_up(), -1e10 - 0.5] {
                assert_eq!(
                    PolygammaRoute::for_arguments(order, x),
                    Unsupported,
                    "at {x:e}"
                );
            }
            for x in [f64::MIN_POSITIVE, 0.5, 1.0, 1e10, f64::INFINITY] {
                assert_eq!(PolygammaRoute::for_arguments(order, x), Hurwitz, "at {x:e}");
            }
            // Every NaN-returning arm agrees on the value, which is why only the route separates
            // them.
            for x in [f64::NAN, 0.0, -2.0, -0.5] {
                assert!(polygamma(order, x).is_nan(), "at {x:e}");
            }
        }
    }
}
