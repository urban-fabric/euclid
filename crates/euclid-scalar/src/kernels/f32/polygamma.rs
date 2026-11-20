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
            sign * (ln_gamma(n + 1.0)).exp() * hurwitz_zeta(n + 1.0, x)
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
mod tests {
    use super::*;

    fn close(a: f32, b: f32, tol: f32) {
        assert!(
            (a - b).abs() <= tol * b.abs().max(1.0),
            "{a:e} vs {b:e} (tol {tol:e})"
        );
    }

    #[test]
    fn polygamma_reduces_and_recurs_in_f32() {
        for i in 1..20 {
            let x = (i as f32) * 0.5;
            close(polygamma(0, x), digamma(x), 1e-5);
            close(polygamma(1, x), trigamma(x), 1e-5);
        }
        // psi^(2)(x+1) = psi^(2)(x) + 2/x^3
        for i in 1..20 {
            let x = (i as f32) * 0.5;
            close(
                polygamma(2, x + 1.0),
                polygamma(2, x) + 2.0 / x.powi(3),
                1e-4,
            );
        }
    }

    #[test]
    fn rgamma_and_exprel_in_f32() {
        assert_eq!(rgamma(0.0f32), 0.0);
        assert_eq!(rgamma(-3.0f32), 0.0);
        for i in 1..20 {
            let x = (i as f32) * 0.5;
            close(rgamma(x), (-ln_gamma(x)).exp(), 1e-5);
        }
        assert_eq!(exprel(0.0f32), 1.0);
        for i in 1..20 {
            let x = (i as f32) * 0.25;
            close(exprel(x), x.exp_m1() / x, 1e-5);
        }
    }

    /// The five reciprocal-gamma routes, asserted as decisions.
    ///
    /// [`RgammaRoute::Pole`] and [`RgammaRoute::AtPositiveInfinity`] return the same `0.0` for
    /// different reasons, so no value assertion can separate them — nor either from a reflection
    /// that happened to round to zero, which is what the exact-zero arm prevents.
    #[test]
    fn the_reciprocal_gamma_routes_split_the_line_at_the_integers() {
        use RgammaRoute::{AtPositiveInfinity, Exponential, Pole, Reflection, Undefined};
        for (x, want) in [
            (f32::NAN, Undefined),
            (f32::NEG_INFINITY, Undefined),
            (f32::INFINITY, AtPositiveInfinity),
            (0.0, Pole),
            (-0.0, Pole),
            (-1.0, Pole),
            (-39.0, Pole),
            (-f32::MAX, Pole),
            ((-1.0_f32).next_up(), Reflection),
            ((-1.0_f32).next_down(), Reflection),
            (-0.5, Reflection),
            (f32::MIN_POSITIVE, Exponential),
            (0.5, Exponential),
            (1.0, Exponential),
            (f32::MAX, Exponential),
        ] {
            assert_eq!(RgammaRoute::for_argument(x), want, "at {x:e}");
        }
        // The positive integers are *not* poles: `1/Γ(n)` is `1/(n−1)!`, finite and nonzero.
        for n in 1..6 {
            assert_eq!(RgammaRoute::for_argument(n as f32), Exponential);
            assert!(rgamma(n as f32) > 0.0);
        }
        // `Γ` overflows in this format between 35 and 36 — `Γ(35) = 2.95e38` against a ceiling of
        // `3.4e38` — while `1/Γ(36)` is `9.7e-41`, subnormal but representable. That window is
        // the whole reason the positive axis goes through the log domain rather than `recip()`.
        assert!(super::super::gamma::gamma(35.0_f32).is_finite());
        assert!(super::super::gamma::gamma(36.0_f32).is_infinite());
        let r = rgamma(36.0_f32);
        assert!(r > 0.0 && r.is_finite(), "rgamma(36) = {r}");
        assert!(r < f32::MIN_POSITIVE, "expected a subnormal, got {r}");
    }

    /// The non-finite arms, which nothing reached — three mutants lived in this comparison.
    ///
    /// `(eˣ − 1)/x` tends to `+∞` as `x → +∞` and to `0` as `x → −∞`, so the two infinities give
    /// different answers and the sign test between them is load-bearing. Swapping it is a defect
    /// a caller sees as a saturating value where it expected a vanishing one.
    #[test]
    fn exprel_separates_the_two_infinities() {
        assert_eq!(exprel(f32::INFINITY), f32::INFINITY);
        assert_eq!(exprel(f32::NEG_INFINITY), 0.0);
        assert!(exprel(f32::NAN).is_nan());
    }

    /// Below the crossover the series must be **correctly rounded**, which `exp_m1(x)/x` is not.
    ///
    /// This is what makes the branch worth having, and the bound is set from the gap rather than
    /// from taste: measured over 400,000 arguments in `(0, 1e-5]`, `1 + x/2` is accurate to
    /// `5.96e-8` and `exp_m1(x)/x` to `1.07e-7`. A bound of `7e-8` admits the first and refuses
    /// the second, so it fails if the crossover is narrowed until the series stops being used —
    /// which is otherwise invisible, since both paths are accurate to the `1e-5` that the
    /// round-trip check above allows.
    ///
    /// The reference is `f64`, four orders finer than f32 can represent, so the comparison is
    /// against the truth rather than against euclid's other path.
    #[test]
    fn exprel_is_correctly_rounded_below_its_crossover() {
        let mut worst = 0.0f64;
        for i in 1..=4_000u32 {
            for sign in [1.0f32, -1.0] {
                #[allow(clippy::cast_possible_truncation)]
                let x = sign * 1e-5 * (f64::from(i) / 4_000.0) as f32;
                let truth = f64::from(x).exp_m1() / f64::from(x);
                let relative = ((f64::from(exprel(x)) - truth) / truth).abs();
                worst = worst.max(relative);
            }
        }
        assert!(
            worst <= 7e-8,
            "exprel below its crossover is off by {worst:.3e}, which is the accuracy of \
             exp_m1(x)/x rather than of the series"
        );
    }

    /// The five `exprel` routes, at and either side of each boundary.
    #[test]
    fn the_exprel_forms_partition_the_line() {
        use ExprelForm::{AtOrigin, Infinite, Quotient, Series, Undefined};
        for (x, want) in [
            (0.0_f32, AtOrigin),
            (-0.0, AtOrigin),
            (f32::NAN, Undefined),
            (f32::INFINITY, Infinite),
            (f32::NEG_INFINITY, Infinite),
            // The magnitude decides, so both signs route alike, and the bound is open.
            (SERIES_MAX.next_down(), Series),
            (-SERIES_MAX.next_down(), Series),
            (1e-30, Series),
            (SERIES_MAX, Quotient),
            (-SERIES_MAX, Quotient),
            (1.0, Quotient),
            (90.0, Quotient),
        ] {
            assert_eq!(ExprelForm::for_argument(x), want, "at {x:e}");
        }
        // Two terms, not the `f64` peer's three: at the cutoff `x²/6` is `1.7e-11` against a
        // result near one, four orders under this format's `1.19e-7` resolution.
        assert_eq!(exprel(0.0), 1.0);
        assert!(exprel(f32::NAN).is_nan());
        assert_eq!(exprel(f32::INFINITY), f32::INFINITY);
        assert_eq!(exprel(f32::NEG_INFINITY), 0.0);
        assert_eq!(exprel(1e-6), 1.0 + 1e-6 / 2.0);

        use ExprelQuotient::{Direct, Shifted};
        assert_eq!(ExprelQuotient::for_numerator(1.0), Direct);
        assert_eq!(ExprelQuotient::for_numerator(f32::INFINITY), Shifted);
        // The band `(88.72, 93.26)`, where `exp_m1` is gone and the quotient is not: it used to
        // return `+∞` for every argument in it.
        assert!(!90.0_f32.exp_m1().is_finite());
        let value = exprel(90.0);
        assert!(value.is_finite() && value > 0.0, "exprel(90) = {value}");
        assert_eq!(value, (90.0_f32 - SHIFT).exp() * (SHIFT.exp() / 90.0));
        // Past the top of the band the quotient overflows too, and `+∞` is the right answer.
        assert_eq!(exprel(94.0), f32::INFINITY);
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
        for x in [f32::NAN, -3.0, -0.5, 0.0, 1.0, 1e10] {
            assert_eq!(PolygammaRoute::for_arguments(0, x), Digamma, "at {x:e}");
            assert_eq!(PolygammaRoute::for_arguments(1, x), Trigamma, "at {x:e}");
        }
        // And order 0 at a negative non-integer really does return a value, which is what makes
        // the ordering load-bearing rather than cosmetic.
        assert!(polygamma(0, -0.5).is_finite());
        assert!(polygamma(2, -0.5).is_nan());

        for order in [2, 3, 7] {
            assert_eq!(PolygammaRoute::for_arguments(order, f32::NAN), Undefined);
            // Poles at the non-positive integers, both signed zeros among them.
            for x in [0.0, -0.0, -1.0, -2.0, -39.0] {
                assert_eq!(PolygammaRoute::for_arguments(order, x), Pole, "at {x:e}");
            }
            // Real and finite there, but not reachable without a reflection formula.
            for x in [-0.5, -1.5, -10.5, (-1.0_f32).next_up()] {
                assert_eq!(
                    PolygammaRoute::for_arguments(order, x),
                    Unsupported,
                    "at {x:e}"
                );
            }
            for x in [f32::MIN_POSITIVE, 0.5, 1.0, 1e10, f32::INFINITY] {
                assert_eq!(PolygammaRoute::for_arguments(order, x), Hurwitz, "at {x:e}");
            }
            // Every NaN-returning arm agrees on the value, which is why only the route separates
            // them.
            for x in [f32::NAN, 0.0, -2.0, -0.5] {
                assert!(polygamma(order, x).is_nan(), "at {x:e}");
            }
        }
    }
}
