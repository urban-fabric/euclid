//! The exponential integrals `E₁` and `Ei`.
//!
//! A child module so the parent stays inside the crate's 400-line budget. Both are the same
//! function on different rays — `Ei(−x) = −E₁(x)` — so they share the continued fraction in
//! [`super`] and the series routes here. The trigonometric pair sit in
//! [`super::trigonometric`], being the *imaginary* ray of the same fraction.

use super::*;

/// Which route evaluates `E₁`.
///
/// Shaped like its sibling [`EiRoute`], and routed through `partial_cmp` for the same reason:
/// ordering against zero names the `NaN`, the negative half-line and the two positive regimes in
/// one exhaustive match, where the equivalent `x.is_nan() || x < 0.0` guard has an operator a
/// mutant can weaken. That particular mutant is *equivalent* rather than live — a negative `x`
/// reaches `ln x` in the series and comes back `NaN` anyway — but an exhaustive match has nothing
/// to weaken in the first place.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum E1Route {
    /// A value the function takes exactly: the pole at the origin, the limit at `+∞`, and the
    /// `NaN` that states the domain.
    Value(f64),
    /// The power series below [`E1_SERIES_MAX`], which cancels `ln x` against a growing sum.
    Series,
    /// `1 ≤ x ≤ 4`: a Chebyshev series, because *both* neighbours are weak there.
    ///
    /// This is not a moved threshold. The series cancels `ln x` against a growing sum and the
    /// fraction has not yet converged, so the error was elevated on both sides of the join —
    /// measured against a 400-bit reference, worst ulp per band with the condition number
    /// beside it:
    ///
    /// ```text
    /// x        [1,1.5]  [1.5,1.9]  [1.9,2]  [2,2.5]  [2.5,4]  [4,10]
    /// was           16         45       44       34       19      11   ulp
    /// kappa       2.07       2.64     2.68     2.81      3.6    5.19
    /// ```
    ///
    /// No boundary between two routes that are both wrong can fix that, which is what separates
    /// this case from `erfc`'s — there the two forms failed in opposite directions and moving
    /// the join was enough. `E₁` is analytic on `[1, 4]`, with its only singularity the
    /// logarithmic branch point at the origin, so a fit converges geometrically.
    Chebyshev,
    /// The continued fraction above it: the stable route in the tail, where `E₁` decays like
    /// `e^{−x}/x` and the series' cancellation would dominate.
    ContinuedFraction,
}

pub(super) fn e1_route(x: f64) -> E1Route {
    use core::cmp::Ordering;

    match x.partial_cmp(&0.0) {
        // The real `E₁` is defined for a positive argument.
        None | Some(Ordering::Less) => E1Route::Value(f64::NAN),
        Some(Ordering::Equal) => E1Route::Value(f64::INFINITY),
        Some(Ordering::Greater) if x.is_infinite() => E1Route::Value(0.0),
        Some(Ordering::Greater) if x < E1_CHEBYSHEV_MIN => E1Route::Series,
        Some(Ordering::Greater) if x <= E1_CHEBYSHEV_MAX => E1Route::Chebyshev,
        Some(Ordering::Greater) => E1Route::ContinuedFraction,
    }
}

/// The window the Chebyshev series covers.
///
/// `[7/8, 4]`. The lower edge is where the power series' cancellation starts to tell — it is
/// `3` ulp below `7/8` and `17` by `1.5` — and the upper is where the continued fraction has
/// converged enough to be the better route at `14` ulp against a condition number of `5.5`,
/// close to its own floor. Widening costs coefficients geometrically, since `E₁`'s branch point
/// at the origin is the nearest singularity: `[7/8, 4]` takes 33 terms where `[7/8, 6]` would
/// take about 45, to fix a band already within three of its floor.
///
/// # Why `7/8` and not `1`
///
/// `1` is where the series' error starts to climb and is the obvious place to put this. It is
/// also an argument the monotonicity suite scans, and `E₁` at `1` is on the *bit-monotone* list
/// — it earned that place when `E1_SERIES_MAX` rose and took the old route boundary off it,
/// removing an inversion of `114` ulp. Putting a new boundary back on `1` gave it a fresh
/// one-ulp inversion for no accuracy reason: the series is `3` ulp at `7/8` just as it is at
/// `1`, so the window can start an eighth lower and leave `1` interior to a single route.
///
/// A route boundary has to go *somewhere*, and nothing says it has to go on a round number that
/// other contracts key on.
///
/// # What this replaces, and why a third threshold was not the answer
///
/// There was a single join, `E1_SERIES_MAX`, and it had already been moved once — from `1` to
/// `2`, because the fraction was at its worst exactly where it took over. Both routes were
/// measured against a 90-digit reference then:
///
/// ```text
/// x       series   fraction
/// 1.00     0.00      0.00
/// 1.10     0.67     21.51   ulp
/// 1.50     0.62     11.87
/// 2.00     0.00      2.56
/// 2.50    28.22      2.51
/// 4.00   349.87      1.55
/// ```
///
/// Read down the columns and no single boundary is good: the series is already poor by `2.5` and
/// the fraction is still poor at `1.5`, so wherever the join goes, one side of it takes a route
/// carrying tens of ulp. That is what distinguishes this case from `erfc`'s, where the two forms
/// failed in *opposite* directions and moving the join was enough. Here the gap had to be
/// filled rather than divided, and `E1_SERIES_MAX` is retired because nothing routes on it any
/// more.
pub(super) const E1_CHEBYSHEV_MIN: f64 = 0.875;
pub(super) const E1_CHEBYSHEV_MAX: f64 = 4.0;

/// `E₁(x)` on `[1, 4]`, by Clenshaw on [`E1_CHEBYSHEV`] and then unscaling.
///
/// # The fit is of `x·e^x·E₁(x)`, and fitting `E₁` directly does not work
///
/// A Chebyshev fit distributes its error **absolutely**, and `E₁` spans two decades across this
/// window — `0.219` down to `3.8e-3`. Clenshaw's intermediates then scale with the series, not
/// with the answer, so at the small end they are about fifty times the result and its own
/// rounding is amplified by that. Measured: fitting `E₁` directly took `[2.5, 4]` from `26` ulp
/// to `36` — *worse* than the continued fraction it replaced — while fixing `[1, 2.5]`.
///
/// `x·e^x·E₁(x)` runs from `0.596` to `0.825` across the same window, so the intermediates are
/// the size of the answer everywhere and the fit needs thirty terms instead of thirty-eight.
/// Unscaling costs one `exp` whose argument is at most `4` — about two ulp at the far end, which
/// is the price of the other thirty-four.
pub(super) fn e1_chebyshev(x: f64) -> f64 {
    // Maps `[7/8, 4]` onto `[−1, 1]`. Both constants are dyadic, so the map is exact.
    let t = (x - 2.4375) / 1.5625;
    let twice = 2.0 * t;
    let (mut nearer, mut further) = (0.0_f64, 0.0_f64);
    for &coefficient in E1_CHEBYSHEV.iter().skip(1).rev() {
        let next = twice * nearer - further + coefficient;
        further = nearer;
        nearer = next;
    }
    let scaled = t * nearer - further + E1_CHEBYSHEV[0];
    scaled * (-x).exp() / x
}

/// Chebyshev coefficients of `x·e^x·E₁(x)` on `[7/8, 4]`, in `t = (2x − 39/8)/(25/8)`.
///
/// Derived rather than transcribed: evaluated in MPFR at Chebyshev nodes with the transform run
/// in 60-digit decimal. Thirty-three terms, where the tail falls below `2e-17` — and because the
/// target is near-constant, that bound is a relative one everywhere in the window rather than
/// only at its large end. See [`e1_chebyshev`] for why the scaling is not cosmetic.
const E1_CHEBYSHEV: [f64; 33] = [
    0.7282592163362371,
    0.11981746337379036,
    -0.02831751007135849,
    0.007085415499763361,
    -0.0018572888773952855,
    0.0005056889357289853,
    -0.00014204245430080045,
    4.093967080525747e-05,
    -1.205627783743597e-05,
    3.6154080805397834e-06,
    -1.101040442293554e-06,
    3.3978558933954066e-07,
    -1.0606995843856546e-07,
    3.3445149101404704e-08,
    -1.0639051950802642e-08,
    3.410856500325503e-09,
    -1.101146657284781e-09,
    3.577125140028559e-10,
    -1.1685887341477802e-10,
    3.8370405295599545e-11,
    -1.2657196817343642e-11,
    4.192857793006883e-12,
    -1.394314473504849e-12,
    4.653225554045439e-13,
    -1.5580108847091806e-13,
    5.2324293518249254e-14,
    -1.7622098879900082e-14,
    5.950424891736579e-15,
    -2.014177100661628e-15,
    6.833408141290552e-16,
    -2.323294796499081e-16,
    7.914828168617792e-17,
    -2.701449779484975e-17,
];

/// `E₁(x) = −γ − ln x + Σ_{k≥1} (−1)^{k+1} x^k/(k·k!)`, for small positive `x`.
pub(super) fn e1_series(x: f64) -> f64 {
    let mut power_over_factorial = 1.0f64; // x^k/k!, carried across terms
    let series = compensated_series(|k| {
        let k1 = f64::from(k) + 1.0;
        power_over_factorial *= x / k1;
        let term = power_over_factorial / k1; // x^k/(k k!)
        Some(if k.is_multiple_of(2) { term } else { -term })
    });
    -EULER_GAMMA - x.ln() + series
}

/// `E₁(x) = ∫ₓ^∞ e^{−t}/t dt`, for `x > 0`.
///
/// Series below `x = 1`, continued fraction above it; see [`E1Route`].
pub(crate) fn exp_int_e1(x: f64) -> f64 {
    match e1_route(x) {
        E1Route::Value(value) => value,
        E1Route::Series => e1_series(x),
        E1Route::Chebyshev => e1_chebyshev(x),
        // On the real axis the imaginary part is exactly zero throughout, so nothing is lost by
        // sharing the complex routine.
        E1Route::ContinuedFraction => e1_continued_fraction((x, 0.0), ((-x).exp(), 0.0)).0,
    }
}

/// `Ei(x) = −PV ∫_{−x}^∞ e^{−t}/t dt`, the exponential integral.
///
/// For positive argument every term of the series is positive, so there is no cancellation and
/// the series is preferred as far as it converges cheaply; the asymptotic expansion takes over
/// where it is comfortably accurate.
pub(crate) fn exp_int_ei(x: f64) -> f64 {
    match ei_route(x) {
        EiRoute::Value(value) => value,
        EiRoute::Reflection => -exp_int_e1(-x),
        EiRoute::Series => ei_series(x),
        EiRoute::Asymptotic => ei_asymptotic(x),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum EiRoute {
    Value(f64),
    Reflection,
    Series,
    Asymptotic,
}

pub(super) fn ei_route(x: f64) -> EiRoute {
    use core::cmp::Ordering;

    match x.partial_cmp(&0.0) {
        None => EiRoute::Value(f64::NAN),
        Some(Ordering::Equal) => EiRoute::Value(f64::NEG_INFINITY),
        Some(Ordering::Less) => EiRoute::Reflection,
        Some(Ordering::Greater) if x.is_infinite() => EiRoute::Value(f64::INFINITY),
        Some(Ordering::Greater) if x < EI_ASYMPTOTIC_MIN => EiRoute::Series,
        Some(Ordering::Greater) => EiRoute::Asymptotic,
    }
}

/// `Ei(x) = gamma + ln x + sum_{k≥1} x^k / (k k!)` for small positive x.
pub(super) fn ei_series(x: f64) -> f64 {
    let mut power_over_factorial = 1.0f64;
    let series = compensated_series(|k| {
        let k1 = f64::from(k) + 1.0;
        power_over_factorial *= x / k1;
        Some(power_over_factorial / k1)
    });
    EULER_GAMMA + x.ln() + series
}

pub(super) fn ei_asymptotic(x: f64) -> f64 {
    // Split e^x into two factors so an intermediate exponential cannot overflow
    // while the final value is still representable.
    let exp_half = (0.5 * x).exp();
    exp_half * ((exp_half / x) * ei_asymptotic_sum(x))
}

/// Sum `k!/x^k` through the least term for the asymptotic expansion of `Ei(x)`.
pub(super) fn ei_asymptotic_sum(x: f64) -> f64 {
    let mut term = 1.0f64;
    let mut total = 1.0f64;
    for k in 1..MAX_TERMS {
        let prev = term;
        term *= f64::from(k) / x;
        if term > prev {
            break; // past the minimum term; the asymptotic series diverges beyond it
        }
        total += term;
        if term_is_negligible(term, total, 1e-18) {
            break;
        }
    }
    total
}

#[cfg(test)]
mod tests;
