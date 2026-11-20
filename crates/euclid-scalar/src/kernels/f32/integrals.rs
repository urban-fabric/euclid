//! The `f32` peer of [`crate::kernels::integrals`]: `E₁`, `Ei`, `Si` and `Ci`.
//!
//! Native single precision. The [module header](super) listed these among the families that
//! widen to `f64`, on the grounds that "the `Si`/`Ci` series cancels against intermediates
//! reaching `4e20`, which `f32` cannot absorb at all". That is true of the series and beside the
//! point for the function: the series is only used below
//! [`TRIG_SERIES_MAX`], and at `x = 45` the `f64` kernel is evaluating a **continued fraction**,
//! not the series. A native `f32` does the same. The quoted intermediates never arise on either
//! carrier, because neither carrier goes near them.
//!
//! Each constant below is rederived at this format rather than inherited, and the answers are
//! not uniform: the two series/fraction joins come out **unchanged**, while every depth and
//! tolerance comes out much shorter — `MAX_TERMS` from 2,000 to 128, `CF_MAX_STEPS` from 400 to
//! 64, `EI_ASYMPTOTIC_MIN` from 40 to 20. That split is the useful thing to know. A join is set
//! by where two *methods* cross, which is mostly a property of the mathematics; a depth is set
//! by where a method reaches the format's resolution, which is not. Rederiving both is how you
//! find out which kind you have.

use crate::constants::f32::EULER_GAMMA;
use crate::math::arithmetic::{term_is_negligible, two_sum};
use crate::math::complex::{c_mul, c_recip};
use crate::math::continued_fraction::{LENTZ_TINY_F32, converged};

/// Below this argument `Si`/`Ci` use the power series; at or above it, the continued fraction.
///
/// **Unchanged from the `f64` peer's `4.0`, and that is the measurement's answer rather than an
/// inheritance.** Every other constant in this module moved; this one does not need to, which is
/// worth recording so the next reader does not assume it was simply copied.
///
/// Measured against the `f64` kernel, in `f32` ulps of `Si`, with the Lentz step count alongside:
///
/// | `x` | 1.0 | 2.0 | 3.0 | **4.0** | 6.0 | 10.0 |
/// |---|---|---|---|---|---|---|
/// | series | `1` | `0` | `0` | **`0`** | `2` | `31` |
/// | fraction | `5` | `0` | `2` | **`0`** | `1` | `0` |
/// | fraction steps | `36` | `20` | `15` | **`12`** | `9` | `6` |
///
/// The join is not delicate: anywhere in `[2, 6]` both arms are within two ulp. What picks `4.0`
/// out of that range is cost rather than accuracy — the series is the cheaper of the two while
/// it is accurate, and the fraction's step count falls as `x` grows, so handing over as late as
/// the series allows minimises the work on both sides. `10.0` would be too late: the series is at
/// 31 ulp by then, its `ε·Σ|tₖ|` floor finally showing.
const TRIG_SERIES_MAX: f32 = 4.0;

/// Below this argument `E₁` uses the power series; at or above it, the continued fraction.
///
/// Unchanged from the `f64` peer's `1.0`, and that is a measurement rather than an inheritance:
/// `E₁`'s series is `−γ − ln x + Σ (−1)^{k+1} x^k/(k·k!)`, whose terms at `x ≤ 1` are bounded by
/// `1/(k·k!)` and so never grow at all. There is nothing for the narrower format to change,
/// because the cancellation the join exists to avoid does not depend on `ε`.
const E1_SERIES_MAX: f32 = 1.0;

/// Above this argument `Ei` switches from its series to its asymptotic expansion.
///
/// **`20`, where the `f64` peer uses `40`.** Every term of `Ei`'s series is positive for `x > 0`,
/// so it suffers no cancellation and is preferred as long as it converges cheaply — but "cheaply"
/// is a different number here. The series needs about `x/ln x` terms to pass its peak, and its
/// error floor is `ε·Σtₖ ≈ ε·Ei(x)`, i.e. an ulp, at any depth. The asymptotic expansion's floor
/// is its least term, `≈ e^x √(2π/x)`, relative `≈ √(2π/x)` — which reaches `f32`'s resolution
/// much later than `f64`'s, so it cannot be relied on as early.
///
/// Measured against the `f64` kernel, in `f32` ulps, with the series' term count alongside:
///
/// | `x` | 5 | 10 | 15 | **20** | 25 | 40 | 60 |
/// |---|---|---|---|---|---|---|---|
/// | series | `0` | `1` | `2` | **`4`** | `5` | `7` | `4` |
/// | asymptotic | `396596` | `4357` | `54` | **`4`** | `2` | `1` | `2` |
/// | series terms | `22` | `32` | `41` | **`49`** | `57` | `79` | `107` |
///
/// They cross at exactly `20`, where both carry four ulp. The asymmetry is the thing to notice:
/// the asymptotic expansion is not merely worse below the join but *catastrophic* — `4357` ulp at
/// `x = 10` and five orders beyond that at `x = 5` — because it is a divergent series summed
/// through its least term, and that term is still large there. Erring low on this threshold is
/// unsafe in a way that erring high is not, and `f64`'s `40` errs high by design.
///
/// `20` also sits well below where `Ei` overflows the format — `e^x/x` passes `f32::MAX` at
/// `x ≈ 91` — so the branch is chosen on accuracy rather than on range.
const EI_ASYMPTOTIC_MIN: f32 = 20.0;

/// Iteration cap for the power series here. The `f64` peer allows 2,000.
///
/// The deepest any of them runs is the `Ei` series just below [`EI_ASYMPTOTIC_MIN`], which
/// terminates in **49** terms at `x = 20`; the trig and `E₁` series are shorter still, their
/// arguments being bounded by `4` and `1`. 128 is a backstop, not a schedule — and unlike the
/// `f64` peer's 2,000 it is close enough to the real depth to notice if one of them stops
/// converging.
const MAX_TERMS: u32 = 128;

/// Iteration cap for Lentz's method. The `f64` peer allows 400.
///
/// Convergence is geometric for `|z| ≥ 2`, and the binding case is not the trig join but `E₁`
/// just above [`E1_SERIES_MAX`]: at `|z| = 1` the fraction takes **36** steps, against 12 at the
/// trig join. 64 leaves room above the worst case without being so loose that a
/// non-converging argument would go unnoticed.
const CF_MAX_STEPS: u32 = 64;

/// Relative size at which Lentz's running factor is taken to have converged.
///
/// `1e-7`, against the `f64` peer's `1e-17`. Asking for more than the format holds does not buy
/// accuracy — the factor stops moving at `ε` and the loop then runs to its cap for nothing,
/// which on a path chosen for speed is the whole cost of the call.
const CF_EPS: f32 = 1e-7;

/// Relative size at which a series term is taken to be negligible against `Σ|tₖ|`.
const SERIES_EPS: f32 = 1e-8;

/// Compensated accumulation of a series whose terms are produced by `next`.
///
/// From the fourth term onward, stops when a term is negligible against the running magnitude.
/// `magnitude` tracks `Σ|tₖ|`, which is what an alternating series has to be compared against —
/// comparing against the running *sum* would stop far too early while the partial sums are still
/// cancelling.
///
/// The compensation matters more here than in `f64`, not less: it removes the `n`-fold growth of
/// rounding error, leaving the `ε·Σ|tₖ|` floor, and that floor is `2^29` larger in this format.
/// It is what keeps [`TRIG_SERIES_MAX`] as high as 4 rather than forcing the fraction down to
/// where it has not converged — at `x = 1` the fraction is both less accurate and three times
/// the work.
fn compensated_series(mut next: impl FnMut(u32) -> Option<f32>) -> f32 {
    let mut total = 0.0f32;
    let mut correction = 0.0f32;
    let mut magnitude = 0.0f32;
    for k in 0..MAX_TERMS {
        let Some(term) = next(k) else { break };
        magnitude += term.abs();
        let (s, e) = two_sum(total, term);
        total = s;
        correction += e;
        if term_is_negligible(term, magnitude, SERIES_EPS) && k > 2 {
            break;
        }
    }
    total + correction
}

// --- complex arithmetic, as (re, im) pairs -------------------------------
// The `f32` peers of the `f64` module's local helpers. Kept local for the same reason: the
// crate carries a `Complex` *trait* but no concrete complex scalar, and the continued fraction
// needs only these three operations.

/// The complex Lentz nudge at this format's floor; see [`crate::math::continued_fraction::nudge`].
fn nudge_at(z: (f32, f32)) -> (f32, f32) {
    crate::math::continued_fraction::nudge(z, LENTZ_TINY_F32)
}

/// `E₁(z)` for `Re z ≥ 0`, `z ≠ 0`, by Lentz's modified continued fraction.
///
/// `exp_neg_z` is `e^{−z}`, supplied by the caller because the two rays this is used on form it
/// differently — `e^{−x}` on the real axis, `cos x − i·sin x` on the imaginary one — and routing
/// both through a complex exponential would be a needless loss of accuracy.
fn e1_continued_fraction(z: (f32, f32), exp_neg_z: (f32, f32)) -> (f32, f32) {
    // Evaluate f = b₀ + a₁/(b₁ + a₂/(b₂ + ⋯)) with b_k = z + 2k + 1 and a_k = −k², so that
    // E₁(z) = e^{−z}/f.
    let mut f = nudge_at((z.0 + 1.0, z.1));
    let mut c = f;
    let mut d = (0.0, 0.0);

    for k in 1..CF_MAX_STEPS {
        let kf = k as f32;
        let a = -kf * kf; // real, so it scales a complex value componentwise
        let b = (z.0 + 2.0 * kf + 1.0, z.1);

        // D ← 1/(b + a·D)
        d = c_recip(nudge_at((b.0 + a * d.0, b.1 + a * d.1)));
        // C ← b + a/C
        let inv_c = c_recip(nudge_at(c));
        c = (b.0 + a * inv_c.0, b.1 + a * inv_c.1);

        let delta = c_mul(c, d);
        f = c_mul(f, delta);
        if converged(delta.0, delta.1, CF_EPS) {
            break;
        }
    }
    c_mul(exp_neg_z, c_recip(f))
}

/// Which route evaluates `E₁`.
///
/// Shaped like its sibling [`EiRoute`], and routed through `partial_cmp` rather than a guard
/// chain for a reason this file used to carry as a comment: `x.is_nan() || x < 0.0` has an
/// operator a mutant can weaken to `&&`, and that mutant is *equivalent* — a negative `x`
/// satisfies `x <= E1_SERIES_MAX`, reaches `ln x`, and comes back `NaN` anyway. The guard stated
/// the domain without creating the answer. An exhaustive match states it with nothing to weaken.
#[derive(Clone, Copy, Debug, PartialEq)]
enum E1Route {
    /// A value the function takes exactly: the pole at the origin, the limit at `+∞`, and the
    /// `NaN` that states the domain.
    Value(f32),
    /// The power series below [`E1_SERIES_MAX`], which cancels `ln x` against a growing sum.
    Series,
    /// The continued fraction above it: the stable route in the tail, where `E₁` decays like
    /// `e^{−x}/x` and the series' cancellation would dominate.
    ContinuedFraction,
}

fn e1_route(x: f32) -> E1Route {
    use core::cmp::Ordering;

    match x.partial_cmp(&0.0) {
        // The real `E₁` is defined for a positive argument.
        None | Some(Ordering::Less) => E1Route::Value(f32::NAN),
        Some(Ordering::Equal) => E1Route::Value(f32::INFINITY),
        Some(Ordering::Greater) if x.is_infinite() => E1Route::Value(0.0),
        Some(Ordering::Greater) if x <= E1_SERIES_MAX => E1Route::Series,
        Some(Ordering::Greater) => E1Route::ContinuedFraction,
    }
}

/// `E₁(x) = −γ − ln x + Σ_{k≥1} (−1)^{k+1} x^k/(k·k!)`, for small positive `x`.
fn e1_series(x: f32) -> f32 {
    let mut power_over_factorial = 1.0f32; // x^k/k!, carried across terms
    let series = compensated_series(|k| {
        let k1 = k as f32 + 1.0;
        power_over_factorial *= x / k1;
        let term = power_over_factorial / k1; // x^k/(k k!)
        Some(if k.is_multiple_of(2) { term } else { -term })
    });
    -EULER_GAMMA - x.ln() + series
}

/// `E₁(x) = ∫ₓ^∞ e^{−t}/t dt`, for `x > 0`.
///
/// Series below [`E1_SERIES_MAX`], continued fraction above it; see [`E1Route`].
pub(crate) fn exp_int_e1(x: f32) -> f32 {
    match e1_route(x) {
        E1Route::Value(value) => value,
        E1Route::Series => e1_series(x),
        // On the real axis the imaginary part is exactly zero throughout, so nothing is lost by
        // sharing the complex routine.
        E1Route::ContinuedFraction => e1_continued_fraction((x, 0.0), ((-x).exp(), 0.0)).0,
    }
}

/// Which route evaluates `Ei`.
#[derive(Clone, Copy, Debug, PartialEq)]
enum EiRoute {
    /// A value the function takes exactly: the limits and the singularity at zero.
    Value(f32),
    /// Negative argument, where `Ei(x) = −E₁(−x)`.
    Reflection,
    /// The all-positive power series, which cannot cancel.
    Series,
    /// The asymptotic expansion, summed through its least term.
    Asymptotic,
}

fn ei_route(x: f32) -> EiRoute {
    use core::cmp::Ordering;

    match x.partial_cmp(&0.0) {
        None => EiRoute::Value(f32::NAN),
        Some(Ordering::Equal) => EiRoute::Value(f32::NEG_INFINITY),
        Some(Ordering::Less) => EiRoute::Reflection,
        Some(Ordering::Greater) if x.is_infinite() => EiRoute::Value(f32::INFINITY),
        Some(Ordering::Greater) if x < EI_ASYMPTOTIC_MIN => EiRoute::Series,
        Some(Ordering::Greater) => EiRoute::Asymptotic,
    }
}

/// `Ei(x) = gamma + ln x + sum_{k≥1} x^k / (k k!)` for small positive x.
fn ei_series(x: f32) -> f32 {
    let mut power_over_factorial = 1.0f32;
    let series = compensated_series(|k| {
        let k1 = k as f32 + 1.0;
        power_over_factorial *= x / k1;
        Some(power_over_factorial / k1)
    });
    EULER_GAMMA + x.ln() + series
}

/// Sum `k!/x^k` through the least term for the asymptotic expansion of `Ei(x)`.
fn ei_asymptotic_sum(x: f32) -> f32 {
    let mut term = 1.0f32;
    let mut total = 1.0f32;
    for k in 1..MAX_TERMS {
        let prev = term;
        term *= k as f32 / x;
        if term > prev {
            break; // past the minimum term; the asymptotic series diverges beyond it
        }
        total += term;
        if term_is_negligible(term, total, SERIES_EPS) {
            break;
        }
    }
    total
}

fn ei_asymptotic(x: f32) -> f32 {
    // Split e^x into two factors so an intermediate exponential cannot overflow while the final
    // value is still representable. `f32` needs this sooner than `f64`: `e^x` alone passes
    // `f32::MAX` at `x ≈ 88`, where `Ei(x) = e^x/x` is still finite to `x ≈ 91`.
    let exp_half = (0.5 * x).exp();
    exp_half * ((exp_half / x) * ei_asymptotic_sum(x))
}

/// `Ei(x)`, the exponential integral.
pub(crate) fn exp_int_ei(x: f32) -> f32 {
    match ei_route(x) {
        EiRoute::Value(value) => value,
        EiRoute::Reflection => -exp_int_e1(-x),
        EiRoute::Series => ei_series(x),
        EiRoute::Asymptotic => ei_asymptotic(x),
    }
}

/// Which route evaluates the pair `(Ci, Si)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TrigIntegralRoute {
    /// The alternating power series, below [`TRIG_SERIES_MAX`].
    Series,
    /// One pass of the continued fraction along the imaginary axis, which returns both.
    Fraction,
}

impl TrigIntegralRoute {
    fn for_argument(x: f32) -> Self {
        if x < TRIG_SERIES_MAX {
            Self::Series
        } else {
            Self::Fraction
        }
    }
}

/// The two trigonometric integrals at one argument, which [`trig_integrals`] produces together.
///
/// Named fields rather than a `(f32, f32)`, for the reason the `f64` peer in
/// [`crate::kernels::integrals`] gives: the pair is two different functions, not a coordinate,
/// and a caller that swapped them would still typecheck.
#[derive(Clone, Copy, Debug, PartialEq)]
struct TrigIntegrals {
    /// `Ci(x) = γ + ln x + ∫₀ˣ (cos t − 1)/t dt`, the cosine integral.
    cosine: f32,
    /// `Si(x) = ∫₀ˣ sin(t)/t dt`, the sine integral.
    sine: f32,
}

/// Both trigonometric integrals at `x > 0`, from a single evaluation.
///
/// The two are the real and imaginary parts of one exponential integral, so taking them together
/// is both cheaper than two passes and guarantees they stay mutually consistent.
fn trig_integrals(x: f32) -> TrigIntegrals {
    debug_assert!(x > 0.0 && x.is_finite());
    match TrigIntegralRoute::for_argument(x) {
        TrigIntegralRoute::Series => {
            // Ci(x) = gamma + ln x + sum_{k>=1} (-1)^k x^{2k} / ((2k)(2k)!)
            let mut power_over_factorial = 1.0f32; // x^{2k}/(2k)!
            let cosine = EULER_GAMMA
                + x.ln()
                + compensated_series(|k| {
                    let m = 2.0 * (k as f32 + 1.0);
                    power_over_factorial *= x * x / ((m - 1.0) * m);
                    let term = power_over_factorial / m;
                    Some(if k.is_multiple_of(2) { -term } else { term })
                });
            // Si(x) = sum_{k>=0} (-1)^k x^{2k+1} / ((2k+1)(2k+1)!)
            let mut power_over_factorial = x; // x^{2k+1}/(2k+1)!
            let sine = compensated_series(|k| {
                if k > 0 {
                    let m = 2.0 * k as f32;
                    power_over_factorial *= x * x / (m * (m + 1.0));
                }
                let term = power_over_factorial / (2.0 * k as f32 + 1.0);
                Some(if k.is_multiple_of(2) { term } else { -term })
            });
            TrigIntegrals { cosine, sine }
        }
        TrigIntegralRoute::Fraction => {
            // E1(ix) = -Ci(x) + i (Si(x) - pi/2), with e^{-ix} = cos x - i sin x.
            let (re, im) = e1_continued_fraction((0.0, x), (x.cos(), -x.sin()));
            TrigIntegrals {
                cosine: -re,
                sine: core::f32::consts::FRAC_PI_2 + im,
            }
        }
    }
}

/// `Si(x) = ∫₀ˣ sin(t)/t dt`, the sine integral. Odd in `x`.
pub(crate) fn sin_int(x: f32) -> f32 {
    match SinIntRoute::for_argument(x) {
        SinIntRoute::Undefined => f32::NAN,
        // Returning the argument rather than a literal zero, so a negative zero survives: `Si`
        // is odd, hence `Si(−0) = −0`.
        SinIntRoute::AtOrigin => x,
        SinIntRoute::AtInfinity => core::f32::consts::FRAC_PI_2.copysign(x),
        SinIntRoute::Quadrature => {
            let si = trig_integrals(x.abs()).sine;
            // A sign-bit test rather than `* x.signum()`: multiplying and dividing by `±1` give
            // the same answer, so that spelling buys mutants with no observable difference. The
            // argument is never zero here — [`SinIntRoute::AtOrigin`] takes that case.
            if x.is_sign_negative() { -si } else { si }
        }
    }
}

/// Which route evaluates `Si` at this argument.
///
/// `Si` is odd and entire, so no arm here is a singularity: the quadrature takes `|x|` and the
/// sign is applied afterwards. What the arms divide is the format — the two ends of it, and the
/// signed zero in the middle that the general route would round away.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SinIntRoute {
    /// Not a number.
    Undefined,
    /// `x = ±0`, where `Si` is zero with the argument's own sign.
    AtOrigin,
    /// `±∞`, where `Si` converges to `±π/2` — the Dirichlet integral.
    AtInfinity,
    /// Everywhere else, by quadrature on `|x|` with the sign applied afterwards.
    Quadrature,
}

impl SinIntRoute {
    fn for_argument(x: f32) -> Self {
        if x.is_nan() {
            Self::Undefined
        } else if x == 0.0 {
            Self::AtOrigin
        } else if x.is_infinite() {
            Self::AtInfinity
        } else {
            Self::Quadrature
        }
    }
}

/// Which route evaluates `Ci` at this argument.
///
/// Unlike [`SinIntRoute`] this one *is* about the function: `Ci` carries a `ln x` term, so it has
/// a genuine singularity at the origin and no real value at all to the left of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CosIntRoute {
    /// `NaN`, and the whole negative half-line including `−∞`, where the `ln x` term puts `Ci` on
    /// the far side of a branch cut.
    Undefined,
    /// `x = 0`, where `ln x` diverges and `Ci` goes to `−∞`.
    AtOrigin,
    /// `+∞`, where `Ci` decays to zero.
    AtInfinity,
    /// Everywhere else, by quadrature.
    Quadrature,
}

impl CosIntRoute {
    fn for_argument(x: f32) -> Self {
        if x.is_nan() || x < 0.0 {
            Self::Undefined
        } else if x == 0.0 {
            Self::AtOrigin
        } else if x.is_infinite() {
            Self::AtInfinity
        } else {
            Self::Quadrature
        }
    }
}

/// `Ci(x) = γ + ln x + ∫₀ˣ (cos t − 1)/t dt`, the cosine integral, for `x > 0`.
pub(crate) fn cos_int(x: f32) -> f32 {
    match CosIntRoute::for_argument(x) {
        CosIntRoute::Undefined => f32::NAN,
        CosIntRoute::AtOrigin => f32::NEG_INFINITY,
        CosIntRoute::AtInfinity => 0.0,
        CosIntRoute::Quadrature => trig_integrals(x).cosine,
    }
}

#[cfg(test)]
mod tests;
