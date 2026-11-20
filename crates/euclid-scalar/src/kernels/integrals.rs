//! Integrals of elementary functions with no elementary antiderivative: the exponential
//! integrals `E₁` and `Ei`, and the trigonometric integrals `Si` and `Ci`.
//!
//! # One continued fraction behind all four
//!
//! Each of these has a power series about the origin and an asymptotic expansion for large
//! argument, with a gap between them where neither reaches full precision. A single continued
//! fraction closes that gap for all four, because all four are the same exponential integral
//! evaluated along different rays:
//!
//! ```text
//! E₁(z) = e^{−z} / (z + 1 − 1²/(z + 3 − 2²/(z + 5 − ⋯)))
//! ```
//!
//! On the positive real axis this gives `E₁` directly, and `Ei(−x) = −E₁(x)`. On the
//! imaginary axis,
//!
//! ```text
//! E₁(ix) = −Ci(x) + i·(Si(x) − π/2)
//! ```
//!
//! so one pass returns *both* trigonometric integrals — which additionally keeps them
//! mutually consistent, rather than letting two independent approximations drift apart.
//!
//! The fraction is evaluated by Lentz's method, which builds it front-to-back and therefore
//! needs no depth chosen in advance: it stops when the running factor stops moving.
//!
//! # Why the `Si`/`Ci` series cannot simply be pushed further
//!
//! The series for `Si`/`Ci` alternates, with terms peaking near `k = x/2`, so its partial sums
//! cancel catastrophically: at `x = 45` the terms reach `4e20` while `Si(45) ≈ 1.56`.
//!
//! Compensated summation does **not** rescue this, and it is worth being exact about why,
//! because the opposite is easy to assume. Compensation removes the growth of rounding error
//! with the *term count* — naive summation drifts like `n·ε·Σ|tₖ|`, compensated summation like
//! `ε·Σ|tₖ|` — but the `ε·Σ|tₖ|` floor itself cannot be removed by any summation rule: each
//! term is built by a recurrence and already carries its own relative error of order `ε`, so
//! the information is gone before the addition ever happens. At `x = 45` that floor is
//! `ε·4e20 ≈ 4e4`, and the series there is pure noise however it is summed.
//!
//! So the honest ranges are narrow. The series holds to about `x = 2`, where its terms peak
//! near `1.5` and nothing has been lost. The asymptotic expansion only reaches machine
//! precision past about `x = 40`, its optimally-truncated error being `≈ e^{−x}√x/x`. The
//! continued fraction covers everything from `x = 2` upward, so the asymptotic expansion is
//! not needed in the implementation at all — it lives in the tests instead, as an
//! *independent* check on the continued fraction at large argument.

use crate::constants::f64::EULER_GAMMA;
use crate::math::arithmetic::{term_is_negligible, two_sum};
use crate::math::complex::{c_mul, c_recip};
use crate::math::continued_fraction::{LENTZ_TINY_F64, converged};

/// Below this argument `Si`/`Ci` use the power series; at or above it, the continued fraction.
///
/// Sited where *both* routes are at machine precision rather than at the first point the
/// series is merely adequate. Measured relative error against 600-digit references: the series
/// is exact through `x = 5` and reaches `2.7e-10` by `x = 20`, while the continued fraction
/// converges slowly for small `|z|` — `6.5e-15` at `x = 0.5`, `2.6e-15` at `x = 2` — and
/// settles to `~1e-16` from `x = 3` on. At `x = 4` both are `~1e-16`, so the join costs
/// nothing from either side.
const TRIG_SERIES_MAX: f64 = 4.0;

/// Above this argument `E₁` leaves the power series for Lentz's continued fraction.
///
/// # `2.0`, raised from `1.0`, because the fraction is worst where it takes over
///
/// Both routes were measured against a 90-digit reference — the alternating series and the
/// continued fraction evaluated independently in `Decimal`, agreeing with each other to between
/// 24 and 75 digits, which is what makes either usable as a referee:
///
/// ```text
/// x       series   fraction
/// 1.00     0.00      0.00
/// 1.05     0.00     16.10   ulp
/// 1.10     0.67     21.51
/// 1.50     0.62     11.87
/// 2.00     0.00      2.56
/// 2.50    28.22      2.51
/// 3.00    76.64      1.20
/// 4.00   349.87      1.55
/// ```
///
/// The two cross at about `x = 2`, and the old threshold of `1.0` sat a decade below that — so
/// every argument in `[1, 2]` took the *worse* of the two routes. The fraction converges slowest
/// exactly where it is handed control, which is the same pattern `erfcx` shows above
/// `ERFCX_CF_MIN` and the modified Bessels showed below `I_ASYMPTOTIC_MIN`.
///
/// Raising it halves the pointwise error on `[1, 2]` — `0`–`11.7` ulp against `12`–`22` — and
/// also repairs the monotonicity that `stability::wiggle` records: the worst inversion was `114`
/// ulp just above `1.0`, and `E₁` is now **exactly monotone** there, with the residual peak `64`
/// ulp near `1.9` where the series' own cancellation is largest.
///
/// It cannot go higher. The series' terms are `xᵏ/(k·k!)`, alternating, and their peak grows with
/// `x` while `E₁` itself decays — so cancellation beats even the compensated summation past about
/// `2.5`, reaching `350` ulp by `x = 4` and `3.7e8` by `x = 12`.
const E1_SERIES_MAX: f64 = 2.0;

/// Above this argument `Ei` switches from its series to its asymptotic expansion. Every term
/// of the series is positive for `x > 0`, so it suffers no cancellation at all and is
/// preferred as far as it keeps converging cheaply.
const EI_ASYMPTOTIC_MIN: f64 = 40.0;

/// Iteration cap for the power series here.
const MAX_TERMS: u32 = 2_000;

/// Iteration cap for Lentz's method. Convergence is geometric for `|z| ≥ 2`, taking a few
/// dozen steps; this only bounds the cost at an argument near the series/fraction join.
const CF_MAX_STEPS: u32 = 400;

/// Relative size at which Lentz's running factor is taken to have converged.
const CF_EPS: f64 = 1e-17;

/// Compensated accumulation of a series whose terms are produced by `next`.
///
/// From the fourth term onward, stops when a term is negligible against the running magnitude.
/// The `magnitude` accumulator tracks `Σ|tₖ|`, which is what the stopping rule has to
/// compare against for an alternating series — comparing against the running *sum* would stop
/// far too early while the partial sums are still cancelling.
///
/// What this buys is the removal of the `n`-fold growth of rounding error, not immunity to
/// cancellation; see the module header. Every caller here is kept inside the range where
/// `ε·Σ|tₖ|` is itself negligible, so the compensation is the only remaining error term.
fn compensated_series(mut next: impl FnMut(u32) -> Option<f64>) -> f64 {
    let mut total = 0.0f64;
    let mut correction = 0.0f64;
    let mut magnitude = 0.0f64;
    for k in 0..MAX_TERMS {
        let Some(term) = next(k) else { break };
        magnitude += term.abs();
        let (s, e) = two_sum(total, term);
        total = s;
        correction += e;
        if term_is_negligible(term, magnitude, 1e-20) && k > 2 {
            break;
        }
    }
    total + correction
}

// --- complex arithmetic, as (re, im) pairs -------------------------------
// `euclid-scalar` carries a `Complex` *trait* but no concrete complex scalar, and the
// continued fraction needs only these few operations, so they are local to this module.

/// The complex Lentz nudge at this format's floor; see [`crate::math::continued_fraction::nudge`].
fn nudge_at(z: (f64, f64)) -> (f64, f64) {
    crate::math::continued_fraction::nudge(z, LENTZ_TINY_F64)
}

/// `E₁(z)` for `Re z ≥ 0`, `z ≠ 0`, by Lentz's modified continued fraction.
///
/// `exp_neg_z` is `e^{−z}`, supplied by the caller because the two rays this is used on form
/// it differently — `e^{−x}` on the real axis, `cos x − i·sin x` on the imaginary one — and
/// routing both through a complex exponential would be a needless loss of accuracy.
fn e1_continued_fraction(z: (f64, f64), exp_neg_z: (f64, f64)) -> (f64, f64) {
    // Evaluate f = b₀ + a₁/(b₁ + a₂/(b₂ + ⋯)) with b_k = z + 2k + 1 and a_k = −k², so that
    // E₁(z) = e^{−z}/f.
    let mut f = nudge_at((z.0 + 1.0, z.1));
    let mut c = f;
    let mut d = (0.0, 0.0);

    for k in 1..CF_MAX_STEPS {
        let kf = f64::from(k);
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
/// Shaped like its sibling [`EiRoute`], and routed through `partial_cmp` for the same reason:
/// ordering against zero names the `NaN`, the negative half-line and the two positive regimes in
/// one exhaustive match, where the equivalent `x.is_nan() || x < 0.0` guard has an operator a
/// mutant can weaken. That particular mutant is *equivalent* rather than live — a negative `x`
/// reaches `ln x` in the series and comes back `NaN` anyway — but an exhaustive match has nothing
/// to weaken in the first place.
#[derive(Clone, Copy, Debug, PartialEq)]
enum E1Route {
    /// A value the function takes exactly: the pole at the origin, the limit at `+∞`, and the
    /// `NaN` that states the domain.
    Value(f64),
    /// The power series below [`E1_SERIES_MAX`], which cancels `ln x` against a growing sum.
    Series,
    /// The continued fraction above it: the stable route in the tail, where `E₁` decays like
    /// `e^{−x}/x` and the series' cancellation would dominate.
    ContinuedFraction,
}

fn e1_route(x: f64) -> E1Route {
    use core::cmp::Ordering;

    match x.partial_cmp(&0.0) {
        // The real `E₁` is defined for a positive argument.
        None | Some(Ordering::Less) => E1Route::Value(f64::NAN),
        Some(Ordering::Equal) => E1Route::Value(f64::INFINITY),
        Some(Ordering::Greater) if x.is_infinite() => E1Route::Value(0.0),
        Some(Ordering::Greater) if x <= E1_SERIES_MAX => E1Route::Series,
        Some(Ordering::Greater) => E1Route::ContinuedFraction,
    }
}

/// `E₁(x) = −γ − ln x + Σ_{k≥1} (−1)^{k+1} x^k/(k·k!)`, for small positive `x`.
fn e1_series(x: f64) -> f64 {
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
enum EiRoute {
    Value(f64),
    Reflection,
    Series,
    Asymptotic,
}

fn ei_route(x: f64) -> EiRoute {
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
fn ei_series(x: f64) -> f64 {
    let mut power_over_factorial = 1.0f64;
    let series = compensated_series(|k| {
        let k1 = f64::from(k) + 1.0;
        power_over_factorial *= x / k1;
        Some(power_over_factorial / k1)
    });
    EULER_GAMMA + x.ln() + series
}

fn ei_asymptotic(x: f64) -> f64 {
    // Split e^x into two factors so an intermediate exponential cannot overflow
    // while the final value is still representable.
    let exp_half = (0.5 * x).exp();
    exp_half * ((exp_half / x) * ei_asymptotic_sum(x))
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
    fn for_argument(x: f64) -> Self {
        if x < TRIG_SERIES_MAX {
            Self::Series
        } else {
            Self::Fraction
        }
    }
}

/// Sum `k!/x^k` through the least term for the asymptotic expansion of `Ei(x)`.
fn ei_asymptotic_sum(x: f64) -> f64 {
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

/// The two trigonometric integrals at one argument, which [`trig_integrals`] produces together.
///
/// Named fields rather than a `(f64, f64)`: the pair is genuinely two different functions, not a
/// coordinate, and every call site previously picked one out by index. `Ci` and `Si` differ in
/// parity, in domain and in what they do at the origin — a reader asking which of `.0` and `.1`
/// was which had to go and look, and a caller that swapped them would still typecheck.
#[derive(Clone, Copy, Debug, PartialEq)]
struct TrigIntegrals {
    /// `Ci(x) = γ + ln x + ∫₀ˣ (cos t − 1)/t dt`, the cosine integral.
    cosine: f64,
    /// `Si(x) = ∫₀ˣ sin(t)/t dt`, the sine integral.
    sine: f64,
}

/// Both trigonometric integrals at `x > 0`, from a single evaluation.
///
/// The two are the real and imaginary parts of one exponential integral, so taking them
/// together is both cheaper than two passes and guarantees they remain mutually consistent.
fn trig_integrals(x: f64) -> TrigIntegrals {
    debug_assert!(x > 0.0 && x.is_finite());
    match TrigIntegralRoute::for_argument(x) {
        TrigIntegralRoute::Series => {
            // Ci(x) = gamma + ln x + sum_{k>=1} (-1)^k x^{2k} / ((2k)(2k)!)
            let mut power_over_factorial = 1.0f64; // x^{2k}/(2k)!
            let cosine = EULER_GAMMA
                + x.ln()
                + compensated_series(|k| {
                    let m = 2.0 * (f64::from(k) + 1.0);
                    power_over_factorial *= x * x / ((m - 1.0) * m);
                    let term = power_over_factorial / m;
                    Some(if k.is_multiple_of(2) { -term } else { term })
                });
            // Si(x) = sum_{k>=0} (-1)^k x^{2k+1} / ((2k+1)(2k+1)!)
            let mut power_over_factorial = x; // x^{2k+1}/(2k+1)!
            let sine = compensated_series(|k| {
                if k > 0 {
                    let m = 2.0 * f64::from(k);
                    power_over_factorial *= x * x / (m * (m + 1.0));
                }
                let term = power_over_factorial / (2.0 * f64::from(k) + 1.0);
                Some(if k.is_multiple_of(2) { term } else { -term })
            });
            TrigIntegrals { cosine, sine }
        }
        TrigIntegralRoute::Fraction => {
            // E1(ix) = -Ci(x) + i (Si(x) - pi/2), with e^{-ix} = cos x - i sin x.
            let (re, im) = e1_continued_fraction((0.0, x), (x.cos(), -x.sin()));
            TrigIntegrals {
                cosine: -re,
                sine: core::f64::consts::FRAC_PI_2 + im,
            }
        }
    }
}

/// `Si(x) = ∫₀ˣ sin(t)/t dt`, the sine integral. Odd in `x`.
pub(crate) fn sin_int(x: f64) -> f64 {
    match SinIntRoute::for_argument(x) {
        SinIntRoute::Undefined => f64::NAN,
        // Returning the argument rather than a literal zero, so a negative zero survives: `Si`
        // is odd, hence `Si(−0) = −0`.
        SinIntRoute::AtOrigin => x,
        SinIntRoute::AtInfinity => core::f64::consts::FRAC_PI_2.copysign(x),
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
    fn for_argument(x: f64) -> Self {
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
    fn for_argument(x: f64) -> Self {
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
pub(crate) fn cos_int(x: f64) -> f64 {
    match CosIntRoute::for_argument(x) {
        CosIntRoute::Undefined => f64::NAN,
        CosIntRoute::AtOrigin => f64::NEG_INFINITY,
        CosIntRoute::AtInfinity => 0.0,
        CosIntRoute::Quadrature => trig_integrals(x).cosine,
    }
}

#[cfg(test)]
mod tests;
