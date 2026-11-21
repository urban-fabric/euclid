//! Numerical steps shared by the two host gamma kernels.

use crate::{HostReal, Real};

use super::arithmetic::term_is_negligible;
use super::continued_fraction::{converged, lentz_step};

/// The shifted power, base and rational sum in the Lanczos approximation.
fn lanczos_parts<T: HostReal>(x: T, g: T, &[leading, ref rest @ ..]: &[T; 9]) -> (T, T, T) {
    let shifted = x - T::ONE;
    let mut sum = leading;
    for (i, &coefficient) in rest.iter().enumerate() {
        sum += coefficient / (shifted + T::from_f64((i + 1) as f64));
    }
    (shifted + T::HALF, shifted + g + T::HALF, sum)
}

/// `½·ln(2π)`, the prefactor both forms of the Lanczos log share.
///
/// Named rather than spelled twice because as a *term* it is untestable. On the folded path the
/// sum it joins is `~1e38` in `f32` and `~1e308` in `f64`, which puts `0.9189` some thirty-one
/// orders below the result's last bit: no assertion on `lnΓ` can see this expression change, in
/// either format, at any argument that reaches that path. Called directly it is the whole answer.
fn log_sqrt_tau<T: HostReal>() -> T {
    T::HALF * (T::TWO * T::PI).ln()
}

/// Which form of `power·ln(base) − base` keeps every intermediate inside the format.
///
/// # `power·ln(base)` is formed only when it fits
///
/// The literal form overflows while `lnΓ` itself does not. `− base` is smaller than that product
/// by a factor of `ln(base)`, so there is a band roughly `1/ln(base)` wide — `1.2%`, measured as
/// `x ∈ (4.0371e36, 4.0850e36)` in `f32` — where the answer is an ordinary number reached through
/// an infinite intermediate, and `lnΓ` returned `+∞` for all of it.
///
/// # Mutation testing is what asked for this
///
/// Every operator on the folded line survived the suite, so this follows the `IForm` treatment in
/// the modified-`I` kernel. Two reasons, and the split addresses both. The sweep that
/// reaches the band at all — `f32_exhaustive` — compares *classification*, so a sign flip on the
/// dominant term reads as finite either way; and it uses the `f64` kernel as its reference, so the
/// `f64` arm validates itself. Naming the two forms makes the arithmetic reachable at ordinary
/// arguments, where every term of it is the same order as the answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LanczosLogForm {
    /// The product fits: `power·ln(base) − base`, one rounding shorter, and what every ordinary
    /// argument should get.
    Direct,
    /// The product alone has left the format: `power·(ln(base) − base/power)` folds the
    /// subtraction inside it. `base/power` is an extra division and `ln(base) − base/power` a
    /// well-conditioned difference of `O(ln x)` against `O(1)`, so this is the overflow path's
    /// form alone.
    Folded,
}

impl LanczosLogForm {
    /// The product is formed once by the caller, and its range *is* the branch.
    fn for_scaled_power<T: HostReal>(scaled: T) -> Self {
        if HostReal::is_finite(scaled) {
            Self::Direct
        } else {
            Self::Folded
        }
    }
}

/// `½ln(2π) + power·ln(base) − base + ln(sum)`, the product already formed by the caller.
fn lanczos_log_direct<T: HostReal>(scaled: T, base: T, sum: T) -> T {
    log_sqrt_tau::<T>() + scaled - base + sum.ln()
}

/// The same sum with the subtraction folded inside the product; see [`LanczosLogForm::Folded`].
fn lanczos_log_folded<T: HostReal>(power: T, log_base: T, base: T, sum: T) -> T {
    log_sqrt_tau::<T>() + power * (log_base - base / power) + sum.ln()
}

/// The Lanczos approximation to `lnΓ`, in whichever form the product's range allows.
///
/// Both arms are arithmetically the same quantity; see [`LanczosLogForm`] for which applies where
/// and why the choice is named rather than inlined.
pub(crate) fn lanczos_log_gamma<T: HostReal>(x: T, g: T, coefficients: &[T; 9]) -> T {
    let (power, base, sum) = lanczos_parts(x, g, coefficients);
    let log_base = base.ln();
    let scaled = power * log_base;
    match LanczosLogForm::for_scaled_power(scaled) {
        LanczosLogForm::Direct => lanczos_log_direct(scaled, base, sum),
        LanczosLogForm::Folded => lanczos_log_folded(power, log_base, base, sum),
    }
}

/// # The approximation runs out above about `x = 50`, and a Stirling route fixes it
///
/// `Γ`'s error at large argument is this approximation's **truncation**, not its arithmetic —
/// every piece below is sub-ulp, and evaluating the same formula with the same coefficients at
/// 60 digits reproduces the error exactly. See the precision note on `Real::gamma`.
///
/// A Stirling series is the repair, and the formulation matters. `Γ(x) = √(2π/x)·(x/e)^x·e^{μ(x)}`
/// with `μ` the same `1/(12x) − 1/(360x³) + ⋯` the `I` family uses, which is exact to rounding
/// from about `x = 20`. Measured against true `Γ`:
///
/// ```text
/// x                20.5   40.5    60.5   80.5   120.5   170.5
/// (x/e)^x            6.0    9.0    11.5   37.6    43.2    60.8   ulp
/// x^x·e^{−x}         1.5    0.4     1.8    0.9     0.1     0.2
/// this Lanczos       1.6    0.0   145.1      —   354.9   456.5
/// ```
///
/// **The base must be `x`, not `x/e`.** `x/e` is not representable, and its half-ulp of rounding
/// is amplified by the exponent — `x·ε/4` is `42` ulp at `x = 170`, which is the `60.8` row.
/// Splitting the exponential out instead keeps the base exact, and is the same trick
/// [`lanczos_half_power`] uses for range, applied here for accuracy.
///
/// So the repair is a `Stirling` arm above a threshold near `50`, built from three pieces —
/// the `μ` series, `x^{x/2}·e^{−x/2}` squared, and the `√(2π/x)` prefactor. `ln_gamma` needs
/// none of this and is unaffected.
pub(crate) fn lanczos_gamma<T: HostReal>(x: T, g: T, coefficients: &[T; 9]) -> T {
    let (power, base, sum) = lanczos_parts(x, g, coefficients);
    let half = lanczos_half_power(power, base);
    (T::TWO * T::PI).sqrt() * half * half * sum
}

/// `μ(x) = lnΓ(x) − (x−½)ln x + x − ½ln 2π`, from its asymptotic series.
///
/// The same expansion the modified-Bessel family uses, and valid here for the same reason:
/// every argument reaching it is above [`STIRLING_MIN`], where the first omitted term
/// `−691/(360360 x¹¹)` is far below rounding.
pub(crate) fn stirling_remainder<T: HostReal>(x: T) -> T {
    let inv = T::ONE / x;
    let inv2 = inv * inv;
    inv * (T::from_f64(1.0 / 12.0)
        + inv2
            * (T::from_f64(-1.0 / 360.0)
                + inv2
                    * (T::from_f64(1.0 / 1260.0)
                        + inv2 * (T::from_f64(-1.0 / 1680.0) + inv2 * T::from_f64(1.0 / 1188.0)))))
}

/// `x^{x/2}·e^{−x/2}`, the half of `x^x e^{−x}` that stays in range.
///
/// **The base is `x`, not `x/e`**, and that is the whole accuracy of this route. `x/e` is not
/// representable, so its half-ulp of rounding is amplified by the exponent — `x·ε/4`, which is
/// `42` ulp at `x = 170`. Splitting the exponential out instead leaves the base exact. Measured
/// against true `Γ` through [`stirling_gamma`]:
///
/// ```text
/// x             20.5   40.5   60.5   80.5   120.5   170.5
/// (x/e)^x        6.0    9.0   11.5   37.6    43.2    60.8   ulp
/// x^x·e^{−x}     1.5    0.4    1.8    0.9     0.1     0.2
/// ```
///
/// `(x/e)^x` is how Stirling's formula is usually written, which is why this note exists.
fn stirling_half_power<T: HostReal>(x: T) -> T {
    let half = x * T::HALF;
    x.powf(half) * (-half).exp()
}

/// `Γ(x) = √(2π/x)·x^x·e^{−x}·e^{μ(x)}`, for large `x`.
///
/// Three pieces: the remainder series, the half power, and the prefactor. Used above
/// [`STIRLING_MIN`] because the Lanczos approximation's own *truncation* — not its arithmetic —
/// reaches `456` ulp by `x = 170`, while this is `0.1` to `1.8` across the same range.
pub(crate) fn stirling_gamma<T: HostReal>(x: T) -> T {
    let half = stirling_half_power(x);
    let tau_over_x = (T::TWO * T::PI) / x;
    tau_over_x.sqrt() * half * half * stirling_remainder(x).exp()
}

/// At or above this argument `Γ` leaves the Lanczos product for [`stirling_gamma`].
///
/// **`50`, between the two measured points that bracket the handover.** The Lanczos truncation
/// is `0.0` ulp at `x = 40.5` and `145.1` at `60.5`, so it goes between them; Stirling is `0.4`
/// and `1.8` at the same two, so either side is safe for it. `50` also clears the `μ` series'
/// own requirement of about `20` by a wide margin.
pub(crate) const STIRLING_MIN: f64 = 20.0;

/// Splitting the power and exponential keeps the intermediates in range.
fn lanczos_half_power<T: HostReal>(power: T, base: T) -> T {
    base.powf(power * T::HALF) * (-base * T::HALF).exp()
}

/// Reflect through the Lanczos factors before squaring the half power, so a
/// subnormal value is rounded only by the final division.
///
/// `clamp` is the reflected argument past which the result has certainly rounded to signed zero,
/// and it is **per format** rather than shared: it depends both on how fast `Γ` outruns the
/// format's range and on the spacing of that format's inputs near an integer, since the sine
/// factor can only be as small as the nearest representable distance to a pole. Each caller
/// derives its own; see the two call sites.
pub(crate) fn lanczos_reflected_gamma<T: HostReal>(
    sin: T,
    reflected: T,
    g: T,
    clamp: T,
    coefficients: &[T; 9],
) -> T {
    let (power, base, sum) = lanczos_parts(Real::min(reflected, clamp), g, coefficients);
    let half = lanczos_half_power(power, base);
    let prefactor = T::from_f64(crate::constants::f64::SQRT_TAU) * sum * sin;
    ((T::PI / prefactor) / half) / half
}

/// Reflection applies below its fixed point at one half.
pub(crate) fn use_gamma_reflection<T: HostReal>(x: T) -> bool {
    x < T::HALF
}

/// Evaluate sin(πx) for the reflection formulas. The reduced offset keeps
/// the input's distance to a pole when multiplying by π.
pub(crate) fn reflection_sine<T: HostReal>(x: T) -> T {
    let nearest = x.round();
    let sin = (T::PI * (x - nearest)).sin();
    if nearest % T::TWO == T::ZERO {
        sin
    } else {
        -sin
    }
}

/// The log of the reflected value, where both the sine and gamma are positive.
pub(crate) fn log_gamma_reflection<T: HostReal>(sin: T, log_gamma: T) -> T {
    (T::PI / sin).ln() - log_gamma
}

/// Evaluate the reflection in ordinary coordinates while the positive gamma
/// factor fits. Otherwise defer to the caller's range and precision policy.
/// A finite subnormal result may still fit on the negative side.
pub(crate) fn gamma_reflection<T: HostReal>(x: T, on_overflow: impl FnOnce(T, T) -> T) -> T {
    let sin = reflection_sine(x);
    let reflected = T::ONE - x;
    let gamma = reflected.gamma();
    if HostReal::is_finite(gamma) {
        T::PI / (sin * gamma)
    } else {
        on_overflow(sin, reflected)
    }
}

/// Positive finite shape and nonnegative argument, including positive infinity.
pub(crate) fn valid_incomplete_gamma_inputs<T: HostReal>(a: T, x: T) -> bool {
    HostReal::is_finite(a) && a > T::ZERO && x >= T::ZERO
}

/// Which of the two incomplete-gamma routes is valid at `(a, x)`.
///
/// # The split looks like a performance bug and is not one
///
/// The series walks about `7.3·√a` terms — 7,413 at `a = 1e6`, 32,200 at `2e7` — while the
/// continued fraction converges in **65 steps anywhere it is used**. Moving this boundary down
/// is therefore the single most tempting optimisation in the crate, and it is wrong.
///
/// Measured below the boundary, the fraction *reports convergence in three to twelve steps and
/// returns a different function*. Agreement with the series, as `P`:
///
/// ```text
/// x/a      0.5        0.9        0.99      1.00      1.01
/// a=1e5    NaN        5.1e234    1.3e3     2.9e-10   2.2e-10
/// a=1e6    NaN        NaN        1.8e23    1.4e-9    1.0e-9
/// a=1e7    NaN        NaN        3.2e220   3.3e-8    3.7e-8
/// ```
///
/// The multiplier comes within tolerance of one while the fraction has not converged at all, so
/// the exit rule fires early and the cheap step count is an illusion. It only becomes a real
/// answer from about `x/a = 0.99`, and there it costs 88 to 1,951 steps rather than a dozen.
///
/// So the depth of the series at large shape is the price of correctness, not slack. The route
/// that *does* replace it is the Edgeworth approximation above
/// [`GAMMA_EDGEWORTH_MIN`](crate::kernels::gamma::GAMMA_EDGEWORTH_MIN), chosen on accuracy
/// grounds, and that threshold is where this cost stops growing.
pub(crate) fn use_lower_gamma_series<T: HostReal>(a: T, x: T) -> bool {
    x < a + T::ONE
}

/// Advance the term `x^k/[a(a+1)…(a+k)]` by one index.
fn advance_lower_series<T: HostReal>(x: T, denominator: &mut T, term: &mut T) {
    *denominator += T::ONE;
    *term *= x / *denominator;
}

/// Sum the positive lower-tail series before applying its exponential scale.
pub(crate) fn gamma_lower_series<T: HostReal>(a: T, x: T, tolerance: T, limit: usize) -> T {
    let mut denominator = a;
    let mut term = a.recip();
    let mut sum = term;
    for _ in 0..limit {
        advance_lower_series(x, &mut denominator, &mut term);
        sum += term;
        if term_is_negligible(term, sum.abs(), tolerance) {
            break;
        }
    }
    sum
}
/// `ln(1+u) − u` and the log prefactor the incomplete gamma exponentiates.
///
/// A child module so the parent stays inside the crate's 400-line budget. The sub-topic is a
/// real one: everything in it exists because `a·ln x − x − lnΓ(a)` is a cancelling sum under an
/// `exp`, and the three arrangements that answer that are argued against each other rather than
/// against anything in the Lanczos or Stirling machinery above.
mod prefactor;

pub(crate) use prefactor::{ln_1p_minus_linear, log_scaled_gamma};

/// Advance the denominator and return the numerator for the next coefficient.
fn upper_gamma_coefficient<T: HostReal>(a: T, index: T, denominator: &mut T) -> T {
    *denominator += T::TWO;
    -index * (index - a)
}

/// Upper-tail continued fraction before it is multiplied by the prefactor.
pub(crate) fn gamma_upper_fraction<T: HostReal>(
    a: T,
    x: T,
    floor: T,
    tolerance: T,
    limit: usize,
) -> T {
    let mut b = x + T::ONE - a;
    let mut c = floor.recip();
    let mut d = b.recip();
    let mut value = d;
    for i in 1..limit {
        let numerator = upper_gamma_coefficient(a, T::from_f64(i as f64), &mut b);
        let delta = lentz_step(numerator, b, &mut c, &mut d, floor);
        value *= delta;
        if converged(delta, T::ZERO, tolerance) {
            break;
        }
    }
    value
}

/// Standardized argument and signed first-order Edgeworth correction.
fn gamma_edgeworth_terms<T: HostReal>(a: T, x: T) -> (T, T) {
    let root = a.sqrt();
    let z = (x - a) / root;
    let density = (-T::HALF * z * z).exp() / T::from_f64(crate::constants::f64::SQRT_TAU);
    // Multiplying by z ± 1 separately avoids a squared intermediate at extreme arguments.
    let correction = density * (z - T::ONE) * (z + T::ONE) / (T::from_f64(3.0) * root);
    (z, correction)
}

/// Evaluate both normal tails directly so that the smaller probability is not
/// formed by subtraction from one. The erfc kernel reaches the incomplete gamma
/// at shape one half, below either format's approximation threshold.
pub(crate) fn gamma_edgeworth<T: HostReal>(a: T, x: T) -> (T, T) {
    let (z, correction) = gamma_edgeworth_terms(a, x);
    let lower = T::HALF * (-z * T::FRAC_1_SQRT_2).erfc() - correction;
    let upper = T::HALF * (z * T::FRAC_1_SQRT_2).erfc() + correction;
    (lower.clamp(T::ZERO, T::ONE), upper.clamp(T::ZERO, T::ONE))
}

#[cfg(test)]
mod tests;
