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

pub(crate) fn lanczos_gamma<T: HostReal>(x: T, g: T, coefficients: &[T; 9]) -> T {
    let (power, base, sum) = lanczos_parts(x, g, coefficients);
    let half = lanczos_half_power(power, base);
    (T::TWO * T::PI).sqrt() * half * half * sum
}

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

/// The common log prefactor for the lower series and upper continued fraction.
pub(crate) fn log_scaled_gamma<T: HostReal>(a: T, x: T, log_scale: T) -> T {
    log_scale - x + a * x.ln() - a.ln_gamma()
}

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
