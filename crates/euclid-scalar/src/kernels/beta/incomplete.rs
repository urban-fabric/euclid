//! The regularized incomplete beta `Iₓ(a, b)`, `x ∈ [0, 1]`, `a, b > 0`.
//!
//! A child module so the parent stays inside the crate's 400-line budget, on the same seam
//! `kernels::gamma` uses: what is left there is the *complete* beta — `ln B(a, b)`, `B(a, b)` and
//! the Stirling machinery they share — while the incomplete `Iₓ(a, b)`, its eight-arm router, the
//! continued fraction and the Edgeworth approximation are this module. The routing is the
//! substance here: most of [`BetaRegMethod`]'s arms are exact forms that exist to keep the
//! continued fraction away from arguments where its intermediates misbehave.

use super::*;

/// Which closed form or approximation evaluates `Iₓ(a, b)`.
///
/// Eight arms, which is what this function is: most of them are exact forms that exist to keep
/// the fraction away from arguments where its intermediates misbehave, and naming them is the
/// only way the ordering stays legible. The order is load-bearing — the endpoint checks precede
/// the shape checks, and the two gamma limits precede Edgeworth, so a pair qualifying for both
/// takes the gamma route.
///
/// Matched rather than compared, so adding an arm is a compile error at the one site that has to
/// say what to do with it. The single-precision peer carries the same enum.
#[derive(Clone, Copy, Debug, PartialEq)]
enum BetaRegMethod {
    /// Not two positive finite shapes.
    Invalid,
    /// `x` outside `(0, 1)`, or both shapes so small the distribution is two endpoint masses.
    Value(f64),
    /// `a = 1`: `I = 1 − (1−x)^b`, exact.
    UnitLower,
    /// `b = 1`: `I = x^a`, exact.
    UnitUpper,
    /// One shape vastly larger than the other, where the beta tends to a gamma in `−ln(1−x)`.
    GammaLimit,
    /// The same limit reflected, which preserves the small tail.
    GammaLimitReflected,
    /// `min(a, b) ≥ `[`BETA_EDGEWORTH_MIN`]: the Edgeworth-corrected normal.
    Edgeworth,
    /// Otherwise the continued fraction, with Edgeworth as its fallback if it fails to converge.
    Fraction,
}

impl BetaRegMethod {
    fn for_arguments(a: f64, b: f64, x: f64) -> Self {
        if !a.is_finite() || !b.is_finite() || a <= 0.0 || b <= 0.0 {
            Self::Invalid
        } else if x <= 0.0 {
            Self::Value(0.0)
        } else if x >= 1.0 {
            Self::Value(1.0)
        } else if a.max(b) <= 1e-22 && !x.is_nan() {
            // The endpoint masses are b/(a+b) at zero and a/(a+b) at one. Throughout the
            // representable interior the relative correction is O(max(a,b)·max(|ln x|,
            // |ln(1−x)|)), below 1e-19 here. The `f32` peer places this bound at `1e-18`.
            Self::Value(b / (a + b))
        } else if a == 1.0 {
            Self::UnitLower
        } else if b == 1.0 {
            Self::UnitUpper
        } else if beta_gamma_rate(a, b).is_some() {
            Self::GammaLimit
        } else if beta_gamma_rate(b, a).is_some() {
            Self::GammaLimitReflected
        } else if a.min(b) >= BETA_EDGEWORTH_MIN {
            Self::Edgeworth
        } else {
            Self::Fraction
        }
    }
}

pub(crate) fn beta_reg(a: f64, b: f64, x: f64) -> f64 {
    match BetaRegMethod::for_arguments(a, b, x) {
        BetaRegMethod::Invalid => f64::NAN,
        BetaRegMethod::Value(value) => value,
        // Integrating the density gives exact forms that also avoid the continued fraction's
        // large intermediates when one shape is much larger than the other.
        BetaRegMethod::UnitLower => -(b * (-x).ln_1p()).exp_m1(),
        BetaRegMethod::UnitUpper => x.powf(a),
        BetaRegMethod::GammaLimit => {
            let rate = beta_gamma_rate(a, b).unwrap_or(b);
            crate::kernels::gamma::gamma_lr(a, -rate * (-x).ln_1p())
        }
        BetaRegMethod::GammaLimitReflected => {
            // Reflection in log coordinates preserves the small tail.
            let rate = beta_gamma_rate(b, a).unwrap_or(a);
            crate::kernels::gamma::gamma_ur(b, -rate * x.ln())
        }
        BetaRegMethod::Edgeworth => beta_edgeworth(a, b, x),
        BetaRegMethod::Fraction => beta_fraction_cdf(a, b, x)
            .unwrap_or_else(|| beta_edgeworth(a, b, x))
            .clamp(0.0, 1.0),
    }
}

/// Evaluate the fraction in the chosen orientation. An unconverged fraction
/// leaves the decision to use an approximation with the caller.
fn beta_fraction_cdf(a: f64, b: f64, x: f64) -> Option<f64> {
    // Above the Stirling threshold the prefactor is formed about the mode, where its three
    // large logarithms cancel symbolically instead of numerically. See
    // `math::beta::beta_power_terms`.
    let front = if use_centred_prefactor(a, b, x) {
        let remainders = stirling_remainder_of_sum(a, b)
            - stirling_remainder_from_one(a)
            - stirling_remainder_from_one(b);
        crate::math::beta::beta_power_terms(a, b, x, remainders)
    } else {
        crate::math::beta::beta_fraction_prefactor(a, b, x, ln_beta(a, b))
    };
    if crate::math::beta::use_lower_beta_fraction(a, b, x) {
        beta_cf(a, b, x).map(|h| front * h / a)
    } else {
        beta_cf(b, a, 1.0 - x).map(|h| 1.0 - front * h / b)
    }
}

/// Whether the prefactor is formed about the mode rather than from `ln B(a, b)`.
///
/// Three conditions, guarding different things: the smaller shape must reach
/// [`stirling_remainder_from_one`]'s climb, `a + b` must be in the Stirling series' range since
/// [`stirling_remainder_of_sum`] takes it directly, and the offset from the mode must be within
/// half of both shapes or the centring is the worse arrangement.
fn use_centred_prefactor(a: f64, b: f64, x: f64) -> bool {
    a.min(b) >= CENTRED_MIN
        && a + b >= STIRLING_MIN
        && crate::math::beta::use_beta_power_terms(a, b, x)
}

/// Smallest shape the centred prefactor is used at — a cost bound, since the climb in
/// [`stirling_remainder_from_one`] is one step per unit.
const CENTRED_MIN: f64 = 1.0;

/// Which side of the Edgeworth expansion's usable range the standardised point falls on.
///
/// # Named because two comparisons inside one guard cannot be tested apart
///
/// This was `if z.abs() > 40.0 { return f64::from(u8::from(z > 0.0)); }`, and mutation testing
/// left three survivors in it — `>` to `>=` and `>` to `==` on the magnitude, and `>` to `>=` on
/// the sign. Every one is unreachable through `beta_reg`: getting `|z|` to land exactly on `40`
/// needs a `(a, b, x)` nobody can solve for, and the sign test is only read once the magnitude
/// test has already passed. The arithmetic was correct and untestable in the same breath.
///
/// Split out, each arm is a value of a function of one argument, and a test names `40` and its
/// two neighbouring doubles directly.
///
/// Past `|z| = 40` the standard normal tail has left the format in both directions: `erfc` has
/// saturated to `0` or `2`, and `e^{−z²/2}` underflowed at `|z| ≈ 38.6`. So returning the
/// certainty changes no value the expansion could have produced, and it keeps `z·z` out of an
/// overflow that would make the correction `NaN`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EdgeworthRange {
    /// `|z| ≤ 40`: the expansion is evaluated.
    Interior,
    /// `z > 40`: the lower tail is `1`.
    Certain,
    /// `z < −40`: the lower tail is `0`.
    Impossible,
}

impl EdgeworthRange {
    /// Each bound is compared against separately, which is both the testable spelling and the
    /// only `NaN`-correct one.
    ///
    /// Going through `|z| ≤ MAX` instead sends a `NaN` to a saturation arm, because every
    /// comparison against it is false — and `beta_reg(NaN, a, b)` then returns `0` where it owes
    /// a `NaN`. `subnormal_beta_shapes_preserve_the_endpoint_mass_ratio` caught exactly that.
    /// Written this way a `NaN` fails both tests, lands in [`Self::Interior`], and the expansion
    /// propagates it as it always did.
    fn for_standardised(z: f64) -> Self {
        if z > EDGEWORTH_Z_MAX {
            Self::Certain
        } else if z < -EDGEWORTH_Z_MAX {
            Self::Impossible
        } else {
            Self::Interior
        }
    }
}

/// Largest `|z|` the Edgeworth expansion is evaluated at.
///
/// `erfc` has saturated by here and `e^{−z²/2}` underflowed at `|z| ≈ 38.6`, so the expansion has
/// no information left to add. A `NaN` is the alternative: `z·z` at a `z` that reached the
/// format's top overflows, and `inf · 0` in the correction is how that shows up.
const EDGEWORTH_Z_MAX: f64 = 40.0;

/// Rate for the gamma approximation to `T = -ln(1-X)`, with `X ~ Beta(a,b)`.
///
/// `(1-e^-t)^(a-1) = t^(a-1) e^(-(a-1)t/2) (sinh(t/2)/(t/2))^(a-1)`.
/// Moving the first exponential into the rate leaves a log-density correction
/// of order `(a-1)t²/24`. Near the mass, `t` is about `a/b`, so the switch
/// bounds this correction at the floating-point scale.
fn beta_gamma_rate(a: f64, b: f64) -> Option<f64> {
    // Once gamma itself would use Edgeworth, standardize directly in beta
    // coordinates. Rounding the transformed argument can erase its distance
    // from the gamma mean, even when the distance is many standard deviations.
    if a >= GAMMA_EDGEWORTH_MIN {
        return None;
    }
    let shift = a + 1.0;
    (b / shift / shift.sqrt() >= 5e7).then_some(b + 0.5 * (a - 1.0))
}

/// Iteration cap for the continued fraction.
///
/// The requirement grows with the parameters, because the fraction converges slowest exactly at
/// `x ≈ a/(a+b)` — the median, where the branch switch below gives no help. Measured iterations
/// to converge at `a = b`, `x = ½`:
///
/// | `a` | 1e2 | 1e3 | 1e4 | 1e5 | 1e6 | 1e7 | 1e8 |
/// |-----|-----|-----|-----|-----|-----|-----|-----|
/// | iterations | 25 | 54 | 115 | 243 | 526 | 1091 | 2282 |
///
/// This used to be `300`, which silently truncated from about `a = 1e4` on and left
/// `I_{1/2}(1e8, 1e8)` returning `0.687` where the true value is exactly `½`. `4000` carries the
/// fraction past [`BETA_EDGEWORTH_MIN`], where it hands over anyway, with room to spare.
const MAX_ITER: usize = 4000;
const EPS: f64 = 1e-15;

/// At or above this `min(a, b)` the Edgeworth-corrected normal replaces the continued fraction.
///
/// # Why `min(a, b)` and not `a + b`
///
/// `Beta(a,b)` approaches a normal only as **both** shape parameters grow; its skewness is
/// `≈ 2/√min(a,b)`, so the expansion parameter is the smaller one alone. Gating on the sum
/// instead admits `Beta(1, 1e8)` — an exponential-shaped density with skewness 2, nothing like
/// a normal — and the approximation is then wrong in the first digit. That mistake was made
/// here first and caught by the `tests::identities::beta_range` suite, which checks the exact
/// `I_x(1, b) = 1 − (1−x)^b`.
///
/// Measured worst `|Edgeworth − converged fraction|` over `z ∈ [−4, 4]`, which confirms the
/// error depends on `min` and essentially not on the ratio `b/a`:
///
/// | `min(a,b)` | ratio 1 | ratio 10 | ratio 1e2 | ratio 1e4 |
/// |---|---|---|---|---|
/// | 1e5 | `6.1e-7` | `4.3e-7` | `6.1e-7` | `6.3e-7` |
/// | 1e6 | `6.0e-8` | `4.3e-8` | `6.1e-8` | `6.3e-8` |
///
/// # Where the crossover falls
///
/// Two independent error sources meet, and neither is fixable by iterating harder:
///
/// - **The fraction's prefactor cancels.** `front` is `exp(a ln x + b ln(1−x) − ln B(a,b))`,
///   three terms of magnitude `≈ 1.386n` for `n = a + b` summing to an `O(1)` exponent. The
///   exponent's absolute error is `≈ 1.386nε`, which `exp` hands back as *relative* error —
///   `3.1e-9` at `n = 1e7`, `3.1e-8` at `n = 1e8`. That is the subtraction, not the truncation.
/// - **The Edgeworth form improves as `≈ 0.0605/min(a,b)`**, from the table above.
///
/// Setting those equal at the worst case `a = b` (so `n = 2min`) gives `min ≈ 1e7`, where both
/// carry about `6e-9`. Below it the fraction is better; above it the approximation is.
///
/// # What it buys
///
/// The symmetric case — the one that was catastrophically wrong — becomes **exact** rather than
/// merely better: `Beta(a,a)` has zero skewness, so the correction vanishes identically and
/// `I_{1/2}(a,a)` is `Φ(0) = ½` to the last bit, at every `a`. Measured against that closed
/// form, `a = 1e7` went from `9.5e-3` to `0`, and `a = 1e8` from `0.187` to `0`.
pub(crate) const BETA_EDGEWORTH_MIN: f64 = 1e7;

/// The signed distance from the mean in units of `1/(a+b)`.
fn beta_mean_residual(a: f64, n: f64, n_error: f64, x: f64) -> f64 {
    // Near the mean, product − a is exact by Sterbenz's lemma. Rounding
    // a/n first can move x by many standard deviations at large shapes.
    let (product, product_error) = x.two_product(n);
    let (offset, offset_error) = (product - a).two_sum(product_error);
    x.mul_add(n_error, offset) + offset_error
}

/// `Iₓ(a, b)` from the Edgeworth-corrected normal approximation; see [`BETA_EDGEWORTH_MIN`].
///
/// The peer of [`crate::kernels::gamma::gamma_lr`]'s `gamma_edgeworth`, and the same expansion: a normal
/// with the distribution's own mean and variance, plus the first skewness correction
/// `−φ(z)·γ₁(z²−1)/6`. For `Beta(a,b)`,
///
/// ```text
/// μ = a/n,   σ² = ab/(n²(n+1)),   γ₁ = 2(b−a)√(n+1)/((n+2)√(ab)),   n = a + b
/// ```
///
/// `Φ` is formed as `½erfc(−z/√2)` so a small lower tail keeps its relative accuracy instead of
/// being lost to a complement. The `erfc` call reaches into [`erf`](crate::kernels::erf), which
/// is built on `gamma_lr` — but that recursion terminates at once, since `erfc` asks for shape
/// `½`, far below any threshold that returns here.
fn beta_edgeworth(a: f64, b: f64, x: f64) -> f64 {
    let (n, n_error) = a.two_sum(b);
    if !n.is_finite() {
        // Even the smaller shape is at least 2^970 here. Near the mean,
        // the lowest bits of the binary64 products in x(a+b) − a are
        // hundreds of bits above the transition width √(ab/(a+b+1)).
        // Halving keeps the sum finite and preserves the residual's sign.
        let half_a = 0.5 * a;
        let (half_n, half_error) = half_a.two_sum(0.5 * b);
        return match beta_mean_residual(half_a, half_n, half_error, x).partial_cmp(&0.0) {
            Some(Ordering::Less) => 0.0,
            Some(Ordering::Equal) => 0.5,
            Some(Ordering::Greater) => 1.0,
            None => f64::NAN,
        };
    }

    // z = (x·n − a)√(n+1)/√(ab). Taking the roots separately keeps the
    // factors in range even when a·b overflows or σ² underflows.
    let inverse_width = (n + 1.0).sqrt() / a.sqrt() / b.sqrt();
    let z = beta_mean_residual(a, n, n_error, x) * inverse_width;
    match EdgeworthRange::for_standardised(z) {
        EdgeworthRange::Certain => return 1.0,
        EdgeworthRange::Impossible => return 0.0,
        EdgeworthRange::Interior => {}
    }
    let skew = 2.0 * ((b - a) / (n + 2.0)) * inverse_width;
    let density = (-0.5 * z * z).exp() / SQRT_TAU;
    let correction = density * skew * (z * z - 1.0) / 6.0;
    let lower = 0.5 * crate::kernels::erf::erfc(-z * core::f64::consts::FRAC_1_SQRT_2) - correction;
    lower.clamp(0.0, 1.0)
}

/// Lentz continued fraction for the incomplete beta (`betacf` in Numerical Recipes).
///
/// Returns `None` when it exhausts [`MAX_ITER`] without converging, rather than handing back a
/// truncated partial sum. Returning the partial sum is what made the large-parameter defect
/// *silent*: the loop exited by either route and the caller could not tell which.
fn beta_cf(a: f64, b: f64, x: f64) -> Option<f64> {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = nonzero_pivot(1.0 - qab * x / qap, LENTZ_TINY_F64).recip();
    let mut h = d;
    // `1..=MAX_ITER`, not `1..MAX_ITER`: the exclusive form ran one step short of the cap, so
    // the constant did not mean what its name said.
    for m in 1..=MAX_ITER {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        h *= lentz_step(aa, 1.0, &mut c, &mut d, LENTZ_TINY_F64);
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        let del = lentz_step(aa, 1.0, &mut c, &mut d, LENTZ_TINY_F64);
        h *= del;
        if converged(del, 0.0, EPS) {
            return Some(h);
        }
    }
    None
}

#[cfg(test)]
mod tests;
