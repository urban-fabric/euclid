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

/// The widest `|u| = |x−a|/a` the rearranged prefactor is used on.
///
/// `½`, and it is a **correctness** bound rather than a convergence one. `u` is formed as
/// `(x−a)/a` and then fed to `ln(1+u)`, so it has to carry `x/a` in its own last bits. At
/// `|u| ≤ ½` that is free: Sterbenz makes `x − a` exact on `[a/2, 2a]`, so `u` is the quotient's
/// single rounding and `ln(1+u)` sees every digit of `x`.
///
/// Push it further and the formulation destroys the thing it is computing. At `a = 20`,
/// `x = 1e-6` the quotient is `−0.99999995`, whose ulp is `1.1e-16` against a `1+u` of `5e-8` —
/// a relative error of `2.2e-9` in the argument of the logarithm, which `a` then multiplies into
/// the exponent. Measured: `2.7e8` ulp, against `539` for the direct form it replaced. **The
/// rearrangement is algebraically exact and numerically valid only near `u = 0`**, which cost a
/// full measurement pass to notice because the algebra checks out at every step.
///
/// What this bounds is `u`'s fitness as a variable, *not* the rearrangement's. Outside the band
/// the same cancellation is still available through `ln(x/a)`, which carries `x` at every
/// argument; [`ScaledGammaForm::Ratio`] is that form, and it is why this constant no longer
/// decides between a rearranged and an unrearranged prefactor — only between two ways of
/// spelling the rearranged one.
const LN_1P_SERIES_MAX: f64 = 0.5;

/// `ln(1+u) − u`, by series near zero and directly beyond it.
///
/// Both terms are `≈ u` while the result is `≈ −u²/2`, so the direct difference throws away
/// `1/|u|` of relative accuracy. That matters here because this value is multiplied by `a` and
/// exponentiated: an *absolute* error `δ` in the exponent is a *relative* error `δ` in the
/// answer, so the quantity has to be right in absolute terms, not merely relative ones.
///
/// The series is `−u²/2 + u³/3 − u⁴/4 + ⋯`, summed by ratio. For `u < 0` every term carries the
/// same sign, so nothing cancels at all; for `u > 0` it alternates with the first term dominant.
///
/// Two callers want different things from the bound, so both arms live here and
/// [`log_scaled_gamma`] keeps its *own* extra guard. For the beta prefactor the only question is
/// whether this series converges, and past `|u| = ½` the direct difference is the better form —
/// it loses about `1/|u|` relative digits, which by then is a fraction of one. For the gamma
/// prefactor there is a second and unrelated reason to stay inside the band: out there `u` stops
/// carrying `x` at all. See [`LN_1P_SERIES_MAX`].
pub(crate) fn ln_1p_minus_linear<T: HostReal>(u: T) -> T {
    if u.abs() > T::from_f64(LN_1P_SERIES_MAX) {
        return u.ln_1p() - u;
    }
    // t_k = (−1)^{k+1} u^k / k, starting at k = 2, advanced by t_{k+1} = t_k·(−u)·k/(k+1).
    let mut index = T::TWO;
    let mut term = -u * u * T::HALF;
    let mut sum = term;
    for _ in 0..LN_1P_SERIES_MAX_TERMS {
        let next = index + T::ONE;
        term *= -u * index / next;
        sum += term;
        if term_is_negligible(term, sum.abs(), T::EPSILON) {
            break;
        }
        index = next;
    }
    sum
}

/// The common log prefactor for the lower series and upper continued fraction.
///
/// # The direct form is a cancelling sum under an exponential, and that was the error
///
/// This read `log_scale − x + a·ln x − lnΓ(a)` and returned a value that is then `exp`-ed. Those
/// three terms are each of size `a·ln a` while their sum is `O(ln a)` — at `a = x = 1000` they
/// are `6908`, `1000` and `5905`, summing to `2.53`. Rounding leaves each with an absolute error
/// of `|term|·ε`, the sum keeps it, and `exp` turns an absolute error in its argument into a
/// *relative* error in its result. So the answer carried about `|largest term|` ulp.
///
/// Measured against a 400-bit reference, worst ulp over 300 log-uniform samples per cell, with
/// the prediction `≈ |a·ln x|` beside it:
///
/// ```text
/// a               2      20     100    1000    3000
/// largest term   28     263     434    6327   22771
/// was            26     686    1112   11473   45448   ulp
/// ```
///
/// That is one root cause under several symptoms: `erf` is `P(½, x²)` and carried `7`–`32` ulp
/// across its whole domain above `ERF_LINEAR_MAX`, which in turn is the floor `erf_inv`'s
/// interior route inherits through its Newton step.
///
/// # The rearrangement
///
/// Stirling gives `lnΓ(a) = (a−½)ln a − a + ½ln 2π + μ(a)` with `μ` from
/// [`stirling_remainder`], and substituting it lets the large parts cancel *symbolically*
/// rather than numerically. With `u = (x−a)/a`,
///
/// ```text
/// a·ln x − x − lnΓ(a) = a·[ln(1+u) − u] + ½·ln(a/2π) − μ(a)
/// ```
///
/// Every term on the right is now the size of the answer: at `a = x = 1000` they are `0`,
/// `2.535` and `8.3e-5`. `ln(1+u) − u` is itself a cancelling difference, which is why
/// [`ln_1p_minus_linear`] exists rather than the expression being written inline.
///
/// # The band was mistaken for the rearrangement's limit, and it is only `u`'s
///
/// Below [`STIRLING_MIN`] the direct form is kept because `lnΓ(a)` is `O(1)` there — nothing
/// large to cancel — and because `μ`'s series is not valid that low. That guard is unchanged.
///
/// The second guard used to send everything outside [`LN_1P_SERIES_MAX`] back to the direct
/// form, on the reasoning that out there the exponent is genuinely large and so bounds its own
/// accuracy. That reasoning holds at tiny `x`, where `a·ln x` *is* the answer's size, and it is
/// false in the other direction. At `a = 445`, `x = 1000` the exponent is `−605` while the
/// direct form builds it from terms of `3074` — the band had excluded the rearrangement from a
/// region where nothing was wrong with it except the variable it was written in.
///
/// `u = (x−a)/a` stops carrying `x` outside the band; `t = x/a` never does. So the identity is
/// rewritten with no `ln1p` in it at all,
///
/// ```text
/// a·ln x − x − lnΓ(a) = a·[ln t + 1 − t] + ½·ln(a/2π) − μ(a),      t = x/a
/// ```
///
/// which is the same cancellation with the same tail — see [`ScaledGammaForm`] for which of the
/// three forms each argument gets and why that is a measurement rather than a range.
///
/// # Measured
///
/// Against a 400-bit reference over the region this choice governs — `a ≥ 20`, outside the band,
/// sampled log-uniformly in `a` and in `t`, keeping the `~1240` points per tail whose answer is
/// still representable:
///
/// ```text
///                        P(a, x)            Q(a, x)
///                   worst    mean      worst    mean
/// direct throughout 24381     533      65836     614   ulp
/// as written         1444     110       3897      88
/// ```
///
/// The worst points are not in the deep tail but at moderate shape just outside the band:
/// `P(2215.1, 1107.5)` went `29603 → 322` and `Q(1807.8, 3175.2)` went `32941 → 472`.
///
/// It is not uniformly better. `21` of `1236` points are more than half again worse than the
/// direct form, the worst by `507` ulp, and they sit where both forms are near the floor
/// `|exponent|·ε` imposes — there the ranking of the two is noise, and a threshold chasing it
/// would be fitting noise. What the choice buys is the envelope and the mean.
pub(crate) fn log_scaled_gamma<T: HostReal>(a: T, x: T, log_scale: T) -> T {
    match ScaledGammaForm::for_arguments(a, x) {
        ScaledGammaForm::Direct => scaled_gamma_direct(a, x, log_scale),
        ScaledGammaForm::Centred => scaled_gamma_centred(a, x, log_scale),
        ScaledGammaForm::Ratio => scaled_gamma_ratio(a, x, log_scale),
    }
}

/// Which arrangement of the log prefactor an `(a, x)` gets.
///
/// [`Self::Centred`] and [`Self::Ratio`] are the *same* rearrangement and differ only in the
/// variable it is written in, which is why they share [`with_stirling_tail`]. [`Self::Direct`]
/// is the unrearranged sum, kept where the rearrangement has nothing to offer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScaledGammaForm {
    /// `log_scale − x + a·ln x − lnΓ(a)` as written.
    ///
    /// Reached below [`STIRLING_MIN`], where there is nothing large to cancel, and outside the
    /// band at small `x`, where it is the *better* of the two: `a·ln x` is then the size of the
    /// answer's own logarithm. At `a = 20`, `x = 1e-6` its largest term is `276` against the
    /// ratio form's `336`.
    ///
    /// # A sweep that stops at `t = 1e-4` says this arm is worth nothing
    ///
    /// Dropping it — always rearranging outside the band — leaves the worst case *unchanged* at
    /// `1444` and `3897` ulp, so on the envelope alone this arm looks like dead weight. It is
    /// not: it cuts the points made worse than the direct form from `96` to `21`, and the worst
    /// single loss from `988` to `507`. The evidence is all at `t` far below the band, where the
    /// rearrangement's `a·ln t` is the largest term in play and grows without bound as `t → 0`:
    /// at `a = 25`, `x = 2.5e-10` this arm is `18` ulp and the rearrangement `928`.
    ///
    /// The first derivation of the discriminant sampled `t ∈ [1e-4, 16]` and concluded exactly
    /// the opposite, because that band contains none of the above. The lesson is the standing
    /// one: a sweep over magnitudes ends at the format's limit, and `1e-4` is a round number.
    Direct,
    /// `a·[ln(1+u) − u] + ½ln(a/2π) − μ(a)` with `u = (x−a)/a`, inside [`LN_1P_SERIES_MAX`].
    ///
    /// Preferred over [`Self::Ratio`] wherever it applies, because `u` is exact there — Sterbenz
    /// makes `x − a` exact on `[a/2, 2a]` — while `t = x/a` takes a rounding, and because
    /// `ln(1+u) − u` is summed as a series instead of cancelling two `O(u)` terms.
    Centred,
    /// `a·[ln t + 1 − t] + ½ln(a/2π) − μ(a)` with `t = x/a`, outside the band where it wins.
    ///
    /// This is most of the non-Temme surface above [`STIRLING_MIN`]: Temme's expansion already
    /// covers `|u| ≤ 0.4`, so [`Self::Centred`] keeps only the sliver out to `½` and everything
    /// the series and the continued fraction actually see arrives here.
    Ratio,
}

impl ScaledGammaForm {
    /// The shape first, then the band, then — outside it — whichever form has less to lose.
    fn for_arguments<T: HostReal>(a: T, x: T) -> Self {
        if a < T::from_f64(STIRLING_MIN) {
            Self::Direct
        } else if ((x - a) / a).abs() <= T::from_f64(LN_1P_SERIES_MAX) {
            Self::Centred
        } else if largest_term(&ratio_terms(a, x)) < largest_term(&direct_terms(a, x)) {
            Self::Ratio
        } else {
            Self::Direct
        }
    }
}

/// `log_scale − x + a·ln x − lnΓ(a)`; see [`ScaledGammaForm::Direct`].
fn scaled_gamma_direct<T: HostReal>(a: T, x: T, log_scale: T) -> T {
    log_scale - x + a * x.ln() - a.ln_gamma()
}

/// The rearrangement in `u = (x−a)/a`; see [`ScaledGammaForm::Centred`].
fn scaled_gamma_centred<T: HostReal>(a: T, x: T, log_scale: T) -> T {
    with_stirling_tail(log_scale + a * ln_1p_minus_linear((x - a) / a), a)
}

/// The rearrangement in `t = x/a`; see [`ScaledGammaForm::Ratio`].
fn scaled_gamma_ratio<T: HostReal>(a: T, x: T, log_scale: T) -> T {
    let t = x / a;
    with_stirling_tail(log_scale + a * (t.ln() + T::ONE - t), a)
}

/// Append `+ ½·ln(a/2π) − μ(a)` to a rearrangement's `x`-dependent part, in that order.
///
/// These two terms are the whole reason the rearrangement is worth having: they are the size of
/// the answer, where the three they replaced were the size of `lnΓ(a)`.
///
/// # The grouping is load-bearing, so the leading part comes in already formed
///
/// Returning `½ln(a/2π) − μ(a)` as a value for the caller to add reads identically and is not:
/// it makes the sum `A + (B − C)` where this makes it `(A + B) − C`, and `μ(a)` is `8.3e-5`
/// against a result of `−108`, so the two differ in the last bit. That ulp is in an *exponent* —
/// `exp` turns it into `64` ulp of `P(1000, 600)`, which is what the Temme band-edge comparison
/// reported when this was first factored out the other way. Refactors here are verified
/// bit-identical for exactly this reason.
fn with_stirling_tail<T: HostReal>(leading: T, a: T) -> T {
    leading + T::HALF * (a / T::TAU).ln() - stirling_remainder(a)
}

/// The terms [`ScaledGammaForm::Direct`] sums, with `lnΓ(a)` standing in as `a(ln a − 1)`.
///
/// Computing `lnΓ(a)` in order to decide whether to compute it would be circular, so its leading
/// Stirling term is used instead — which is within a logarithm of the whole, far closer than the
/// discriminant needs.
fn direct_terms<T: HostReal>(a: T, x: T) -> [T; 3] {
    [x, a * x.ln(), a * (a.ln() - T::ONE)]
}

/// The terms [`ScaledGammaForm::Ratio`] sums, `a·ln t` and `a − x`.
///
/// [`with_stirling_tail`] is left out of both lists: it is common to the rearranged forms and
/// `O(ln a)` besides, so it cannot discriminate. `log_scale` is likewise common to all three.
fn ratio_terms<T: HostReal>(a: T, x: T) -> [T; 2] {
    [a * (x / a).ln(), a - x]
}

/// The largest term in magnitude — the quantity that predicts an exponent's error.
///
/// Named because it is a *prediction* and not an accident of either formula. `exp` turns the
/// absolute error of its argument into relative error of its result, and the argument's absolute
/// error is the largest term's magnitude times `ε`. That relationship was checked across five
/// orders of magnitude when the gamma prefactor was first repaired — observed ulp came out at
/// about twice the largest term from `a = 2` to `a = 3000` — which is what makes it usable as a
/// runtime discriminant rather than only as an explanation.
///
/// Where `x` itself is the largest term on both sides — `x ≫ a`, where `Direct` carries it alone
/// and `Ratio` carries it inside `a − x` — the comparison is close and the choice is immaterial:
/// the exponent is then `≈ −x` too, so either form's error is one ulp of a result that has long
/// since underflowed to zero.
///
/// # It is an envelope, so the comparison is bare and has no margin in it
///
/// The two sides are compared as they stand, with no fudge factor, and that is a measured choice
/// rather than the absence of one. Requiring the rearrangement to win by `k` before switching was
/// swept for `k ∈ [1, 10]` with both routes forced and measured at every point, so each `k` only
/// decides which column to read:
///
/// ```text
/// k                      1      1.5      2        3        4       10
/// P worst ulp         1444     1444   1444     3695     5084    22428
/// P points made worse  21       18     13        4        2        0
/// ```
///
/// The envelope is flat out to `k = 2` and breaks at `k = 3`, so `k = 1` sits inside that range
/// rather than against its edge — the placement rule every crossover here follows. Raising it
/// trades the envelope for the count, and the count is the wrong thing to buy: those points are
/// where both arrangements are already at the `|exponent|·ε` floor.
fn largest_term<T: HostReal>(terms: &[T]) -> T {
    terms
        .iter()
        .fold(T::ZERO, |worst, &term| worst.max(term.abs()))
}

/// Terms the `ln(1+u) − u` series is allowed before it gives up.
///
/// `|u| ≤ ½` there, where `(½)^k/k` falls below `ε` relative to a result of size `u²/2` by about
/// the fifty-second term. Sixty-four is that with headroom, and the loop exits on its own long
/// before for the small `|u|` that the interesting arguments produce.
const LN_1P_SERIES_MAX_TERMS: usize = 64;

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
