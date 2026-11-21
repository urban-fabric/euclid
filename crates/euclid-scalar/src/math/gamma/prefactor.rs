//! `ln(1+u) − u`, and the log prefactor `a·ln x − x − lnΓ(a)` that the incomplete gamma
//! exponentiates.
//!
//! A child module so the parent stays inside the crate's 400-line budget; the parent keeps the
//! Lanczos approximation, the Stirling series and the reflection. What is here is one question:
//! that prefactor is a sum of `lnΓ`-sized terms with an `O(ln a)` answer, and `exp` turns the
//! surviving absolute error into relative error. The three arrangements below are the answers —
//! argued against each other rather than against anything in the Lanczos or Stirling machinery
//! above — and [`ScaledGammaForm`] is which one applies where.

use super::*;

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
    match LnOnePlusForm::for_offset(u) {
        LnOnePlusForm::Series => ln_1p_minus_linear_series(u),
        LnOnePlusForm::Difference => u.ln_1p() - u,
    }
}

/// Which form evaluates `ln(1+u) − u` at this offset.
///
/// # Named because the band's own comparison was untestable
///
/// This was an early return inside [`ln_1p_minus_linear`], and mutation testing left two
/// survivors in its single comparison: `>` to `>=`, which moves the boundary by one representable
/// value, and `>` to `==`, which sends everything *past* the radius to the series. Neither is
/// visible through the function's value — at `|u| = ½` the two forms agree to a few ulp by
/// construction, and the series still converges a little beyond the radius, so routing `0.6`
/// through it is wrong without being obviously wrong.
///
/// Split out, the choice is a value rather than a control-flow edge, and a test names `½` and
/// `0.6` directly. This is the same move [`LanczosLogForm`] was given for the same reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LnOnePlusForm {
    /// `|u| ≤ `[`LN_1P_SERIES_MAX`]: the alternating series in `u`, which never forms `ln(1+u)`
    /// and so never cancels it against `u`.
    Series,
    /// Past the radius: the direct difference, which by then has lost a fraction of one digit and
    /// is the cheaper of the two.
    Difference,
}

impl LnOnePlusForm {
    /// The radius is inclusive — at exactly `½` the series is still the better form, which is
    /// what separates `>` from `>=`.
    fn for_offset<T: HostReal>(u: T) -> Self {
        if u.abs() > T::from_f64(LN_1P_SERIES_MAX) {
            Self::Difference
        } else {
            Self::Series
        }
    }
}

/// `ln(1+u) − u` by its alternating series; see [`LnOnePlusForm::Series`].
///
/// `t_k = (−1)^{k+1} u^k / k` from `k = 2`, advanced by `t_{k+1} = t_k·(−u)·k/(k+1)`. For `u < 0`
/// every term carries the same sign so nothing cancels at all; for `u > 0` it alternates with the
/// first term dominant.
fn ln_1p_minus_linear_series<T: HostReal>(u: T) -> T {
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
        } else if prefers_ratio(
            largest_term(&ratio_terms(a, x)),
            largest_term(&direct_terms(a, x)),
        ) {
            Self::Ratio
        } else {
            Self::Direct
        }
    }
}

/// Whether the rearranged form's largest term is the smaller of the two.
///
/// A function of the two magnitudes rather than of `(a, x)`, because that is what makes the
/// comparison testable. Inside [`ScaledGammaForm::for_arguments`] it was `<` between two
/// `largest_term` calls, and `<` mutated to `<=` differs only when the two are bit-equal — an
/// `(a, x)` nobody can solve for, and one where either arrangement is correct anyway. Given the
/// two scalars directly, the tie is one line of a test.
///
/// Ties go to [`ScaledGammaForm::Direct`], which is the arm that needs no `μ` series and is
/// therefore the one to prefer when nothing distinguishes them.
fn prefers_ratio<T: HostReal>(ratio: T, direct: T) -> bool {
    ratio < direct
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

#[cfg(test)]
mod tests;
