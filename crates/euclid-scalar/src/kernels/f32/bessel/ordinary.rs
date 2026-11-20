//! The `f32` peer of the ordinary Bessel functions: `J₀`, `J₁`, `Y₀`, `Y₁`.
//!
//! Native single precision throughout. The module header once grouped all four as needing `f64`
//! because "a cancelling series, an outward march, and a divergent asymptotic expansion" all
//! have error floors set by the working precision. Taken mechanism by mechanism:
//!
//! - **`J`** reaches its values through a series, an *aliased trapezoidal quadrature* and the
//!   Hankel expansion. The quadrature's node count and the expansion's handover are both fixed
//!   by equating an error bound to a target accuracy, so both retune to `f32` arithmetically —
//!   the same situation as `K`, and the same roughly threefold saving. That is this file's
//!   native half.
//! - **`Y`** has no comparable quadrature, and closes its middle gap by *marching Bessel's
//!   equation outward* from the series. A march is the one mechanism here whose floor is not a
//!   tunable: it accumulates a rounding of order `ε` per step, so the narrower format pays for
//!   every step and no retuning recovers it.
//!
//!   Measured, it costs less than that argument suggests, for a reason that only shows up once
//!   `J`'s handover has moved: the march runs from [`Y_SERIES_MAX`] to [`HANKEL_MIN`], and that
//!   interval is `[2, 7]` here against `f64`'s `[4, 18]` — about six steps either way. Forward
//!   integration is stable here because `J` and `Y` both decay like `x^{−1/2}`, so neither
//!   dominates and an error introduced at one step is not amplified at the next. The finished
//!   `Y` holds to `2.9e-7` of its envelope across `x ∈ (0, 300]`, under three ulp.
//!
//!   The concern was real and the conclusion drawn from it was not. A per-step `ε` matters when
//!   the step count is large; bounding the count is what makes the mechanism affordable, and
//!   that is a property of where the neighbouring routes hand over.

use crate::constants::f32::EULER_GAMMA;
use core::f32::consts::{PI, TAU};

/// At or above this argument `J` uses the Hankel expansion; below it, the series or the aliased
/// trapezoid.
///
/// **`7.0`, where the `f64` peer uses `18.0`.** The expansion is divergent and truncated at its
/// least term, so its floor is the size of that term — a function of `x` alone, not of `ε`. It
/// therefore becomes usable at a *smaller* argument in a coarser format, and the handover moves
/// down to meet it.
///
/// Measured as a fraction of the envelope `√(2/πx)`, which is the right scale here because `J`
/// has infinitely many zeros and an ulp count collapses at every one of them:
///
/// | `x` | 3 | 4 | 5 | 6 | **7** | 8 | 10 | 12 | 18 |
/// |---|---|---|---|---|---|---|---|---|---|
/// | Hankel | `5.1e-4` | `4.1e-5` | `2.0e-6` | `1.0e-6` | **`1.5e-7`** | `1.9e-7` | `1.4e-8` | `3.1e-7` | `3.2e-7` |
/// | aliased | `1.7e-7` | `3.2e-7` | `3.6e-7` | `4.9e-8` | **`2.5e-7`** | `5.5e-7` | `3.1e-7` | `9.0e-7` | `5.4e-8` |
///
/// They cross just under `7`. Confirmed end to end: the worst error of the finished `J` over
/// `x ∈ (0, 300]` is `9.1e-7` of the envelope with the handover at `6`, `5.4e-7` at `7`, and
/// `7.3e-7` at `8`.
///
/// Moving it down from `18` also saves the quadrature's most expensive band, since its node
/// count grows like `2x` — though the saving is smaller than it looks, because [`ALIAS_EPS`]
/// already cut the count: 16–50 nodes across `x ∈ [3, 25]` against the `f64` constants' 26–68,
/// a ratio of about `1.4` rather than the `3` the `K` quadrature gave.
const HANKEL_MIN: f32 = 7.0;

/// Relative size below which an aliased term is taken to vanish.
///
/// `1e-8` against the `f64` peer's `1e-19`. This sets the node count directly — the walk stops
/// at the smallest `m` with `(x/2)^m/m! < ALIAS_EPS` — so it is the main lever on the
/// quadrature's cost, and asking for eleven more orders than the format holds simply pays for
/// nodes whose contribution is never representable.
const ALIAS_EPS: f32 = 1e-8;

/// Cap on the aliasing node count, which grows like `2x`; only reached above [`HANKEL_MIN`],
/// where this route is not used.
const ALIAS_MAX_NODES: u32 = 64;

/// Iteration cap for the power series. The `f64` peer allows 1,000.
const MAX_TERMS: u32 = 32;

/// Relative size at which a series term stops contributing.
const TERM_EPS: f32 = 1e-8;

/// Term cap for the Hankel expansion; the minimum-term check normally stops it far sooner.
const HANKEL_TERMS: u32 = 32;

/// At or below this argument `J₀`/`J₁` use the power series rather than the quadrature.
///
/// **Unchanged from the `f64` peer's `2.0`**, because the reason for it is not an accuracy
/// trade. The quadrature's summand is `cos(x·sin θ − nθ)`, and for small `x` the `x·sin θ` part
/// is swallowed by `nθ`: at a subnormal `x`, `x − π/2` rounds to exactly `−π/2`, so the sum
/// returns the rounding of `cos(−π/2)` instead of anything resembling `J₁(x) ≈ x/2`. The
/// absolute error stays tiny while every digit of the relative answer is gone — and `J₁(x) ~ x/2`
/// is an exactly-conditioned limit that ought to come back exactly. That failure is about the
/// *representation* of the argument, so it happens in both formats and at the same place.
const J_SERIES_MAX: f32 = 2.0;

/// Node count making the aliasing error of [`j_aliased`] negligible.
///
/// The error is `J_{N−n}(x) + J_{N+n}(x) + ⋯`, dominated by the first term, so this finds the
/// smallest `m` with `(x/2)^m/m! < ALIAS_EPS` and returns `N = m + n` rounded up to even (the
/// sum exploits a symmetry that needs an even node count).
///
/// The `m <= half` clause keeps the walk going until it is past the peak at `m ≈ x/2`: the terms
/// *grow* before they decay, so stopping at the first small value would otherwise return `m = 1`
/// for every argument above a few.
fn alias_nodes(n: u32, x: f32) -> u32 {
    let half = x * 0.5;
    let mut m = 1.0f32;
    let mut term = half; // (x/2)^m / m!
    while m <= half || term > ALIAS_EPS {
        m += 1.0;
        term *= half / m;
        if m >= ALIAS_MAX_NODES as f32 {
            break;
        }
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let needed = m as u32 + n + 2;
    needed + needed % 2
}

/// `Jₙ(x)` for small `x ≥ 0` and `n ∈ {0, 1}`, by the defining power series
/// `Σ (−1)^k (x/2)^{2k+n} / (k!(k+n)!)`.
fn j_series(n: u32, x: f32) -> f32 {
    let h = x * 0.5;
    let order = n as f32;
    // The k = 0 term is (x/2)^n / n!, and n is 0 or 1, so n! is 1 either way.
    let mut term = if n == 0 { 1.0 } else { h };
    let mut total = term;
    for k in 1..MAX_TERMS {
        let kf = k as f32;
        term *= -h * h / (kf * (kf + order));
        total += term;
        if term.abs() < total.abs() * TERM_EPS {
            break;
        }
    }
    total
}

/// `Jₙ(x)` for `x ≥ 0` and `n ∈ {0, 1}`, by the aliased trapezoidal rule.
///
/// The summand is symmetric under `θ → 2π − θ`, so only half the period is walked.
fn j_aliased(n: u32, x: f32) -> f32 {
    debug_assert!(x >= 0.0);
    let nodes = alias_nodes(n, x);
    let count = nodes as f32;
    let step = TAU / count;
    let order = n as f32;
    // The j = 0 and j = nodes/2 nodes are their own mirror images; everything between is
    // counted twice.
    let mut total = 1.0; // theta = 0, where the summand is cos(0) = 1
    total += (x * PI.sin() - order * PI).cos();
    let mut paired = 0.0f32;
    for j in 1..nodes / 2 {
        let theta = step * j as f32;
        paired += (x * theta.sin() - order * theta).cos();
    }
    (total + 2.0 * paired) / count
}

/// `(Jν, Yν)` from the Hankel expansions, for large `x`.
///
/// ```text
/// Jν ~ √(2/πx)[cos ω·P − sin ω·Q],   Yν ~ √(2/πx)[sin ω·P + cos ω·Q]
/// ```
///
/// with `ω = x − (ν/2 + ¼)π`, `a₀ = 1`, `a_k = a_{k−1}(4ν² − (2k−1)²)/(8k)`, `P` the
/// even-indexed terms of the alternating sum and `Q` the odd-indexed ones. The coefficients are
/// generated in the loop, so nothing is transcribed.
///
/// Both series are divergent and truncated at their minimum term; continuing past it makes them
/// worse, which the growth check watches for. Returning the pair costs nothing — `P`, `Q` and
/// `ω` are shared — and keeps `J` and `Y` consistent in the tail.
pub(in crate::kernels::f32) fn hankel(nu: f32, x: f32) -> (f32, f32) {
    debug_assert!(x > 0.0);
    let four_nu_sq = 4.0 * nu * nu;
    let mut a = 1.0f32;
    let mut inv_power = 1.0f32; // x^{-m}
    let (mut p, mut q) = (0.0f32, 0.0f32);
    let mut previous = f32::INFINITY;
    for m in 0..HANKEL_TERMS {
        if m > 0 {
            let mf = m as f32;
            let odd = 2.0 * mf - 1.0;
            a *= (four_nu_sq - odd * odd) / (8.0 * mf);
            inv_power /= x;
        }
        let magnitude = (a * inv_power).abs();
        if magnitude >= previous {
            break; // past the minimum term of a divergent expansion
        }
        previous = magnitude;
        // Sign is (-1)^{m/2} for the even (P) terms and (-1)^{(m-1)/2} for the odd (Q) ones.
        let signed = if (m / 2) % 2 == 0 {
            a * inv_power
        } else {
            -a * inv_power
        };
        if m % 2 == 0 {
            p += signed;
        } else {
            q += signed;
        }
        if magnitude < TERM_EPS {
            break;
        }
    }
    // The phase is `ω = x − (ν/2 + ¼)π`, and **it must not be formed as a single `f32`.**
    //
    // `x` is exact, but `x − c` rounds to `ulp(x)`, which *is* the angle's absolute error: at
    // `x = 227` that is `7.6e-6` radians, and an oscillating function returns it as `7.6e-6` of
    // its envelope. Measured that way, forming `ω` directly cost `6.7e-6` of the envelope at
    // `x = 227` — fifty-six times the format's resolution, and growing linearly with `x`.
    //
    // The angle-subtraction identities avoid it entirely. `sin x` and `cos x` are evaluated at
    // the exact argument, where the library's own reduction keeps them to an ulp however large
    // `x` is, and the shift is applied through exact constants afterwards:
    //
    // ```text
    // sin ω = sin x·cos c − cos x·sin c,    cos ω = cos x·cos c + sin x·sin c
    // ```
    //
    // For `ν ∈ {0, 1}`, `c` is `π/4` or `3π/4`, so `|cos c| = |sin c| = √2/2` and the whole
    // shift is one multiplication by a constant that is correctly rounded. The `f64` peer forms
    // `ω` directly and can afford to — `ulp(x)` is `3e-14` there at the same argument.
    let (sin_x, cos_x) = x.sin_cos();
    // `c = (ν/2 + ¼)π` is `O(1)` whatever `ν` is, so its sine and cosine are accurate to an ulp
    // computed directly — the precision problem is `x`, never the shift. Evaluating it for
    // general `ν` rather than special-casing `|cos c| = |sin c| = √2/2` at `ν ∈ {0, 1}` costs one
    // `sin_cos` and lets the Airy family reach this through `ν = ±⅓, ±⅔`.
    let (sin_c, cos_c) = ((nu * 0.5 + 0.25) * PI).sin_cos();
    let sin_omega = sin_x * cos_c - cos_x * sin_c;
    let cos_omega = cos_x * cos_c + sin_x * sin_c;
    let amplitude = (2.0 / (PI * x)).sqrt();
    (
        amplitude * (cos_omega * p - sin_omega * q),
        amplitude * (sin_omega * p + cos_omega * q),
    )
}

/// Which route evaluates `J₀`/`J₁` at a given magnitude.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JRoute {
    /// `|x| ≥ `[`HANKEL_MIN`]: the divergent Hankel expansion, cheapest and most accurate here.
    Hankel,
    /// `|x| ≤ `[`J_SERIES_MAX`]: the power series — required rather than preferred, see that
    /// constant.
    Series,
    /// The middle band, by aliased trapezoidal quadrature.
    Aliased,
}

impl JRoute {
    /// Takes the already-absolute magnitude: `J₀` is even and `J₁` odd, so both reduce to it.
    fn for_magnitude(ax: f32) -> Self {
        if ax >= HANKEL_MIN {
            Self::Hankel
        } else if ax <= J_SERIES_MAX {
            Self::Series
        } else {
            Self::Aliased
        }
    }
}

/// `Jₙ(|x|)` for `n ∈ {0, 1}`, routed by magnitude.
fn j_magnitude(n: u32, ax: f32) -> f32 {
    match JRoute::for_magnitude(ax) {
        JRoute::Hankel => hankel(n as f32, ax).0,
        JRoute::Series => j_series(n, ax),
        JRoute::Aliased => j_aliased(n, ax),
    }
}

/// The Bessel function of the first kind `J₀(x)`, even in `x`.
pub(crate) fn j0(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    let ax = x.abs();
    if ax == 0.0 {
        return 1.0;
    }
    if ax.is_infinite() {
        return 0.0; // J0 decays like x^{-1/2}
    }
    j_magnitude(0, ax)
}

/// The Bessel function of the first kind `J₁(x)`, odd in `x`.
pub(crate) fn j1(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    let ax = x.abs();
    if ax == 0.0 {
        // Exactly, and preserving the sign of zero: the quadrature's mirror-image nodes cancel
        // only to rounding, leaving a residue where the true value is 0.
        return x;
    }
    if ax.is_infinite() {
        return 0.0;
    }
    let magnitude = j_magnitude(1, ax);
    if x < 0.0 { -magnitude } else { magnitude }
}

// ===== second kind ========================================================

/// At or below this argument `Y₀`/`Y₁` use their power series; above it, the march.
///
/// **`2.0`, where the `f64` peer uses `4.0`.** The series' error is `≈ ε·ln(x)·I₀(x)`, and `I₀`
/// grows exponentially, so this threshold scales with the format rather than transferring. At
/// `x = 4` that bound is `1e-15` in `f64` — the value the peer cites — and `1.9e-6` here, which
/// against `Y₀`'s envelope of `0.4` is forty times the format's resolution. Worse, `Y₀` has a
/// zero at `3.958`, so `x = 4` is exactly where an absolute error of `1.9e-6` is least
/// affordable: the relative error there is `1e-4`.
///
/// At `2.0` the bound is `1.9e-7`, a few ulp of the envelope, and the cost is a longer march —
/// cheap here for the reason given on [`MARCH_TERMS`].
///
/// Confirmed end to end, as the worst error of the finished `Y` over `x ∈ (0, 300]` measured
/// against its envelope: `2.9e-7` at each of `1.5`, `2.0` and `3.0` — all three limited by the
/// Hankel region rather than by this threshold — and `5.4e-7` at `f64`'s `4.0`, where the worst
/// case moves to `x = 5.75` and is the march running from a less accurate seed.
const Y_SERIES_MAX: f32 = 2.0;

/// Fraction of the current abscissa used as a Taylor step. The radius of convergence is `x₀`,
/// so this makes the coefficients decay like `4^{−k}`.
const MARCH_RATIO: f32 = 0.25;

/// Taylor coefficients per march step. The `f64` peer uses 40.
///
/// With a `4^{−k}` decay, `f64`'s 40 terms reach `4^{−40} ≈ 8e-25`. This format needs only
/// `4^{−k} < 6e-8`, so `k > 11.9`; **16** gives `2.3e-10` with margin. Each step builds a
/// coefficient array of this length, so it is the march's whole cost — two and a half times
/// cheaper per step.
///
/// The march is also much shorter here than in `f64`, which is what makes a lower
/// [`Y_SERIES_MAX`] affordable. It runs from `Y_SERIES_MAX` to [`HANKEL_MIN`], and that interval
/// is `[2, 7]` rather than `f64`'s `[4, 18]` — about six steps at this ratio against seven. So
/// the one mechanism on this tree's exception list whose error is *not* a tunable, a march
/// accumulating `ε` per step, is bounded by roughly the same step count as `f64`'s while each
/// step costs less.
const MARCH_TERMS: usize = 16;

/// Iteration cap for the `Y` power series.
const Y_MAX_TERMS: u32 = 32;

/// `Y₀(x)` by its power series, for small `x`.
///
/// `Y₀ = (2/π)(ln(x/2) + γ)J₀(x) + (2/π)Σ_{k≥1}(−1)^{k+1}H_k(x²/4)^k/(k!)²`.
fn y0_series(x: f32) -> f32 {
    y0_series_terms(x).0
}

/// `Y₀` from its power series, with the number of terms it summed.
///
/// # Why the count is returned at all
///
/// The break is a convergence test: the terms it stops after are below `TERM_EPS` of the running
/// sum, so narrowing `<` to `==` — which never fires and runs every call to [`Y_MAX_TERMS`] —
/// leaves the value identical and costs the remaining iterations. `<=` differs only when a term
/// lands exactly on the tolerance, and is an equivalent mutant. The count separates the first
/// from the second.
fn y0_series_terms(x: f32) -> (f32, u32) {
    let q = x * x * 0.25;
    let mut term = 1.0f32; // (x²/4)^k/(k!)²
    let mut harmonic = 0.0f32;
    let mut sum = 0.0f32;
    let mut terms = 0;
    for k in 1..Y_MAX_TERMS {
        let kf = k as f32;
        term *= q / (kf * kf);
        harmonic += 1.0 / kf;
        let t = term * harmonic;
        sum += if k % 2 == 1 { t } else { -t };
        terms += 1;
        if t < sum.abs() * TERM_EPS {
            break;
        }
    }
    (
        2.0 / PI * (ln_half_x(x) + EULER_GAMMA) * j0(x) + 2.0 / PI * sum,
        terms,
    )
}

/// `Y₁(x)` by its power series, for small `x`.
///
/// The `−2/(πx)` carries the pole, so it is kept as a separate exact term rather than folded in.
fn y1_series(x: f32) -> f32 {
    y1_series_terms(x).0
}

/// `Y₁` from its power series, with the number of terms it summed.
///
/// Instrumented for the reason [`y0_series_terms`] records; the `k > 1` guard on its break adds
/// three more mutants of the same kind, each of which forces the loop to its cap.
fn y1_series_terms(x: f32) -> (f32, u32) {
    let q = x * x * 0.25;
    let mut term = 1.0f32; // (x²/4)^k/(k!(k+1)!)
    let mut h_k = 0.0f32;
    let mut sum = 0.0f32;
    let mut terms = 0;
    for k in 0..Y_MAX_TERMS {
        let kf = k as f32;
        if k > 0 {
            term *= q / (kf * (kf + 1.0));
            h_k += 1.0 / kf;
        }
        let h_k1 = h_k + 1.0 / (kf + 1.0);
        let t = term * (h_k + h_k1 - 2.0 * EULER_GAMMA);
        sum += if k % 2 == 0 { t } else { -t };
        terms += 1;
        if t.abs() < sum.abs() * TERM_EPS && k > 1 {
            break;
        }
    }
    (
        -2.0 / (PI * x) + 2.0 / PI * ln_half_x(x) * j1(x) - x / (2.0 * PI) * sum,
        terms,
    )
}

/// `ln(x/2)`, formed as `ln x − ln 2` so the product form cannot underflow at a subnormal.
fn ln_half_x(x: f32) -> f32 {
    x.ln() - core::f32::consts::LN_2
}

/// One Taylor step of Bessel's equation `x²y″ + xy′ + (x² − n²)y = 0`.
///
/// The coefficient recurrence is seeded by `a₀ = y(x₀)` and `a₁ = y′(x₀)` and needs no
/// derivatives of the equation's coefficients. Both the value and the derivative are returned,
/// by Horner on `Σaₖδᵏ` and `Σk·aₖδ^{k−1}`.
fn taylor_step(x0: f32, y: f32, dy: f32, delta: f32, order_sq: f32) -> (f32, f32) {
    let mut a = [0.0f32; MARCH_TERMS];
    a[0] = y;
    a[1] = dy;
    let x0_sq = x0 * x0;
    for m in 0..MARCH_TERMS - 2 {
        let mf = m as f32;
        let prev1 = if m >= 1 { a[m - 1] } else { 0.0 };
        let prev2 = if m >= 2 { a[m - 2] } else { 0.0 };
        let numerator = x0 * (mf + 1.0) * (2.0 * mf + 1.0) * a[m + 1]
            + (mf * mf + x0_sq - order_sq) * a[m]
            + 2.0 * x0 * prev1
            + prev2;
        a[m + 2] = -numerator / (x0_sq * (mf + 2.0) * (mf + 1.0));
    }
    let mut value = 0.0f32;
    for k in (0..MARCH_TERMS).rev() {
        value = value * delta + a[k];
    }
    let mut derivative = 0.0f32;
    for k in (1..MARCH_TERMS).rev() {
        derivative = derivative * delta + (k as f32) * a[k];
    }
    (value, derivative)
}

/// `(Y₀(x), Y₁(x))` for `x > Y_SERIES_MAX`, by marching out from the series.
///
/// `Y₀′ = −Y₁`, so the `n = 0` march carries both. Forward integration is stable here precisely
/// because `J` and `Y` both decay like `x^{−1/2}`: neither dominates, so an error introduced at
/// one step is not amplified at the next.
fn y_march(target: f32) -> (f32, f32) {
    let mut x = Y_SERIES_MAX;
    let mut value = y0_series(x);
    let mut derivative = -y1_series(x);
    while x < target {
        let step = (x * MARCH_RATIO).min(target - x);
        (value, derivative) = taylor_step(x, value, derivative, step, 0.0);
        x += step;
    }
    (value, -derivative)
}

/// Which route evaluates `Y₀`/`Y₁`.
#[derive(Clone, Copy, Debug, PartialEq)]
enum YRoute {
    /// A value the function takes exactly: the singularity at zero and the limit at infinity.
    Value(f32),
    /// `x ≤ `[`Y_SERIES_MAX`]: the power series.
    Series,
    /// `x ≥ `[`HANKEL_MIN`]: the Hankel expansion, shared with `J`.
    Hankel,
    /// The middle gap, by marching Bessel's equation outward from the series.
    March,
}

impl YRoute {
    fn for_argument(x: f32) -> Self {
        if x.is_nan() || x < 0.0 {
            Self::Value(f32::NAN) // the real Y is defined for positive argument
        } else if x == 0.0 {
            Self::Value(f32::NEG_INFINITY) // logarithmic singularity for Y0, a pole for Y1
        } else if x.is_infinite() {
            Self::Value(0.0)
        } else if x <= Y_SERIES_MAX {
            Self::Series
        } else if x >= HANKEL_MIN {
            Self::Hankel
        } else {
            Self::March
        }
    }
}

/// The Bessel function of the second kind `Y₀(x)`, for `x > 0`.
pub(crate) fn y0(x: f32) -> f32 {
    match YRoute::for_argument(x) {
        YRoute::Value(value) => value,
        YRoute::Series => y0_series(x),
        YRoute::Hankel => hankel(0.0, x).1,
        YRoute::March => y_march(x).0,
    }
}

/// The Bessel function of the second kind `Y₁(x)`, for `x > 0`.
pub(crate) fn y1(x: f32) -> f32 {
    match YRoute::for_argument(x) {
        YRoute::Value(value) => value,
        YRoute::Series => y1_series(x),
        YRoute::Hankel => hankel(1.0, x).1,
        YRoute::March => y_march(x).1,
    }
}

#[cfg(test)]
mod tests;
