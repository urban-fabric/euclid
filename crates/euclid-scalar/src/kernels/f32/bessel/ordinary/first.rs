//! `J₀` and `J₁`, the Bessel functions of the first kind.
//!
//! A child module so the parent stays inside the crate's 400-line budget. The two kinds share
//! the alias quadrature's constants in [`super`] and little else: `J` is entire and bounded,
//! reached by series, aliased quadrature or the Hankel expansion, while `Y` has a logarithmic
//! singularity at the origin and is marched outward from a series. See [`JRoute`].

use super::*;

/// Node count making the aliasing error of [`j_aliased`] negligible.
///
/// The error is `J_{N−n}(x) + J_{N+n}(x) + ⋯`, dominated by the first term, so this finds the
/// smallest `m` with `(x/2)^m/m! < ALIAS_EPS` and returns `N = m + n` rounded up to even (the
/// sum exploits a symmetry that needs an even node count).
///
/// The `m <= half` clause keeps the walk going until it is past the peak at `m ≈ x/2`: the terms
/// *grow* before they decay, so stopping at the first small value would otherwise return `m = 1`
/// for every argument above a few.
use crate::math::pair::Solutions;

pub(super) fn alias_nodes(n: u32, x: f32) -> u32 {
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
pub(super) fn j_series(n: u32, x: f32) -> f32 {
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
pub(super) fn j_aliased(n: u32, x: f32) -> f32 {
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
pub(in crate::kernels::f32) fn hankel(nu: f32, x: f32) -> Solutions<f32> {
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
    Solutions::new(
        amplitude * (cos_omega * p - sin_omega * q),
        amplitude * (sin_omega * p + cos_omega * q),
    )
}

/// Which route evaluates `J₀`/`J₁` at a given magnitude.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum JRoute {
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
    pub(super) fn for_magnitude(ax: f32) -> Self {
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
pub(super) fn j_magnitude(n: u32, ax: f32) -> f32 {
    match JRoute::for_magnitude(ax) {
        JRoute::Hankel => hankel(n as f32, ax).first_kind,
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
