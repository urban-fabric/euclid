//! `Y₀` and `Y₁`, the Bessel functions of the second kind.
//!
//! A child module alongside [`super::first`]. These need their own machinery entirely: a
//! logarithmic singularity at the origin, a series whose term count is returned so the march can
//! be started where it is still accurate, and a Taylor transport outward from there. See
//! [`YRoute`].

use super::*;

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
pub(super) const Y_SERIES_MAX: f32 = 2.0;

/// Fraction of the current abscissa used as a Taylor step. The radius of convergence is `x₀`,
/// so this makes the coefficients decay like `4^{−k}`.
pub(super) const MARCH_RATIO: f32 = 0.25;

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
pub(super) const MARCH_TERMS: usize = 16;

/// Iteration cap for the `Y` power series.
pub(super) const Y_MAX_TERMS: u32 = 32;

/// `Y₀(x)` by its power series, for small `x`.
///
/// `Y₀ = (2/π)(ln(x/2) + γ)J₀(x) + (2/π)Σ_{k≥1}(−1)^{k+1}H_k(x²/4)^k/(k!)²`.
pub(super) fn y0_series(x: f32) -> f32 {
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
pub(super) fn y0_series_terms(x: f32) -> (f32, u32) {
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
pub(super) fn y1_series(x: f32) -> f32 {
    y1_series_terms(x).0
}

/// `Y₁` from its power series, with the number of terms it summed.
///
/// Instrumented for the reason [`y0_series_terms`] records; the `k > 1` guard on its break adds
/// three more mutants of the same kind, each of which forces the loop to its cap.
pub(super) fn y1_series_terms(x: f32) -> (f32, u32) {
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
pub(super) fn ln_half_x(x: f32) -> f32 {
    x.ln() - core::f32::consts::LN_2
}

/// One Taylor step of Bessel's equation `x²y″ + xy′ + (x² − n²)y = 0`.
///
/// The coefficient recurrence is seeded by `a₀ = y(x₀)` and `a₁ = y′(x₀)` and needs no
/// derivatives of the equation's coefficients. Both the value and the derivative are returned,
/// by Horner on `Σaₖδᵏ` and `Σk·aₖδ^{k−1}`.
pub(super) fn taylor_step(x0: f32, y: f32, dy: f32, delta: f32, order_sq: f32) -> (f32, f32) {
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
pub(super) fn y_march(target: f32) -> (f32, f32) {
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
pub(super) enum YRoute {
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
    pub(super) fn for_argument(x: f32) -> Self {
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
