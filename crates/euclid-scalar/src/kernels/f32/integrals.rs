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

mod exponential;
mod trigonometric;

// Re-exported so the family keeps one path per function: every caller reached
// `kernels::integrals::sin_int` before the split and still does.
pub(crate) use exponential::{exp_int_e1, exp_int_ei};
pub(crate) use trigonometric::{cos_int, sin_int};

#[cfg(test)]
mod tests;
