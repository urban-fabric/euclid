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

mod exponential;
pub(super) mod trigonometric;

// Re-exported so the family keeps one path per function: every caller reached
// `kernels::integrals::sin_int` before the split and still does.
pub(crate) use exponential::{exp_int_e1, exp_int_ei};
pub(crate) use trigonometric::{cos_int, sin_int};

#[cfg(test)]
mod tests;
