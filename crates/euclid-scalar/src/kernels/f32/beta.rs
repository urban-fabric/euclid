//! The `f32` beta family, native single precision throughout.
//!
//! `ln_beta` and `beta` are well-conditioned at every magnitude, having been rewritten around
//! the rising factorial to remove the subtraction that used to wreck them. `beta_reg` was the
//! last function in this tree to narrow the `f64` implementation, on the grounds that its error
//! floor is set by the working precision — which is true of its continued fraction and not of
//! the function, because the Edgeworth arm *improves* where the fraction degrades. See
//! [`BETA_EDGEWORTH_MIN`].

#[cfg(test)]
mod tests;

use super::gamma::{ln_gamma, ln_rising_factorial};
use crate::Real;
use crate::math::continued_fraction::{LENTZ_TINY_F32, converged, lentz_step, nonzero_pivot};

/// `ln B(a, b) = ln Γ(a) + ln Γ(b) − ln Γ(a + b)`.
///
/// # Why it is not written that way
///
/// Written literally, that expression subtracts two nearly-equal large logarithms whenever
/// `a + b ≫ min(a, b)` — and the result is `O(ln)` while the terms are `O(a ln a)`. The damage
/// is total, not gradual. Against the exact `ln B(a, 1) = −ln a`:
///
/// ```text
/// a = 1e2   1.7e-14 relative        a = 1e15   4e-2 relative
/// a = 1e8   1.4e-9  relative        a = 1e16   ln B returned 0, and B returned 1
/// ```
///
/// Past `a = 1/ε` the sum `a + b` rounds back to `a`, so `lnΓ(a + b)` and `lnΓ(a)` become
/// *bit-identical* and the whole answer cancels away. `B(1e300, 1)` came back as `1` where the
/// true value is `1e-300`.
///
/// Rewriting it around the rising factorial removes the subtraction entirely:
///
/// ```text
/// ln B(a, b) = ln Γ(b) − ln (a)_b,     (a)_b = Γ(a+b)/Γ(a)
/// ```
///
/// which is the same identity read differently — and
/// [`ln_rising_factorial`] already computes
/// that ratio without ever forming the two logs apart, by an exact product for small integer
/// `b` and an integrated-digamma asymptotic for large `a`. The larger argument is passed as
/// the base so the increment is the short one.
pub(crate) fn ln_beta(a: f32, b: f32) -> f32 {
    let (base, increment) = if a >= b { (a, b) } else { (b, a) };
    match LnBetaForm::for_arguments(base, increment) {
        LnBetaForm::RisingFactorial => ln_gamma(increment) - ln_rising_factorial(base, increment),
        LnBetaForm::CombinedStirling => {
            // Both Stirling expansions combined before any large term is formed. Written with
            // `ln_1p` on the ratio so the `base + increment` that would overflow the format is
            // never materialised, and so the two leading contributions — which share a sign —
            // are not subtracted from one another.
            let ratio = increment / base;
            let log_sum_ratio = ratio.ln_1p();
            increment * (ratio.ln() - log_sum_ratio) - base * log_sum_ratio
                + (0.5 * (log_sum_ratio - increment.ln()) + crate::constants::f32::HALF_LN_TAU)
        }
    }
}

/// Which of the two forms of `ln B` is evaluated.
///
/// Native, where the `CombinedStirling` arm used to narrow the `f64` kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LnBetaForm {
    /// `lnΓ(b) − ln (a)_b`, the rising-factorial identity. Exact for a small integer increment
    /// and an integrated-digamma asymptotic otherwise, and it never forms the two logs apart.
    RisingFactorial,
    /// Both Stirling series combined symbolically, for when *both* shapes are large.
    CombinedStirling,
}

impl LnBetaForm {
    /// The boundary is [`LN_BETA_STIRLING_MIN`], on the smaller shape.
    fn for_arguments(base: f32, increment: f32) -> Self {
        if increment >= LN_BETA_STIRLING_MIN && base.is_finite() {
            Self::CombinedStirling
        } else {
            Self::RisingFactorial
        }
    }
}

/// Smaller shape at or above which both Stirling expansions are combined instead.
///
/// **Rederived at `f32`, not inherited.** The `f64` peer switches at `1e8`, which is about
/// `f64`'s mantissa rather than about the beta function — and `1e8` is not even representable as
/// a distinct `f32` shape in the way the branch needs.
///
/// The metric is absolute error in `ln B`, because `exp(ln B)` turns that into *relative* error
/// in `B`, expressed in ulps of the result so that the largest shapes — where `f32` simply
/// cannot hold `ln B(1e6, 1e6) = −1.386e6` to better than `0.125` — do not dominate. Both shapes
/// equal, which is the rising factorial's worst case:
///
/// | `a = b` | 128 | 256 | 512 | **1024** | 2048 | 1e4 | 1e6 |
/// |---|---|---|---|---|---|---|---|
/// | rising factorial | `0.5` | `0.1` | `1.6` | **`2.3`** | `6.2` | `3.5` | `6.1` |
/// | combined Stirling | `46.1` | `11.6` | `2.8` | **`0.6`** | `0.4` | `0.0` | `0.0` |
///
/// They cross near `600`. `1024` is the power of two that keeps the degrading arm out of each
/// region: the rising factorial stays within `2.3` ulp everywhere below it, and the Stirling form
/// within `0.6` everywhere above.
///
/// Two things the table does not say, and should. The Stirling arm is *catastrophic* at a small
/// increment — `4.3e6` ulp if the gate is removed entirely — because it assumes both shapes are
/// large; the gate exists to prevent that, not to tune the last ulp. And the worst case of the
/// finished function is at neither extreme: it is `89.6` ulp at `a = 1e-2, b = 1e2`, on the
/// rising-factorial arm, and no choice of this threshold moves it.
const LN_BETA_STIRLING_MIN: f32 = 1024.0;

/// The beta function `B(a, b)`.
pub(crate) fn beta(a: f32, b: f32) -> f32 {
    ln_beta(a, b).exp()
}

mod incomplete;

// Re-exported so the family keeps one path per function: every caller reached
// `kernels::f32::beta::beta_reg` before the split and still does.
pub(crate) use incomplete::beta_reg;
