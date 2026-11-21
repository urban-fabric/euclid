//! `erfcx` below `x = 2`, by a Chebyshev series rather than by either neighbour.
//!
//! # One route fixes the accurate-but-slow side and the fast-but-wrong side at once
//!
//! `erfcx` had two arms either side of [`ERFCX_CF_MIN`](super::ERFCX_CF_MIN) and each was
//! unsatisfactory in its own direction. Below it the value came from `e^{x²}·erfc(x)` with
//! `erfc` itself formed as `1 − erf`, whose relative error is `ε/erfc(x)` — a staircase that
//! reaches **23 ulp by `x = 1.36`** against a condition number of `0.74`, so almost all of it is
//! defect rather than conditioning. Above it the continued fraction is accurate and runs **145
//! backward levels**, which the benchmark prices at **313 ns**.
//!
//! `erfcx` is an entire function. A Chebyshev series on a bounded interval therefore converges
//! faster than geometrically, and twenty-four coefficients cover `[0, 2]` to a quarter ulp — so
//! the same route is both the accurate one and the cheap one. There is no trade here to site a
//! threshold against, which is why this replaces both arms below `2` rather than joining them.
//!
//! The fraction keeps everything above `2`, where its own ladder is already short — five to
//! eleven levels — and where a Chebyshev fit would need a new interval for every decade.

use super::erfcx_cf;

/// Which form evaluates `erfcx` at this magnitude.
///
/// Separate from [`super::ErfcForm`], which `erfc` uses, because the two functions no longer
/// want the same split: `erfc` near zero is best left as `1 − erf`, where that subtraction is
/// harmless and exact, while `erfcx` near zero has no such option and needs this series.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ErfcxForm {
    /// `|x| < `[`ERFCX_CHEBYSHEV_MAX`]: the Chebyshev series.
    Chebyshev,
    /// At or above it: the backward continued fraction.
    Fraction,
}

impl ErfcxForm {
    pub(super) fn for_argument(x: f64) -> Self {
        if x < ERFCX_CHEBYSHEV_MAX {
            Self::Chebyshev
        } else {
            Self::Fraction
        }
    }
}

/// Where the Chebyshev series hands over to the continued fraction.
///
/// `2`, and both sides of the choice are shallow — which is why it is a round number rather than
/// a measured crossover. Widening the fit costs coefficients slowly (24 for `[0,2]`, 28 for
/// `[0,3]`) because the function is entire; narrowing it hands work back to a fraction that is
/// cheap by then anyway, since its ladder has dropped to eleven levels by `x = 20`. `2` keeps
/// the fit short while covering every depth bucket the ladder prices above 100 ns.
const ERFCX_CHEBYSHEV_MAX: f64 = 2.0;

/// `erfcx(x)` for `0 ≤ x < `[`ERFCX_CHEBYSHEV_MAX`], by Clenshaw on [`ERFCX_CHEBYSHEV`].
pub(super) fn erfcx_chebyshev(x: f64) -> f64 {
    // `t = x − 1` maps `[0, 2]` onto `[−1, 1]`, and is exact on the whole interval.
    let t = x - 1.0;
    let twice = 2.0 * t;
    let (mut nearer, mut further) = (0.0_f64, 0.0_f64);
    for &coefficient in ERFCX_CHEBYSHEV.iter().skip(1).rev() {
        let next = twice * nearer - further + coefficient;
        further = nearer;
        nearer = next;
    }
    t * nearer - further + ERFCX_CHEBYSHEV[0]
}

/// Chebyshev coefficients of `erfcx` on `[0, 2]`, in `t = x − 1`.
///
/// Derived rather than transcribed, by `erfcx_cheb.py`, which evaluates `e^{x²}erfc(x)` in MPFR
/// at Chebyshev nodes and runs the transform in 60-digit decimal — the node values and the `π`
/// in the node angles both have to leave `f64` or the coefficients floor at `1e-16` and look
/// converged when they are not.
///
/// Twenty-four terms, where the tail drops below `3e-18`. The fit is checked against the two
/// values `erfcx` is known at outright: `erfcx(0) = 1` and `erfcx(1) = 0.4275835761558070`.
const ERFCX_CHEBYSHEV: [f64; 24] = [
    0.5213608704996062,
    -0.3447707454725235,
    0.09975977926820097,
    -0.026063665256478758,
    0.00626846349431901,
    -0.001406122385887065,
    0.00029698557246549,
    -5.948555390106724e-05,
    1.136296076113552e-05,
    -2.079407863414776e-06,
    3.6590821201669714e-07,
    -6.210726136362159e-08,
    1.0195316699270051e-08,
    -1.622322780246961e-09,
    2.5073406746662083e-10,
    -3.7703813593742715e-11,
    5.5248768982320525e-12,
    -7.89991297902184e-13,
    1.1036213151407695e-13,
    -1.50799316325038e-14,
    2.0174311859216233e-15,
    -2.6449521072902544e-16,
    3.4011233662469575e-17,
    -4.29287736642484e-18,
];

/// `erfcx(x)` for non-negative finite `x`.
pub(super) fn erfcx_positive(x: f64) -> f64 {
    match ErfcxForm::for_argument(x) {
        ErfcxForm::Chebyshev => erfcx_chebyshev(x),
        ErfcxForm::Fraction => erfcx_cf(x),
    }
}

#[cfg(test)]
mod tests;
