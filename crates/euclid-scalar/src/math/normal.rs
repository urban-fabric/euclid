//! Evaluation steps shared by the two standard normal kernels.

use crate::HostReal;
use crate::constants::f64::HALF_LN_TAU;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LogCdfRegime {
    UpperTail,
    Central,
    LowerTail,
}

fn log_cdf_regime<T: HostReal>(x: T) -> LogCdfRegime {
    if x > T::ZERO {
        LogCdfRegime::UpperTail
    } else if x > -T::ONE {
        LogCdfRegime::Central
    } else {
        LogCdfRegime::LowerTail
    }
}

pub(crate) fn ndtr<T: HostReal>(x: T) -> T {
    T::HALF * (-x * T::FRAC_1_SQRT_2).erfc()
}

pub(crate) fn log_ndtr<T: HostReal>(x: T) -> T {
    if HostReal::is_nan(x) {
        return T::NAN;
    }
    match log_cdf_regime(x) {
        LogCdfRegime::UpperTail => (-ndtr(-x)).ln_1p(),
        LogCdfRegime::Central => ndtr(x).ln(),
        LogCdfRegime::LowerTail => {
            let z = -x * T::FRAC_1_SQRT_2;
            z.erfcx().ln() - z * z - T::ln_2()
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum QuantileRoute<T> {
    Value(T),
    Lower(T),
    Reflected(T),
}

fn quantile_route<T: HostReal>(p: T) -> QuantileRoute<T> {
    if !(T::ZERO..=T::ONE).contains(&p) {
        QuantileRoute::Value(T::NAN)
    } else if p == T::ZERO {
        QuantileRoute::Value(T::NEG_INFINITY)
    } else if p == T::ONE {
        QuantileRoute::Value(T::INFINITY)
    } else if p > T::HALF {
        // This subtraction is exact on the reflected half of the unit interval.
        QuantileRoute::Reflected(T::ONE - p)
    } else if p < T::HALF {
        QuantileRoute::Lower(p)
    } else {
        QuantileRoute::Value(T::ZERO)
    }
}

/// Asymptotic seed from `ln(p) ≈ -x²/2 - ln(|x|) - ln(sqrt(2π))`.
pub(crate) fn tail_start_from_log<T: HostReal>(ln_p: T) -> T {
    let t = (-T::TWO * ln_p).sqrt();
    -(t - (t.ln() + T::from_f64(HALF_LN_TAU)) / t)
}

/// Central seed from `Phi⁻¹(p) = sqrt(2) erf⁻¹(2p - 1)`.
pub(crate) fn central_start<T: HostReal>(p: T) -> T {
    (T::TWO * p - T::ONE).erf_inv() / T::FRAC_1_SQRT_2
}

fn lower_seed<T: HostReal>(p: T, ln_p: T, central_min: T) -> T {
    if p >= central_min {
        let central = central_start(p);
        if HostReal::is_finite(central) {
            return central;
        }
    }
    tail_start_from_log(ln_p)
}

/// Newton correction on `ln Phi(x) - ln p`, whose derivative is `phi(x)/Phi(x)`.
fn normal_newton_step<T: HostReal>(x: T, ln_p: T, log_cdf: T) -> T {
    let log_density = -T::HALF * x * x - T::from_f64(HALF_LN_TAU);
    (log_cdf - ln_p) * (log_cdf - log_density).exp()
}

/// Polish a seed with Newton on `ln Phi(x) - ln p`, stopping at a fixed point *or a cycle*.
///
/// # Why a cycle is the normal ending, not a failure
///
/// `next == x` alone is not enough. [`normal_newton_step`] forms `log_cdf - ln_p`, a difference
/// of two nearly equal logarithms, so once `x` is within an ulp or two of the root the step is
/// set by that cancellation rather than by the residual, and the iterate alternates between two
/// adjacent values instead of settling. Measured over a uniform sweep of `p`, **26.9% of calls
/// ran all six passes without reaching a fixed point, and 61% of those were exact two-cycles.**
///
/// This is benign, which is worth stating because it looks alarming. The *backward* error of the
/// returned value — `|Phi(x) - p| / p`, the only accuracy measure available without a wider
/// float — is at most `8.9e-15` across that sweep and `4.9e-13` in the far tail. Several
/// adjacent floats all map to the same `p` to within `Phi`'s own accuracy, so each of them is a
/// correct answer and the iteration is choosing between equals. The 19-ulp spread of the
/// iterates is the width of that set, not error.
///
/// So the cycle test buys time rather than accuracy: **14.8% fewer passes** on a uniform sweep
/// and 12.9% in the tail, each pass being a `log_ndtr` and therefore an `ndtr` or an `erfcx`,
/// which dominate everything else here. The returned value can differ from the six-pass result
/// by up to 18 ulp, and the maximum backward error is **unchanged to four significant figures**
/// in both regions, which is the sense in which the two answers are equally good.
///
/// Newton on `Phi(x) - p` directly was measured as the alternative and is much worse: the
/// subtraction cancels for small `p`, taking the tail's backward error from `4.9e-13` to
/// `2.8e-4`. The logarithmic form is the right one; it just needs to know when to stop.
pub(crate) fn refine_lower<T: HostReal>(ln_p: T, mut x: T) -> T {
    let mut previous = x;
    for _ in 0..6 {
        let step = normal_newton_step(x, ln_p, log_ndtr(x));
        if !HostReal::is_finite(step) {
            break;
        }
        let next = x - step;
        if next == x || next == previous {
            return next;
        }
        previous = x;
        x = next;
    }
    x
}

fn ndtri_lower<T: HostReal>(p: T, central_min: T) -> T {
    let ln_p = p.ln();
    refine_lower(ln_p, lower_seed(p, ln_p, central_min))
}

pub(crate) fn ndtri<T: HostReal>(p: T, central_min: T) -> T {
    match quantile_route(p) {
        QuantileRoute::Value(x) => x,
        QuantileRoute::Lower(p) => ndtri_lower(p, central_min),
        QuantileRoute::Reflected(p) => -ndtri_lower(p, central_min),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_cdf_routes_have_explicit_boundaries() {
        let left = -1.0_f64;
        for (x, route) in [
            (f64::NEG_INFINITY, LogCdfRegime::LowerTail),
            (left.next_down(), LogCdfRegime::LowerTail),
            (left, LogCdfRegime::LowerTail),
            (left.next_up(), LogCdfRegime::Central),
            (-0.0, LogCdfRegime::Central),
            (0.0, LogCdfRegime::Central),
            (f64::from_bits(1), LogCdfRegime::UpperTail),
            (f64::INFINITY, LogCdfRegime::UpperTail),
        ] {
            assert_eq!(log_cdf_regime(x), route);
        }
        assert_eq!(log_cdf_regime((-1.0_f32).next_up()), LogCdfRegime::Central);
        assert_eq!(log_cdf_regime(f32::from_bits(1)), LogCdfRegime::UpperTail);
    }

    #[test]
    fn quantile_routes_preserve_endpoints_and_the_reflected_tail() {
        let half = 0.5_f64;
        let one = 1.0_f64;
        assert_eq!(
            quantile_route(-0.0_f64),
            QuantileRoute::Value(f64::NEG_INFINITY)
        );
        assert_eq!(
            quantile_route(half.next_down()),
            QuantileRoute::Lower(half.next_down())
        );
        assert_eq!(quantile_route(half), QuantileRoute::Value(0.0));
        assert_eq!(
            quantile_route(half.next_up()),
            QuantileRoute::Reflected(half - f64::EPSILON / 2.0)
        );
        assert_eq!(
            quantile_route(one.next_down()),
            QuantileRoute::Reflected(f64::EPSILON / 2.0)
        );
        assert_eq!(quantile_route(one), QuantileRoute::Value(f64::INFINITY));
        for p in [
            f64::NAN,
            f64::INFINITY,
            (-0.0_f64).next_down(),
            one.next_up(),
        ] {
            assert!(matches!(quantile_route(p), QuantileRoute::Value(x) if x.is_nan()));
        }
    }

    #[test]
    fn seed_selection_respects_the_cutoff_and_a_nonfinite_central_seed() {
        let p = 0.25_f64;
        assert_eq!(lower_seed(p, p.ln(), p), central_start(p));
        assert_eq!(
            lower_seed(p, p.ln(), p.next_up()),
            tail_start_from_log(p.ln())
        );
        let p = f32::from_bits(1);
        assert!(central_start(p).is_infinite());
        assert_eq!(lower_seed(p, p.ln(), p), tail_start_from_log(p.ln()));
    }

    #[test]
    fn normal_newton_step_matches_a_closed_form_at_zero() {
        // At zero, Phi is 1/2 and phi is 1/sqrt(2π). For target p=1/4,
        // the correction is ln(2) * sqrt(2π) / 2.
        let ln_2 = core::f64::consts::LN_2;
        let expected = ln_2 * (2.0 * core::f64::consts::PI).sqrt() * 0.5;
        assert!((normal_newton_step(0.0, -2.0 * ln_2, -ln_2) - expected).abs() <= f64::EPSILON);
        assert_eq!(normal_newton_step(0.0, -ln_2, -ln_2), 0.0);
    }
}
