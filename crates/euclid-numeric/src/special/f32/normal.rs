//! Single-precision peer of [`super::super::normal`]; see that module for the algorithms and
//! the reason the log-CDF is routed through `erfcx`.
//!
//! Evaluated in `f32` throughout, with no `f64` round-trip, matching the rest of this tree.

use super::erf::{erf_inv, erfc, erfcx};

const LN_2: f32 = core::f32::consts::LN_2;
const FRAC_1_SQRT_2: f32 = core::f32::consts::FRAC_1_SQRT_2;
use crate::constants::f32::HALF_LN_TAU;

/// The standard normal CDF.
pub(crate) fn ndtr(x: f32) -> f32 {
    0.5 * erfc(-x * FRAC_1_SQRT_2)
}

/// `ln Phi(x)`. The `f32` exponent range is much narrower, so the `erfcx` route matters
/// sooner: `Phi` underflows by about `x = -13` here rather than `x = -38`.
pub(crate) fn log_ndtr(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    let z = -x * FRAC_1_SQRT_2;
    if x > -1.0 {
        (0.5 * erfc(z)).ln()
    } else {
        erfcx(z).ln() - z * z - LN_2
    }
}

fn tail_start(p: f32) -> f32 {
    let t = (-2.0 * p.ln()).sqrt();
    -(t - (t.ln() + HALF_LN_TAU) / t)
}

/// The standard normal quantile.
pub(crate) fn ndtri(p: f32) -> f32 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f32::NAN;
    }
    if p == 0.0 {
        return f32::NEG_INFINITY;
    }
    if p == 1.0 {
        return f32::INFINITY;
    }
    if p == 0.5 {
        return 0.0;
    }
    if p > 0.5 {
        return -ndtri_lower(1.0 - p);
    }
    ndtri_lower(p)
}

fn ndtri_lower(p: f32) -> f32 {
    let ln_p = p.ln();
    // `2p - 1` stops carrying information much earlier in f32 than in f64.
    let mut x = if p >= 1e-6 {
        erf_inv(2.0 * p - 1.0) / FRAC_1_SQRT_2
    } else {
        tail_start(p)
    };
    if !x.is_finite() {
        x = tail_start(p);
    }
    for _ in 0..6 {
        let l_phi = log_ndtr(x);
        let ln_pdf = -0.5 * x * x - HALF_LN_TAU;
        let step = (l_phi - ln_p) * (l_phi - ln_pdf).exp();
        if !step.is_finite() {
            break;
        }
        let next = x - step;
        if next == x {
            break;
        }
        x = next;
    }
    x
}

/// The inverse complementary error function.
pub(crate) fn erfc_inv(y: f32) -> f32 {
    if y.is_nan() || !(0.0..=2.0).contains(&y) {
        return f32::NAN;
    }
    if y == 0.0 {
        return f32::INFINITY;
    }
    if y == 2.0 {
        return f32::NEG_INFINITY;
    }
    -ndtri(0.5 * y) * FRAC_1_SQRT_2
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, tol: f32) {
        assert!(
            (a - b).abs() <= tol * b.abs().max(1.0),
            "{a:e} vs {b:e} (tol {tol:e})"
        );
    }

    #[test]
    fn ndtr_hits_its_known_values() {
        close(ndtr(0.0), 0.5, 1e-6);
        close(ndtr(1.0), 0.841_344_7, 1e-5);
        close(ndtr(-1.0), 0.158_655_25, 1e-5);
    }

    #[test]
    fn log_ndtr_stays_finite_past_the_f32_underflow() {
        assert_eq!(ndtr(-20.0), 0.0, "Phi underflows in f32 here");
        let v = log_ndtr(-20.0);
        assert!(v.is_finite() && v < 0.0, "log_ndtr(-20) = {v}");
        close(v, -200.0 - 20.0f32.ln() - HALF_LN_TAU, 1e-2);
    }

    #[test]
    fn ndtri_round_trips() {
        for i in 1..20 {
            let p = (i as f32) * 0.05;
            close(ndtr(ndtri(p)), p, 1e-4);
        }
        for &p in &[1e-10f32, 1e-20, 1e-30] {
            let x = ndtri(p);
            assert!(x.is_finite() && x < 0.0, "ndtri({p}) = {x}");
            close(log_ndtr(x), p.ln(), 1e-4);
        }
    }

    #[test]
    fn erfc_inv_inverts_erfc() {
        for i in 1..20 {
            let y = (i as f32) * 0.1;
            close(erfc(erfc_inv(y)), y, 1e-4);
        }
    }
}
