//! The native `f32` kernels against their measured ulp budgets.
//!
//! Split from the delegate checks because these kernels have no delegate: they are evaluated in
//! single precision throughout, so the `f64` kernel is a *reference* rather than the thing they
//! are required to match bit for bit. Each budget is measured; see the parent module.

use super::SWEEP;
use crate::Real;

/// Check one kernel's f32 implementation against its f64 one across a domain.
macro_rules! agrees_to {
    ($name:literal, $lo:expr, $hi:expr, $ulps:expr, $wide:expr, $narrow:expr) => {{
        for i in 0..=SWEEP {
            #[allow(clippy::cast_precision_loss)]
            let x = $lo + ($hi - $lo) * (i as f64) / (SWEEP as f64);
            let outcome =
                euclid_stability::relation::precision_agrees($name, x, $wide, $narrow, $ulps);
            assert!(
                outcome.is_ok(),
                "{}",
                outcome.err().map(|f| f.message).unwrap_or_default()
            );
        }
    }};
}

// --- cross-precision agreement ------------------------------------------

/// The f32 kernels with their own implementation, each at its measured budget.
#[test]
fn the_independent_f32_kernels_stay_inside_their_budgets() {
    agrees_to!("erf", -4.0, 4.0, 8, Real::erf, Real::erf);
    agrees_to!("erfc", -4.0, 4.0, 7, Real::erfc, Real::erfc);
    agrees_to!("erfcx", 0.01, 10.0, 7, Real::erfcx, Real::erfcx);
    agrees_to!("erfc_inv", 0.0, 2.0, 8, Real::erfc_inv, Real::erfc_inv);
    agrees_to!(
        "exp_int_e1",
        0.1,
        10.0,
        13,
        Real::exp_int_e1,
        Real::exp_int_e1
    );
    // `Ei` has a zero at `x ≈ 0.3725`, inside this sweep, so the budget is dominated by the
    // metric collapsing there rather than by the kernel — the same effect noted for
    // `ln_gamma`, `digamma` and `atanh` below. Measured 49; the native `Ei` holds to 4 ulp
    // against the `f64` kernel away from the zero.
    agrees_to!(
        "exp_int_ei",
        0.1,
        10.0,
        49,
        Real::exp_int_ei,
        Real::exp_int_ei
    );
    agrees_to!("erf_inv", -0.99, 0.99, 15, Real::erf_inv, Real::erf_inv);
    agrees_to!("gamma", 0.1, 8.0, 35, Real::gamma, Real::gamma);
    agrees_to!("trigamma", 0.1, 20.0, 3, Real::trigamma, Real::trigamma);
    agrees_to!("bessel_i0", 0.0, 15.0, 48, Real::bessel_i0, Real::bessel_i0);
    agrees_to!(
        "bessel_i0e",
        0.0,
        30.0,
        44,
        Real::bessel_i0e,
        Real::bessel_i0e
    );
    agrees_to!("bessel_i1", 0.0, 15.0, 50, Real::bessel_i1, Real::bessel_i1);
    agrees_to!("exprel", -2.0, 2.0, 1, Real::exprel, Real::exprel);
    agrees_to!("cos_m1", -3.0, 3.0, 2, Real::cos_m1, Real::cos_m1);
    agrees_to!(
        "sigmoid",
        -10.0,
        10.0,
        1,
        |v: f64| Real::sigmoid(&v),
        |v: f32| Real::sigmoid(&v)
    );
    agrees_to!(
        "ln_1p_exp",
        -10.0,
        10.0,
        1,
        |v: f64| Real::ln_1p_exp(&v),
        |v: f32| Real::ln_1p_exp(&v)
    );

    // Measured in a second pass over the surface the first one missed.
    agrees_to!("sinc", -8.0, 8.0, 1, Real::sinc, Real::sinc);
    agrees_to!(
        "ln_1m_exp",
        -8.0,
        -0.01,
        1,
        |v: f64| Real::ln_1m_exp(&v),
        |v: f32| Real::ln_1m_exp(&v)
    );
    agrees_to!(
        "bessel_i1_over_i0",
        0.1,
        20.0,
        7,
        Real::bessel_i1_over_i0,
        Real::bessel_i1_over_i0
    );
    agrees_to!(
        "bessel_i1e",
        0.0,
        30.0,
        43,
        Real::bessel_i1e,
        Real::bessel_i1e
    );
}

/// The kernels whose budget is dominated by a zero inside the range, not by the kernel.
///
/// See the note at the top of this section: an ulp count against a function that passes through
/// zero measures the metric's collapse rather than the implementation's error. Pinned anyway,
/// because a *rise* here is still a regression worth failing on.
///
/// **But the collapse is not a licence to stop looking.** `ln_gamma` was pinned at 1514 and
/// `digamma` at 205 on exactly this reasoning, and both were also carrying a genuine
/// cancellation defect — the two causes are not exclusive, and the metric's noise was large
/// enough to hide the kernel's. At 72 and 31 the remaining width really is the zero.
#[test]
fn the_f32_kernels_with_a_root_in_range_are_pinned_loosely() {
    agrees_to!("ln_gamma", 0.1, 20.0, 72, Real::ln_gamma, Real::ln_gamma);
    agrees_to!("digamma", 0.1, 20.0, 3, Real::digamma, Real::digamma);
    // `atanh(0) = 0`, so the same collapse applies — 17 ulp here is the metric, not the kernel.
    agrees_to!("atanh", -0.99, 0.99, 17, Real::atanh, Real::atanh);
}
