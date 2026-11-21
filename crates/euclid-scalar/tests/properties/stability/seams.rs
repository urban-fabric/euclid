//! Seams: no kernel may jump at an algorithm crossover.

use super::*;

// ===== Seams ==============================================================

/// A kernel and the crossovers its own source names, for the seam scan.
type Seamed = (&'static str, fn(f64) -> f64, &'static [f64]);

/// Kernels whose documented crossovers are scanned *in addition to* [`LIKELY_THRESHOLDS`].
///
/// The generic list is a set of round numbers, and a real crossover is only on it by luck.
/// `ERFCX_CF_MIN` is `27/16`; `MARCH_MAX` and `BI_SERIES_MAX` are `9`; `HANKEL_MIN` is `18`;
/// `EI_ASYMPTOTIC_MIN` is `40`; `I_ASYMPTOTIC_MIN` is `20`. None of those is in it, and the
/// Bessel, Airy, integral and transcendental families were absent from the scan altogether —
/// which is where nearly every crossover in this crate lives.
fn seam_kernels() -> Vec<Seamed> {
    vec![
        // erf: ERF_LINEAR_MAX, and ERFCX_CF_MIN at 27/16 where the complement hands over to
        // the continued fraction.
        ("erf", Real::erf, &[1e-8, 1.6875]),
        ("erfc", Real::erfc, &[1e-8, 1.6875]),
        ("erfcx", Real::erfcx, &[1.6875]),
        ("erf_inv", Real::erf_inv, &[]),
        ("erfc_inv", Real::erfc_inv, &[1.0]),
        // gamma: SMALL_SHAPE, GAMMA_OVERFLOW_MIN, REFLECTED_ZERO_MIN.
        ("ln_gamma", Real::ln_gamma, &[1e-3, 171.7, 200.0]),
        ("gamma", Real::gamma, &[1e-3, 171.7]),
        ("rgamma", Real::rgamma, &[1e-3, 200.0]),
        // polygamma: EM_SPLIT at 16.
        ("digamma", Real::digamma, &[16.0]),
        ("trigamma", Real::trigamma, &[16.0]),
        // normal: the log-CDF regimes meet at -1 and 0.
        ("ndtr", Real::ndtr, &[-1.0, 0.0]),
        ("log_ndtr", Real::log_ndtr, &[-1.0, 0.0]),
        // modified Bessel: I1_NO_LOG_MAX, STIRLING_ASYMPTOTIC_MIN's peak, I_ASYMPTOTIC_MIN.
        ("bessel_i0", Real::bessel_i0, &[1.0, 20.0, 40.0]),
        ("bessel_i1", Real::bessel_i1, &[1.0, 20.0, 40.0]),
        ("bessel_i0e", Real::bessel_i0e, &[1.0, 20.0]),
        ("bessel_i1e", Real::bessel_i1e, &[1.0, 20.0]),
        ("ln_bessel_i0", Real::ln_bessel_i0, &[1.0, 20.0]),
        // K: K_SERIES_MAX.
        ("bessel_k0", Real::bessel_k0, &[2.0]),
        ("bessel_k1", Real::bessel_k1, &[2.0]),
        ("bessel_k0e", Real::bessel_k0e, &[2.0]),
        ("bessel_k1e", Real::bessel_k1e, &[2.0]),
        // J/Y: J_SERIES_MAX, Y_SERIES_MAX, HANKEL_MIN.
        ("bessel_j0", Real::bessel_j0, &[2.0, 18.0]),
        ("bessel_j1", Real::bessel_j1, &[2.0, 18.0]),
        ("bessel_y0", Real::bessel_y0, &[4.0, 18.0]),
        ("bessel_y1", Real::bessel_y1, &[4.0, 18.0]),
        // Airy: AI_SERIES_MAX, BI_SERIES_MAX, SERIES_MAX and MARCH_MAX on the negative axis,
        // LEADING_MIN. These are the eight entry points whose halves were just separated, so a
        // seam here would be the first thing that change could have broken.
        ("airy_ai", Real::airy_ai, &[1.0, 9.0, -2.0, -9.0, 1e10]),
        ("airy_bi", Real::airy_bi, &[1.0, 9.0, -2.0, -9.0]),
        (
            "airy_ai_prime",
            Real::airy_ai_prime,
            &[1.0, 9.0, -2.0, -9.0],
        ),
        (
            "airy_bi_prime",
            Real::airy_bi_prime,
            &[1.0, 9.0, -2.0, -9.0],
        ),
        ("airy_ai_e", Real::airy_ai_e, &[1.0, 9.0, -2.0, -9.0, 1e10]),
        ("airy_bi_e", Real::airy_bi_e, &[1.0, 9.0, -2.0, -9.0, 1e10]),
        // integrals: TRIG_SERIES_MAX, E1_SERIES_MAX, EI_ASYMPTOTIC_MIN.
        ("sin_int", Real::sin_int, &[4.0]),
        ("cos_int", Real::cos_int, &[4.0]),
        ("exp_int_ei", Real::exp_int_ei, &[1.0, 40.0]),
        ("exp_int_e1", Real::exp_int_e1, &[1.0, 40.0]),
        // transcendental: the branch point of `W`.
        ("lambert_w", Real::lambert_w, &[-0.367_879_441_171_442_33]),
        // elementary, carried over from the original list.
        ("exp_m1", |x| Real::exp_m1(&x), &[]),
        ("ln_1p", |x| Real::ln_1p(&x), &[]),
        ("cos_m1", |x| Real::cos_m1(x), &[]),
        ("sinc", Real::sinc, &[]),
        ("ln_1p_exp", |x| Real::ln_1p_exp(&x), &[]),
        ("ln_exp_m1", |x| Real::ln_exp_m1(&x), &[]),
        ("cbrt", Real::cbrt, &[]),
        ("hypot with 1", |x| Real::hypot(x, 1.0), &[]),
    ]
}

/// Kernels with a genuine singularity, scanned **only** at their own crossovers.
///
/// `ζ` has a pole at `s = 1`, and `1.0` is in [`LIKELY_THRESHOLDS`]. Scanning it there reports a
/// step `2.8e11` times the derivative — correctly, because `ζ → −∞` approaching the pole from
/// below, and this heuristic cannot tell a pole from a discontinuity. So `ζ` gets its real route
/// boundaries, `s = −1` where the eta series hands over to reflection and `s = 0` where the
/// kernel returns `−½` exactly, and the pole is excluded by construction rather than by a
/// tolerance wide enough to hide a real seam next to it.
fn seam_kernels_with_poles() -> Vec<Seamed> {
    vec![("zeta", Real::zeta, &[-1.0, 0.0])]
}

/// No kernel may have a discontinuity at an algorithm crossover.
///
/// This is the strongest assertion in the suite and the one with no caveats: a continuous
/// function cannot jump, so any jump is a bug. The scan needs no reference value — it only
/// requires that stepping the input one ulp cannot move the output a thousand times further than
/// the derivative allows.
///
/// It does require knowing *where* to look, and that is what this test got wrong. It scanned
/// [`LIKELY_THRESHOLDS`], a list of round numbers, over fifteen kernels that were mostly the erf
/// and gamma families — so the crossovers of the Bessel, Airy, integral and transcendental
/// kernels were never visited, and neither were the several real thresholds that are not round.
/// `hunt_seams`'s own documentation says "passing the kernel's documented crossovers does better
/// still"; now it does. Forty-three kernels, each at the generic list plus the constants its own
/// source names.
#[test]
fn no_kernel_has_a_seam_at_a_likely_threshold() {
    let mut reported = Vec::new();
    let mut scanned = 0usize;

    for (name, f, documented) in seam_kernels() {
        let mut candidates: Vec<f64> = LIKELY_THRESHOLDS.to_vec();
        candidates.extend_from_slice(documented);
        scanned += 1;
        // 1000x the expected step. A smooth kernel scores about 1 and rarely exceeds 10, so
        // this leaves three orders of magnitude of headroom and still catches a branch
        // disagreement, which scores in the millions.
        for seam in hunt_seams(f, &candidates, 256, 1_000.0) {
            reported.push(format!(
                "  {name}: {:e} -> {:e} gives {:e} -> {:e}, {:.2e} times the expected step",
                seam.at, seam.next, seam.values.0, seam.values.1, seam.ratio
            ));
        }
    }
    for (name, f, documented) in seam_kernels_with_poles() {
        scanned += 1;
        for seam in hunt_seams(f, documented, 256, 1_000.0) {
            reported.push(format!(
                "  {name}: {:e} -> {:e} gives {:e} -> {:e}, {:.2e} times the expected step",
                seam.at, seam.next, seam.values.0, seam.values.1, seam.ratio
            ));
        }
    }

    assert!(
        scanned >= 43,
        "the scan lost kernels: only {scanned} covered"
    );
    assert!(
        reported.is_empty(),
        "discontinuities found at algorithm crossovers:\n{}",
        reported.join("\n")
    );
}
