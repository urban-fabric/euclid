//! Agreement between the f64 and f32 carriers.

use crate::Real;

// --- cross-precision agreement ------------------------------------------

#[test]
fn f32_and_f64_agree_to_single_precision() {
    // The same formula on both carriers must land within f32's own resolution. This is the
    // check that catches a divergent f32 code path rather than a mere precision difference.
    for i in 1..60 {
        let x64 = f64::from(i) * 0.1;
        let x32 = x64 as f32;
        let pairs: [(f64, f32); 8] = [
            (Real::erf(x64), Real::erf(x32)),
            (Real::erfc(x64), Real::erfc(x32)),
            (Real::ln_gamma(x64), Real::ln_gamma(x32)),
            (Real::digamma(x64), Real::digamma(x32)),
            (Real::trigamma(x64), Real::trigamma(x32)),
            (Real::sigmoid(&x64), Real::sigmoid(&x32)),
            (Real::ln_1p_exp(&x64), Real::ln_1p_exp(&x32)),
            (Real::bessel_i0e(x64), Real::bessel_i0e(x32)),
        ];
        for (idx, (a, b)) in pairs.iter().enumerate() {
            // Mixed criterion: `ln_gamma(1) = 0`, where a relative comparison between two
            // different roundings of zero is meaningless. f32 carries ~1.2e-7 resolution, so
            // the absolute floor has to accommodate it.
            let err = (f64::from(*b) - *a).abs();
            let bound = 1e-5 * a.abs().max(1.0);
            assert!(
                err <= bound,
                "f32/f64 diverged at x={x64}, fn index {idx}: {a} vs {b} \
                 (abs err {err:.3e} > {bound:.3e})"
            );
        }
    }
}

// ===== The f32 kernels against their f64 counterparts =====================
//
// # Why the test above is not enough
//
// `f32_and_f64_agree_to_single_precision` compares eight functions at a relative tolerance of
// `1e-5`. f32 carries about `1.2e-7` of resolution, so that bound is roughly eighty times
// looser than the format — wide enough that a mutated polynomial coefficient can sit inside it
// unnoticed. Mutation testing measured the consequence: around 480 surviving mutants in the f32
// tree, with `kernels/f32/quantile.rs` (118), `impls/real_f32.rs` (117),
// `kernels/f32/gamma.rs` (91), `kernels/f32/polygamma.rs` (77) and `kernels/erf/f32.rs` (74) the
// worst. `<impl Real for f32>::asin` could return `0.0` for every input.
//
// `euclid_stability::relation::precision_agrees` is the instrument built for exactly this, and
// it was called **nowhere in this crate**. Its docstring describes the risk precisely: an f32
// kernel "is written second, exercised less, and its constants are usually the f64 coefficients
// rounded — which is the right starting point and not always the right answer, because a
// polynomial fitted for f64 can be the wrong polynomial at f32".
//
// What it adds is the right bound. The wide result *rounded to f32* is the best the narrow
// kernel could possibly return, so the comparison is in f32 ulps against that — rather than a
// relative tolerance someone picked.
//
// # These are budgets, not ideals
//
// The numbers below are measured, in the same spirit as the monotonicity counts in
// `tests/stability.rs`: "the honest treatment is to measure and pin rather than to assert an
// ideal nobody promised". Today's behaviour passes; a kernel that drifts fails.
//
// Three things the measurement shows that are worth reading off it:
//
//   - **Several kernels agree bit-exactly** (`bessel_j0`, `k0`, `k1`, `y0`, `airy_ai`,
//     `airy_bi`, `exp_int_ei`, `exp_int_e1`, `exp_m1`). A budget of zero across four hundred
//     arguments is not luck — these compute in f64 and round once. Pinning zero keeps that
//     true, and would catch someone "optimising" one into a genuine f32 path.
//   - **The separately-implemented kernels are the loose ones**, and they are exactly the files
//     the survivors concentrated in: `gamma` 35, `bessel_i0` 54, `bessel_i1` 48, `erfc` 69,
//     `erfcx` 62.
//   - **`ln_gamma` at 1514 ulp and `digamma` at 205 are not kernel defects.** Both have a zero
//     inside the swept range — `ln_gamma(1) = 0` and `digamma` vanishes near `1.4616` — and ulp
//     spacing collapses towards zero while the function's derivative does not. This is the same
//     effect `euclid_stability::numeric::seam_ratio` documents for a raw ulp count, and the
//     reason that function divides by an expected step. A tighter budget here would be a claim
//     about the metric rather than about the code.

/// Arguments swept per kernel. Fixed, because the budgets below are the maximum gap observed
/// over *this* sweep — changing it changes which arguments are visited and therefore the
/// measurement.
const SWEEP: usize = 400;

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

/// The f32 kernels that compute in f64 and round once: bit-exact, and pinned there.
#[test]
fn the_delegating_f32_kernels_are_bit_exact() {
    agrees_to!("bessel_j0", 0.0, 20.0, 0, Real::bessel_j0, Real::bessel_j0);
    agrees_to!("bessel_j1", 0.0, 20.0, 0, Real::bessel_j1, Real::bessel_j1);
    agrees_to!("bessel_k0", 0.1, 15.0, 0, Real::bessel_k0, Real::bessel_k0);
    agrees_to!("bessel_k1", 0.1, 15.0, 0, Real::bessel_k1, Real::bessel_k1);
    agrees_to!("bessel_y0", 0.5, 20.0, 0, Real::bessel_y0, Real::bessel_y0);
    agrees_to!("airy_ai", -5.0, 5.0, 0, Real::airy_ai, Real::airy_ai);
    agrees_to!("airy_bi", -5.0, 2.0, 0, Real::airy_bi, Real::airy_bi);
    agrees_to!(
        "exp_int_ei",
        0.1,
        10.0,
        0,
        Real::exp_int_ei,
        Real::exp_int_ei
    );
    agrees_to!(
        "exp_int_e1",
        0.1,
        10.0,
        0,
        Real::exp_int_e1,
        Real::exp_int_e1
    );
    agrees_to!(
        "exp_m1",
        -2.0,
        2.0,
        0,
        |v: f64| Real::exp_m1(&v),
        |v: f32| Real::exp_m1(&v)
    );
}

/// The rest of the `Real` surface at f32: the delegating half.
///
/// A second pass over the surface the first one missed. `impls/real_f32.rs` held 117 survivors —
/// the largest single file in the crate — because the suites instantiate at `f64` and that impl
/// block has its own one-line body per method. `asin` could return `0.0` for every input.
///
/// Measured at zero ulp across four hundred arguments each, which says these compute in f64 and
/// round once, like the kernel family above. Pinning zero keeps that true.
#[test]
fn the_delegating_f32_real_methods_are_bit_exact() {
    agrees_to!("asin", -0.99, 0.99, 0, Real::asin, Real::asin);
    agrees_to!("acos", -0.99, 0.99, 0, Real::acos, Real::acos);
    agrees_to!("atan", -8.0, 8.0, 0, Real::atan, Real::atan);
    agrees_to!("asinh", -8.0, 8.0, 0, Real::asinh, Real::asinh);
    agrees_to!("acosh", 1.01, 10.0, 0, Real::acosh, Real::acosh);
    agrees_to!("sinh", -5.0, 5.0, 0, Real::sinh, Real::sinh);
    agrees_to!("cosh", -5.0, 5.0, 0, Real::cosh, Real::cosh);
    agrees_to!("tanh", -5.0, 5.0, 0, Real::tanh, Real::tanh);
    agrees_to!("cbrt", -20.0, 20.0, 0, Real::cbrt, Real::cbrt);
    agrees_to!("exp2", -10.0, 10.0, 0, Real::exp2, Real::exp2);
    agrees_to!(
        "bessel_k0e",
        0.1,
        15.0,
        0,
        Real::bessel_k0e,
        Real::bessel_k0e
    );
    agrees_to!(
        "bessel_k1e",
        0.1,
        15.0,
        0,
        Real::bessel_k1e,
        Real::bessel_k1e
    );
}

/// The mathematical constants must agree to the last bit of `f32`.
///
/// These are the simplest mutation target in the crate — a constant replaced by a default — and
/// nothing was checking them. There is no accuracy argument available here: the f32 value is the
/// f64 value correctly rounded, or it is wrong.
#[test]
fn the_constants_agree_across_precisions() {
    #[allow(clippy::cast_possible_truncation)]
    let narrowed = |wide: f64| wide as f32;

    assert_eq!(f32::apery(), narrowed(f64::apery()), "Apery's constant");
    assert_eq!(
        f32::catalan(),
        narrowed(f64::catalan()),
        "Catalan's constant"
    );
    assert_eq!(f32::e(), narrowed(f64::e()), "e");
    assert_eq!(
        f32::euler_gamma(),
        narrowed(f64::euler_gamma()),
        "the Euler-Mascheroni constant"
    );
}

/// The f32 kernels with their own implementation, each at its measured budget.
#[test]
fn the_independent_f32_kernels_stay_inside_their_budgets() {
    agrees_to!("erf", -4.0, 4.0, 8, Real::erf, Real::erf);
    agrees_to!("erfc", -4.0, 4.0, 69, Real::erfc, Real::erfc);
    agrees_to!("erfcx", 0.01, 10.0, 62, Real::erfcx, Real::erfcx);
    agrees_to!("erf_inv", -0.99, 0.99, 15, Real::erf_inv, Real::erf_inv);
    agrees_to!("gamma", 0.1, 8.0, 35, Real::gamma, Real::gamma);
    agrees_to!("trigamma", 0.1, 20.0, 3, Real::trigamma, Real::trigamma);
    agrees_to!("bessel_i0", 0.0, 15.0, 54, Real::bessel_i0, Real::bessel_i0);
    agrees_to!(
        "bessel_i0e",
        0.0,
        30.0,
        44,
        Real::bessel_i0e,
        Real::bessel_i0e
    );
    agrees_to!("bessel_i1", 0.0, 15.0, 48, Real::bessel_i1, Real::bessel_i1);
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
    agrees_to!("erfc_inv", 0.01, 1.99, 17, Real::erfc_inv, Real::erfc_inv);
    agrees_to!(
        "bessel_i1e",
        0.0,
        30.0,
        42,
        Real::bessel_i1e,
        Real::bessel_i1e
    );
}

/// The two kernels whose budget is dominated by a zero inside the range, not by the kernel.
///
/// See the note at the top of this section: an ulp count against a function that passes through
/// zero measures the metric's collapse rather than the implementation's error. Pinned anyway,
/// because a *rise* here is still a regression worth failing on.
#[test]
fn the_f32_kernels_with_a_root_in_range_are_pinned_loosely() {
    agrees_to!("ln_gamma", 0.1, 20.0, 1_514, Real::ln_gamma, Real::ln_gamma);
    agrees_to!("digamma", 0.1, 20.0, 205, Real::digamma, Real::digamma);
    // `atanh(0) = 0`, so the same collapse applies — 17 ulp here is the metric, not the kernel.
    agrees_to!("atanh", -0.99, 0.99, 17, Real::atanh, Real::atanh);
}
