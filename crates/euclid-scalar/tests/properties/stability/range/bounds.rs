//! Documented bounds, and the point at which a value is allowed to underflow.
//!
//! Part of the [`range`](super) suite. The underflow law is the sharper of the two: a kernel may
//! return zero only once its true value is below the format's smallest subnormal, so returning
//! zero early — which looks correct to any relative comparison — is a failure here.

use super::*;

/// Every kernel with a closed-form bound respects it, across the whole format.
///
/// A bound is the cheapest correctness statement there is — it needs no reference, no wider
/// float and no tolerance — and it catches a whole class of defect that an accuracy test cannot:
/// a probability above one, a correlation past unity, a `J₀` outside `[−1, 1]`. Each of these is
/// a mathematical fact about the function, not a property of the implementation, so a violation
/// is unambiguous.
///
/// The one-sided bounds are stated as **non-strict**. `erfc` and `E₁` are strictly positive in
/// exact arithmetic but underflow to zero at a finite argument — `erfc(34.4) ≈ 1e-514` and
/// `E₁(1003) ≈ 1e-439`, both below `f64::MIN_POSITIVE` — so `> 0` would fail on correct
/// saturation. That was a bug in the first draft of this test, not in the kernels.
#[test]
fn every_bounded_kernel_respects_its_documented_bounds() {
    // (name, kernel, inclusive lower bound, inclusive upper bound, both signs)
    let rows: Vec<Bounded> = vec![
        ("erf", Real::erf, -1.0, 1.0, true),
        ("erfc", Real::erfc, 0.0, 2.0, true),
        ("ndtr", Real::ndtr, 0.0, 1.0, true),
        ("log_ndtr", Real::log_ndtr, f64::NEG_INFINITY, 0.0, true),
        (
            "bessel_i1_over_i0",
            Real::bessel_i1_over_i0,
            -1.0,
            1.0,
            true,
        ),
        ("bessel_j0", Real::bessel_j0, -1.0, 1.0, true),
        ("bessel_j1", Real::bessel_j1, -0.581_9, 0.581_9, true),
        ("sinc", Real::sinc, -0.217_4, 1.0, true),
        ("erfcx", Real::erfcx, 0.0, f64::INFINITY, false),
        ("bessel_i0e", Real::bessel_i0e, 0.0, 1.0, true),
        ("bessel_i1e", Real::bessel_i1e, -0.5, 0.5, true),
        ("bessel_k0e", Real::bessel_k0e, 0.0, f64::INFINITY, false),
        ("exp_int_e1", Real::exp_int_e1, 0.0, f64::INFINITY, false),
        ("airy_ai_e", Real::airy_ai_e, 0.0, 0.355_1, false),
        ("gamma", Real::gamma, 0.0, f64::INFINITY, false),
        ("trigamma", Real::trigamma, 0.0, f64::INFINITY, false),
    ];

    let mut failures = Vec::new();
    for (name, f, lower, upper, both_signs) in &rows {
        for &x in &whole_format_probes() {
            if !x.is_finite() || (!both_signs && x <= 0.0) {
                continue;
            }
            let y = f(x);
            if y.is_finite() && (y < *lower || y > *upper) {
                failures.push(format!(
                    "  {name}({x:e}) = {y:e}, outside [{lower:e}, {upper:e}]"
                ));
            }
        }
    }
    // `ζ(s) ≥ 1` for every `s > 1`, which needs the argument shifted rather than sign-filtered.
    for &x in &whole_format_probes() {
        if !x.is_finite() || x <= 0.0 {
            continue;
        }
        let s = 1.0 + x;
        let y = Real::zeta(s);
        if y.is_finite() && y < 1.0 {
            failures.push(format!("  zeta({s:e}) = {y:e}, below 1"));
        }
    }
    assert!(rows.len() >= 16, "the sweep lost kernels: {}", rows.len());
    assert!(
        failures.is_empty(),
        "kernels left their documented bounds:\n{}",
        failures.join("\n")
    );
}

/// A kernel, the argument up to which it owes a non-zero value, and whether to sweep negatives.
type Underflows = (&'static str, fn(f64) -> f64, f64, bool);

/// A scaled kernel and whether both signs are in scope; it owes a non-zero value throughout.
type Scaled = (&'static str, fn(f64) -> f64, bool);

/// No kernel underflows to zero before its true value does.
///
/// # The third corner of the range contract
///
/// The sibling tests cover `NaN` where the value is real, and infinity where the value is
/// representable. This is the remaining corner: **zero where the value is representable**. It is
/// the same defect as premature overflow seen from the other end — an intermediate that
/// underflows, or an exponential formed before a cancellation instead of after — and it is the
/// quietest of the three, because zero is a plausible answer for a decaying function and nothing
/// about it looks wrong.
///
/// Each boundary below is an analytic prediction, checked against where the kernel actually first
/// returns zero over a 20,000-point sweep:
///
/// ```text
/// kernel          predicted   measured first zero
/// erfc             ~26.6            27.24
/// exp_int_e1        ~708           738.58
/// bessel_k0/k1      ~708           742.68
/// airy_ai           ~104           107.55
/// ndtr             ~-38.5          -38.477
/// ```
///
/// The predictions come from the leading asymptotic — `erfc(x) ≈ e^{−x²}/(x√π)` underflows once
/// `x² > 708`, `K₀(x) ≈ √(π/2x)e^{−x}` once `x > 708`, `Ai(x) ≈ e^{−ζ}/(2√π x^{1/4})` once
/// `ζ = ⅔x^{3/2} > 708`. Each kernel reaches a little past its prediction because of the
/// algebraic prefactor, which is the right direction: it means none of them gives up early. The
/// assertion boundary is set just inside the measured value, so the test fails if a kernel starts
/// underflowing sooner and does not depend on the exact last representable argument.
///
/// # The scaled forms must never reach zero at all
///
/// `erfcx`, `K₀ᵉ`, `K₁ᵉ`, `I₀ᵉ`, `Aiᵉ` and `Biᵉ` exist so that the decay is divided out. A zero
/// from one of them would mean the scaling had failed to do the one thing it is for, so they are
/// asserted non-zero across the whole format rather than up to a boundary.
#[test]
fn no_kernel_underflows_before_its_value_does() {
    let rows: Vec<Underflows> = vec![
        ("erfc", Real::erfc, 26.0, false),
        ("exp_int_e1", Real::exp_int_e1, 735.0, false),
        ("bessel_k0", Real::bessel_k0, 740.0, false),
        ("bessel_k1", Real::bessel_k1, 740.0, false),
        ("airy_ai", Real::airy_ai, 107.0, false),
        // `Φ(x)` underflows on the *left*, so this one is swept negative.
        ("ndtr", Real::ndtr, 38.4, true),
    ];

    let mut failures = Vec::new();
    for (name, f, boundary, negate) in &rows {
        for i in 1..=4_000u32 {
            let magnitude = f64::from(i) * *boundary / 4_000.0;
            let x = if *negate { -magnitude } else { magnitude };
            if f(x) == 0.0 {
                failures.push(format!(
                    "  {name}({x:e}) = 0, but the value is representable until |x| ~ {boundary}"
                ));
                break;
            }
        }
    }

    // The scaled forms, across the whole format. `I₁ᵉ` is excluded: it is genuinely zero at the
    // origin, because `I₁(0) = 0`.
    let scaled: Vec<Scaled> = vec![
        ("erfcx", Real::erfcx, false),
        ("bessel_k0e", Real::bessel_k0e, false),
        ("bessel_k1e", Real::bessel_k1e, false),
        ("bessel_i0e", Real::bessel_i0e, true),
        ("airy_ai_e", Real::airy_ai_e, false),
        ("airy_bi_e", Real::airy_bi_e, false),
    ];
    for (name, f, both_signs) in &scaled {
        for &x in &whole_format_probes() {
            if !x.is_finite() || x == 0.0 || (!both_signs && x < 0.0) {
                continue;
            }
            if f(x) == 0.0 {
                failures.push(format!(
                    "  {name}({x:e}) = 0, but a scaled form has no decay left to underflow"
                ));
                break;
            }
        }
    }

    assert!(rows.len() + scaled.len() >= 12, "the sweep lost kernels");
    assert!(
        failures.is_empty(),
        "kernels underflowed before their value did:\n{}",
        failures.join("\n")
    );
}
