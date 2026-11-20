//! The range contract: what every kernel owes across the whole format.

use super::*;

// ===== Range contract =====================================================

/// A kernel, its domain, and where `NaN` is the correct answer rather than a defect.
type Ranged = (
    &'static str,
    fn(f64) -> f64,
    fn(f64) -> bool,
    fn(f64) -> bool,
);

/// `x` is a pole of `Γ` and `ψ`: zero or a negative integer.
fn is_nonpositive_integer(x: f64) -> bool {
    x <= 0.0 && x.fract() == 0.0
}

/// No kernel returns `NaN` for a finite argument inside its domain.
///
/// # Why `NaN` specifically, and over the whole format
///
/// Accuracy cannot be checked here without a wider float, but the *range contract* can, and it is
/// where this crate's historical defects live: an intermediate that overflows while the result is
/// an ordinary number, an underflow that turns a product into `0 * inf`, a cancellation that
/// leaves a logarithm of zero. Each of those surfaces as `NaN` or a wrong infinity, and each is
/// invisible to a test that samples a comfortable interval.
///
/// So this sweeps every decade of the format at four multipliers, both signs, plus the
/// subnormals, `MIN_POSITIVE`, `MAX` and the neighbours of `±1`. Infinity is permitted — a
/// kernel whose true value overflows *should* saturate — but `NaN` is permitted only where the
/// function genuinely has no real value, and each such case is named below rather than tolerated
/// in bulk.
///
/// # What it found
///
/// `ζ(s)` returned `NaN` for every `s` above `1.0548e28`, where the true value is exactly `1`.
/// The Euler–Maclaurin tail in `hurwitz_zeta` formed `inf * 0`: the power `b^{-s-2j+1}`
/// underflows to zero for any large `s` while the rising factorial `(s)_{2j-1}` overflows, and
/// one of the six Bernoulli terms landed on both at once. Fixed at the source by stopping the
/// tail once the power underflows, which is exact — every later term is zero anyway.
///
/// The first draft of this sweep missed it twice over. It built arguments only as `±10^e` and
/// `±3·10^e`, so every negative sample was a negative *integer* — precisely the poles of `Γ` and
/// `ψ` — and the `NaN`s they produce masked the question of what happens between the poles. The
/// multipliers above exist for that reason.
#[test]
fn no_kernel_returns_nan_where_its_value_is_real() {
    let rows: Vec<Ranged> = vec![
        // `Γ` and `ψ` have simple poles at the non-positive integers and return `NaN` there.
        ("gamma", Real::gamma, |_| true, is_nonpositive_integer),
        ("digamma", Real::digamma, |_| true, is_nonpositive_integer),
        // `lnΓ` is real only where `Γ` is positive, which excludes the alternating negative
        // intervals as well as the poles.
        (
            "ln_gamma",
            Real::ln_gamma,
            |_| true,
            |x: f64| is_nonpositive_integer(x) || Real::gamma(x) <= 0.0,
        ),
        // Everything else must be real across its whole domain.
        ("rgamma", Real::rgamma, |_| true, never),
        ("trigamma", Real::trigamma, |_| true, never),
        ("erf", Real::erf, |_| true, never),
        ("erfc", Real::erfc, |_| true, never),
        ("erfcx", Real::erfcx, |_| true, never),
        (
            "erf_inv",
            Real::erf_inv,
            |x: f64| (-1.0..=1.0).contains(&x),
            never,
        ),
        (
            "erfc_inv",
            Real::erfc_inv,
            |x: f64| (0.0..=2.0).contains(&x),
            never,
        ),
        ("ndtr", Real::ndtr, |_| true, never),
        ("log_ndtr", Real::log_ndtr, |_| true, never),
        (
            "ndtri",
            Real::ndtri,
            |x: f64| (0.0..=1.0).contains(&x),
            never,
        ),
        ("bessel_i0", Real::bessel_i0, |_| true, never),
        ("bessel_i1", Real::bessel_i1, |_| true, never),
        ("bessel_i0e", Real::bessel_i0e, |_| true, never),
        ("bessel_i1e", Real::bessel_i1e, |_| true, never),
        ("ln_bessel_i0", Real::ln_bessel_i0, |_| true, never),
        (
            "bessel_i1_over_i0",
            Real::bessel_i1_over_i0,
            |_| true,
            never,
        ),
        ("bessel_k0", Real::bessel_k0, |x: f64| x > 0.0, never),
        ("bessel_k1", Real::bessel_k1, |x: f64| x > 0.0, never),
        ("bessel_k0e", Real::bessel_k0e, |x: f64| x > 0.0, never),
        ("bessel_k1e", Real::bessel_k1e, |x: f64| x > 0.0, never),
        ("bessel_j0", Real::bessel_j0, |_| true, never),
        ("bessel_j1", Real::bessel_j1, |_| true, never),
        ("bessel_y0", Real::bessel_y0, |x: f64| x > 0.0, never),
        ("bessel_y1", Real::bessel_y1, |x: f64| x > 0.0, never),
        ("airy_ai", Real::airy_ai, |_| true, never),
        ("airy_bi", Real::airy_bi, |_| true, never),
        ("airy_ai_prime", Real::airy_ai_prime, |_| true, never),
        ("airy_bi_prime", Real::airy_bi_prime, |_| true, never),
        ("airy_ai_e", Real::airy_ai_e, |_| true, never),
        ("airy_bi_e", Real::airy_bi_e, |_| true, never),
        ("airy_ai_e_prime", Real::airy_ai_e_prime, |_| true, never),
        ("airy_bi_e_prime", Real::airy_bi_e_prime, |_| true, never),
        ("sin_int", Real::sin_int, |_| true, never),
        ("cos_int", Real::cos_int, |x: f64| x > 0.0, never),
        ("exp_int_ei", Real::exp_int_ei, |x: f64| x != 0.0, never),
        ("exp_int_e1", Real::exp_int_e1, |x: f64| x > 0.0, never),
        (
            "lambert_w",
            Real::lambert_w,
            |x: f64| x >= -0.367_879_441_171_442_33,
            never,
        ),
        // `ζ` has its pole at `s = 1`, where it returns an infinity rather than `NaN`.
        ("zeta", Real::zeta, |_| true, never),
        ("sinc", Real::sinc, |_| true, never),
        ("exprel", Real::exprel, |_| true, never),
        ("cos_m1", Real::cos_m1, |_| true, never),
        ("atanh", Real::atanh, |x: f64| x > -1.0 && x < 1.0, never),
        ("asinh", Real::asinh, |_| true, never),
        ("acosh", Real::acosh, |x: f64| x >= 1.0, never),
    ];

    let probes = whole_format_probes();
    let mut failures: Vec<String> = Vec::new();
    for (name, f, in_domain, nan_is_real) in &rows {
        for &x in &probes {
            if !x.is_finite() || !in_domain(x) || nan_is_real(x) {
                continue;
            }
            let y = f(x);
            if y.is_nan() {
                failures.push(format!(
                    "  {name}({x:e}) = NaN, but the value is real there"
                ));
            }
        }
    }
    assert!(
        rows.len() >= 47,
        "the sweep lost kernels: only {} covered",
        rows.len()
    );
    assert!(
        probes.len() >= 2_000,
        "the probe set shrank to {}",
        probes.len()
    );
    assert!(
        failures.is_empty(),
        "kernels returned NaN inside their domain:\n{}",
        failures.join("\n")
    );
}

/// A kernel, and the lowest argument at which it owes a finite value, with whether both signs
/// are in scope.
type Finite = (&'static str, fn(f64) -> f64, f64, bool);

/// A kernel and its closed-form bounds, inclusive, with whether both signs are in scope.
type Bounded = (&'static str, fn(f64) -> f64, f64, f64, bool);

/// The point below which `K₁(x) ≈ 1/x` exceeds the format, so an infinity is the right answer.
const K1_OVERFLOW_POINT: f64 = 5.562_684_646_268_004e-309;

/// The scaled and log forms stay finite wherever their value is representable.
///
/// # The dual of the `NaN` sweep, and the sharper half
///
/// [`no_kernel_returns_nan_where_its_value_is_real`] permits an infinity anywhere, because a
/// kernel whose true value overflows should saturate. That leaves the opposite defect unchecked:
/// an infinity where the value is an ordinary number, from an intermediate that overflowed on the
/// way. This crate has shipped exactly that — `2πx` overflowing inside `I₀ᵉ` "while the *result*
/// is still an entirely ordinary number", which cost `I₀ᵉ(1e38)` its value outright in `f32`.
///
/// The scaled and log forms are where it matters, because being finite everywhere **is their
/// purpose**. `I₀ᵉ`, `K₀ᵉ`, `Aiᵉ`, `Biᵉ` and `ln I₀` exist precisely so a caller can work past
/// the argument where the plain form leaves the format; one of them returning an infinity is a
/// contradiction in terms rather than a loss of accuracy.
///
/// # The one exception, which is not one
///
/// `K₁ᵉ` goes infinite below `x ≈ 5.56e-309`, and that is correct: `K₁(x) ~ 1/x` near zero, so
/// the true value passes `f64::MAX` there. Measured, it tracks the reciprocal to the digit —
/// `K₁ᵉ(5.57e-309) = 1.7953e308` against `1/x = 1.7953e308`, and infinite one step below, with
/// the boundary landing on `1/f64::MAX` exactly. So the bound is stated as that reciprocal rather
/// than as a tolerated exception.
#[test]
fn the_scaled_and_log_forms_stay_finite_where_their_value_is_representable() {
    // (name, kernel, lowest argument at which finiteness is owed, both signs)
    let rows: Vec<Finite> = vec![
        ("bessel_i0e", Real::bessel_i0e, 0.0, true),
        ("bessel_i1e", Real::bessel_i1e, 0.0, true),
        ("bessel_k0e", Real::bessel_k0e, f64::MIN_POSITIVE, false),
        ("bessel_k1e", Real::bessel_k1e, K1_OVERFLOW_POINT, false),
        ("ln_bessel_i0", Real::ln_bessel_i0, 0.0, true),
        ("bessel_i1_over_i0", Real::bessel_i1_over_i0, 0.0, true),
        ("airy_ai_e", Real::airy_ai_e, 0.0, true),
        ("airy_bi_e", Real::airy_bi_e, 0.0, true),
        ("airy_ai_e_prime", Real::airy_ai_e_prime, 0.0, true),
        ("airy_bi_e_prime", Real::airy_bi_e_prime, 0.0, true),
        // `erfcx(x) = e^{x²}erfc(x)` is bounded on the positives and legitimately overflows on
        // the negatives, so only the positive half is owed.
        ("erfcx", Real::erfcx, 0.0, false),
        // Bounded by construction, so finiteness is unconditional.
        ("erf", Real::erf, 0.0, true),
        ("erfc", Real::erfc, 0.0, true),
        ("ndtr", Real::ndtr, 0.0, true),
        ("bessel_j0", Real::bessel_j0, 0.0, true),
        ("bessel_j1", Real::bessel_j1, 0.0, true),
    ];

    let mut failures = Vec::new();
    for (name, f, lowest, both_signs) in &rows {
        for &x in &whole_format_probes() {
            if !x.is_finite() || x.abs() < *lowest || (!both_signs && x <= 0.0) {
                continue;
            }
            let y = f(x);
            if !y.is_finite() {
                failures.push(format!(
                    "  {name}({x:e}) = {y:e}, but the value is representable there"
                ));
            }
        }
    }
    assert!(rows.len() >= 16, "the sweep lost kernels: {}", rows.len());
    assert!(
        failures.is_empty(),
        "scaled or log forms left the format early:\n{}",
        failures.join("\n")
    );
}

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
