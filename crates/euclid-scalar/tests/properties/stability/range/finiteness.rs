//! No kernel returns `NaN`, and no scaled or log form leaves the format, where the true value is
//! representable.
//!
//! Part of the [`range`](super) suite, whose preamble holds the `Ranged`/`Bounded` row shapes and
//! the `is_nonpositive_integer` pole test. This is the half that catches a *missing* value rather
//! than an inaccurate one: a kernel that gives up and returns `NaN` passes every accuracy budget
//! in the tree.

use super::*;

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
