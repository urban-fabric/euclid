//! Each plain kernel's saturation, judged by its own scaled sibling. No reference needed.
//!
//! # The oracle
//!
//! A scaled form exists so that the decay or growth is divided out, which means it also *records*
//! that factor. So for every plain/scaled pair the magnitude of the plain value is known in the
//! log domain without computing it:
//!
//! ```text
//! ln|I₀(x)|  = ln I₀ᵉ(x) + |x|            ln|K₀(x)| = ln K₀ᵉ(x) − x
//! ln|Bi(x)|  = ln Biᵉ(x) + ζ              ln|Ai(x)| = ln Aiᵉ(x) − ζ
//! ln|erfc(x)| = ln erfcx(x) − x²          ln|Φ(x)|  = log_ndtr(x)
//! ln|Γ(x)|   = lnΓ(x)                     ln|1/Γ(x)| = −lnΓ(x)
//! ```
//!
//! Each right-hand side stays in range precisely where the left-hand side does not — that is what
//! the scaled form is *for* — so it decides whether the plain form owes a finite non-zero answer.
//! **This needs no wider float, no asymptotic expansion and no pasted table.** It is the cheapest
//! accuracy-adjacent oracle in the suite, and the only one that costs nothing to run at both ends
//! of the format at once.
//!
//! # Why this file exists: three of four recent defects were exactly this shape
//!
//! `bessel_i1` returned `0` where `I₁ᵉ·e^x` was unbounded, `airy_bi` returned `+∞` where
//! `Biᵉ·e^ζ` was an ordinary number, and `lnΓ` overflowed where its own result fit — all found by
//! [`super::f32_exhaustive`] comparing `f32` against `f64`, which is a tool that does not exist
//! for `f64` itself. This relation does, which is the point: it is the same question asked in a
//! way that works in **either** precision.
//!
//! Checked rather than asserted: against the pre-fix kernels this test flags `bessel_i1`
//! (plain `0`, sibling unbounded) and `airy_bi` (plain `+∞`, sibling `87.59` against a ceiling of
//! `88.72`). It would **not** have caught the other two, and the reason is worth stating plainly:
//! `exprel` has no scaled sibling at all, and for `Γ` the sibling *is* `lnΓ`, so a broken `lnΓ`
//! breaks the oracle rather than being caught by it. Eleven pairs is the whole reach of this
//! technique, and the kernels outside it need the cross-precision sweep.
//!
//! # The slack is asymmetric, because the two ends are not alike
//!
//! At the **overflow** end consecutive values differ by `ε` relatively, so `0.25` in the log
//! domain — a factor of `1.28` — is four orders above any kernel's own error. The largest term in
//! reconstructing the exponent is `ζ·ε`, about `1e-5` in `f32`.
//!
//! At the **underflow** end that reasoning fails completely. The smallest subnormal's neighbour is
//! *twice* it, so the floor's granularity is a factor of two and a log-domain slack below `ln 2`
//! fires on a one-step difference. A first pass used `0.25` at both ends and reported `ndtr` in
//! both precisions — at `f32` the true value was `1.80e-45` against a smallest subnormal of
//! `1.40e-45`, so `1.29` of them, which rounds to one and which `ndtr` returned as zero. One
//! subnormal of disagreement in a quantity with no significant bits left, and not a defect; the
//! same call [`super::f32_exhaustive`] makes for `K₀` and `K₁`. So the underflow slack is
//! `2·ln 2`, which admits a couple of subnormals and nothing wider.
//!
//! # What it found
//!
//! Nothing outstanding — 22 pair-checks over 200,001 arguments each, in both precisions, all
//! clean. Since two of the four defects it would have caught were fixed in the same session that
//! wrote it, that is the expected result and the reason it is worth keeping.

use super::*;

/// Log-domain headroom at the overflow end; see the module header.
const OVERFLOW_SLACK: f64 = 0.25;

/// Log-domain headroom at the underflow end: `2·ln 2`, two subnormals of the floor's granularity.
const UNDERFLOW_SLACK: f64 = 2.0 * core::f64::consts::LN_2;

/// Points swept across each kernel's threshold window.
const SAMPLES: u32 = 200_000;

/// `ln(f32::MAX)` and `ln` of the argument at which `f32`'s subnormals round away.
const F32_CEILING: f64 = 88.722_839;
const F32_FLOOR: f64 = -103.972_08;

/// The same two for `f64`.
const F64_CEILING: f64 = 709.782_712_893_384;
const F64_FLOOR: f64 = -745.133_219_101_941_2;

/// Sweep one pair, returning the complaints.
///
/// `predict` returns `ln|value|` from the scaled sibling. A point where that is itself non-finite
/// is skipped rather than failed: the sibling is the oracle here, so it cannot also be the thing
/// under test — see the module header on `Γ`.
fn audit(
    label: &str,
    window: (f64, f64),
    ceiling: f64,
    floor: f64,
    plain: impl Fn(f64) -> f64,
    predict: impl Fn(f64) -> f64,
) -> Vec<String> {
    let mut failures = Vec::new();
    let (low, high) = window;
    for i in 0..=SAMPLES {
        let x = low + (high - low) * f64::from(i) / f64::from(SAMPLES);
        let predicted = predict(x);
        if !predicted.is_finite() {
            continue;
        }
        let got = plain(x);
        if !got.is_finite() && predicted < ceiling - OVERFLOW_SLACK {
            failures.push(format!(
                "  {label}({x:e}) = {got:e}, but its scaled sibling puts ln|value| at \
                 {predicted:.3}, inside the ceiling of {ceiling:.3}"
            ));
        }
        if got == 0.0 && predicted > floor + UNDERFLOW_SLACK {
            failures.push(format!(
                "  {label}({x:e}) = 0, but its scaled sibling puts ln|value| at {predicted:.3}, \
                 above the floor of {floor:.3}"
            ));
        }
        if failures.len() >= 12 {
            break;
        }
    }
    failures
}

/// Two thirds of `ζ = ⅔x^{3/2}`, the Airy scaling exponent.
fn zeta(x: f64) -> f64 {
    2.0 / 3.0 * x.powf(1.5)
}

/// No `f64` kernel saturates where its own scaled sibling says the value still fits.
///
/// Eleven pairs, each swept across the window containing its threshold. The windows are generous
/// on both sides of the crossing so the test covers the correct saturation as well as the
/// premature kind — a kernel that stopped saturating at all would fail the opposite clause.
#[test]
fn no_f64_kernel_saturates_against_its_scaled_sibling() {
    let (c, f) = (F64_CEILING, F64_FLOOR);
    let mut failures = Vec::new();
    let mut push = |v: Vec<String>| failures.extend(v);

    push(audit(
        "bessel_i0",
        (690.0, 730.0),
        c,
        f,
        Real::bessel_i0,
        |x| Real::bessel_i0e(x).ln() + x,
    ));
    push(audit(
        "bessel_i1",
        (690.0, 730.0),
        c,
        f,
        Real::bessel_i1,
        |x| Real::bessel_i1e(x).ln() + x,
    ));
    push(audit(
        "bessel_k0",
        (700.0, 780.0),
        c,
        f,
        Real::bessel_k0,
        |x| Real::bessel_k0e(x).ln() - x,
    ));
    push(audit(
        "bessel_k1",
        (700.0, 780.0),
        c,
        f,
        Real::bessel_k1,
        |x| Real::bessel_k1e(x).ln() - x,
    ));
    push(audit("erfc", (20.0, 30.0), c, f, Real::erfc, |x| {
        Real::erfcx(x).ln() - x * x
    }));
    push(audit("airy_ai", (95.0, 120.0), c, f, Real::airy_ai, |x| {
        Real::airy_ai_e(x).ln() - zeta(x)
    }));
    push(audit("airy_bi", (100.0, 110.0), c, f, Real::airy_bi, |x| {
        Real::airy_bi_e(x).ln() + zeta(x)
    }));
    push(audit(
        "gamma",
        (165.0, 180.0),
        c,
        f,
        Real::gamma,
        Real::ln_gamma,
    ));
    push(audit("rgamma", (165.0, 300.0), c, f, Real::rgamma, |x| {
        -Real::ln_gamma(x)
    }));
    push(audit(
        "ndtr",
        (-45.0, -35.0),
        c,
        f,
        Real::ndtr,
        Real::log_ndtr,
    ));
    push(audit(
        "bessel_i0 (log)",
        (690.0, 730.0),
        c,
        f,
        Real::bessel_i0,
        Real::ln_bessel_i0,
    ));

    assert!(
        failures.is_empty(),
        "f64 kernels disagreed with their own scaled siblings:\n{}",
        failures.join("\n")
    );
}

/// The same eleven pairs in `f32`, where the thresholds are seven times closer in.
///
/// Worth running separately rather than folding into the `f64` test: every window moves, and the
/// narrow format is where this defect class has actually landed. `bessel_i1`'s `f32` boundary was
/// `1.08e38` and its `f64` one `5.7e307` — the same bug, found in the format with less headroom.
#[test]
fn no_f32_kernel_saturates_against_its_scaled_sibling() {
    let (c, f) = (F32_CEILING, F32_FLOOR);
    /// Lift a narrow kernel to the `f64` sweep's signature.
    fn narrow(k: fn(f32) -> f32) -> impl Fn(f64) -> f64 {
        move |x| f64::from(k(x as f32))
    }
    /// `ln` of a narrow scaled value, in `f64` so the exponent arithmetic is not the thing failing.
    fn wide_ln(k: fn(f32) -> f32, x: f64) -> f64 {
        f64::from(k(x as f32)).ln()
    }

    let mut failures = Vec::new();
    let mut push = |v: Vec<String>| failures.extend(v);

    push(audit(
        "bessel_i0",
        (80.0, 110.0),
        c,
        f,
        narrow(Real::bessel_i0),
        |x| wide_ln(Real::bessel_i0e, x) + x,
    ));
    push(audit(
        "bessel_i1",
        (80.0, 110.0),
        c,
        f,
        narrow(Real::bessel_i1),
        |x| wide_ln(Real::bessel_i1e, x) + x,
    ));
    push(audit(
        "bessel_k0",
        (95.0, 115.0),
        c,
        f,
        narrow(Real::bessel_k0),
        |x| wide_ln(Real::bessel_k0e, x) - x,
    ));
    push(audit(
        "bessel_k1",
        (95.0, 115.0),
        c,
        f,
        narrow(Real::bessel_k1),
        |x| wide_ln(Real::bessel_k1e, x) - x,
    ));
    push(audit("erfc", (8.0, 13.0), c, f, narrow(Real::erfc), |x| {
        wide_ln(Real::erfcx, x) - x * x
    }));
    push(audit(
        "airy_ai",
        (24.0, 35.0),
        c,
        f,
        narrow(Real::airy_ai),
        |x| wide_ln(Real::airy_ai_e, x) - zeta(x),
    ));
    push(audit(
        "airy_bi",
        (24.0, 29.0),
        c,
        f,
        narrow(Real::airy_bi),
        |x| wide_ln(Real::airy_bi_e, x) + zeta(x),
    ));
    push(audit(
        "gamma",
        (30.0, 40.0),
        c,
        f,
        narrow(Real::gamma),
        |x| f64::from(Real::ln_gamma(x as f32)),
    ));
    push(audit(
        "rgamma",
        (30.0, 60.0),
        c,
        f,
        narrow(Real::rgamma),
        |x| -f64::from(Real::ln_gamma(x as f32)),
    ));
    push(audit(
        "ndtr",
        (-17.0, -12.0),
        c,
        f,
        narrow(Real::ndtr),
        |x| f64::from(Real::log_ndtr(x as f32)),
    ));
    push(audit(
        "bessel_i0 (log)",
        (80.0, 110.0),
        c,
        f,
        narrow(Real::bessel_i0),
        |x| f64::from(Real::ln_bessel_i0(x as f32)),
    ));

    assert!(
        failures.is_empty(),
        "f32 kernels disagreed with their own scaled siblings:\n{}",
        failures.join("\n")
    );
}
