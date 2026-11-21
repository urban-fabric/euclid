//! Where `f32` kernels reach the edge of their format — and that it is not sooner than it must be.
//!
//! # The boundaries are derived, not tabulated
//!
//! [`super::log_domain`] found that five `f64` kernels saturate at one shared argument,
//! `|x| = 745.133`, because all five route through the same `e^{−|x|}` and that is where it
//! reaches half the smallest subnormal. The same five in `f32` saturate at `103.972`. Neither
//! number is interesting on its own; what is interesting is that **both are
//! `ln(2 / smallest_subnormal)` for their own format**, so one formula predicts both:
//!
//! ```text
//! boundary                      formula                 f64        f32
//! log-domain family -> 0/-0     ln(2 / subnormal_min)   745.133    103.972
//! sigmoid -> 1                  ln(2 / EPSILON)          36.737     16.636
//! exp_m1 -> -1                  ln(EPSILON / 4)         -37.430    -17.329
//! exp_m1 -> inf                 ln(MAX)                 709.783     88.723
//! ```
//!
//! Measured, every `f32` entry matches its formula **to all seven digits `f32` has**. So the
//! assertions below are written as the formulas rather than as the constants, which is what makes
//! them a statement about the kernels being format-aware rather than a table of magic numbers
//! someone would have to re-derive. A kernel that hard-coded an `f64` threshold would fail here
//! and nowhere else.
//!
//! # The margins are the reason this matters more in `f32`
//!
//! Every decay boundary arrives roughly seven times sooner in `f32`, and the underflow points are
//! close enough to ordinary arguments to be reached by accident:
//!
//! ```text
//! kernel        f64 first zero    f32 first zero     predicted from the f32 asymptotic
//! erfc               27.24             10.06          e^{-x²}/(x√π) under 1.4e-45
//! ndtr              -38.48            -14.14          e^{-x²/2}/(x√2π) under 1.4e-45
//! bessel_k0         742.68            101.77          √(π/2x)e^{-x} under 1.4e-45
//! airy_ai           107.55             28.56          e^{-ζ}/(2√π x^{1/4}), ζ = ⅔x^{3/2}
//! exp_int_e1        738.58             99.36          e^{-x}/x under 1.4e-45
//! ```
//!
//! Each `f32` value sits slightly *inside* its exponential-only prediction because the algebraic
//! prefactor is below one and pulls the product under the subnormal floor early — which is the
//! correct direction and the same relationship [`super::range`] records for `f64`.

use super::*;

/// A kernel and the argument up to which it owes a non-zero value, with whether to sweep negative.
type NarrowUnderflow = (&'static str, fn(f32) -> f32, f32, bool);

/// A named `f32` kernel with whether the negative half is in scope.
type NarrowSigned = (&'static str, fn(f32) -> f32, bool);

/// A named `f32` kernel with no extra columns.
type NarrowNamed = (&'static str, fn(f32) -> f32);

/// `ln(2 / smallest subnormal)`, where `e^{−|x|}` rounds to zero. The `f32` twin of `745.133`.
fn subnormal_exhausted() -> f32 {
    (2.0 / f64::from(f32::from_bits(1))).ln() as f32
}

/// The scaled and log forms stay finite wherever their value is representable.
///
/// Being finite everywhere **is their purpose** — `I₀ᵉ`, `K₀ᵉ`, `Aiᵉ`, `Biᵉ` and `ln I₀` exist so
/// a caller can work past the argument where the plain form leaves the format — so an infinity
/// from one of them is a contradiction in terms rather than a loss of accuracy.
///
/// `K₁ᵉ` is excluded, as in [`super::range`]: `K₁(x) ~ 1/x` near zero, so its true value really
/// does pass `f32::MAX` at small enough `x` and an infinity there is correct.
#[test]
fn the_f32_scaled_and_log_forms_stay_finite_where_their_value_is_representable() {
    let rows: Vec<NarrowSigned> = vec![
        ("bessel_i0e", Real::bessel_i0e, true),
        ("bessel_i1e", Real::bessel_i1e, true),
        ("ln_bessel_i0", Real::ln_bessel_i0, true),
        ("bessel_i1_over_i0", Real::bessel_i1_over_i0, true),
        ("airy_ai_e", Real::airy_ai_e, true),
        ("airy_bi_e", Real::airy_bi_e, true),
        ("airy_ai_e_prime", Real::airy_ai_e_prime, true),
        ("airy_bi_e_prime", Real::airy_bi_e_prime, true),
        ("erf", Real::erf, true),
        ("erfc", Real::erfc, true),
        ("ndtr", Real::ndtr, true),
        ("bessel_j0", Real::bessel_j0, true),
        ("bessel_j1", Real::bessel_j1, true),
        ("sigmoid", |x| Real::sigmoid(&x), true),
        ("ln_sigmoid", |x| Real::ln_sigmoid(&x), true),
        ("ln_1m_sigmoid", |x| Real::ln_1m_sigmoid(&x), true),
        ("ln_1p_exp", |x| Real::ln_1p_exp(&x), true),
        // Positive half only: `erfcx` and `K₀ᵉ` legitimately overflow on the negatives.
        ("erfcx", Real::erfcx, false),
        ("bessel_k0e", Real::bessel_k0e, false),
    ];

    let mut failures = Vec::new();
    for (name, f, both_signs) in &rows {
        for &x in &narrow_format_probes() {
            if !x.is_finite() || (!both_signs && x <= 0.0) {
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
    assert!(rows.len() >= 19, "the sweep lost kernels: {}", rows.len());
    assert!(
        failures.is_empty(),
        "f32 scaled or log forms left the format early:\n{}",
        failures.join("\n")
    );
}

/// The `f32` overflow this crate once shipped is repaired, and stays repaired.
///
/// [`super::range`] records the defect in prose: `2πx` overflowed inside `I₀ᵉ` "while the
/// *result* is still an entirely ordinary number", and it "cost `I₀ᵉ(1e38)` its value outright in
/// `f32`". Prose is not a test, and a fixed bug with no regression pinning it is a bug waiting for
/// a refactor.
///
/// So this asserts the repair against the closed form rather than against a recorded output.
/// `I₀ᵉ(x) → 1/√(2πx)` for large `x`, and the agreement is exact to `f32`'s precision at every
/// one of these arguments — `I₀ᵉ(1e38) = 3.9894226e-20` against `1/√(2π·1e38) = 3.9894228e-20`.
/// Checking the value and not merely finiteness matters: a kernel that returned `0` instead of
/// `inf` would also be "finite", and would be just as broken.
///
/// `ln I₀` and `I₁/I₀` ride the same intermediate and are included for that reason: `ln I₀(x) → x`
/// and `I₁/I₀(x) → 1` at these arguments, both exactly representable, both cheap to state.
#[test]
fn the_repaired_f32_overflow_stays_repaired() {
    let tau = core::f64::consts::TAU;
    for &x in &[1e20f32, 1e30, 1e35, 1e38, f32::MAX] {
        let scaled = Real::bessel_i0e(x);
        let want = (1.0 / (tau * f64::from(x)).sqrt()) as f32;
        assert!(
            scaled.is_finite() && scaled > 0.0,
            "bessel_i0e({x:e}) = {scaled:e}; the scaled form has no decay left to lose and 2*pi*x \
             must not be formed directly"
        );
        let relative = f64::from((scaled - want) / want).abs();
        assert!(
            relative <= 8.0 * f64::from(f32::EPSILON),
            "bessel_i0e({x:e}) = {scaled:e} against 1/sqrt(2*pi*x) = {want:e}, relative \
             {relative:e}"
        );
        // `ln I₀(x) = x + ln I₀ᵉ(x)`, and the second term is negligible against `x` here.
        assert_eq!(
            Real::ln_bessel_i0(x),
            x,
            "ln_bessel_i0({x:e}) must round to x itself at this magnitude"
        );
        assert_eq!(
            Real::bessel_i1_over_i0(x),
            1.0,
            "bessel_i1_over_i0({x:e}) must saturate to 1, its supremum"
        );
    }
}

/// No `f32` kernel saturates before the format forces it to.
///
/// # The two halves, and why both are formulas
///
/// The first half is the log-domain family, which turns over at
/// [`subnormal_exhausted`] — derived from `f32::from_bits(1)`, so it is the *format's* number and
/// not a transcribed constant. The second is the decaying special functions, whose boundaries come
/// from their own asymptotics and are listed in the module header with their predictions.
///
/// Assertions sit just inside each measured boundary, so the test fails if a kernel gives up early
/// without depending on the exact last representable argument. The `sigmoid → 1` and
/// `exp_m1 → −1` boundaries are **exclusive**, as in [`super::log_domain`]: `ln(2/ε)` is itself
/// the first argument at which the rounding has happened, so asserting *at* it would be asserting
/// that correct rounding has not occurred.
#[test]
fn no_f32_kernel_saturates_before_the_format_forces_it() {
    let mut failures = Vec::new();
    let edge = subnormal_exhausted() - 0.2;

    // The log-domain family, decaying towards zero from the negative side.
    let decaying: Vec<NarrowNamed> = vec![
        ("sigmoid", |x| Real::sigmoid(&x)),
        ("ln_1m_sigmoid", |x| Real::ln_1m_sigmoid(&x)),
        ("ln_1p_exp", |x| Real::ln_1p_exp(&x)),
        ("ln_1m_exp", |x| Real::ln_1m_exp(&x)),
    ];
    for (name, f) in &decaying {
        for i in 0..=400 {
            let x = -(17.0 + (edge - 17.0) * (i as f32) / 400.0);
            if f(x) == 0.0 {
                failures.push(format!(
                    "  {name}({x:e}) = 0 already, but the value is representable to -{edge}"
                ));
                break;
            }
        }
    }
    // `ln_sigmoid` is `ln_1m_sigmoid` mirrored, so it decays on the positive side.
    for i in 0..=400 {
        let x = 17.0 + (edge - 17.0) * (i as f32) / 400.0;
        if Real::ln_sigmoid(&x) == 0.0 {
            failures.push(format!(
                "  ln_sigmoid({x:e}) = -0 already, but the value is representable to {edge}"
            ));
            break;
        }
    }

    // `sigmoid` must not reach `1` before `1 − e^{−x}` rounds to it. Exclusive bound.
    let sigmoid_saturates = (2.0 / f64::from(f32::EPSILON)).ln() as f32;
    for i in 0..400 {
        let x = sigmoid_saturates * (i as f32) / 400.0;
        if Real::sigmoid(&x) >= 1.0 {
            failures.push(format!(
                "  sigmoid({x:e}) = 1 already, but 1 - e^-x is distinguishable until \
                 {sigmoid_saturates}"
            ));
            break;
        }
    }
    // `exp_m1` must not reach `−1` before `e^x` falls below half an ulp of `1`. Exclusive.
    let rounds_to_minus_one = (0.25 * f64::from(f32::EPSILON)).ln() as f32;
    for i in 0..400 {
        let x = rounds_to_minus_one * (i as f32) / 400.0;
        if Real::exp_m1(&x) <= -1.0 {
            failures.push(format!(
                "  exp_m1({x:e}) = -1 already, but e^x is distinguishable from 0 until \
                 {rounds_to_minus_one}"
            ));
            break;
        }
    }

    // The decaying special functions, each to just inside its measured first zero.
    let rows: Vec<NarrowUnderflow> = vec![
        ("erfc", Real::erfc, 10.0, false),
        ("exp_int_e1", Real::exp_int_e1, 99.0, false),
        ("bessel_k0", Real::bessel_k0, 101.0, false),
        ("bessel_k1", Real::bessel_k1, 101.0, false),
        ("airy_ai", Real::airy_ai, 28.0, false),
        // `Φ(x)` underflows on the *left*, so this one is swept negative.
        ("ndtr", Real::ndtr, 14.1, true),
    ];
    for (name, f, boundary, negate) in &rows {
        for i in 1..=2_000u16 {
            let magnitude = f32::from(i) * *boundary / 2_000.0;
            let x = if *negate { -magnitude } else { magnitude };
            if f(x) == 0.0 {
                failures.push(format!(
                    "  {name}({x:e}) = 0, but the value is representable until |x| ~ {boundary}"
                ));
                break;
            }
        }
    }

    // And the scaled forms must never reach zero at all, across the whole format.
    let scaled: Vec<NarrowSigned> = vec![
        ("erfcx", Real::erfcx, false),
        ("bessel_k0e", Real::bessel_k0e, false),
        ("bessel_k1e", Real::bessel_k1e, false),
        ("bessel_i0e", Real::bessel_i0e, true),
        ("airy_ai_e", Real::airy_ai_e, false),
        ("airy_bi_e", Real::airy_bi_e, false),
    ];
    for (name, f, both_signs) in &scaled {
        for &x in &narrow_format_probes() {
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

    assert!(
        failures.is_empty(),
        "f32 kernels saturated before the format did:\n{}",
        failures.join("\n")
    );
}

/// The `f32` saturation boundaries are the format's own numbers, not transcribed `f64` ones.
///
/// The sharpest statement in this file, and the one that would catch a hard-coded threshold. Each
/// boundary below is located by bisection and compared against the formula from the module
/// header — not against a constant written down here. A kernel carrying an `f64` threshold would
/// saturate at `745` instead of `104` and fail immediately; one carrying no threshold at all and
/// relying on the hardware would pass, which is the point.
///
/// All five log-domain members share one boundary because they share one `e^{−|x|}`, so the test
/// also pins that routing: a member drifting off the shared value is the signal that its route
/// changed.
#[test]
fn the_f32_saturation_points_match_their_closed_forms() {
    /// Bisect for where `pred` first holds, between a known-false and known-true argument.
    fn boundary(mut lo: f32, mut hi: f32, pred: impl Fn(f32) -> bool) -> f32 {
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if mid == lo || mid == hi {
                break;
            }
            if pred(mid) { hi = mid } else { lo = mid }
        }
        hi
    }

    let exhausted = subnormal_exhausted();
    let shared: Vec<(&str, f32)> = vec![
        (
            "sigmoid -> 0",
            boundary(-80.0, -200.0, |x| Real::sigmoid(&x) == 0.0),
        ),
        (
            "ln_1m_sigmoid -> -0",
            boundary(-17.0, -200.0, |x| Real::ln_1m_sigmoid(&x) == 0.0),
        ),
        (
            "ln_1p_exp -> 0",
            boundary(-17.0, -200.0, |x| Real::ln_1p_exp(&x) == 0.0),
        ),
        (
            "ln_1m_exp -> -0",
            boundary(-17.0, -200.0, |x| Real::ln_1m_exp(&x) == 0.0),
        ),
    ];
    for (name, at) in &shared {
        assert!(
            (at.abs() - exhausted).abs() <= 0.01,
            "{name} turns over at |{at}|, but ln(2/subnormal_min) for f32 is {exhausted}; a \
             kernel off this shared point has changed route or hard-coded a threshold"
        );
    }
    // The positive-side mirror.
    let ln_sigmoid_at = boundary(17.0, 200.0, |x| Real::ln_sigmoid(&x) == 0.0);
    assert!(
        (ln_sigmoid_at - exhausted).abs() <= 0.01,
        "ln_sigmoid turns over at {ln_sigmoid_at}, expected {exhausted}"
    );

    let sigmoid_one = boundary(10.0, 40.0, |x| Real::sigmoid(&x) == 1.0);
    let predicted_one = (2.0 / f64::from(f32::EPSILON)).ln() as f32;
    assert!(
        (sigmoid_one - predicted_one).abs() <= 0.01,
        "sigmoid reaches 1 at {sigmoid_one}, but ln(2/EPSILON) is {predicted_one}"
    );

    let exp_m1_minus_one = boundary(-10.0, -40.0, |x| Real::exp_m1(&x) == -1.0);
    let predicted_minus_one = (0.25 * f64::from(f32::EPSILON)).ln() as f32;
    assert!(
        (exp_m1_minus_one - predicted_minus_one).abs() <= 0.01,
        "exp_m1 reaches -1 at {exp_m1_minus_one}, but ln(EPSILON/4) is {predicted_minus_one}"
    );

    let exp_m1_overflow = boundary(80.0, 120.0, |x| Real::exp_m1(&x).is_infinite());
    let predicted_overflow = f64::from(f32::MAX).ln() as f32;
    assert!(
        (exp_m1_overflow - predicted_overflow).abs() <= 0.01,
        "exp_m1 overflows at {exp_m1_overflow}, but ln(f32::MAX) is {predicted_overflow}"
    );

    // And the f64 twins differ, which is what makes the formulas load-bearing rather than
    // decorative: a shared hard-coded constant could not satisfy both files.
    assert!(
        exhausted < 110.0 && f64::from(exhausted) > 100.0,
        "the f32 boundary should be near 104, not f64's 745: got {exhausted}"
    );
}
