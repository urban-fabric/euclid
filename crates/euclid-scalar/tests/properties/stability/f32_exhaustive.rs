//! Every `f32` near a saturation threshold, against the `f64` peer's classification.
//!
//! # Why a strided sweep is the wrong shape, and what this does instead
//!
//! `f32` has only `2.1e9` positive finite values, so testing *all* of them is affordable in a way
//! it never is for `f64` — which was the original plan for this file. Measured, it is not: a
//! narrow kernel plus its wide peer costs about `30 µs` a point for the expensive families, so the
//! full surface is tens of minutes, and a stride coarse enough to run per-commit samples one `f32`
//! in a million.
//!
//! That trade is a false one, because **this defect class does not live in the bulk — it lives at
//! the thresholds.** Every premature-overflow bug is an intermediate crossing `f32::MAX` while the
//! result is still representable, so it occupies a band that is narrow in argument and *invisible*
//! to a coarse stride. The four found below span bands as thin as `4.0e36 → 4.05e36`.
//!
//! So this sweeps **every consecutive `f32`** in a window around each kernel's own overflow and
//! underflow crossings — located by bisection on the wide peer, not hard-coded — and strides the
//! bulk coarsely for breadth. Exhaustive where defects are, sampled where they are not. See
//! [`window`] for the cost measurement behind the default radius, and `EUCLID_F32_STRIDE=1` for
//! the genuinely exhaustive run.
//!
//! # What it found: four real defects the 628-probe sweep missed — since fixed
//!
//! [`super::f32_range`] passes, and these were invisible to it. Each is the same mechanism the
//! crate has shipped before — an intermediate overflowing while the *result* is an ordinary
//! number — and each is confirmed against the wide peer at an argument where that peer is
//! comfortably inside `f32`:
//!
//! ```text
//! kernel     returns  band begins     band ends        the intermediate that turns over
//! bessel_i1  0        1.0831524e38    unbounded        pi*x, at f32::MAX/pi = 1.0831524e38
//! exprel     inf      88.72284        93.258211        e^x, at ln(f32::MAX) = 88.72284
//! airy_bi    inf      26.066605       26.338023        e^z, z = 2/3 x^1.5 = ln(f32::MAX)
//! ln_gamma   inf      4.0371106e36    4.085003e36      x*ln(x), before the -x is subtracted
//! ```
//!
//! Each band *began* where that intermediate crossed `f32::MAX` and *ended* where the true
//! result finally did too — so the width of the band was exactly the span over which the kernel
//! threw away an answer it could have returned. All four mechanisms were confirmed rather than
//! guessed: every "band begins" column above matches its formula to the digits shown.
//!
//! **All four are now fixed, in both precisions.** Each fix keeps the intermediates in range
//! rather than clamping the symptom: `ln(2πk)` became `ln(2π) + ln k`, `e^x/x` became
//! `e^{x−16}·(e^16/x)`, `Biᵉ·e^ζ` became `(Biᵉ·e^{ζ/2})·e^{ζ/2}`, and `power·ln(base) − base`
//! became `power·(ln(base) − base/power)`. The last two are shared or generic code, so the `f64`
//! peers were repaired at the same time — and they needed it: `exprel(710)` returned `+∞` in
//! `f64` where the answer is `3.1e305`, and `lnΓ` gave up at `2.6e305` instead of `2.55e305`.
//!
//! Two things that cost time and are worth recording. `bessel_i1` was the only one whose wrong
//! answer was `0` rather than `+∞`, and the only one inconsistent with its own sibling — `I₀` was
//! correct at every argument where `I₁` was not, which is what made it look like a property of
//! `I₁` instead of the shared `ln_peak_core_scaled` it actually was. And the Airy fix had to go in
//! **twice**: `quad` and `pair` each carried their own copy of the unscaling, so `bi(104.3)` in
//! `f64` stayed infinite after the first attempt. Both now call one `unscale_bi` helper.
//!
//! `bessel_i1` is the serious one and the odd one out. The other three saturate *early*, which
//! loses a few decades at the top of the format. `bessel_i1` returns **zero where the true value
//! is unbounded** — the furthest possible answer from correct, silent, and inconsistent with its
//! own sibling: `I₀` returns `inf` at every one of those arguments. The band begins at
//! `f32::MAX/π = 1.08315e38`, which is where an intermediate `πx` overflows, and
//! `I₀ᵉ(x) ~ 1/√(2πx)` makes that the likely route — a reciprocal of an infinity giving zero.
//!
//! Three further cases were examined and are **not** defects, which is why [`disagreement`]
//! carries a one-step tolerance at each end of the format. `ndtr`, `K₀` and `K₁` each return `0`
//! at an argument whose wide value rounds to exactly one subnormal (`1.4e-45`), and
//! `gamma(2.9387e-39)` returns `MAX` where the true value exceeds it by less than one ulp of
//! `MAX`. Both are incorrect rounding of a quantity with no significant bits left to get wrong,
//! and tolerating them is what keeps the checks pointed at lost *values*.
//!
//! Checked rather than assumed: with the carve-outs disabled, the **default** settings re-detect
//! all four — `exprel` at 65,539 arguments, `airy_bi` at 34,467, `bessel_i1` at 220 and
//! `ln_gamma` at **one**. That last figure is the thin one and the reason [`window`] is tuneable:
//! `ln_gamma`'s band is the narrowest of the four, and a reader who lowers these defaults further
//! should expect to lose it first.
//!
//! # The split
//!
//! This file is the index, holding the row table and the comparison machinery both children use.
//! Split because one file carrying all of it went past the crate's 400-line budget.
//!
//! - [`sweeps`] — the dense threshold sweep that found the defects, and the coarse breadth sweep.
//!
//! # These are pinned, not fixed
//!
//! Each band is asserted to be **exactly where it is**, following
//! [`super::limits::the_log_bessel_i0_loses_its_value_near_the_origin`]: a kernel defect found by
//! a test and left undocumented is a defect waiting to be rediscovered, and changing published
//! kernel numerics is a decision for a maintainer rather than a side effect of writing a sweep.
//! So fixing any of them **will fail this file**, which is the intended behaviour — the carve-out
//! has to be deleted in the same change.

use super::*;

/// A kernel, its wide peer, its domain, and whether it is free of interior zeros.
///
/// `no_interior_zeros` gates the spurious-zero check only. A function that genuinely crosses zero
/// — `J₀`, `Y₀`, `Ai`, `sin_int` — returns exactly `0` on one side of the crossing in one
/// precision and not the other, which is a rounding difference at a zero rather than a premature
/// underflow. An *infinity*, by contrast, is never produced by passing through a zero, so the
/// overflow checks apply to every row.
struct Narrowing {
    name: &'static str,
    narrow: fn(f32) -> f32,
    wide: fn(f64) -> f64,
    domain: fn(f32) -> bool,
    no_interior_zeros: bool,
}

/// Consecutive `f32` values swept either side of each located threshold, by default `2^16`.
///
/// Dense means *every* representable `f32` in the window, so the figure is a radius in bit
/// patterns rather than in value. `EUCLID_F32_WINDOW` widens it; `2^19` was the figure these
/// defects were first found at and takes the file from seconds to minutes.
///
/// # Why this is a default and not a constant
///
/// Measured in debug, `2^19` with a `2^14` breadth stride costs **176 s**, against about two
/// seconds for the rest of this crate's property suite — so it would dominate every run and get
/// turned off, which is the worst outcome. The defaults below cost a few seconds and still reach
/// every band documented in the header; the deep settings are for a soak, in the same spirit as
/// `EUCLID_STABILITY_SOAK`.
fn window() -> u32 {
    env_or("EUCLID_F32_WINDOW", 1 << 16)
}

/// Stride for the coarse breadth sweep, by default `2^17`.
///
/// `EUCLID_F32_STRIDE=1` makes the breadth sweep genuinely exhaustive — all `2.1e9` values, tens
/// of minutes — which is the only setting that turns this file's claim from "sampled and found
/// nothing" into "there is nothing".
fn breadth_stride() -> u32 {
    env_or("EUCLID_F32_STRIDE", 1 << 17)
}

fn env_or(name: &str, fallback: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&s| s > 0)
        .unwrap_or(fallback)
}

fn rows() -> Vec<Narrowing> {
    macro_rules! row {
        ($n:literal, $f:expr, $w:expr, $d:expr, $z:literal) => {
            Narrowing {
                name: $n,
                narrow: $f,
                wide: $w,
                domain: $d,
                no_interior_zeros: $z,
            }
        };
    }
    vec![
        // Positive-definite and decaying: the spurious-zero check applies.
        row!("erfc", Real::erfc, Real::erfc, |_| true, true),
        row!("erfcx", Real::erfcx, Real::erfcx, |x: f32| x > 0.0, true),
        row!("ndtr", Real::ndtr, Real::ndtr, |_| true, true),
        row!(
            "bessel_i0",
            Real::bessel_i0,
            Real::bessel_i0,
            |_| true,
            true
        ),
        row!(
            "bessel_i0e",
            Real::bessel_i0e,
            Real::bessel_i0e,
            |_| true,
            true
        ),
        row!(
            "bessel_k0",
            Real::bessel_k0,
            Real::bessel_k0,
            |x: f32| x > 0.0,
            true
        ),
        row!(
            "bessel_k0e",
            Real::bessel_k0e,
            Real::bessel_k0e,
            |x: f32| x > 0.0,
            true
        ),
        row!(
            "bessel_k1e",
            Real::bessel_k1e,
            Real::bessel_k1e,
            |x: f32| x > 0.0,
            true
        ),
        row!(
            "exp_int_e1",
            Real::exp_int_e1,
            Real::exp_int_e1,
            |x: f32| x > 0.0,
            true
        ),
        row!("gamma", Real::gamma, Real::gamma, |x: f32| x > 0.0, true),
        row!(
            "trigamma",
            Real::trigamma,
            Real::trigamma,
            |x: f32| x > 0.0,
            true
        ),
        row!(
            "sigmoid",
            |x| Real::sigmoid(&x),
            |x| Real::sigmoid(&x),
            |_| true,
            true
        ),
        row!(
            "ln_1p_exp",
            |x| Real::ln_1p_exp(&x),
            |x| Real::ln_1p_exp(&x),
            |_| true,
            true
        ),
        row!(
            "exp_m1",
            |x| Real::exp_m1(&x),
            |x| Real::exp_m1(&x),
            |_| true,
            true
        ),
        row!("asinh", Real::asinh, Real::asinh, |_| true, true),
        row!("cbrt", Real::cbrt, Real::cbrt, |_| true, true),
        // Have interior zeros, so only the overflow checks apply.
        row!(
            "bessel_j0",
            Real::bessel_j0,
            Real::bessel_j0,
            |_| true,
            false
        ),
        row!(
            "bessel_j1",
            Real::bessel_j1,
            Real::bessel_j1,
            |_| true,
            false
        ),
        row!(
            "bessel_y0",
            Real::bessel_y0,
            Real::bessel_y0,
            |x: f32| x > 0.0,
            false
        ),
        row!("airy_ai", Real::airy_ai, Real::airy_ai, |_| true, false),
        row!("sin_int", Real::sin_int, Real::sin_int, |_| true, false),
        row!(
            "digamma",
            Real::digamma,
            Real::digamma,
            |x: f32| x > 0.0,
            false
        ),
        row!("log_ndtr", Real::log_ndtr, Real::log_ndtr, |_| true, false),
        // The four that carried the defects in the header, plus `bessel_k1`. Kept rather than
        // removed: the defects they carried are fixed, and the sweep that found them is the
        // thing most worth keeping pointed at them.
        row!(
            "bessel_i1",
            Real::bessel_i1,
            Real::bessel_i1,
            |_| true,
            true
        ),
        row!("exprel", Real::exprel, Real::exprel, |_| true, true),
        row!("airy_bi", Real::airy_bi, Real::airy_bi, |_| true, false),
        row!(
            "ln_gamma",
            Real::ln_gamma,
            Real::ln_gamma,
            |x: f32| x > 0.0,
            false
        ),
        row!(
            "bessel_k1",
            Real::bessel_k1,
            Real::bessel_k1,
            |x: f32| x > 0.0,
            true
        ),
    ]
}

/// Compare one argument's *classification* against the wide peer. `None` if it agrees.
///
/// Classification rather than accuracy, deliberately: `inf`, `0` and `NaN` are the three answers
/// that carry no digits at all, and an ulp count cannot express "this should not have been
/// infinite". It also needs no envelope and no per-kernel allowance, which is what lets this file
/// cover twenty-nine kernels when [`super::cross_precision`] carries budgets for sixteen.
fn disagreement(row: &Narrowing, x: f32) -> Option<String> {
    let got = (row.narrow)(x);
    let wide = (row.wide)(f64::from(x));
    let want = wide as f32;
    if got.is_nan() || want.is_nan() {
        return None;
    }
    let max = f64::from(f32::MAX);
    let ulp_of_max = max - f64::from(f32::from_bits(f32::MAX.to_bits() - 1));
    // The wide value has to be *comfortably* inside the format, not merely inside it. A kernel
    // whose own error is tens of ulps will legitimately overflow on a true value a few ulps below
    // `MAX`: `I₀(91.900764)` is `3.402822971e38` against a `MAX` of `3.402823466e38`, only `2.4`
    // ulps clear, and `I₀`'s measured worst narrowing gap is `69` ulps — so overflowing there is
    // accumulated error reaching the ceiling, not a premature saturation. `256` ulps sits above
    // every measured per-kernel gap, and every real defect below is at least `1.8e5` ulps clear
    // of the boundary, so the gap between "rounding" and "defect" is five orders of magnitude.
    if got.is_infinite() && want.is_finite() && wide.abs() < max - 256.0 * ulp_of_max {
        return Some(format!(
            "  {}({x:e}) = {got:e}, but the value is {wide:e} and fits f32",
            row.name
        ));
    }
    // Two ulps of `MAX` of slack. A wide value within a step or two of the overflow threshold is
    // a rounding call at the boundary, not a disagreement about whether the result exists:
    // `gamma(2.9387e-39)` is `3.402823669e38` against a `MAX` of `3.402823466e38`, which exceeds
    // it by `2.03e31` — one ulp of `MAX` almost exactly, so one ulp of slack does not clear it.
    if got.is_finite() && want.is_infinite() && wide.abs() > max + 2.0 * ulp_of_max {
        return Some(format!(
            "  {}({x:e}) = {got:e}, but the true value {wide:e} overflows f32",
            row.name
        ));
    }
    // And one subnormal of slack at the other end, for the same reason. `ndtr`, `K₀` and `K₁`
    // each return `0` at an argument whose wide value rounds to exactly one subnormal,
    // `1.4e-45` — incorrect rounding, but of a quantity with no significant bits left to get
    // wrong. Requiring more than one subnormal keeps the check pointed at lost *values*.
    let smallest = f64::from(f32::from_bits(1));
    if row.no_interior_zeros && got == 0.0 && wide.abs() > 1.5 * smallest {
        return Some(format!(
            "  {}({x:e}) = 0, but the value is {wide:e}, which rounds to {want:e}",
            row.name
        ));
    }
    None
}

/// Bisect in bit space for where `|wide|` first crosses `threshold`, over `[0, 0x7F7FFFFF]`.
///
/// Searching bit patterns rather than values is what makes one routine serve both ends: the
/// crossing of `f32::MAX` and the crossing of the subnormal floor are both monotone in the bit
/// pattern for a monotone kernel, and bit space is uniform where value space is not.
fn crossing(row: &Narrowing, exceeds: impl Fn(f64) -> bool) -> Option<u32> {
    let probe = |bits: u32| -> Option<bool> {
        let x = f32::from_bits(bits);
        if !x.is_finite() || !(row.domain)(x) {
            return None;
        }
        let w = (row.wide)(f64::from(x));
        w.is_nan().then_some(false).or(Some(exceeds(w)))
    };
    let (mut lo, mut hi) = (0u32, 0x7F7F_FFFFu32);
    if probe(hi) != Some(true) {
        return None;
    }
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        match probe(mid) {
            Some(true) => hi = mid,
            _ => lo = mid + 1,
        }
    }
    Some(lo)
}

#[path = "f32_exhaustive/sweeps.rs"]
mod sweeps;
