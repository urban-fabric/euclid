//! Every odd kernel negates exactly, and every even one repeats exactly. No reference needed.
//!
//! # The cheapest oracle in the suite, and it found a defect
//!
//! `f(−x) = −f(x)` for an odd function and `f(−x) = f(x)` for an even one are **exact** facts, not
//! approximations: they hold in any arithmetic that has a sign bit, because negating an argument
//! and negating a result are both exact operations. So this needs no reference value, no tolerance
//! and no wider float — and it can be asserted at **zero ulps**, which almost nothing else here
//! can.
//!
//! That sharpness is the whole point. `identities::symmetry` already covers most of these
//! kernels, with a tolerance; a tolerance cannot see a function that is *nearly* symmetric, and
//! nearly symmetric is exactly what a one-sided accuracy defect looks like.
//!
//! # What it found: `atanh`, in both precisions
//!
//! Fifteen of the sixteen kernels below are bit-exactly symmetric at every one of 100,000
//! arguments. `atanh` was not:
//!
//! ```text
//!        non-odd arguments   worst relative asymmetry
//! f32    64,192 / 100,000    3.9e-3
//! f64    63,344 / 100,000    4.6e-12
//! ```
//!
//! And the asymmetry was one-sided rather than a rounding tie, so the negative half was simply
//! less accurate — by up to **2500×** in `f32`: `atanh(0.999)` sat `1.9e-9` from the wide value
//! and `atanh(−0.999)` sat `4.8e-6`.
//!
//! The cause was outside this crate: `Real::atanh` delegated straight to `std`, whose `atanh` is
//! not odd. The repair is in [`crate`]'s own implementation — evaluate on `|x|` and sign the
//! result — which restores the exact symmetry *and* hands the negative half the better branch's
//! accuracy. Measured after: both halves report `1.097e-8` at `±0.9` and `1.864e-9` at `±0.999`,
//! identical to the digit, with `atanh(−0.0) = −0.0` and the domain boundaries unmoved.
//!
//! The generalisable part: **a delegated kernel inherits the delegate's defects, and an exact
//! symmetry law is the cheapest way to notice.** Fifteen kernels euclid computes itself were
//! already perfect; the one it handed to someone else was not.
//!
//! # The signed-zero corollary — and a convention this crate has two of
//!
//! `f(−0.0)` is the same law at a single point, and the sweeps above cannot reach it: they start
//! at `i = 1`, because `+0.0 == −0.0` is *true* and an `assert_eq!` at zero passes whatever the
//! sign bit says. Asked directly, four odd kernels returned `+0.0` for `−0.0` in both precisions:
//! `erf_inv`, `I₁`, `I₁ᵉ` and `I₁/I₀`.
//!
//! **Only one of those was a defect.** euclid holds two deliberate and opposite conventions here,
//! and finding that out is the useful part of the exercise:
//!
//! - **The `erf` family signs its zeros.** `erf/f32.rs` says so in as many words — "`if x == 0.0
//!   { 0.0 }` dropped the sign of the zero, which an odd function should not" — and `erf` itself
//!   was fixed accordingly. `erf_inv`, three functions below it, was missed at the time, so it
//!   disagreed with its own family. That one is repaired, with `copysign`.
//! - **The first-kind Bessel family does not.** `zero_is_unsigned_and_the_first_kind_is_odd`
//!   asserts `+0` at *either* zero, and it exists because `cargo mutants` found a `<`-to-`<=`
//!   mutation in `i1e`'s sign step that flipped `+0.0` to `−0.0` and survived every other
//!   assertion. `I₁`, `I₁ᵉ` and `I₁/I₀` are behaving as specified, and an earlier pass of this
//!   file "fixed" them before that test caught the mistake.
//!
//! So the table below encodes **both** conventions rather than imposing one. Which is right is a
//! design question — IEEE-754 and most libms sign the zero of an odd function, so the `erf`
//! family's choice is the conventional one — but it is a maintainer's call, not a sweep's.
//!
//! The mechanisms are worth keeping either way, since both recur. `if x < 0.0 { −m } else { m }`
//! takes the `else` arm for `−0.0`, which is `atanh`'s fold hazard again and why that fix used
//! `is_sign_negative`. And `if x == 0.0 { return 0.0; }` returns the *literal*, discarding the
//! argument's sign — `I₁/I₀` already used `copysign` for its infinite case, reasoning that "zero
//! returned early above, so there is no `−0.0` case to disagree about", which was true and beside
//! the point: the early return was where the sign went.

use super::*;

/// Arguments per kernel, swept over `(0, hi]`.
const SAMPLES: u32 = 100_000;

/// Assert `f(−x) = −f(x)` at zero ulps across `(0, hi]`.
fn assert_odd_f64(name: &str, f: fn(f64) -> f64, hi: f64) {
    for i in 1..=SAMPLES {
        let x = hi * f64::from(i) / f64::from(SAMPLES);
        let (positive, negative) = (f(x), f(-x));
        if !positive.is_finite() || !negative.is_finite() {
            continue;
        }
        assert_eq!(
            negative, -positive,
            "f64 {name} is not odd at x = {x:e}: f(-x) = {negative:e} against -f(x) = {:e}. \
             An odd function negates exactly in any arithmetic with a sign bit, so this is a \
             one-sided evaluation rather than a rounding difference.",
            -positive
        );
    }
}

/// The `f32` peer of [`assert_odd_f64`].
fn assert_odd_f32(name: &str, f: fn(f32) -> f32, hi: f32) {
    for i in 1..=SAMPLES {
        let x = hi * (i as f32) / (SAMPLES as f32);
        let (positive, negative) = (f(x), f(-x));
        if !positive.is_finite() || !negative.is_finite() {
            continue;
        }
        assert_eq!(
            negative, -positive,
            "f32 {name} is not odd at x = {x:e}: f(-x) = {negative:e} against -f(x) = {:e}",
            -positive
        );
    }
}

/// Assert `f(−x) = f(x)` at zero ulps across `(0, hi]`.
fn assert_even_f64(name: &str, f: fn(f64) -> f64, hi: f64) {
    for i in 1..=SAMPLES {
        let x = hi * f64::from(i) / f64::from(SAMPLES);
        let (positive, negative) = (f(x), f(-x));
        if !positive.is_finite() || !negative.is_finite() {
            continue;
        }
        assert_eq!(
            negative, positive,
            "f64 {name} is not even at x = {x:e}: f(-x) = {negative:e} against f(x) = {positive:e}"
        );
    }
}

/// The `f32` peer of [`assert_even_f64`].
fn assert_even_f32(name: &str, f: fn(f32) -> f32, hi: f32) {
    for i in 1..=SAMPLES {
        let x = hi * (i as f32) / (SAMPLES as f32);
        let (positive, negative) = (f(x), f(-x));
        if !positive.is_finite() || !negative.is_finite() {
            continue;
        }
        assert_eq!(
            negative, positive,
            "f32 {name} is not even at x = {x:e}: f(-x) = {negative:e} against f(x) = {positive:e}"
        );
    }
}

/// Every odd kernel negates exactly, in both precisions.
///
/// `atanh` leads the list because it is the one that failed; see the module header. Its window
/// stops just inside `1` because the function diverges there, and the sweep needs finite values on
/// both sides to compare.
#[test]
fn every_odd_kernel_negates_exactly() {
    assert_odd_f64("atanh", Real::atanh, 0.999_999);
    assert_odd_f32("atanh", Real::atanh, 0.999_999);
    assert_odd_f64("asinh", Real::asinh, 1e8);
    assert_odd_f32("asinh", Real::asinh, 1e8);
    assert_odd_f64("erf", Real::erf, 6.0);
    assert_odd_f32("erf", Real::erf, 6.0);
    assert_odd_f64("erf_inv", Real::erf_inv, 0.999_999);
    assert_odd_f32("erf_inv", Real::erf_inv, 0.999_999);
    assert_odd_f64("cbrt", Real::cbrt, 1e8);
    assert_odd_f32("cbrt", Real::cbrt, 1e8);
    assert_odd_f64("sin_int", Real::sin_int, 30.0);
    assert_odd_f32("sin_int", Real::sin_int, 30.0);
    // `I₁` and its relatives are odd; the plain form's window stops short of its overflow.
    assert_odd_f64("bessel_i1", Real::bessel_i1, 700.0);
    assert_odd_f32("bessel_i1", Real::bessel_i1, 80.0);
    assert_odd_f64("bessel_i1e", Real::bessel_i1e, 1e6);
    assert_odd_f32("bessel_i1e", Real::bessel_i1e, 1e6);
    assert_odd_f64("bessel_i1_over_i0", Real::bessel_i1_over_i0, 1e6);
    assert_odd_f32("bessel_i1_over_i0", Real::bessel_i1_over_i0, 1e6);
    assert_odd_f64("bessel_j1", Real::bessel_j1, 30.0);
    assert_odd_f32("bessel_j1", Real::bessel_j1, 30.0);
}

/// Every even kernel repeats exactly, in both precisions.
///
/// Fewer surprises available here than among the odd ones — an even kernel that reduces on `|x|`
/// is even by construction, and most of these do. Asserted anyway, at zero ulps, because the
/// cost is a loop and the alternative is assuming.
#[test]
fn every_even_kernel_repeats_exactly() {
    assert_even_f64("bessel_i0", Real::bessel_i0, 700.0);
    assert_even_f32("bessel_i0", Real::bessel_i0, 80.0);
    assert_even_f64("bessel_i0e", Real::bessel_i0e, 1e6);
    assert_even_f32("bessel_i0e", Real::bessel_i0e, 1e6);
    assert_even_f64("ln_bessel_i0", Real::ln_bessel_i0, 1e6);
    assert_even_f32("ln_bessel_i0", Real::ln_bessel_i0, 1e6);
    assert_even_f64("bessel_j0", Real::bessel_j0, 30.0);
    assert_even_f32("bessel_j0", Real::bessel_j0, 30.0);
    assert_even_f64("cos_m1", Real::cos_m1, 30.0);
    assert_even_f32("cos_m1", Real::cos_m1, 30.0);
    assert_even_f64("sinc", Real::sinc, 30.0);
    assert_even_f32("sinc", Real::sinc, 30.0);
}

/// Each odd kernel's behaviour at `±0.0`, per the convention its family actually holds.
///
/// Checked separately from the sweeps above because they cannot reach it: `+0.0 == -0.0` is
/// *true*, so an `assert_eq!` at `x = 0` passes whatever the sign bit says. This asks about the
/// bit directly.
///
/// **Two conventions, both deliberate** — see the module header. The `erf` family signs its
/// zeros; the first-kind Bessel family returns `+0` at either. The second group is listed here
/// anyway, asserting the *unsigned* behaviour, so that a change in either direction is visible
/// from one place rather than only from a kernel's own unit tests.
#[test]
fn every_odd_kernel_handles_a_signed_zero_as_its_family_specifies() {
    macro_rules! odd_at_zero {
        ($name:literal, $wide:expr, $narrow:expr) => {{
            let wide: fn(f64) -> f64 = $wide;
            let narrow: fn(f32) -> f32 = $narrow;
            assert_eq!(wide(0.0), 0.0, "f64 {} should be zero at zero", $name);
            assert!(
                wide(0.0).is_sign_positive(),
                "f64 {}(+0.0) must be +0.0, not -0.0",
                $name
            );
            assert_eq!(wide(-0.0), 0.0, "f64 {} should be zero at -0.0", $name);
            assert!(
                wide(-0.0).is_sign_negative(),
                "f64 {}(-0.0) must be -0.0: an odd function carries the sign of a zero, and \
                 `x < 0.0` is false for -0.0 while `copysign` is not",
                $name
            );
            assert!(
                narrow(0.0).is_sign_positive() && narrow(-0.0).is_sign_negative(),
                "f32 {} lost the sign of a zero",
                $name
            );
        }};
    }

    odd_at_zero!("erf", Real::erf, Real::erf);
    odd_at_zero!("erf_inv", Real::erf_inv, Real::erf_inv);
    odd_at_zero!("asinh", Real::asinh, Real::asinh);
    odd_at_zero!("atanh", Real::atanh, Real::atanh);
    odd_at_zero!("cbrt", Real::cbrt, Real::cbrt);
    odd_at_zero!("sin_int", Real::sin_int, Real::sin_int);
    odd_at_zero!("bessel_j1", Real::bessel_j1, Real::bessel_j1);
    odd_at_zero!("exp_m1", |x| Real::exp_m1(&x), |x| Real::exp_m1(&x));
    odd_at_zero!("ln_1p", |x| Real::ln_1p(&x), |x| Real::ln_1p(&x));

    // The first-kind Bessels, which return `+0` at either zero by this crate's choice.
    for (name, wide, narrow) in [
        (
            "bessel_i1",
            Real::bessel_i1 as fn(f64) -> f64,
            Real::bessel_i1 as fn(f32) -> f32,
        ),
        ("bessel_i1e", Real::bessel_i1e, Real::bessel_i1e),
        (
            "bessel_i1_over_i0",
            Real::bessel_i1_over_i0,
            Real::bessel_i1_over_i0,
        ),
    ] {
        for signed_zero in [0.0_f64, -0.0] {
            let got = wide(signed_zero);
            assert_eq!(got, 0.0, "f64 {name}({signed_zero}) must be zero");
            assert!(
                got.is_sign_positive(),
                "f64 {name}({signed_zero}) must be +0: the first-kind family returns an \
                 unsigned zero, per zero_is_unsigned_and_the_first_kind_is_odd. If that \
                 convention changed, change it there too."
            );
        }
        for signed_zero in [0.0_f32, -0.0] {
            assert!(
                narrow(signed_zero).is_sign_positive(),
                "f32 {name}({signed_zero}) must be +0, as in f64"
            );
        }
    }
}

/// `erfc(t) + erfc(−t) = 2` to **zero** ulps, because the negatives reflect.
///
/// This law used to hold at one ulp for the wrong reason. `erfc` evaluated both signs
/// independently — `1 − erf(t)` and `1 − erf(−t)` — so the sum was `2` with `erf`'s error
/// cancelling *identically*. That is not accuracy, it is agreement, and it broke the moment the
/// positive side was made more accurate than the negative one: deepening the continued fraction
/// drifted this to one or two ulp and failed `identities::normal`'s reflection property.
///
/// `erfc` now computes `2 − erfc(|x|)` on the negatives, which makes the sum exact by
/// construction and keeps it exact however good `erfc(t)` gets. Measured over 200,000 arguments
/// spanning `[0, 40]`: **0.000 ulp**, so it is asserted at zero rather than at one.
///
/// The same structure carries `Φ(x) + Φ(−x) = 1`, since `Φ(x) = erfc(−x/√2)/2`.
#[test]
fn the_complementary_error_function_reflects_exactly() {
    for i in 0..=20_000 {
        let t = 40.0 * f64::from(i) / 20_000.0;
        let sum = Real::erfc(t) + Real::erfc(-t);
        assert_eq!(
            sum, 2.0,
            "erfc({t}) + erfc(-{t}) must be exactly 2: the negatives reflect as 2 - erfc(|x|), \
             so this is structural and does not depend on erfc's own accuracy"
        );
        let reflected = Real::ndtr(t) + Real::ndtr(-t);
        assert_eq!(
            reflected, 1.0,
            "Phi({t}) + Phi(-{t}) must be exactly 1, by the same reflection"
        );
    }
}

/// `atanh`'s repair holds at the edges the fold could have broken.
///
/// Folding to `|x|` and signing afterwards is only correct if the sign is carried by the *sign
/// bit* rather than by a comparison — `atanh(−0.0)` must stay `−0.0`, not become `+0.0`, and the
/// domain boundaries must survive. Pinned separately because an implementation using `self < 0.0`
/// instead of `is_sign_negative` passes every sweep above and loses the signed zero.
#[test]
fn the_atanh_fold_preserves_its_edges() {
    assert_eq!(Real::atanh(0.0_f64), 0.0);
    assert!(Real::atanh(-0.0_f64).is_sign_negative());
    assert_eq!(Real::atanh(-0.0_f64), 0.0, "-0.0 compares equal to 0.0");
    assert_eq!(Real::atanh(1.0_f64), f64::INFINITY);
    assert_eq!(Real::atanh(-1.0_f64), f64::NEG_INFINITY);
    assert!(Real::atanh(2.0_f64).is_nan());
    assert!(Real::atanh(-2.0_f64).is_nan());

    assert_eq!(Real::atanh(0.0_f32), 0.0);
    assert!(Real::atanh(-0.0_f32).is_sign_negative());
    assert_eq!(Real::atanh(1.0_f32), f32::INFINITY);
    assert_eq!(Real::atanh(-1.0_f32), f32::NEG_INFINITY);
    assert!(Real::atanh(2.0_f32).is_nan());
    assert!(Real::atanh(-2.0_f32).is_nan());
}
