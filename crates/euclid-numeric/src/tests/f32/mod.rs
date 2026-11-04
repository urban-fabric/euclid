//! **Is the `f32` path as good as the `f64` one?**
//!
//! For most of the surface `f32` is not a thin cast over `f64`: `kernels/f32/` is a *separate*
//! implementation, evaluated in single precision throughout, so it can drift independently.
//! That is what this suite is for.
//!
//! The `f64` implementation is the reference throughout, which is sound because it is accurate
//! to ~1e-15 across the whole surface — four orders of magnitude tighter than `f32::EPSILON`
//! (1.19e-7). Tolerances are quoted in ulp of `f32` so they are legible as "how many trailing
//! bits does single precision lose here".
//!
//! [`classical`] holds the exception: the functions that deliberately evaluate in `f64` and
//! narrow once, where agreement is near-exact by construction and the bar is correspondingly
//! tighter.

use crate::Real;
use euclid_utils::{assert_close_f32, rel};

mod classical;

// --- special functions --------------------------------------------------

#[test]
fn f32_erf_family_tracks_f64() {
    // Measured worst: erf 5.5e-7 (~5 ulp), erfc 5.6e-6 (~47 ulp), erfcx 5.5e-6 (~46 ulp).
    // erfc and erfcx are weaker because the mid-range forms complement, as in f64.
    for i in 1..80 {
        let x32 = (i as f32) * 0.1;
        let x64 = f64::from(x32);
        assert_close_f32(Real::erf(x32), Real::erf(x64), 16.0, &format!("erf({x32})"));
        assert_close_f32(
            Real::erfc(x32),
            Real::erfc(x64),
            128.0,
            &format!("erfc({x32})"),
        );
        assert_close_f32(
            Real::erfcx(x32),
            Real::erfcx(x64),
            128.0,
            &format!("erfcx({x32})"),
        );
    }
}

#[test]
fn f32_erf_inv_tracks_f64() {
    // Measured worst 1.1e-6 (~10 ulp).
    for i in 1..20 {
        let p32 = (i as f32) * 0.05;
        assert_close_f32(
            Real::erf_inv(p32),
            Real::erf_inv(f64::from(p32)),
            32.0,
            &format!("erf_inv({p32})"),
        );
    }
}

#[test]
fn f32_gamma_family_tracks_f64() {
    // Measured worst: gamma 2.5e-6 (~21 ulp), digamma 1.6e-6 (~14 ulp),
    // trigamma 1.9e-7 (~2 ulp). ln_gamma is checked separately because it has a zero.
    for i in 1..60 {
        let x32 = (i as f32) * 0.25;
        let x64 = f64::from(x32);
        assert_close_f32(
            Real::gamma(x32),
            Real::gamma(x64),
            64.0,
            &format!("gamma({x32})"),
        );
        assert_close_f32(
            Real::digamma(x32),
            Real::digamma(x64),
            64.0,
            &format!("digamma({x32})"),
        );
        assert_close_f32(
            Real::trigamma(x32),
            Real::trigamma(x64),
            32.0,
            &format!("trigamma({x32})"),
        );
    }
}

#[test]
fn f32_ln_gamma_tracks_f64_including_across_its_zeros() {
    // `ln Γ` vanishes at 1 and 2, so a relative comparison there is meaningless — the mixed
    // criterion in `agrees` is what makes this test possible at all. (A purely relative check
    // reports a 5e8 "error" at x = 1, comparing two different roundings of zero.)
    for i in 1..200 {
        let x32 = (i as f32) * 0.25;
        assert_close_f32(
            Real::ln_gamma(x32),
            Real::ln_gamma(f64::from(x32)),
            64.0,
            &format!("ln_gamma({x32})"),
        );
    }
    // Both carriers put the zeros in the same places.
    assert!(Real::ln_gamma(1.0f32).abs() < 1e-6);
    assert!(Real::ln_gamma(2.0f32).abs() < 1e-6);
}

#[test]
fn f32_incomplete_gamma_and_beta_track_f64() {
    // Measured worst: gamma_lr 7.3e-7 (~6 ulp), beta_reg 1.3e-6 (~11 ulp).
    for &a in &[0.5f32, 1.0, 2.5, 10.0] {
        for i in 1..100 {
            let x32 = (i as f32) * 0.2;
            assert_close_f32(
                Real::gamma_lr(x32, a),
                Real::gamma_lr(f64::from(x32), f64::from(a)),
                64.0,
                &format!("gamma_lr({x32}, {a})"),
            );
        }
    }
    for &(a, b) in &[(1.0f32, 1.0f32), (2.0, 3.0), (5.0, 2.0)] {
        for i in 1..100 {
            let x32 = (i as f32) * 0.01;
            assert_close_f32(
                Real::beta_reg(x32, a, b),
                Real::beta_reg(f64::from(x32), f64::from(a), f64::from(b)),
                64.0,
                &format!("beta_reg({x32}, {a}, {b})"),
            );
        }
    }
}

#[test]
fn f32_bessel_tracks_f64_within_the_shared_polynomial_error() {
    // Both carriers now use the peak-factored series, each evaluated in its own precision, so
    // the f32 path is limited by f32 rather than by a shared approximation error. They
    // previously agreed to ~1 ulp for the wrong reason: both inherited the same 1e-7
    // polynomial inaccuracy.
    for i in 0..200 {
        let x32 = (i as f32) * 0.25;
        let x64 = f64::from(x32);
        assert_close_f32(
            Real::bessel_i0e(x32),
            Real::bessel_i0e(x64),
            16.0,
            &format!("bessel_i0e({x32})"),
        );
        if i > 0 {
            assert_close_f32(
                Real::bessel_i1e(x32),
                Real::bessel_i1e(x64),
                16.0,
                &format!("bessel_i1e({x32})"),
            );
        }
    }
}

#[test]
fn f32_log_domain_helpers_track_f64() {
    // Measured worst: ln_1p_exp 6.7e-8, sigmoid 9.3e-8, ln_1m_exp 8.1e-8 — all under one ulp.
    for i in -60..60 {
        let x32 = (i as f32) * 0.5;
        let x64 = f64::from(x32);
        assert_close_f32(
            Real::ln_1p_exp(&x32),
            Real::ln_1p_exp(&x64),
            8.0,
            &format!("ln_1p_exp({x32})"),
        );
        assert_close_f32(
            Real::sigmoid(&x32),
            Real::sigmoid(&x64),
            8.0,
            &format!("sigmoid({x32})"),
        );
    }
    for i in 1..60 {
        let x32 = -((i as f32) * 0.2);
        assert_close_f32(
            Real::ln_1m_exp(&x32),
            Real::ln_1m_exp(&f64::from(x32)),
            8.0,
            &format!("ln_1m_exp({x32})"),
        );
    }
}

// --- the f32 carrier has the same stability properties ------------------

#[test]
fn f32_softplus_and_sigmoid_do_not_overflow() {
    // f32 overflows at ~3.4e38, so e^x dies by x ≈ 89 — much earlier than f64. The stable
    // forms must still work there.
    let big = 100.0f32;
    assert!(big.exp().is_infinite(), "naive e^x overflows in f32 here");
    assert!(rel(f64::from(Real::ln_1p_exp(&big)), f64::from(big)) < 1e-6);

    for &x in &[-200.0f32, -50.0, 0.0, 50.0, 200.0] {
        let s = Real::sigmoid(&x);
        assert!((0.0..=1.0).contains(&s), "sigmoid({x}) = {s}");
        assert!(Real::ln_1p_exp(&x).is_finite() || x > 1e38);
    }
    // ln sigmoid stays finite where sigmoid itself has underflowed.
    assert_eq!(Real::sigmoid(&-200.0f32), 0.0);
    assert!(rel(f64::from(Real::ln_sigmoid(&-200.0f32)), -200.0) < 1e-5);
}

#[test]
fn f32_ln_add_exp_beats_the_naive_form() {
    let (a, b) = (100.0f32, 100.0f32);
    assert!(
        (a.exp() + b.exp()).ln().is_infinite(),
        "naive log-sum-exp overflows in f32"
    );
    let want = 100.0 + f64::from(core::f32::consts::LN_2);
    assert!(rel(f64::from(Real::ln_add_exp(a, b)), want) < 1e-6);

    // And at the underflow end. Note f32 has subnormals down to ~1.4e-45, so e^-100 is
    // still representable (3.8e-44); it takes about x = -110 for the exponential to reach a
    // true zero.
    assert!(
        (-100.0f32).exp() > 0.0,
        "e^-100 is subnormal in f32, not zero"
    );
    let (c, d) = (-120.0f32, -130.0f32);
    assert_eq!(c.exp() + d.exp(), 0.0, "naive form underflows in f32");
    let got = Real::ln_add_exp(c, d);
    assert!(got.is_finite() && (got - c).abs() < 1.0);
}

#[test]
fn f32_cancellation_helpers_keep_their_digits() {
    // f32::EPSILON is 1.19e-7, so `1 + x` collapses much sooner than in f64.
    let x = 1e-10f32;
    assert_eq!((1.0f32 + x).ln(), 0.0, "naive form collapses in f32");
    assert!(rel(f64::from(Real::ln_1p(&x)), f64::from(x)) < 1e-6);
    assert_eq!(x.exp() - 1.0, 0.0, "naive form collapses in f32");
    assert!(rel(f64::from(Real::exp_m1(&x)), f64::from(x)) < 1e-6);

    // xlogy's absorbing zero holds on f32 too.
    assert!((0.0f32 * 0.0f32.ln()).is_nan());
    assert_eq!(Real::xlogy(0.0f32, 0.0), 0.0);
    assert_eq!(Real::xlog1py(0.0f32, -1.0), 0.0);
}

#[test]
fn f32_erfc_tail_and_erfcx_survive_the_narrower_range() {
    // In f32 the erfc tail underflows by x ≈ 10 rather than 27.
    assert_eq!(
        1.0f32 - Real::erf(5.0f32),
        0.0,
        "naive complement underflows"
    );
    assert!(Real::erfc(5.0f32) > 0.0, "erfc keeps the tail");
    // erfcx stays usable far beyond where erfc has gone to zero.
    assert_eq!(Real::erfc(20.0f32), 0.0);
    let scaled = Real::erfcx(20.0f32);
    assert!(scaled > 0.0 && scaled.is_finite());
    let asym = 1.0 / (20.0 * core::f32::consts::PI.sqrt());
    assert!(rel(f64::from(scaled), f64::from(asym)) < 1e-2);
}

#[test]
fn f32_scaled_bessel_survives_where_the_unscaled_form_overflows() {
    assert!(100.0f32.exp().is_infinite());
    for &x in &[50.0f32, 100.0, 1e4] {
        let v = Real::bessel_i0e(x);
        assert!(v > 0.0 && v.is_finite(), "bessel_i0e({x}) = {v}");
        let asym = 1.0 / (core::f32::consts::TAU * x).sqrt();
        assert!(rel(f64::from(v), f64::from(asym)) < 1e-2);
    }
}

#[test]
fn f32_cdfs_stay_within_the_unit_interval() {
    for i in 0..200 {
        let x = (i as f32) * 0.25;
        for &a in &[0.5f32, 1.0, 2.5, 10.0] {
            let (p, q) = (Real::gamma_lr(x, a), Real::gamma_ur(x, a));
            assert!((0.0..=1.0).contains(&p), "gamma_lr({x},{a}) = {p}");
            assert!((0.0..=1.0).contains(&q), "gamma_ur({x},{a}) = {q}");
            assert!((p + q - 1.0).abs() < 1e-6, "P+Q != 1 at ({x},{a})");
        }
    }
    for i in 0..=100 {
        let x = (i as f32) * 0.01;
        for &(a, b) in &[(0.5f32, 0.5f32), (2.0, 5.0), (20.0, 10.0)] {
            let v = Real::beta_reg(x, a, b);
            assert!((0.0..=1.0).contains(&v), "beta_reg({x},{a},{b}) = {v}");
        }
    }
}

#[test]
fn f32_identities_hold_at_single_precision() {
    for i in -40..40 {
        let x = (i as f32) * 0.25;
        // erf + erfc = 1
        assert!((Real::erf(x) + Real::erfc(x) - 1.0).abs() < 1e-6);
        // sin² + cos² = 1
        let (s, c) = Real::sin_cos(x);
        assert!((s * s + c * c - 1.0).abs() < 1e-6);
        // sigmoid reflection
        assert!((Real::sigmoid(&-x) - (1.0 - Real::sigmoid(&x))).abs() < 1e-6);
        // ln sigmoid = -softplus(-x)
        assert!((Real::ln_sigmoid(&x) + Real::ln_1p_exp(&-x)).abs() < 1e-5);
    }
    // ln Γ recurrence
    for i in 1..100 {
        let x = (i as f32) * 0.25;
        let lhs = Real::ln_gamma(x + 1.0);
        let rhs = Real::ln_gamma(x) + Real::ln(&x);
        assert!(
            (lhs - rhs).abs() < 1e-5 * lhs.abs().max(1.0),
            "f32 ln_gamma recurrence at {x}"
        );
    }
}

#[test]
fn f32_integer_and_natural_bridges_use_the_narrow_real() {
    use crate::{Integer, Natural};
    // The width pairing documented on the traits: i32/u32 bridge to f32, not f64.
    let r: f32 = <i32 as Integer>::to_real(&7);
    assert_eq!(r, 7.0f32);
    let n: f32 = <u32 as Natural>::to_real(&9);
    assert_eq!(n, 9.0f32);
    // Round trip.
    assert_eq!(<i32 as Integer>::from_real(-7.0f32), -7);
    assert_eq!(<u32 as Natural>::from_real(9.0f32), 9);
    // And saturation at the f32 extremes.
    assert_eq!(<i32 as Integer>::from_real(f32::MAX), i32::MAX);
    assert_eq!(<u32 as Natural>::from_real(-1.0f32), 0);
}
