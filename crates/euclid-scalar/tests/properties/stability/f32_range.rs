//! The range contract in `f32`: no `NaN` inside the domain, and every closed-form bound held.
//!
//! # Why this is not [`super::range`] with a narrower type
//!
//! Because none of the numbers are the same. `f32` runs to `3.4e38` rather than `1.8e308`, its
//! smallest subnormal is `1.4e-45` rather than `4.9e-324`, and its `EPSILON` is `1.19e-7` rather
//! than `2.2e-16`. Every boundary a range contract is *about* therefore sits somewhere different,
//! and the margins are thinner everywhere: an intermediate that has three hundred decades of
//! headroom in `f64` has thirty-eight in `f32`.
//!
//! That is not a hypothetical difference. The crate's documented history of premature-overflow
//! defects is specifically an `f32` history — [`super::range`]'s own prose records `2πx`
//! overflowing inside `I₀ᵉ` "while the *result* is still an entirely ordinary number", which cost
//! `I₀ᵉ(1e38)` its value outright in `f32` and not in `f64`. So the narrow surface is where this
//! class of bug actually lands, and it had no systematic sweep: `f32::MAX` appears in seven test
//! files, always as a spot check, never as a contract over the format.
//!
//! # What the sweep found
//!
//! **Fifty-four kernels, 628 probes, nothing.** No `NaN` anywhere inside a domain, no bound
//! broken. The `I₀ᵉ` defect above is repaired and is pinned by
//! [`the_repaired_f32_overflow_stays_repaired`] in [`super::f32_saturation`] so it cannot return
//! quietly.
//!
//! Fifty-four is also more kernels than the `f64` sweep covers, which is the other reason this
//! file is worth its length: `cross_precision` reaches sixteen `f32` kernels and `src/tests/f32`
//! is organised by family rather than by contract, so most of the narrow surface had never been
//! asked the two cheapest questions there are.

use super::*;

/// A kernel and the predicate for where it has a real value.
type NarrowDomain = (&'static str, fn(f32) -> f32, fn(f32) -> bool);

/// A kernel and its inclusive bounds, with whether the negative half is in scope.
type NarrowBound = (&'static str, fn(f32) -> f32, f32, f32, bool);

/// `x` is a pole of `Γ` and `ψ`: zero or a negative integer.
fn is_nonpositive_integer(x: f32) -> bool {
    x <= 0.0 && x.fract() == 0.0
}

/// No `f32` kernel returns `NaN` for a finite argument inside its domain.
///
/// The cheapest correctness statement available — it needs no reference value, no wider float and
/// no tolerance — applied to the whole narrow surface. `NaN` is permitted only where the function
/// has no real value, and every such case is a named domain predicate below rather than a
/// tolerated exception.
///
/// The domains are where the content is. `Γ` and `ψ` have poles at the non-positive integers;
/// `lnΓ` additionally has none on the alternating negative intervals where `Γ` is negative, but
/// `f32`'s coarser grid means `x.fract() == 0.0` catches integers that `f64` would have resolved
/// as nearby non-integers — so the predicate is written against the value actually passed, not
/// against the real number it approximates.
#[test]
fn no_f32_kernel_returns_nan_where_its_value_is_real() {
    let rows: Vec<NarrowDomain> = vec![
        // Total on the reals.
        ("erf", Real::erf, |_| true),
        ("erfc", Real::erfc, |_| true),
        ("erfcx", Real::erfcx, |_| true),
        ("ndtr", Real::ndtr, |_| true),
        ("log_ndtr", Real::log_ndtr, |_| true),
        ("bessel_i0", Real::bessel_i0, |_| true),
        ("bessel_i1", Real::bessel_i1, |_| true),
        ("bessel_i0e", Real::bessel_i0e, |_| true),
        ("bessel_i1e", Real::bessel_i1e, |_| true),
        ("ln_bessel_i0", Real::ln_bessel_i0, |_| true),
        ("bessel_i1_over_i0", Real::bessel_i1_over_i0, |_| true),
        ("bessel_j0", Real::bessel_j0, |_| true),
        ("bessel_j1", Real::bessel_j1, |_| true),
        ("airy_ai", Real::airy_ai, |_| true),
        ("airy_bi", Real::airy_bi, |_| true),
        ("airy_ai_e", Real::airy_ai_e, |_| true),
        ("airy_bi_e", Real::airy_bi_e, |_| true),
        ("airy_ai_prime", Real::airy_ai_prime, |_| true),
        ("airy_bi_prime", Real::airy_bi_prime, |_| true),
        ("airy_ai_e_prime", Real::airy_ai_e_prime, |_| true),
        ("airy_bi_e_prime", Real::airy_bi_e_prime, |_| true),
        ("sin_int", Real::sin_int, |_| true),
        ("sinc", Real::sinc, |_| true),
        ("exprel", Real::exprel, |_| true),
        ("cos_m1", Real::cos_m1, |_| true),
        ("asinh", Real::asinh, |_| true),
        ("cbrt", Real::cbrt, |_| true),
        ("rgamma", Real::rgamma, |_| true),
        ("trigamma", Real::trigamma, |_| true),
        ("zeta", Real::zeta, |_| true),
        ("sigmoid", |x| Real::sigmoid(&x), |_| true),
        ("ln_sigmoid", |x| Real::ln_sigmoid(&x), |_| true),
        ("ln_1m_sigmoid", |x| Real::ln_1m_sigmoid(&x), |_| true),
        ("ln_1p_exp", |x| Real::ln_1p_exp(&x), |_| true),
        ("exp_m1", |x| Real::exp_m1(&x), |_| true),
        // Positive half only.
        ("bessel_k0", Real::bessel_k0, |x: f32| x > 0.0),
        ("bessel_k1", Real::bessel_k1, |x: f32| x > 0.0),
        ("bessel_k0e", Real::bessel_k0e, |x: f32| x > 0.0),
        ("bessel_k1e", Real::bessel_k1e, |x: f32| x > 0.0),
        ("bessel_y0", Real::bessel_y0, |x: f32| x > 0.0),
        ("bessel_y1", Real::bessel_y1, |x: f32| x > 0.0),
        ("cos_int", Real::cos_int, |x: f32| x > 0.0),
        ("exp_int_e1", Real::exp_int_e1, |x: f32| x > 0.0),
        ("ln_exp_m1", |x| Real::ln_exp_m1(&x), |x: f32| x > 0.0),
        ("exp_int_ei", Real::exp_int_ei, |x: f32| x != 0.0),
        // Genuine domains.
        ("acosh", Real::acosh, |x: f32| x >= 1.0),
        ("atanh", Real::atanh, |x: f32| x > -1.0 && x < 1.0),
        ("gamma", Real::gamma, |x: f32| !is_nonpositive_integer(x)),
        ("digamma", Real::digamma, |x: f32| {
            !is_nonpositive_integer(x)
        }),
        ("erf_inv", Real::erf_inv, |x: f32| (-1.0..=1.0).contains(&x)),
        ("erfc_inv", Real::erfc_inv, |x: f32| {
            (0.0..=2.0).contains(&x)
        }),
        ("ndtri", Real::ndtri, |x: f32| (0.0..=1.0).contains(&x)),
        ("lambert_w", Real::lambert_w, |x: f32| x >= -0.367_879_45),
        (
            "logit",
            |x| Real::logit(&x),
            |x: f32| (0.0..=1.0).contains(&x),
        ),
        ("ln_1m", |x| Real::ln_1m(&x), |x: f32| x <= 1.0),
        ("ln_1p", |x| Real::ln_1p(&x), |x: f32| x >= -1.0),
        ("ln_1m_exp", |x| Real::ln_1m_exp(&x), |x: f32| x <= 0.0),
    ];

    let probes = narrow_format_probes();
    let mut failures = Vec::new();
    for (name, f, in_domain) in &rows {
        for &x in &probes {
            if !x.is_finite() || !in_domain(x) {
                continue;
            }
            if f(x).is_nan() {
                failures.push(format!(
                    "  {name}({x:e}) = NaN, but the value is real there"
                ));
            }
        }
    }

    assert!(
        rows.len() >= 54,
        "the sweep lost kernels: only {} covered",
        rows.len()
    );
    assert!(
        probes.len() >= 600,
        "the probe set shrank to {}",
        probes.len()
    );
    assert!(
        failures.is_empty(),
        "f32 kernels returned NaN inside their domain:\n{}",
        failures.join("\n")
    );
}

/// Every `f32` kernel with a closed-form bound respects it across the whole format.
///
/// A bound is a mathematical fact about the function rather than a property of the
/// implementation, so a violation is unambiguous and needs no tolerance to judge. These catch a
/// class an accuracy test cannot: a probability above one, a `J₀` outside `[−1, 1]`, a negative
/// `Γ` on the positive axis.
///
/// The bounds are **non-strict** for the same reason as in [`super::range`], and the reason bites
/// harder here. `erfc` and `E₁` are strictly positive in exact arithmetic but underflow to zero
/// at a finite argument — and in `f32` that argument is `10.06` and `99.36` rather than `27.2`
/// and `738.6`, so a strict `> 0` would fail roughly twenty-seven times sooner than it would in
/// `f64`. Thinner margins are the whole character of the narrow surface.
#[test]
fn every_bounded_f32_kernel_respects_its_bounds() {
    let rows: Vec<NarrowBound> = vec![
        ("erf", Real::erf, -1.0, 1.0, true),
        ("erfc", Real::erfc, 0.0, 2.0, true),
        ("ndtr", Real::ndtr, 0.0, 1.0, true),
        ("log_ndtr", Real::log_ndtr, f32::NEG_INFINITY, 0.0, true),
        ("bessel_i0e", Real::bessel_i0e, 0.0, 1.0, true),
        ("bessel_i1e", Real::bessel_i1e, -0.5, 0.5, true),
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
        ("sigmoid", |x| Real::sigmoid(&x), 0.0, 1.0, true),
        (
            "ln_sigmoid",
            |x| Real::ln_sigmoid(&x),
            f32::NEG_INFINITY,
            0.0,
            true,
        ),
        (
            "ln_1m_sigmoid",
            |x| Real::ln_1m_sigmoid(&x),
            f32::NEG_INFINITY,
            0.0,
            true,
        ),
        (
            "ln_1p_exp",
            |x| Real::ln_1p_exp(&x),
            0.0,
            f32::INFINITY,
            true,
        ),
        ("exp_m1", |x| Real::exp_m1(&x), -1.0, f32::INFINITY, true),
        // Positive half only.
        ("airy_ai_e", Real::airy_ai_e, 0.0, 0.355_1, false),
        ("gamma", Real::gamma, 0.0, f32::INFINITY, false),
        ("trigamma", Real::trigamma, 0.0, f32::INFINITY, false),
        ("erfcx", Real::erfcx, 0.0, f32::INFINITY, false),
        ("exp_int_e1", Real::exp_int_e1, 0.0, f32::INFINITY, false),
        ("bessel_k0e", Real::bessel_k0e, 0.0, f32::INFINITY, false),
    ];

    let probes = narrow_format_probes();
    let mut failures = Vec::new();
    for (name, f, lower, upper, both_signs) in &rows {
        for &x in &probes {
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
    for &x in &probes {
        if !x.is_finite() || x <= 0.0 {
            continue;
        }
        let s = 1.0 + x;
        let y = Real::zeta(s);
        if y.is_finite() && y < 1.0 {
            failures.push(format!("  zeta({s:e}) = {y:e}, below 1"));
        }
    }

    assert!(rows.len() >= 21, "the sweep lost kernels: {}", rows.len());
    assert!(
        failures.is_empty(),
        "f32 kernels left their documented bounds:\n{}",
        failures.join("\n")
    );
}
