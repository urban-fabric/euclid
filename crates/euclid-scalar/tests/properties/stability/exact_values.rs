//! Values that are known exactly, at arguments that are representable exactly.
//!
//! # Why this is a distinct law from everything else here
//!
//! Most of this suite compares a kernel against a reference it has to *construct* — an asymptotic
//! expansion, a wider float, a scaled sibling — and so carries that construction's error. A
//! handful of points need none of that: `Γ(½) = √π`, `ζ(2) = π²/6`, `lnΓ(1) = 0`,
//! `Ai(0) = 3^{−2/3}/Γ(⅔)`. The argument is representable, the answer is a closed form, and the
//! only question is how many ulps apart they are.
//!
//! That makes these the sharpest assertions available, and the **zeros** among them sharper still:
//! at a zero there is no relative tolerance to hide in, because any non-zero answer is infinitely
//! wrong. `lnΓ` has exactly two zeros and they are the reason this file exists.
//!
//! # What it found
//!
//! `lnΓ(1)` returned **`−8.88e-16`** where the answer is `ln 1 = 0`, in `f64` only — `f32` routed
//! through a series that happened to be exact there. It propagated: `ln_beta(1, 1)` carried the
//! same value, and a log-density normalisation at shape one — a Gamma that is an Exponential, a
//! Beta that is a uniform — carries it into every term.
//!
//! The repair is a branch rather than a special case, because the neighbourhood was wrong too.
//! Beside a zero the general route delivers an absolute error where a caller needs a relative
//! one, so `lnΓ(1 + ulp)` came back `0` when the answer is `−γ·ulp = −1.28e-16`. The kernel now
//! evaluates `a·S(a)` with `S = lnΓ(1+a)/a` within `SMALL_SHAPE` of either zero, which is exact at
//! `a = 0` and so needs no equality test, and correct to `0.00e0` relative across the
//! neighbourhood where it used to be wrong by everything.
//!
//! # And a tolerance that was passing for the wrong reason
//!
//! Fixing it broke `the_log_beta_matches_its_exact_closed_forms_over_the_whole_range`, which is
//! the useful part. `B(½,½) = Γ(½)²/Γ(1) = π` was inside `1e-15` only because `lnΓ(1)`'s
//! `−8.88e-16` was *subtracted*, adding a compensating term that cancelled part of `lnΓ(½)`'s own
//! `2.5` ulp. Remove the bug and the real `5` ulp appears. **A tolerance that passes because two
//! errors cancel is not measuring what it appears to** — that bound is now `2e-15` with the
//! reason recorded beside it.
//!
//! # The non-finite arguments, which no other sweep reaches
//!
//! `super::range`'s probe set filters `!x.is_finite()`, so nothing in this suite asked what any
//! kernel does at `±∞` or `NaN` until this file. Those answers are exactly specified too — a
//! limit or a `NaN` — and they are asserted below.
//!
//! That found one more: **`I₀(±∞)` and `I₁(±∞)` returned `NaN`** where the limits are `+∞` and
//! `±∞`. The general route reaches infinity through `∞ − ∞` in its peak-factored core, and the
//! plain forms had no infinite branch — while `I₀ᵉ`, `I₁ᵉ`, `ln I₀` and `I₁/I₀` all had one.
//! A kernel whose own scaled siblings handle a limit it does not is the shape of defect this
//! file exists to catch.

use super::*;

/// How far a closed form may sit from its kernel, in ulps of the value.
///
/// `8` covers every row measured. Most are `0.00`; the worst is `B(½,½) = π` at `5.09`, which is
/// `2·lnΓ(½)` doubled by the squaring and is the figure that only became visible once `lnΓ(1)`'s
/// exact zero stopped compensating for it — see the module header. Next worst is
/// `digamma(1) = −γ` at `2.60`.
const EXACT_ULPS: f64 = 8.0;

/// A named single-argument kernel, for the `NaN`-propagation sweep.
type Named = (&'static str, fn(f64) -> f64);

/// Ulps between two finite values of the same sign, for a non-zero reference.
fn ulps_apart(got: f64, want: f64) -> f64 {
    ((got - want) / want).abs() / f64::EPSILON
}

/// Every exactly-known value sits within a few ulps of its closed form, and the zeros are exact.
///
/// The zeros are asserted with `assert_eq!` and the rest with [`EXACT_ULPS`], because those are
/// genuinely different claims: `lnΓ(1) = 0` admits no tolerance at all, while `Γ(½) = √π` is a
/// transcendental the kernel has to land on.
#[test]
fn the_exactly_known_values_are_exact() {
    let pi = core::f64::consts::PI;

    // The two zeros of `lnΓ`, which must be exactly zero. This is the assertion that failed.
    assert_eq!(
        Real::ln_gamma(1.0_f64),
        0.0,
        "lnGamma(1) = ln 1 = 0 exactly"
    );
    assert_eq!(
        Real::ln_gamma(2.0_f64),
        0.0,
        "lnGamma(2) = ln 1 = 0 exactly"
    );
    assert_eq!(Real::ln_gamma(1.0_f32), 0.0, "f32 lnGamma(1) = 0 exactly");
    assert_eq!(Real::ln_gamma(2.0_f32), 0.0, "f32 lnGamma(2) = 0 exactly");
    // And the one that inherits them.
    assert_eq!(
        Real::ln_beta(1.0_f64, 1.0),
        0.0,
        "ln B(1,1) = ln 1 = 0, and it carried lnGamma(1)'s error before that was fixed"
    );

    // Other exact zeros, at arguments where the kernel should not have to approximate.
    assert_eq!(Real::erf(0.0_f64), 0.0, "erf(0) = 0");
    assert_eq!(Real::sin_int(0.0_f64), 0.0, "Si(0) = 0");
    assert_eq!(Real::logit(&0.5_f64), 0.0, "logit(1/2) = ln 1 = 0");
    assert_eq!(Real::bessel_i1(0.0_f64), 0.0, "I1(0) = 0");

    // Exact unit values.
    assert_eq!(Real::erfc(0.0_f64), 1.0, "erfc(0) = 1");
    assert_eq!(Real::erfcx(0.0_f64), 1.0, "erfcx(0) = 1");
    assert_eq!(Real::bessel_i0(0.0_f64), 1.0, "I0(0) = 1");
    assert_eq!(Real::bessel_i0e(0.0_f64), 1.0, "I0e(0) = 1");
    assert_eq!(Real::bessel_j0(0.0_f64), 1.0, "J0(0) = 1");
    assert_eq!(Real::sinc(0.0_f64), 1.0, "sinc(0) = 1");
    assert_eq!(Real::exprel(0.0_f64), 1.0, "exprel(0) = 1");
    assert_eq!(Real::ndtr(0.0_f64), 0.5, "Phi(0) = 1/2");
    assert_eq!(Real::sigmoid(&0.0_f64), 0.5, "sigmoid(0) = 1/2");

    // Closed forms the kernel has to land on rather than return.
    let rows: [(&str, f64, f64); 13] = [
        ("gamma(1/2) = sqrt(pi)", Real::gamma(0.5_f64), pi.sqrt()),
        ("gamma(5) = 4!", Real::gamma(5.0_f64), 24.0),
        ("gamma(11) = 10!", Real::gamma(11.0_f64), 3_628_800.0),
        ("zeta(2) = pi^2/6", Real::zeta(2.0_f64), pi * pi / 6.0),
        ("zeta(4) = pi^4/90", Real::zeta(4.0_f64), pi.powi(4) / 90.0),
        (
            "zeta(6) = pi^6/945",
            Real::zeta(6.0_f64),
            pi.powi(6) / 945.0,
        ),
        (
            "trigamma(1) = pi^2/6",
            Real::trigamma(1.0_f64),
            pi * pi / 6.0,
        ),
        (
            "digamma(1) = -euler",
            Real::digamma(1.0_f64),
            -core::f64::consts::EULER_GAMMA,
        ),
        (
            "airy_ai(0) = 3^(-2/3)/Gamma(2/3)",
            Real::airy_ai(0.0_f64),
            0.355_028_053_887_817_2,
        ),
        (
            "airy_bi(0) = 3^(-1/6)/Gamma(2/3)",
            Real::airy_bi(0.0_f64),
            0.614_926_627_446_000_7,
        ),
        (
            "airy_ai_prime(0) = -3^(-1/3)/Gamma(1/3)",
            Real::airy_ai_prime(0.0_f64),
            -0.258_819_403_792_806_8,
        ),
        (
            "airy_bi_prime(0) = 3^(1/6)/Gamma(1/3)",
            Real::airy_bi_prime(0.0_f64),
            0.448_288_357_353_826_4,
        ),
        ("beta(1/2, 1/2) = pi", Real::beta(0.5_f64, 0.5), pi),
    ];
    let mut failures = Vec::new();
    for (label, got, want) in rows {
        let ulps = ulps_apart(got, want);
        if ulps > EXACT_ULPS {
            failures.push(format!(
                "  {label}: got {got:.17e}, want {want:.17e}, {ulps:.2} ulp"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "closed forms missed by more than {EXACT_ULPS} ulp:\n{}",
        failures.join("\n")
    );
}

/// `lnΓ` is relatively accurate *beside* its zeros, not merely exact at them.
///
/// The half of the repair that a point assertion cannot see. Before the near-zero branch,
/// `lnΓ(1 + ulp)` returned `0` against a true `−γ·ulp`, so the exact zero could have been fixed
/// on its own and left the neighbourhood wrong by everything.
///
/// The reference is the leading behaviour `lnΓ(1+a) = −γa + (π²/12 + γ²/2)a² + ⋯`, which is
/// exact to well under an ulp for the `a` swept here.
#[test]
fn the_log_gamma_is_relatively_accurate_beside_its_zeros() {
    const EULER: f64 = core::f64::consts::EULER_GAMMA;
    /// `π²/12 + γ²/2`, the second coefficient of `lnΓ(1+a)`.
    const SECOND: f64 = 0.822_467_033_424_113_2;

    let mut failures = Vec::new();
    for steps in [1_u64, 2, 8, 64, 1_024, 65_536, 1 << 24] {
        // Around `x = 1`: `lnΓ(1+a) = −γa + c·a²`.
        let x = f64::from_bits(1.0_f64.to_bits() + steps);
        let a = x - 1.0;
        let want = -EULER * a + SECOND * a * a;
        let ulps = ulps_apart(Real::ln_gamma(x), want);
        if ulps > 8.0 {
            failures.push(format!(
                "  lnGamma(1 + {steps} ulp) = {:e}, against {want:e}, {ulps:.2} ulp",
                Real::ln_gamma(x)
            ));
        }
        // Around `x = 2`: `lnΓ(2+b) = ln(1+b) + lnΓ(1+b) = (1−γ)b + (c−½)b² + ⋯`.
        let x = f64::from_bits(2.0_f64.to_bits() + steps);
        let b = x - 2.0;
        let want = (1.0 - EULER) * b + (SECOND - 0.5) * b * b;
        let ulps = ulps_apart(Real::ln_gamma(x), want);
        if ulps > 8.0 {
            failures.push(format!(
                "  lnGamma(2 + {steps} ulp) = {:e}, against {want:e}, {ulps:.2} ulp",
                Real::ln_gamma(x)
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "lnGamma lost relative accuracy beside a zero:\n{}",
        failures.join("\n")
    );
}

/// `ψ` is relatively accurate at and beside its only positive zero.
///
/// The `digamma` counterpart to [`the_log_gamma_is_relatively_accurate_beside_its_zeros`], and the
/// same defect: the general route recurses `x` up to `PSI_ASYMPTOTIC_MIN` accumulating `−1/x` and
/// adds `ln x`, which at `x ≈ 1.46` differences `2.3` against `2.86` to land near zero — so the
/// result carried a flat **absolute** error of about `3ε` however close to the zero it got.
/// Measured before the repair: `ψ(x₀) = 1.2e-15`, and **479% relative** one ulp away.
///
/// The kernel now evaluates a ten-term Taylor series about the zero within `2e-2` of it. The
/// references below come from the same 80-digit generator described on `psi_near_zero`, evaluated
/// at the **exact `f64`** each argument is — not at its shortest decimal repr, which differs by
/// up to an ulp and was enough to make a correct branch look `20%` wrong on first measurement —
/// and then, pasted from that first run, to make the *test* fail against a kernel that was right.
/// Both halves of that mistake are the same rule: the reference belongs at the argument the test
/// actually passes.
///
/// `x₀` is the `f64` nearest the zero, so `ψ(x₀)` is *not* zero: the true zero sits `9.55e-17`
/// above it and `ψ(x₀) = −9.2413e-17`. Asserting that value rather than zero is the point — a
/// kernel returning an exact zero here would be wrong by most of an ulp of the argument. It is
/// also a check on the branch's own constant term, which must equal `ψ′(x₀)` times that offset:
/// `0.96767 × −9.55e-17` reproduces it to every digit.
#[test]
fn the_digamma_is_relatively_accurate_at_its_zero() {
    // (argument, psi at that exact f64, to 80 digits then rounded)
    let rows: [(f64, f64); 10] = [
        (1.461_632_144_968_362_2, -9.241_265_521_729_427e-17),
        (1.461_632_144_968_362_5, 1.224_537_462_200_406_8e-16),
        (1.461_632_144_968_364, 1.626_518_556_281_384_2e-15),
        (1.461_632_144_968_589_6, 2.199_307_824_165_908_3e-13),
        (1.461_632_144_982_914_2, 1.408_139_207_184_821e-11),
        (1.461_632_148_693_652_5, 3.604_859_929_499_651e-9),
        (1.461_632_144_968_362, -3.072_790_566_546_293e-16),
        (1.461_632_144_968_360_5, -1.811_343_866_715_975_7e-15),
        (1.461_632_144_968_134_9, -2.201_156_077_270_712e-13),
        (1.461_632_144_953_810_3, -1.408_157_689_734_616_1e-11),
    ];
    let mut failures = Vec::new();
    for (x, want) in rows {
        let got = Real::digamma(x);
        let ulps = ulps_apart(got, want);
        if ulps > 4.0 {
            failures.push(format!(
                "  digamma({x:?}) = {got:.17e}, want {want:.17e}, {ulps:.2} ulp"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "digamma lost relative accuracy at its zero — the Taylor branch is what keeps it:\n{}",
        failures.join("\n")
    );
    // And the branch must not have cost anything away from the zero.
    for (x, want) in [
        (0.5_f64, -1.963_510_026_021_423_5),
        (1.0, -core::f64::consts::EULER_GAMMA),
        (3.0, 0.922_784_335_098_467_1),
        (100.0, 4.600_161_852_738_088),
    ] {
        let ulps = ulps_apart(Real::digamma(x), want);
        assert!(
            ulps <= 8.0,
            "digamma({x}) is {ulps:.2} ulp out, away from the zero"
        );
    }
}

/// Every kernel's value at `±∞` and `NaN`, which no other sweep in this suite reaches.
///
/// [`super::range`]'s probe set filters `!x.is_finite()` — reasonably, since its subject is the
/// finite format — so these arguments were untested. They are exactly specified: a limit where
/// one exists, `NaN` where it does not, and `NaN` in always propagates.
#[test]
fn the_non_finite_arguments_give_their_limits() {
    let (inf, ninf, nan) = (f64::INFINITY, f64::NEG_INFINITY, f64::NAN);

    // Saturating to a limit.
    assert_eq!(Real::erf(inf), 1.0);
    assert_eq!(Real::erf(ninf), -1.0);
    assert_eq!(Real::erfc(inf), 0.0);
    assert_eq!(Real::erfc(ninf), 2.0);
    assert_eq!(Real::ndtr(inf), 1.0);
    assert_eq!(Real::ndtr(ninf), 0.0);
    assert_eq!(Real::sigmoid(&inf), 1.0);
    assert_eq!(Real::sigmoid(&ninf), 0.0);
    assert_eq!(Real::bessel_j0(inf), 0.0, "the envelope decays to zero");
    assert_eq!(Real::bessel_i1_over_i0(inf), 1.0);
    assert_eq!(Real::bessel_i0e(inf), 0.0);
    assert_eq!(Real::bessel_i1_over_i0(inf), 1.0);
    assert_eq!(Real::bessel_i1_over_i0(ninf), -1.0);
    assert_eq!(Real::bessel_k0(inf), 0.0);
    assert_eq!(Real::airy_ai(inf), 0.0);
    assert_eq!(Real::airy_bi(inf), inf);
    assert_eq!(Real::zeta(inf), 1.0, "zeta(s) -> 1");

    // Growing without bound. `I₀` and `I₁` are here because they were the ones that failed:
    // both returned `NaN`, reaching the limit through `∞ − ∞` in the general route, while their
    // scaled and log siblings branched on infinity correctly.
    assert_eq!(Real::bessel_i0(inf), inf);
    assert_eq!(Real::bessel_i0(ninf), inf, "I0 is even");
    assert_eq!(Real::bessel_i1(inf), inf);
    assert_eq!(Real::bessel_i1(ninf), ninf, "I1 is odd");
    assert_eq!(Real::ln_bessel_i0(inf), inf);
    assert_eq!(Real::gamma(inf), inf);
    assert_eq!(Real::ln_gamma(inf), inf);
    assert_eq!(Real::asinh(inf), inf);
    assert_eq!(Real::acosh(inf), inf);
    assert_eq!(Real::exp_m1(&inf), inf);
    assert_eq!(Real::ln_1p(&inf), inf);

    // No real limit, so `NaN`.
    assert!(Real::gamma(ninf).is_nan(), "Gamma oscillates in sign");
    assert!(Real::ln_gamma(ninf).is_nan());
    assert!(Real::acosh(ninf).is_nan(), "outside the domain");
    assert!(Real::atanh(inf).is_nan());
    assert!(Real::bessel_k0(ninf).is_nan());

    // And `NaN` propagates through everything.
    let kernels: [Named; 16] = [
        ("erf", Real::erf),
        ("erfc", Real::erfc),
        ("erfcx", Real::erfcx),
        ("erf_inv", Real::erf_inv),
        ("ndtr", Real::ndtr),
        ("gamma", Real::gamma),
        ("ln_gamma", Real::ln_gamma),
        ("digamma", Real::digamma),
        ("trigamma", Real::trigamma),
        ("bessel_i0", Real::bessel_i0),
        ("bessel_i1", Real::bessel_i1),
        ("bessel_j0", Real::bessel_j0),
        ("bessel_k0", Real::bessel_k0),
        ("airy_ai", Real::airy_ai),
        ("airy_bi", Real::airy_bi),
        ("lambert_w", Real::lambert_w),
    ];
    for (name, f) in kernels {
        assert!(f(nan).is_nan(), "{name}(NaN) must be NaN");
    }
}
