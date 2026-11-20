//! Cost per **regime**, one row per arm of each kernel's route table.
//!
//! The point of measuring this way is that a piecewise kernel has no single cost: `erfcx` spends
//! 145 recurrence levels just above its crossover and 5 in the far tail, and an average over a
//! sweep would report neither. Each row here names the enum variant it is timing, so a threshold
//! that moves shows up as a row that moves.

use super::*;
use euclid_scalar::Real;

/// `gamma_lr` / `gamma_ur`, over [`IncompleteGammaMethod`]'s arms.
pub(super) fn incomplete_gamma() {
    section("incomplete gamma");
    // `x < a + 1` takes the series; above it the continued fraction. A shape past
    // `GAMMA_EDGEWORTH_MIN = 3e7` takes the Edgeworth normal approximation, which is O(1) rather
    // than O(√a) terms — so it should be *cheaper* than the mid-range series, not dearer.
    row2(
        "gamma_lr",
        "series (x < a+1)",
        2.0,
        1.0,
        <f64 as Real>::gamma_lr,
    );
    row2(
        "gamma_lr",
        "fraction (x > a+1)",
        2.0,
        12.0,
        <f64 as Real>::gamma_lr,
    );
    row2(
        "gamma_lr",
        "series, large a",
        1.0e4,
        1.0e4,
        <f64 as Real>::gamma_lr,
    );
    row2(
        "gamma_lr",
        "edgeworth (a>3e7)",
        1.0e8,
        1.0e8,
        <f64 as Real>::gamma_lr,
    );
    row2(
        "gamma_lr",
        "small shape",
        1.0e-4,
        0.5,
        <f64 as Real>::gamma_lr,
    );
    row2(
        "gamma_ur",
        "fraction tail",
        2.0,
        40.0,
        <f64 as Real>::gamma_ur,
    );
    row1("ln_gamma", "lanczos", 7.5, <f64 as Real>::ln_gamma);
    row1("ln_gamma", "reflected", -7.5, <f64 as Real>::ln_gamma);
    row1("digamma", "recurrence+series", 0.75, <f64 as Real>::digamma);
    row1("digamma", "reflected", -7.25, <f64 as Real>::digamma);
    row1("polygamma_5", "hurwitz euler-maclaurin", 3.0, |x| {
        <f64 as Real>::polygamma(x, 5)
    });
}

/// `erf`/`erfc`, over [`ErfForm`] and [`ErfcForm`].
pub(super) fn error_function() {
    section("error function");
    row1("erf", "linear (|x|<1e-8)", 1.0e-12, <f64 as Real>::erf);
    row1("erf", "incomplete gamma", 0.75, <f64 as Real>::erf);
    row1("erfc", "complement (x<1.375)", 0.5, <f64 as Real>::erfc);
    row1("erfc", "fraction (x>=1.375)", 3.0, <f64 as Real>::erfc);
    row1(
        "erf_inv",
        "central seed + newton",
        0.5,
        <f64 as Real>::erf_inv,
    );
    row1(
        "erf_inv",
        "tail seed + newton",
        0.999,
        <f64 as Real>::erf_inv,
    );
}

/// `erfcx` across [`ErfcxDepth`]'s nine buckets.
///
/// The declared work per bucket is 145, 131, 102, 72, 44, 20, 11, 7 and 5 backward-recurrence
/// levels, so the measured cost should fall monotonically across these rows by about 29x end to
/// end. This row group is the benchmark's own sanity check: the levels are a counted quantity and
/// the timings are not, so a ladder whose costs stop tracking its depths means either the
/// recurrence stopped being the cost or this file stopped measuring what it claims to.
pub(super) fn erfcx_depth_ladder() {
    section("erfcx depth ladder (145, 131, 102, 72, 44, 20, 11, 7, 5 levels)");
    row1("erfcx", "under 3/2    (145)", 1.4, <f64 as Real>::erfcx);
    row1("erfcx", "under 7/4    (131)", 1.6, <f64 as Real>::erfcx);
    row1("erfcx", "under 2      (102)", 1.9, <f64 as Real>::erfcx);
    row1("erfcx", "under 3       (72)", 2.5, <f64 as Real>::erfcx);
    row1("erfcx", "under 5       (44)", 4.0, <f64 as Real>::erfcx);
    row1("erfcx", "under 10      (20)", 7.0, <f64 as Real>::erfcx);
    row1("erfcx", "under 20      (11)", 15.0, <f64 as Real>::erfcx);
    row1("erfcx", "under 50       (7)", 30.0, <f64 as Real>::erfcx);
    row1("erfcx", "at least 50    (5)", 100.0, <f64 as Real>::erfcx);
    // Below the crossover `erfcx` goes through `erfc`'s complement form instead, which is the
    // one arm of the ladder that is not a recurrence at all.
    row1("erfcx", "complement (x<1.375)", 0.5, <f64 as Real>::erfcx);
}

/// The Bessel family, over [`JRoute`], [`YRoute`] and [`IForm`].
pub(super) fn bessel() {
    section("bessel");
    // `J` has the sharpest regime split in the crate: a power series at small argument, aliased
    // trapezoidal quadrature in the middle band, and the divergent Hankel expansion past
    // `HANKEL_MIN`. The quadrature is the expensive one by construction — its node count grows
    // with the argument — and it is the band the other two cannot reach.
    row1(
        "bessel_j0",
        "series (|x|<=2)",
        1.5,
        <f64 as Real>::bessel_j0,
    );
    row1(
        "bessel_j0",
        "aliased quadrature",
        6.0,
        <f64 as Real>::bessel_j0,
    );
    row1(
        "bessel_j0",
        "hankel (|x|>=18)",
        30.0,
        <f64 as Real>::bessel_j0,
    );
    row1("bessel_y0", "series (x<=4)", 1.5, <f64 as Real>::bessel_y0);
    row1("bessel_y0", "march (4..18)", 6.0, <f64 as Real>::bessel_y0);
    row1(
        "bessel_i0",
        "peak-shifted sum",
        5.0,
        <f64 as Real>::bessel_i0,
    );
    row1(
        "bessel_i0",
        "asymptotic (x>=20)",
        500.0,
        <f64 as Real>::bessel_i0,
    );
    row1("bessel_i0e", "scaled, mid", 5.0, <f64 as Real>::bessel_i0e);
    row1("bessel_k0", "series (x<=2)", 0.5, <f64 as Real>::bessel_k0);
    row1(
        "bessel_k0",
        "quadrature (~40 nodes)",
        20.0,
        <f64 as Real>::bessel_k0,
    );
    row1(
        "bessel_i1_over_i0",
        "ratio",
        5.0,
        <f64 as Real>::bessel_i1_over_i0,
    );
    // The two dear rows above are dear *by method*, and neither is a defect. `K` has no
    // asymptotic arm at all — its middle range is a trapezoidal quadrature on the integral
    // representation, about forty nodes of `cosh` and `exp`, chosen because it is the only route
    // that is uniformly accurate there. `Y`'s march integrates Bessel's equation outward from the
    // series, one Taylor step per node. Both buy accuracy in a band where the cheap routes have
    // none, which is the trade their own module headers argue for.
}

/// Airy, over [`AiryRegime`].
pub(super) fn airy() {
    section("airy");
    row1("airy_ai", "series (|x|<=2)", 1.0, <f64 as Real>::airy_ai);
    row1("airy_ai", "march (2..9)", 5.0, <f64 as Real>::airy_ai);
    row1("airy_ai", "K quadrature (>9)", 20.0, <f64 as Real>::airy_ai);
    row1(
        "airy_ai",
        "negative asymptotic",
        -20.0,
        <f64 as Real>::airy_ai,
    );
    row1(
        "airy_ai",
        "leading order (>1e10)",
        1.0e12,
        <f64 as Real>::airy_ai,
    );
    row1("airy_bi", "march", 5.0, <f64 as Real>::airy_bi);
    // Past `MARCH_MAX` the positive axis is *not* an asymptotic expansion: it is the same
    // trapezoidal quadrature `K` uses, reached through `Ai(x) = π⁻¹√(x/3)·K_{1/3}(⅔x^{3/2})`.
    // That is why the row costs about what `bessel_k0`'s quadrature does rather than the handful
    // of terms "asymptotic" would suggest — the label used to say the latter and was wrong.
}

/// Lambert `W` and `ζ`, over [`LambertSeed`] and [`ZetaRoute`].
pub(super) fn transcendental() {
    section("transcendental");
    // The three `W` seeds differ in how many Halley steps they cost, not in accuracy: the
    // measured worst case is 22 steps at the top of the Taylor band against about 5 elsewhere,
    // which `W_STEPS = 60` absorbs. This is where that shows up as time.
    row1(
        "lambert_w",
        "branch point seed",
        -0.3678,
        <f64 as Real>::lambert_w,
    );
    row1("lambert_w", "taylor seed", 1.0, <f64 as Real>::lambert_w);
    row1(
        "lambert_w",
        "taylor seed, top",
        3.0,
        <f64 as Real>::lambert_w,
    );
    row1(
        "lambert_w",
        "logarithmic seed",
        1.0e6,
        <f64 as Real>::lambert_w,
    );
    row1("zeta", "hurwitz (s>1)", 2.5, <f64 as Real>::zeta);
    row1("zeta", "eta accel (strip)", 0.5, <f64 as Real>::zeta);
    row1("zeta", "reflection (s<-1)", -3.5, <f64 as Real>::zeta);
}

/// The non-elementary integrals, over [`EiRoute`], [`E1Route`] and [`TrigIntegralRoute`].
pub(super) fn integrals() {
    section("integrals");
    row1(
        "exp_int_e1",
        "series (x<=2)",
        1.0,
        <f64 as Real>::exp_int_e1,
    );
    row1(
        "exp_int_e1",
        "continued fraction",
        10.0,
        <f64 as Real>::exp_int_e1,
    );
    row1(
        "exp_int_ei",
        "series (x<40)",
        20.0,
        <f64 as Real>::exp_int_ei,
    );
    row1(
        "exp_int_ei",
        "asymptotic (x>=40)",
        60.0,
        <f64 as Real>::exp_int_ei,
    );
    row1("sin_int", "series (x<4)", 2.0, <f64 as Real>::sin_int);
    row1(
        "sin_int",
        "continued fraction",
        20.0,
        <f64 as Real>::sin_int,
    );
    row1(
        "cos_int",
        "continued fraction",
        20.0,
        <f64 as Real>::cos_int,
    );
}

/// The root-finders, which are the most expensive surface in the crate.
///
/// Each call runs a bracketed Newton solve whose every step evaluates the forward CDF — itself a
/// series or continued fraction of `O(√a)` terms — so these rows should be two to three orders
/// above the direct kernels. That is inherent to inversion, not a defect; what the rows are for
/// is catching a regime where the *bracket* search rather than the solve dominates.
pub(super) fn inverses() {
    section("inverses (bracketed newton over a CDF)");
    row2(
        "gamma_lr_inv",
        "mid shape",
        0.5,
        2.0,
        <f64 as Real>::gamma_lr_inv,
    );
    row2(
        "gamma_lr_inv",
        "upper tail",
        0.95,
        2.0,
        <f64 as Real>::gamma_lr_inv,
    );
    row2(
        "gamma_lr_inv",
        "large shape",
        0.5,
        1.0e6,
        <f64 as Real>::gamma_lr_inv,
    );
    row2(
        "gamma_lr_inv",
        "small shape",
        0.5,
        1.0e-3,
        <f64 as Real>::gamma_lr_inv,
    );
    row2("beta_reg_inv", "symmetric", 0.5, 2.0, |p, a| {
        f64::beta_reg_inv(p, a, 3.0)
    });
    row2("beta_reg_inv", "u-shaped", 0.5, 0.1, |p, a| {
        f64::beta_reg_inv(p, a, 0.1)
    });
    row1("ndtri", "central", 0.3, <f64 as Real>::ndtri);
    row1("ndtri", "far tail", 1.0e-300, <f64 as Real>::ndtri);
    row1(
        "erfc_inv",
        "subnormal input",
        f64::from_bits(1),
        <f64 as Real>::erfc_inv,
    );
}
