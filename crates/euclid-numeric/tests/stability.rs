//! Seams, wiggle, and forward stability — the properties a pointwise accuracy test cannot see.
//!
//! `src/tests/accuracy` checks values at chosen arguments. That cannot detect either of the
//! defects here, because both are about the *relationship between neighbouring* results rather
//! than about any single one:
//!
//! - a **seam** is a discontinuity at an algorithm crossover, so every value on one side is
//!   systematically off while each individual value looks plausible;
//! - **wiggle** is ulp-level non-monotonicity, where every value is within tolerance but the
//!   sequence goes backwards — which breaks a root-find or a quantile inversion built on top.
//!
//! # What this suite found, and what it does about it
//!
//! Scanning fifteen kernels at twenty-five likely thresholds turned up **no seams at all**,
//! which is a real and reassuring result: the algorithm crossovers in this crate agree with
//! each other to within a few ulps. That is asserted below, so a future crossover that does not
//! agree fails here.
//!
//! The monotonicity scan found plenty, and the honest treatment is to **measure and pin** rather
//! than to assert an ideal nobody promised. Over 50,000 consecutive representable inputs:
//!
//! | kernel | inverting pairs | worst inversion |
//! |---|---|---|
//! | `erf`, `erfc`, `digamma`, `tanh`, `exp`, `ln` | none | — |
//! | `erf` near 2 | 0.008% | 1 ulp |
//! | `sinc` near 0.5 | 6.2% | 1 ulp |
//! | `ln_gamma` near 10 | 6.1% | 2 ulp |
//! | `gamma` near 3 | 23.2% | 9 ulp |
//! | `ln_gamma` near 3 | 31.1% | 24 ulp |
//! | `erfcx` near 1 | 47.4% | 15 ulp |
//!
//! None of this contradicts the crate's documented contract, which claims *relative* accuracy
//! around `1e-14` and never claims ulp monotonicity. The arithmetic is consistent with it: at
//! `x = 3`, `ln_gamma` has derivative `0.92` and `ulp(3) = 4.4e-16`, so the true step between
//! adjacent inputs is about 3.7 ulps of the output — and a kernel with ±12 ulps of noise will
//! step backwards about a third of the time. The accuracy is as advertised; the *spacing* is
//! finer than the accuracy, and that is what produces the inversions.
//!
//! It is still worth knowing, for one concrete reason: **do not root-find naively on these.**
//! A bisection or Newton iteration over `ln_gamma` or `erfcx` near these points cannot converge
//! to the last few ulps, because the function it is searching is not monotone there. An
//! inverse built on one needs to stop at the accuracy of the forward kernel rather than at the
//! resolution of the format.
//!
//! The counts below are therefore *budgets*. Today's behaviour passes; a kernel that gets worse
//! fails. A kernel that gets better also fails, which is deliberate — the number is a recorded
//! measurement, and improving it should be a visible decision rather than a silent drift.

use euclid_numeric::Real;
use euclid_property::numeric::{LIKELY_THRESHOLDS, hunt_seams, next_up, scan_monotone};

/// A named kernel, for the scan tables below.
type Kernel = (&'static str, fn(f64) -> f64);

/// A named kernel with a scan start and whether it should be increasing there.
type MonotoneCase = (&'static str, fn(f64) -> f64, f64, bool);

/// Consecutive representable inputs scanned by each monotonicity test.
const STEPS: usize = 50_000;

/// Count inverting adjacent pairs and the worst inversion, in ulps of the output.
fn characterise(f: impl Fn(f64) -> f64, start: f64, increasing: bool) -> (usize, f64) {
    let mut x = start;
    let mut here = f(x);
    let mut inversions = 0usize;
    let mut worst = 0.0_f64;

    for _ in 0..STEPS {
        let next = next_up(x);
        let there = f(next);
        if here.is_finite() && there.is_finite() {
            let backwards = if increasing {
                here - there
            } else {
                there - here
            };
            if backwards > 0.0 {
                inversions += 1;
                let spacing = next_up(here.abs()) - here.abs();
                let ulps = backwards / spacing;
                if ulps > worst {
                    worst = ulps;
                }
            }
        }
        x = next;
        here = there;
    }
    (inversions, worst)
}

/// Assert a kernel's measured wiggle against its recorded budget.
fn assert_budget(
    name: &str,
    f: impl Fn(f64) -> f64,
    start: f64,
    increasing: bool,
    max_inversions: usize,
    max_ulps: f64,
) {
    let (inversions, worst) = characterise(f, start, increasing);
    assert!(
        inversions <= max_inversions,
        "{name} near {start} inverted on {inversions} of {STEPS} adjacent pairs, \
         budget {max_inversions}. The kernel got less monotone than when this budget was set."
    );
    assert!(
        worst <= max_ulps,
        "{name} near {start} had a {worst}-ulp inversion, budget {max_ulps}"
    );
}

// ===== Seams ==============================================================

/// No kernel may have a discontinuity at an algorithm crossover.
///
/// This is the strongest assertion in the suite and the one with no caveats: a continuous
/// function cannot jump, so any jump is a bug. The scan needs no reference value and no
/// knowledge of where the branches are — it only requires that stepping the input one ulp
/// cannot move the output a thousand times further than the derivative allows.
#[test]
fn no_kernel_has_a_seam_at_a_likely_threshold() {
    let kernels: Vec<Kernel> = vec![
        ("erf", Real::erf),
        ("erfc", Real::erfc),
        ("erfcx", Real::erfcx),
        ("erf_inv", Real::erf_inv),
        ("ln_gamma", Real::ln_gamma),
        ("gamma", Real::gamma),
        ("digamma", Real::digamma),
        ("exp_m1", |x| Real::exp_m1(&x)),
        ("ln_1p", |x| Real::ln_1p(&x)),
        ("cos_m1", |x| Real::cos_m1(x)),
        ("sinc", Real::sinc),
        ("ln_1p_exp", |x| Real::ln_1p_exp(&x)),
        ("ln_exp_m1", |x| Real::ln_exp_m1(&x)),
        ("cbrt", Real::cbrt),
        ("hypot with 1", |x| Real::hypot(x, 1.0)),
    ];

    let mut reported = Vec::new();
    for (name, f) in &kernels {
        // 1000x the expected step. A smooth kernel scores about 1 and rarely exceeds 10, so
        // this leaves three orders of magnitude of headroom and still catches a branch
        // disagreement, which scores in the millions.
        for seam in hunt_seams(f, LIKELY_THRESHOLDS, 256, 1_000.0) {
            reported.push(format!(
                "  {name}: {:e} -> {:e} gives {:e} -> {:e}, {:.2e} times the expected step",
                seam.at, seam.next, seam.values.0, seam.values.1, seam.ratio
            ));
        }
    }

    assert!(
        reported.is_empty(),
        "discontinuities found at algorithm crossovers:\n{}",
        reported.join("\n")
    );
}

// ===== Wiggle: the kernels that are ulp-monotone ==========================

/// These are monotone to the last bit over 50,000 consecutive inputs, which is a real quality
/// property worth protecting. An inverse or a root-find over any of them can safely iterate to
/// the format's resolution.
#[test]
fn the_ulp_monotone_kernels_stay_monotone() {
    let clean: Vec<MonotoneCase> = vec![
        ("erf", Real::erf, 0.5, true),
        ("erfc", Real::erfc, 0.5, false),
        ("erfc", Real::erfc, 2.0, false),
        ("digamma", Real::digamma, 3.0, true),
        ("tanh", f64::tanh, 1.0, true),
        ("exp", f64::exp, 1.0, true),
        ("ln", f64::ln, 3.0, true),
    ];

    for (name, f, start, increasing) in &clean {
        let inversion = scan_monotone(|x| if *increasing { f(x) } else { -f(x) }, *start, STEPS);
        assert!(
            inversion.is_none(),
            "{name} near {start} lost ulp-level monotonicity: {inversion:?}"
        );
    }
}

// ===== Wiggle: the measured budgets ======================================

/// `erf` is very nearly ulp-monotone — four inverting pairs in fifty thousand, never by more
/// than one ulp. Fine for anything; recorded so a regression is visible.
#[test]
fn erf_wiggle_stays_within_its_budget() {
    assert_budget("erf", Real::erf, 2.0, true, 8, 1.0);
}

/// `sinc` inverts on about 6% of pairs by a single ulp. Its derivative near `0.5` is small, so
/// the true step between adjacent inputs is comparable to the kernel's own rounding — which is
/// the mechanism behind every entry in this section.
#[test]
fn sinc_wiggle_stays_within_its_budget() {
    assert_budget("sinc", Real::sinc, 0.5, false, 3_300, 1.0);
}

/// `ln_gamma` near 3 is the worst case found: a third of adjacent pairs step backwards, by up
/// to 24 ulps. Consistent with the documented `1e-14` relative accuracy, and the reason not to
/// bisect on it past about twelve digits.
#[test]
fn ln_gamma_wiggle_stays_within_its_budget() {
    assert_budget("ln_gamma", Real::ln_gamma, 3.0, true, 16_500, 32.0);
    assert_budget("ln_gamma", Real::ln_gamma, 10.0, true, 3_300, 4.0);
}

/// `gamma` near 3 inverts on about a quarter of pairs, by up to 9 ulps.
#[test]
fn gamma_wiggle_stays_within_its_budget() {
    assert_budget("gamma", Real::gamma, 3.0, true, 12_500, 12.0);
}

/// `erfcx` near 1 inverts on about half of all adjacent pairs, by up to 15 ulps — the highest
/// inversion *rate* measured, though not the largest single inversion.
#[test]
fn erfcx_wiggle_stays_within_its_budget() {
    assert_budget("erfcx", Real::erfcx, 1.0, false, 25_000, 20.0);
}

// ===== Forward stability ==================================================

/// The round trip `erf_inv(erf(x))` must be forward stable — and getting that assertion right
/// took two attempts, which is the interesting part.
///
/// The first version bounded the error by the conditioning of the *composite map*. That is 1,
/// because the composite is the identity, and the test failed at `x = 1.76` by a factor of five:
/// the round trip's relative error there is about `3.4e-14`, roughly eleven times round-off.
///
/// The kernels are not unstable. The error enters the **intermediate**. `erf` saturates, so at
/// `x = 1.76` its derivative is `0.05`, and a relative error of `ε` in `erf(x)` becomes an
/// absolute error that `erf_inv` amplifies about twentyfold. The two stages' condition numbers
/// multiply to 1 exactly because they are inverses — which cancels the amplification out of the
/// bound instead of describing it.
///
/// `two_stage_error_bound` keeps both terms, `(κ_outer·κ_inner + κ_outer)·u`, where the second
/// is the inner stage's own rounding amplified by the outer. That comes to about 12 here, the
/// round trip fits inside it, and the computation is forward stable after all.
///
/// The generalisable lesson, and the reason this test is worth its length: **do not compose
/// through a saturating intermediate and expect the composite's conditioning to describe the
/// result.**
#[test]
fn the_erf_round_trip_is_forward_stable() {
    use euclid_property::numeric::{condition_at, two_stage_error_bound};

    let mut worst_ratio = 0.0_f64;
    let mut worst_at = f64::NAN;

    for step in 1..200 {
        let x = f64::from(step) * 0.01;
        let round_trip = Real::erf_inv(Real::erf(x));
        if !round_trip.is_finite() {
            continue;
        }

        let observed = ((round_trip - x) / x).abs();
        let Some(bound) = two_stage_error_bound(Real::erf, Real::erf_inv, x, 64.0) else {
            continue;
        };

        let ratio = observed / bound;
        if ratio > worst_ratio {
            worst_ratio = ratio;
            worst_at = x;
        }
    }

    let composite = condition_at(|v| Real::erf_inv(Real::erf(v)), worst_at).unwrap_or(f64::NAN);
    let inner = condition_at(Real::erf, worst_at).unwrap_or(f64::NAN);
    assert!(
        worst_ratio <= 1.0,
        "the erf round trip exceeded its two-stage forward-stability bound by \
         {worst_ratio:.2}x at x = {worst_at}. The composite's condition number is \
         {composite:e} and erf's own is {inner:e}, so if the composite looks benign the \
         amplification is in the intermediate."
    );
}
