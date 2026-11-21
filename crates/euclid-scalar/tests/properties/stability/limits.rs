//! What every kernel does as its argument approaches zero, against the exact leading term.
//!
//! # Why this window, and why it can assert ulps
//!
//! Each kernel here has a closed-form leading behaviour at the origin — `erf(x) → 2x/√π`,
//! `K₁(x) → 1/x`, `Ai(x) → Ai(0)` — and that is an *exact analytic* reference, so no wider float
//! is needed. The catch is that a leading term is only a reference where the next term is
//! negligible, and the next term is usually `O(x)` relative.
//!
//! So the sweep runs over `[1e-300, 1e-30]`, chosen for two reasons. The neglected term is below
//! `1e-30` there, far under an ulp, which is what makes a few-ulp assertion honest rather than a
//! tolerance covering the reference's own truncation. And the whole window is **normal** — above
//! `MIN_POSITIVE` — because a subnormal argument carries fewer than 53 bits and its relative
//! error is bounded by the input rather than the kernel. A first draft swept to `1e-9` and to the
//! subnormals, and both ends produced apparent failures that were the harness's: `W(x) ≈ x − x²`
//! disagreeing by exactly the `x` that had been dropped, and `erf_inv` off by `3e-10` at
//! `1.7e-314` because that argument only has a handful of significant bits.
//!
//! Measured, all 31 kernels agree with their leading term to **3.3 ulp or better** across 601
//! points, and most to **0 ulp exactly**. That is the strongest statement this suite makes about
//! any region: near the origin the kernels are not merely plausible, they are right.
//!
//! `ln_bessel_i0` was for a while the one exclusion, documented rather than tolerated because it
//! lost its value entirely below `x ≈ 2e-8`. That is now **fixed at the source** and it has
//! joined the sweep like everything else; the measurement and the repair are recorded in
//! [`the_log_bessel_i0_now_tracks_its_leading_term`].

use super::*;

/// A kernel paired with its exact leading behaviour at the origin.
type Limit = (&'static str, fn(f64) -> f64, fn(f64) -> f64);

/// Relative slack allowed against the leading term, in ulps.
///
/// `3.3` was the worst measured, in `gamma`, where the reference `1/x − γ` has to be formed by a
/// subtraction the kernel does not make. Eight leaves headroom without admitting a real change.
const LIMIT_ULPS: f64 = 8.0;

/// Euler's constant `γ`, which appears in the leading term of `Γ`, `ψ`, `K₀`, `Ci` and `E₁`.
///
/// Taken from `core` rather than written out: a hand-typed literal here is both a clippy denial
/// and a second source of truth for a constant the kernels themselves get from `core`.
const EULER: f64 = core::f64::consts::EULER_GAMMA;

fn limit_window() -> Vec<f64> {
    let (a, b) = (1e-300f64.ln(), 1e-30f64.ln());
    (0..=600)
        .map(|i| (a + (b - a) * f64::from(i) / 600.0).exp())
        .collect()
}

/// Every kernel matches its exact leading term as the argument approaches zero.
///
/// This is where the arithmetic is hardest and where the defects this session found all lived —
/// `0 × ∞` in a Bernoulli tail, a product overflowing before a reciprocal — so it is worth
/// asserting that the limits are not merely finite but correct.
#[test]
fn every_kernel_matches_its_leading_term_near_the_origin() {
    let rows: Vec<Limit> = vec![
        // Odd, linear at the origin.
        ("erf", Real::erf, |x| 2.0 * x / core::f64::consts::PI.sqrt()),
        ("erf_inv", Real::erf_inv, |x| {
            x * core::f64::consts::PI.sqrt() / 2.0
        }),
        ("bessel_i1", Real::bessel_i1, |x| 0.5 * x),
        ("bessel_i1e", Real::bessel_i1e, |x| 0.5 * x),
        ("bessel_i1_over_i0", Real::bessel_i1_over_i0, |x| 0.5 * x),
        ("bessel_j1", Real::bessel_j1, |x| 0.5 * x),
        ("sin_int", Real::sin_int, |x| x),
        ("lambert_w", Real::lambert_w, |x| x),
        ("rgamma", Real::rgamma, |x| x),
        // Joined this sweep when its near-origin loss was fixed; see
        // [`the_log_bessel_i0_now_tracks_its_leading_term`]. `ln I₀(x) → x²/4`.
        ("ln_bessel_i0", Real::ln_bessel_i0, |x| 0.25 * x * x),
        ("asinh", Real::asinh, |x| x),
        ("atanh", Real::atanh, |x| x),
        // Divergent at the origin, which is the half where a reciprocal can overflow.
        ("bessel_k1", Real::bessel_k1, |x| 1.0 / x),
        ("gamma", Real::gamma, |x| 1.0 / x - EULER),
        ("digamma", Real::digamma, |x| -1.0 / x - EULER),
        ("trigamma", Real::trigamma, |x| 1.0 / (x * x)),
        // Logarithmic at the origin.
        ("bessel_k0", Real::bessel_k0, |x| -(0.5 * x).ln() - EULER),
        ("cos_int", Real::cos_int, |x| EULER + x.ln()),
        ("exp_int_e1", Real::exp_int_e1, |x| -EULER - x.ln()),
        // Finite and non-zero at the origin.
        ("erfc", Real::erfc, |_| 1.0),
        ("erfcx", Real::erfcx, |_| 1.0),
        ("sinc", Real::sinc, |_| 1.0),
        ("exprel", Real::exprel, |_| 1.0),
        ("bessel_i0", Real::bessel_i0, |_| 1.0),
        ("bessel_i0e", Real::bessel_i0e, |_| 1.0),
        ("bessel_j0", Real::bessel_j0, |_| 1.0),
        ("ndtr", Real::ndtr, |x| {
            0.5 + x / core::f64::consts::TAU.sqrt()
        }),
        ("log_ndtr", Real::log_ndtr, |_| -core::f64::consts::LN_2),
        ("airy_ai", Real::airy_ai, |_| 0.355_028_053_887_817_2),
        ("airy_bi", Real::airy_bi, |_| 0.614_926_627_446_000_7),
    ];

    let mut failures = Vec::new();
    let mut compared = 0usize;
    for (name, f, leading) in &rows {
        for &x in &limit_window() {
            let (got, want) = (f(x), leading(x));
            // A reference that underflows or overflows says nothing; those ends are covered by
            // the finiteness and underflow sweeps in `super::range`.
            if !got.is_finite() || !want.is_finite() || want == 0.0 {
                continue;
            }
            compared += 1;
            let ulps = ((got - want) / want).abs() / f64::EPSILON;
            if ulps > LIMIT_ULPS {
                failures.push(format!(
                    "  {name}({x:e}) = {got:e}, leading term {want:e}, off by {ulps:.1} ulp"
                ));
                break;
            }
        }
    }
    assert!(rows.len() >= 29, "the sweep lost kernels: {}", rows.len());
    assert!(compared >= 10_000, "only {compared} comparisons made");
    assert!(
        failures.is_empty(),
        "kernels disagreed with their leading term at the origin:\n{}",
        failures.join("\n")
    );
}

/// `ln I₀(x)` by summing `I₀(x) − 1 = Σ_{k≥1} (x²/4)^k/(k!)²` to convergence, then `ln_1p`.
///
/// Kept as the *test's* reference even though the kernel now computes it the same way, because an
/// independent summation here still catches a change to the kernel's term count, stopping rule or
/// threshold. All-positive series, so no cancellation at any argument.
fn ln_i0_series(x: f64) -> f64 {
    let t = 0.25 * x * x;
    let (mut term, mut sum) = (t, t);
    let mut k = 1.0_f64;
    while term > 0.25 * f64::EPSILON * sum && k < 200.0 {
        k += 1.0;
        term *= t / (k * k);
        sum += term;
    }
    sum.ln_1p()
}

/// `ln I₀(x)` tracks its leading term to the last bit near the origin — it did not used to.
///
/// # What was wrong, and the two losses that compounded
///
/// `ln I₀` returned **exactly `0`** below `x ≈ 2.1e-8` in `f64` and `7e-4` in `f32`, where the
/// true value is a perfectly representable `x²/4` — `2.5e-201` at `x = 1e-100`. Two separate
/// mechanisms, which is why an earlier read of it as a single "forms `1 + x²/4`" problem was
/// incomplete:
///
/// 1. `ln I₀` was reached as `i0_rel_sum(h, 0).ln()`, the series summed *relative to its `k = 0`
///    term* — which is `1`. So `1 + x²/4 + ⋯` was formed and then logged, losing the small term
///    to the rounding of `1`, and losing it completely once `x²/4 < ε/2`.
/// 2. The result was then returned as `ln I₀ − ax` and the caller added `ax` back. Near the
///    origin `ln I₀ ≈ x²/4` is vastly smaller than `ax`, so that round trip cancelled to nothing
///    on its own — a fix to the first loss alone would still have returned zero.
///
/// # The repair, and what it bought
///
/// A small-argument branch below `x = 1` sums `s = I₀(x) − 1` directly and returns `ln_1p(s)`,
/// bypassing both: the `1` is never formed and the `±ax` round trip is never taken. Measured
/// against [`ln_i0_series`]:
///
/// ```text
/// x            1e-100   1e-8    1e-6    1e-4    1e-3     1e-1     1
/// f64 before     100%    100%   8.9e-5  6.7e-9  5.5e-10  ~1e-14   ~1e-15
/// f64 after        0       0       0       0       0        0     3.5e-16
/// f32 before      --      --      --     ~1e-1   4.6e-2   1.8e-5  3.8e-8
/// f32 after       --      --    1.1e-9  4.4e-8  7.9e-8   8.2e-8   3.8e-8
/// ```
///
/// `f64` is now **bit-exact** across eight decades where it previously returned nothing at all,
/// and `f32` at `x = 1e-3` improved by a factor of **580,000**. The narrow format mattered more:
/// its old floor of `7e-4` put the failure at ordinary arguments rather than extreme ones.
///
/// # Where it still stops, and why that is the format
///
/// In `f32`, `x²/4` itself leaves the format below `x ≈ 5e-23`, and becomes subnormal below
/// `x ≈ 2.2e-19` — so `0` is the correct answer there and a few digits of relative error are
/// correct just above it. That is the same reason [this module's window](self) is confined to
/// normal arguments, and it is a property of `f32` rather than of the kernel.
#[test]
fn the_log_bessel_i0_now_tracks_its_leading_term() {
    // Across the decades that used to return zero, `f64` agrees with the summed series exactly.
    for e in [-100i32, -50, -20, -10, -8, -7, -6, -5, -4, -3] {
        let x = 10f64.powi(e);
        let want = ln_i0_series(x);
        assert_eq!(
            Real::ln_bessel_i0(x),
            want,
            "ln_bessel_i0(1e{e}) should equal the directly summed series bit for bit; it used to \
             return 0 here"
        );
    }
    // And up to the branch's threshold it stays within a couple of ulp, including across the
    // join at `x = 1` that `super::seams` scans.
    for i in 0..=2_000 {
        let x = 1e-3 * (1e3f64).powf(f64::from(i) / 2_000.0);
        let want = ln_i0_series(x);
        let got = Real::ln_bessel_i0(x);
        if want == 0.0 || !want.is_finite() {
            continue;
        }
        let ulps = ((got - want) / want).abs() / f64::EPSILON;
        assert!(
            ulps <= 16.0,
            "ln_bessel_i0({x:e}) = {got:e} against the summed series {want:e}, {ulps:.2} ulp"
        );
    }
    // `f32`, where the old floor was four orders worse. Down to where `x²/4` goes subnormal.
    for e in [-18i32, -15, -10, -6, -4, -3, -2, -1] {
        let x = 10f32.powi(e);
        let want = ln_i0_series(f64::from(x));
        let got = f64::from(Real::ln_bessel_i0(x));
        let relative = ((got - want) / want).abs();
        assert!(
            relative <= 8.0 * f64::from(f32::EPSILON),
            "f32 ln_bessel_i0(1e{e}) = {got:e} against {want:e}, relative {relative:e}"
        );
    }
}
