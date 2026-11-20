//! Cost of the scalar kernels, **per named regime**, so "fast" is a measurement rather than a
//! claim.
//!
//! Run with `cargo bench -p euclid-scalar`. There is no criterion and no `#[bench]`: the target
//! is `harness = false`, so this is a plain `fn main` that times loops and prints a table. That
//! keeps the workspace's dependency-free property intact, for the reason
//! [`euclid-stability`'s own benchmark](../../euclid-stability/benches/engine.rs) gives — a
//! benchmark dependency is still a dependency, and `deny.toml` would have to grow an exception
//! for it.
//!
//! # Why the rows are regimes rather than functions
//!
//! Every kernel in this crate routes between two and seven approximations, and the choice is
//! named: `IncompleteGammaMethod`, `ErfcForm`, `ErfcxDepth`, `JRoute`, `IForm`, `LambertSeed`,
//! `AiryRegime`, `ZetaRoute`. A single ns/call figure for `gamma_lr` is therefore an average
//! over routes that differ by more than an order of magnitude, and it hides exactly the thing
//! worth knowing: *which arm is the expensive one*.
//!
//! So each row below pins one argument into one arm. A regime that costs far more than its
//! neighbours is then visible directly, and — because the arms are named in the source — the row
//! label says where to go and look.
//!
//! # What these numbers are, and what they are not
//!
//! The lesson from the engine benchmark applies here unchanged and is worth restating, because
//! it is the easiest thing to get wrong about this file:
//!
//! **Do not read a diff off two separate runs.** On this machine the between-run variation —
//! frequency scaling, thermal state, whatever else is resident — exceeded a factor of two on a
//! 16-nanosecond loop, while each run was internally self-consistent to 1.0×. A min-of-seven
//! harness fixes *within*-run noise and cannot touch that. Absolute rows here are for **scale
//! and ranking**: they answer "which regime is expensive" and "is this kernel nanoseconds or
//! microseconds", which is what the rows are for, and they do not answer "did my change make it
//! 8% faster".
//!
//! Two kinds of number are trustworthy, and both appear below:
//!
//! - **Counted quantities**, exact and identical on every machine. [`iteration_depths`] reports
//!   how deep each bounded loop actually runs against the cap that bounds it. That is the figure
//!   that governs kernel cost, it is reproducible, and it is the one to put in a commit message.
//! - **Interleaved A/B**, where both arms run in the same process microseconds apart and so see
//!   the same machine. [`ab`] is here for when a specific change needs defending.
//!
//! # The ratios within a run are the result
//!
//! Because the arms of one kernel are measured in the same pass, their *ratio* is far more
//! robust than either absolute. `erfcx` is the clearest case and doubles as a check that the
//! instrument works at all: [`ErfcxDepth`] spends 160 recurrence levels below `x = 2` and 10
//! above `x = 20`, a 16× difference in declared work, so the measured ratio across that ladder
//! should track it. If it ever does not, either the ladder stopped being the cost or this
//! benchmark stopped measuring what it claims to.

use std::hint::black_box;
use std::time::{Duration, Instant};

use euclid_scalar::Real;

/// Calls per measured loop. Large enough to swamp timer resolution — the cheapest row here is a
/// few nanoseconds — small enough that the whole benchmark stays under a few seconds.
const CALLS: usize = 20_000;

/// Timed repetitions per row, plus one discarded warm-up. The **minimum** is reported, because
/// noise only ever adds: the fastest observed pass is the closest estimate of the work itself,
/// while a mean over a contended machine measures the contention. Spread is printed alongside so
/// a pass that was too noisy to trust says so rather than looking authoritative.
const REPEATS: usize = 7;

fn main() {
    println!("euclid-scalar kernel cost, by named regime");
    println!("(minimum of {REPEATS} timed passes after one warm-up; ratios within a kernel are");
    println!(" the trustworthy part — see this file's header before diffing absolutes)\n");
    println!(
        "{:<26} {:<22} {:>11} {:>8}",
        "kernel", "regime", "ns/call", "spread"
    );
    println!("{}", "-".repeat(70));

    incomplete_gamma();
    error_function();
    erfcx_depth_ladder();
    bessel();
    airy();
    transcendental();
    integrals();
    inverses();

    iteration_depths();
}

// ===== the regime tables =================================================

/// `gamma_lr` / `gamma_ur`, over [`IncompleteGammaMethod`]'s arms.
fn incomplete_gamma() {
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
fn error_function() {
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

/// `erfcx` across [`ErfcxDepth`]'s six buckets.
///
/// The declared work per bucket is 160, 72, 44, 24, 14 and 10 backward-recurrence levels, so the
/// measured cost should fall monotonically across these rows by roughly 16× end to end. This row
/// group is the benchmark's own sanity check; see the file header.
fn erfcx_depth_ladder() {
    section("erfcx depth ladder (160, 72, 44, 24, 14, 10 levels)");
    row1("erfcx", "under 2     (160)", 1.5, <f64 as Real>::erfcx);
    row1("erfcx", "under 3      (72)", 2.5, <f64 as Real>::erfcx);
    row1("erfcx", "under 5      (44)", 4.0, <f64 as Real>::erfcx);
    row1("erfcx", "under 10     (24)", 7.0, <f64 as Real>::erfcx);
    row1("erfcx", "under 20     (14)", 15.0, <f64 as Real>::erfcx);
    row1("erfcx", "at least 20  (10)", 50.0, <f64 as Real>::erfcx);
    // Below the crossover `erfcx` goes through `erfc`'s complement form instead, which is the
    // one arm of the ladder that is not a recurrence at all.
    row1("erfcx", "complement (x<1.375)", 0.5, <f64 as Real>::erfcx);
}

/// The Bessel family, over [`JRoute`], [`YRoute`] and [`IForm`].
fn bessel() {
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
fn airy() {
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
fn transcendental() {
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
fn integrals() {
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
fn inverses() {
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

// ===== counted quantities ================================================

/// How deep each bounded loop actually runs, against the cap that bounds it.
///
/// **This is the reproducible half of the benchmark.** Every figure here is a count, identical on
/// any machine and in any build, so it is the number to quote in a commit message — unlike the
/// timings above, which are a property of this box.
///
/// What it is for: a cap far above the measured depth is not free. The `f32` Lambert work records
/// the reasoning — "a backstop seven times larger than any argument needs is one that hides a
/// non-converging case instead of reporting it" — and the same applies to every row here. A cap
/// that no argument approaches cannot distinguish a slow argument from a broken one.
///
/// The depth is recovered by replaying each kernel's own convergence rule, which is why this
/// function carries the rule rather than importing it: the kernels are `pub(crate)` and a
/// benchmark sees only the public surface. Each replay is marked with the constant it mirrors, so
/// a change to one without the other shows up as a disagreement between this table and the
/// kernel's own documented depth.
fn iteration_depths() {
    println!("\n\ncounted loop depth vs cap (exact; identical on every machine)\n");
    println!(
        "{:<34} {:>9} {:>9} {:>9}",
        "loop", "worst", "cap", "headroom"
    );
    println!("{}", "-".repeat(64));

    // `W`'s Halley iteration, replaying the exit rule `next == w`. Mirrors `W_STEPS = 60`.
    let mut worst_w = 0;
    for i in 0..8_000 {
        let x = -0.367_879_441_171_442_32 + f64::from(i) * 1e-3;
        worst_w = worst_w.max(halley_steps(x));
    }
    depth_row("lambert_w halley steps", worst_w, 60);

    // The Cohen–Rodriguez Villegas–Zagier eta accelerator is a fixed 24 terms with no exit, so
    // its "depth" is its cap by construction. Recorded to make that explicit rather than absent.
    depth_row("zeta eta terms (fixed)", 24, 24);

    // `erfcx`'s backward recurrence is likewise a fixed depth per bucket, chosen per magnitude.
    depth_row("erfcx levels, x<2", 160, 160);
    depth_row("erfcx levels, x>=20", 10, 160);

    // The incomplete gamma's two routes, replaying their own stopping rules. These are the
    // numbers that explain the dearest rows in the table above, and they are the reason
    // `MAX_ITER` is 100_000 rather than something tidier: the lower series walks about
    // `7.3·√a` terms, so the cap is only ~2.5x the worst case a caller can reach before the
    // Edgeworth route takes over at `a = 3e7`. The fraction, by contrast, never needs more than
    // a hundred steps anywhere — its share of that cap is pure slack.
    let (mut worst_series, mut worst_fraction) = (0, 0);
    for e in -6..=7 {
        for m in [1.0, 2.0, 5.0] {
            let a = m * 10f64.powi(e);
            if a >= 3e7 {
                continue; // Edgeworth takes over; neither loop runs
            }
            for j in 0..200 {
                let x = a * (0.01 + 0.03 * f64::from(j));
                if x <= 0.0 || !x.is_finite() {
                    continue;
                }
                if x < a + 1.0 {
                    worst_series = worst_series.max(lower_series_terms(a, x));
                } else {
                    worst_fraction = worst_fraction.max(upper_fraction_steps(a, x));
                }
            }
        }
    }
    depth_row("gamma lower series terms", worst_series, 100_000);
    depth_row("gamma upper fraction steps", worst_fraction, 100_000);

    println!(
        "\nThe two `fixed` rows have no exit test: their depth is the cap, by design, because a\n\
         backward recurrence has to be entered at a chosen level. The headroom column is blank of\n\
         meaning for them and is shown only so the table reads uniformly."
    );
}

/// Halley steps for `W(x)`, replaying [`lambert_w`](euclid_scalar::Real::lambert_w)'s exit rules —
/// both of them, including the period-two cycle check that keeps the loop off its cap.
///
/// A deliberate duplicate of the kernel's loop: the benchmark cannot see inside it, and the
/// alternative — threading a counter through the public surface — would put test scaffolding in
/// a shipped signature. The duplication is the cost of keeping the count honest, and it is
/// bounded: if this drifts from the kernel the reported depth stops matching the kernel's own
/// documented worst case, which is itself the signal.
fn halley_steps(x: f64) -> u32 {
    const NEG_INV_E: f64 = -0.367_879_441_171_442_32;
    if x.is_nan() || x < NEG_INV_E || x == 0.0 || x.is_infinite() {
        return 0;
    }
    // The three seeds, as `LambertSeed` selects them.
    let mut w = if x < NEG_INV_E + 1e-3 {
        let p = (2.0 * (1.0 + core::f64::consts::E * x)).max(0.0).sqrt();
        -1.0 + p - p * p / 3.0 + 11.0 / 72.0 * p * p * p
    } else if x > 3.0 {
        let l = x.ln();
        l - l.ln()
    } else {
        x * (1.0 - x + 1.5 * x * x)
    };
    let mut previous = f64::NAN;
    let mut steps = 0;
    for _ in 0..60 {
        steps += 1;
        let ew = w.exp();
        let (f, denom_newton) = if x > 0.0 {
            (w - x / ew, 1.0 + w)
        } else {
            (w * ew - x, ew * (1.0 + w))
        };
        if f == 0.0 {
            return steps;
        }
        let correction = f * (w + 2.0) / (2.0 * w + 2.0);
        let denom = denom_newton - correction;
        let step = if denom == 0.0 || !denom.is_finite() {
            f / denom_newton
        } else {
            f / denom
        };
        if !step.is_finite() {
            break;
        }
        let next = w - step;
        // The period-two cycle check, which is what keeps this off the cap.
        if next == w || next == previous {
            break;
        }
        previous = w;
        w = next;
    }
    steps
}

/// Terms the lower incomplete-gamma series takes, replaying its own stopping rule.
///
/// Mirrors `gamma_lower_series` with `EPS = 1e-15`. The depth is about `7.3·√a` near `x = a`,
/// which is where it peaks — 17 terms at `a = 1`, 781 at `1e4`, 7,413 at `1e6`, 32,200 at `2e7`.
fn lower_series_terms(a: f64, x: f64) -> u32 {
    const EPS: f64 = 1e-15;
    let mut denominator = a;
    let mut term = a.recip();
    let mut sum = term;
    for k in 0..100_000u32 {
        denominator += 1.0;
        term *= x / denominator;
        sum += term;
        if term.abs() < sum.abs() * EPS {
            return k + 1;
        }
    }
    100_000
}

/// Steps the upper incomplete-gamma continued fraction takes, replaying Lentz's exit rule.
///
/// Mirrors `gamma_upper_fraction`. **Do not be tempted to use this below `x = a + 1`.** It
/// reports convergence there in three to twelve steps and returns a value that is wrong by
/// factors of `1e23` and worse — the multiplier comes within tolerance of one while the fraction
/// has not converged at all. Measured agreement with the series, as `P`: `1e-12` at `x/a = 0.99`
/// and above, and `3e85`, `NaN`, `5e234` below it. The classical `x < a + 1` split is load-bearing
/// and the cheap-looking step count on the wrong side of it is a trap.
fn upper_fraction_steps(a: f64, x: f64) -> u32 {
    const EPS: f64 = 1e-15;
    const TINY: f64 = 1e-300;
    let pivot = |v: f64| if v.abs() < TINY { TINY } else { v };
    let mut b = x + 1.0 - a;
    let mut c = TINY.recip();
    let mut d = b.recip();
    for i in 1..100_000u32 {
        b += 2.0;
        let numerator = -f64::from(i) * (f64::from(i) - a);
        d = pivot(numerator * d + b).recip();
        c = pivot(b + numerator / c);
        if (d * c - 1.0).abs() < EPS {
            return i;
        }
    }
    100_000
}

// ===== harness ===========================================================

fn section(title: &str) {
    println!("\n{title}");
}

/// Time a one-argument kernel at a single argument, pinned into one regime.
fn row1(kernel: &str, regime: &str, x: f64, f: impl Fn(f64) -> f64) {
    let (min, spread) = measure(|| {
        black_box(f(black_box(x)));
    });
    print_row(kernel, regime, min, spread);
}

/// Time a two-argument kernel, pinned into one regime.
fn row2(kernel: &str, regime: &str, a: f64, b: f64, f: impl Fn(f64, f64) -> f64) {
    let (min, spread) = measure(|| {
        black_box(f(black_box(a), black_box(b)));
    });
    print_row(kernel, regime, min, spread);
}

fn print_row(kernel: &str, regime: &str, min: Duration, spread: Duration) {
    #[allow(clippy::cast_precision_loss)]
    let per = min.as_secs_f64() / CALLS as f64 * 1e9;
    let ratio = if min.as_nanos() == 0 {
        1.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        let r = (min + spread).as_secs_f64() / min.as_secs_f64();
        r
    };
    println!("{kernel:<26} {regime:<22} {per:>11.1} {ratio:>7.2}x");
}

fn depth_row(what: &str, worst: u32, cap: u32) {
    let headroom = if worst == 0 {
        String::from("-")
    } else {
        #[allow(clippy::cast_precision_loss)]
        let h = f64::from(cap) / f64::from(worst);
        format!("{h:.1}x")
    };
    println!("{what:<34} {worst:>9} {cap:>9} {headroom:>9}");
}

/// Run `body` `CALLS` times per pass, `REPEATS` passes after a warm-up, and return the minimum
/// pass together with the spread between fastest and slowest.
fn measure(mut body: impl FnMut()) -> (Duration, Duration) {
    for _ in 0..CALLS {
        body();
    }
    let mut min = Duration::MAX;
    let mut max = Duration::ZERO;
    for _ in 0..REPEATS {
        let start = Instant::now();
        for _ in 0..CALLS {
            body();
        }
        let elapsed = start.elapsed();
        min = min.min(elapsed);
        max = max.max(elapsed);
    }
    (min, max - min)
}

/// Interleaved A/B, for when a specific change needs defending.
///
/// Both arms run in the same process microseconds apart, so they see the same machine state —
/// which is the only comparison this file trusts. Unused while nothing is under optimisation;
/// kept because the alternative, when the question next comes up, is a cross-run diff.
#[allow(dead_code)]
fn ab(label: &str, mut before: impl FnMut(), mut after: impl FnMut()) {
    let (b, _) = measure(&mut before);
    let (a, _) = measure(&mut after);
    let ratio = b.as_secs_f64() / a.as_secs_f64();
    #[allow(clippy::cast_precision_loss)]
    let delta = (b.as_secs_f64() - a.as_secs_f64()) / CALLS as f64 * 1e9;
    println!("{label:<48} {delta:>8.1} ns/call {ratio:>7.2}x");
}
