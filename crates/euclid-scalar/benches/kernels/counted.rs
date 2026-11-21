//! Cost as a function of **iteration depth**, for the loops whose depth is a tuning decision.
//!
//! These are the measurements the depth constants are sized against — `W_STEPS`, `MAX_ITER`,
//! `MARCH_MAX_STEPS`, the `erfcx` ladder. A depth is only defensible with a number beside it
//! saying what the next level would have cost, and this is where that number comes from.

use super::*;

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
pub(super) fn iteration_depths() {
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
    // The deepest buckets are no longer *reached*: below `x = 2` the Chebyshev series runs
    // instead, so the ladder now starts at the 72-level bucket. Reporting 145 here would price
    // a loop nothing enters, which is the same thing the gamma series row had to stop doing.
    depth_row("erfcx levels, 2 <= x < 3", 72, 145);
    depth_row("erfcx levels, x>=50", 5, 145);

    // The incomplete gamma's two routes, replaying their own stopping rules — and only at the
    // arguments that still *reach* them. The uniform expansion now intercepts `a ≥ 20` with
    // `|x − a| ≤ 0.4a`, which is exactly where the lower series was at its deepest: its term
    // ratio is `x/(a+k)`, so it is long only while `x` is close to `a`, and it walked about
    // `8.5·√a` terms there — 32200 at the top of the range, which is what made
    // `gamma_lr series, large a` the dearest row in the table above.
    //
    // Replaying the series over arguments it no longer sees would report a cost nothing pays.
    // Skipping them is what keeps this row the answer to "how deep does this loop actually go".
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
                if a >= 20.0 && ((x - a) / a).abs() <= 0.4 {
                    continue; // the uniform expansion runs here, not either loop
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
pub(super) fn halley_steps(x: f64) -> u32 {
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
pub(super) fn lower_series_terms(a: f64, x: f64) -> u32 {
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
pub(super) fn upper_fraction_steps(a: f64, x: f64) -> u32 {
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
