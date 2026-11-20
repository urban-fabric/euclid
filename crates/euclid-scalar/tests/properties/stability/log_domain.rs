//! The log-domain family's range contract, over the whole format.
//!
//! # Why this file exists when `identities::log_domain` already covers these methods
//!
//! It covers them as *algebra* — that `ln_add_exp` is symmetric, that `ln_1m_sigmoid(x)` is
//! `ln_sigmoid(−x)` to the bit, that softplus dominates both `0` and `x` — and it tests them over
//! `[−700, 700]`, which is a comfortable interval. None of that is a range contract. What was
//! missing is the half this file asserts: **no `NaN` inside the domain, no infinity where the
//! value is representable, no zero before the value underflows, and the closed-form bounds** —
//! at the subnormals, at `±MAX`, and at every decade between.
//!
//! The gap mattered more here than anywhere else in the suite, because this family's *entire
//! reason for existing* is that the naive composition saturates. A kernel named `ln_1p_exp` whose
//! answer is `inf` has not lost accuracy, it has failed to be the thing it is for. And one of
//! them, [`Real::ln_1m_exp`], had no coverage anywhere in the crate's tests before this file.
//!
//! # What the sweep found: nothing, and the nothing is the result
//!
//! All ten are clean over 2,500 probes. That is worth recording rather than shrugging at, because
//! three of the four contracts below are false for the obvious implementation:
//!
//! ```text
//! x        ln_1p_exp(x)   (1 + e^x).ln()      ln_sigmoid(x)   sigmoid(x).ln()
//! 1e3      1e3            inf                 -1e3            -inf
//! 1e300    1e300          inf                 -1e300          -inf
//! MAX      1.798e308      inf                 -1.798e308      -inf
//! ```
//!
//! So the assertions are not ceremony around working code; they are the difference between these
//! kernels and `std`, pinned at the arguments where the difference appears.
//!
//! # The one number this file turns on
//!
//! Five of these saturate at exactly the same argument, `|x| = 745.133`, and that is the right
//! one. `e^{−x}` reaches half of the smallest subnormal at `ln(2/4.94e-324) = 745.133`, below
//! which rounding to nearest has nowhere to go — so a kernel that keeps a non-zero value to
//! `745.1` has used the format down to its last bit, including the whole subnormal range. The
//! prediction from the normal floor alone would be `744.44`; each kernel reaches `ln 2` past it,
//! which is round-to-nearest working in their favour and not something to leave untested.

use super::*;

/// A kernel in the family, and the predicate for where it has a real value.
///
/// `NaN` is never the correct answer inside the domain for any member — unlike `Γ` and `ψ` in
/// [`super::range`], none of these has a pole — so there is no fourth "`NaN` is real here" column.
type LogDomain = (&'static str, fn(f64) -> f64, fn(f64) -> bool);

/// A named kernel with no extra columns, for the sweeps that need only "call it everywhere".
type Named = (&'static str, fn(f64) -> f64);

/// A named kernel with its inclusive lower and upper bounds.
type BoundedLog = (&'static str, fn(f64) -> f64, f64, f64);

/// The argument at which `e^{−|x|}` reaches half the smallest subnormal and rounds to zero.
///
/// `ln(2 / 4.94e-324)`. See the module header: this is the measured saturation point of five of
/// the ten kernels, to all the digits printed, and the assertions below sit just inside it.
const SUBNORMAL_EXHAUSTED: f64 = 745.133_219_101_941_2;

/// The ten single-argument members, with their domains.
///
/// Four are total — `sigmoid`, `ln_sigmoid`, `ln_1m_sigmoid` and `ln_1p_exp` have a real value at
/// every real argument, which is most of why they are preferred to the forms they replace. The
/// other six carry a genuine domain, and the boundary of each is a point where the value is an
/// infinity rather than a `NaN`: `ln_1m(1)`, `ln_1p(−1)`, `ln_1m_exp(0)`, `ln_exp_m1(0)` and
/// `logit` at both ends.
fn family() -> Vec<LogDomain> {
    vec![
        ("sigmoid", |x| Real::sigmoid(&x), |_| true),
        ("ln_sigmoid", |x| Real::ln_sigmoid(&x), |_| true),
        ("ln_1m_sigmoid", |x| Real::ln_1m_sigmoid(&x), |_| true),
        ("ln_1p_exp", |x| Real::ln_1p_exp(&x), |_| true),
        ("exp_m1", |x| Real::exp_m1(&x), |_| true),
        // `ln(e^x − 1)` is real only where `e^x > 1`.
        ("ln_exp_m1", |x| Real::ln_exp_m1(&x), |x: f64| x > 0.0),
        // `ln(1 − e^x)` is real only where `e^x < 1`; `x = 0` gives `−∞`, which is the limit
        // from inside and is what the trait documents.
        ("ln_1m_exp", |x| Real::ln_1m_exp(&x), |x: f64| x <= 0.0),
        ("ln_1m", |x| Real::ln_1m(&x), |x: f64| x <= 1.0),
        ("ln_1p", |x| Real::ln_1p(&x), |x: f64| x >= -1.0),
        (
            "logit",
            |x| Real::logit(&x),
            |x: f64| (0.0..=1.0).contains(&x),
        ),
    ]
}

/// No member of the family returns `NaN` for a finite argument inside its domain.
///
/// The same contract as [`super::range::no_kernel_returns_nan_where_its_value_is_real`], applied
/// to the family that sweep omitted. It is a cheaper assertion here than there, because none of
/// these functions has a pole: every `NaN` is a defect, with no exceptions to carve out.
///
/// The domains are the interesting part, and two of them are easy to get wrong in opposite
/// directions. `ln_1m_exp` is defined on `x ≤ 0` and `ln_exp_m1` on `x > 0`, so they are near
/// mirror images — and `0` belongs to the first and not the second, because `ln(1 − e^0)` has a
/// limit of `−∞` from inside the domain while `ln(e^0 − 1)` is approached from outside it. Both
/// return `−∞` at `0` in practice; only one of them is in-domain there.
#[test]
fn no_log_domain_kernel_returns_nan_where_its_value_is_real() {
    let rows = family();
    let probes = whole_format_probes();
    let mut failures = Vec::new();

    for (name, f, in_domain) in &rows {
        for &x in &probes {
            if !x.is_finite() || !in_domain(x) {
                continue;
            }
            let y = f(x);
            if y.is_nan() {
                failures.push(format!(
                    "  {name}({x:e}) = NaN, but the value is real there"
                ));
            }
        }
    }

    assert_eq!(rows.len(), 10, "the sweep lost kernels");
    assert!(
        probes.len() >= 2_000,
        "the probe set shrank to {}",
        probes.len()
    );
    assert!(
        failures.is_empty(),
        "log-domain kernels returned NaN inside their domain:\n{}",
        failures.join("\n")
    );
}

/// Every member with a closed-form bound respects it across the whole format.
///
/// Each bound below is a mathematical fact about the function rather than a property of the
/// implementation, so it needs no reference value and no tolerance, and a violation is
/// unambiguous. They are stated **non-strictly** for the same reason the bounds in
/// [`super::range`] are: `sigmoid` is strictly inside `(0, 1)` in exact arithmetic but reaches
/// both ends in `f64` — `0` below `x = −745.13` and `1` above `x = 36.74` — and both are correct
/// saturation rather than a broken bound. Where each end is first reached is asserted separately,
/// in [`the_log_domain_family_uses_the_format_to_its_last_bit`].
///
/// `exp_m1 ≥ −1` is the one that earns its place at the extremes: `e^x − 1` cannot go below `−1`,
/// but an implementation that computed `x.exp() - 1.0` with a pre-rounded `exp` could, and the
/// probe set reaches `x = MIN` where every bit of the answer comes from the `−1`.
#[test]
fn every_bounded_log_domain_kernel_respects_its_bounds() {
    // (name, kernel, inclusive lower bound, inclusive upper bound)
    let rows: Vec<BoundedLog> = vec![
        ("sigmoid", |x| Real::sigmoid(&x), 0.0, 1.0),
        (
            "ln_sigmoid",
            |x| Real::ln_sigmoid(&x),
            f64::NEG_INFINITY,
            0.0,
        ),
        (
            "ln_1m_sigmoid",
            |x| Real::ln_1m_sigmoid(&x),
            f64::NEG_INFINITY,
            0.0,
        ),
        ("ln_1p_exp", |x| Real::ln_1p_exp(&x), 0.0, f64::INFINITY),
        ("ln_1m_exp", |x| Real::ln_1m_exp(&x), f64::NEG_INFINITY, 0.0),
        ("exp_m1", |x| Real::exp_m1(&x), -1.0, f64::INFINITY),
    ];

    // The bounded members are all total except `ln_1m_exp`, which owes its bound on `x ≤ 0`.
    let in_domain = |name: &str, x: f64| name != "ln_1m_exp" || x <= 0.0;
    let mut failures = Vec::new();
    for (name, f, lower, upper) in &rows {
        for &x in &whole_format_probes() {
            if !x.is_finite() || !in_domain(name, x) {
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

    assert_eq!(rows.len(), 6, "the sweep lost kernels");
    assert!(
        failures.is_empty(),
        "log-domain kernels left their bounds:\n{}",
        failures.join("\n")
    );
}

/// The four total members are finite at **every** finite argument, which the naive forms are not.
///
/// This is the headline contract of the family and the one with no caveats. `ln(1 + e^x)` and
/// `ln(sigmoid(x))` are finite for every real `x` in mathematics; composed out of `std` they are
/// not, and they fail in opposite directions — the first overflows its intermediate `e^x` above
/// `x = 710`, the second underflows its intermediate `sigmoid(x)` below `x = −745` and takes the
/// logarithm of zero. Each failure is total rather than gradual: `inf` and `−inf`, no digits left.
///
/// So the test asserts finiteness over the whole format **and** that the naive composition it
/// replaces really does break, at `±MAX`. Pinning the second half matters: without it, a future
/// reader has no way to tell whether these kernels are load-bearing or merely spelled differently,
/// and the assertion would be as satisfied by `|x| (1.0 + x.exp()).ln()` as by the real thing.
#[test]
fn the_total_log_domain_kernels_are_finite_at_every_finite_argument() {
    let total: Vec<Named> = vec![
        ("sigmoid", |x| Real::sigmoid(&x)),
        ("ln_sigmoid", |x| Real::ln_sigmoid(&x)),
        ("ln_1m_sigmoid", |x| Real::ln_1m_sigmoid(&x)),
        ("ln_1p_exp", |x| Real::ln_1p_exp(&x)),
    ];

    let mut failures = Vec::new();
    for (name, f) in &total {
        for &x in &whole_format_probes() {
            if !x.is_finite() {
                continue;
            }
            let y = f(x);
            if !y.is_finite() {
                failures.push(format!(
                    "  {name}({x:e}) = {y:e}, but the value is representable there"
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "a total log-domain kernel left the format:\n{}",
        failures.join("\n")
    );

    // And the forms they exist to replace do leave it, which is why the above is a contract and
    // not a coincidence. If either of these ever becomes finite, `std` has changed under us and
    // the justification in this file's header needs rewriting rather than quietly keeping.
    assert!(
        (1.0 + f64::MAX.exp()).ln().is_infinite(),
        "the naive softplus no longer overflows at MAX; re-derive this module's premise"
    );
    assert!(
        Real::sigmoid(&f64::MIN).ln().is_infinite(),
        "the naive log-sigmoid no longer underflows at MIN; re-derive this module's premise"
    );
    assert!(
        Real::ln_1p_exp(&f64::MAX).is_finite() && Real::ln_sigmoid(&f64::MIN).is_finite(),
        "the stable forms must be finite exactly where the naive ones are not"
    );
}

/// No member saturates before the format forces it to — measured to the last subnormal.
///
/// # The two boundaries, and why both are exact rather than approximate
///
/// This is the family's version of
/// [`super::range::no_kernel_underflows_before_its_value_does`], and it can be sharper than that
/// test because both boundaries here are closed forms rather than asymptotic predictions:
///
/// - **`|x| = 745.133`**, where `e^{−|x|}` falls below half the smallest subnormal. Five kernels
///   turn over here — `sigmoid` to `0`, `ln_sigmoid` and `ln_1m_sigmoid` to `−0`, `ln_1p_exp` to
///   `0`, `ln_1m_exp` to `−0` — and all five at the same measured argument to seven digits,
///   because all five are routed through the same `e^{−|x|}`. Shared routing is exactly what one
///   would want to see, and a kernel drifting off this point is the signal that its route changed.
/// - **`x = ln(2/ε) = 36.7368`**, where `1 − e^{−x}` rounds to `1`. This is `sigmoid` reaching
///   its upper bound, and it is unavoidable in the representation rather than a kernel limitation
///   — which is the whole reason `ln_sigmoid` exists as a separate entry point, and is recorded
///   as such in `identities::log_domain`'s header.
///
/// `exp_m1` has a third, at `x = ln(ε/4) = −37.43`, where `e^x` falls below half an ulp of `1` and
/// the result rounds to exactly `−1`. Also unavoidable, and also the reason the kernel exists at
/// all — in the other direction, where `exp_m1` keeps the digits that `x.exp() - 1.0` loses.
///
/// The assertions sit just inside each measured boundary, so the test fails if a kernel gives up
/// early and does not depend on the last representable argument being exactly reproduced.
#[test]
fn the_log_domain_family_uses_the_format_to_its_last_bit() {
    let mut failures = Vec::new();

    // Just inside the subnormal-exhaustion point, from both directions.
    let edge = SUBNORMAL_EXHAUSTED - 0.2;
    let decaying: Vec<Named> = vec![
        ("sigmoid", |x| Real::sigmoid(&x)),
        ("ln_1m_sigmoid", |x| Real::ln_1m_sigmoid(&x)),
        ("ln_1p_exp", |x| Real::ln_1p_exp(&x)),
        ("ln_1m_exp", |x| Real::ln_1m_exp(&x)),
    ];
    // 401 points from `x = −36`, where the decay starts to bite, down to the edge — which is
    // around 308 decades of *value*, since the argument is the exponent. A sweep rather than a
    // spot check at the boundary, so a kernel that gave up at 600 instead of 745 is caught in
    // the middle of the range rather than slipping between two probes.
    for (name, f) in &decaying {
        for i in 0..=400 {
            let x = -(36.0 + (edge - 36.0) * f64::from(i) / 400.0);
            let y = f(x);
            if y == 0.0 {
                failures.push(format!(
                    "  {name}({x:e}) = {y:e} already, but the value is representable to -{edge}"
                ));
                break;
            }
        }
    }
    // `ln_sigmoid` decays towards `−0` on the *positive* side, being `ln_1m_sigmoid` mirrored.
    for i in 0..=400 {
        let x = 36.0 + (edge - 36.0) * f64::from(i) / 400.0;
        if Real::ln_sigmoid(&x) == 0.0 {
            failures.push(format!(
                "  ln_sigmoid({x:e}) = -0 already, but the value is representable to {edge}"
            ));
            break;
        }
    }

    // `sigmoid` must not reach its upper bound before `1 − e^{−x}` rounds to `1`.
    //
    // Both of the closed forms below are **exclusive** bounds — `ln(2/ε)` is itself the first
    // argument at which `sigmoid` returns exactly `1`, to every digit `f64` has, and likewise
    // `ln(ε/4)` for `exp_m1` returning `−1`. So the sweeps stop one step short of them. Asserting
    // *at* the boundary would be asserting that correct rounding has not happened yet, which is
    // how the first draft of this test failed.
    let saturates_at = (2.0 / f64::EPSILON).ln();
    for i in 0..400 {
        let x = saturates_at * f64::from(i) / 400.0;
        if Real::sigmoid(&x) >= 1.0 {
            failures.push(format!(
                "  sigmoid({x:e}) = 1 already, but 1 - e^-x is distinguishable from 1 until \
                 {saturates_at}"
            ));
            break;
        }
    }

    // `exp_m1` must not reach `−1` before `e^x` falls below half an ulp of `1`.
    let rounds_to_minus_one = (0.25 * f64::EPSILON).ln();
    for i in 0..400 {
        let x = rounds_to_minus_one * f64::from(i) / 400.0;
        if Real::exp_m1(&x) <= -1.0 {
            failures.push(format!(
                "  exp_m1({x:e}) = -1 already, but e^x is distinguishable from 0 until \
                 {rounds_to_minus_one}"
            ));
            break;
        }
    }

    assert!(
        failures.is_empty(),
        "log-domain kernels saturated before the format did:\n{}",
        failures.join("\n")
    );
}
