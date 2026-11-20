//! The two-argument log-domain members' range contract, over the whole format.
//!
//! # The half [`super::log_domain`] left out
//!
//! That file covers the ten *single*-argument members. These five take two arguments, and the
//! difference is not cosmetic: the whole point of `ln_add_exp` is to combine two numbers whose
//! exponentials would each overflow, so the failure mode lives in the *pair* and a one-argument
//! sweep cannot reach it. A grid is needed, and this one is `2,476 × 2,476`.
//!
//! `identities::log_domain` already proves the algebra — `ln_add_exp` is bit-exactly symmetric,
//! it sits in `[max, max + ln 2]`, `xlogy(0, y)` is `0` for every `y` including `NaN`. What it
//! does not do is reach the arguments where the stable formulation earns its keep: its properties
//! generate over `[−600, 600]`, which is comfortably inside the range where the naive
//! `(e^a + e^b).ln()` still works. Everything below runs to `±MAX` and down into the subnormals.
//!
//! # What the sweep found
//!
//! All five are clean. Over 6,130,576 ordered pairs:
//!
//! - `ln_add_exp` returns **no `NaN` and no infinity at all**, and sits inside
//!   `[max(a,b), max(a,b) + ln 2]` at every one of them — including `(MAX, MAX)`, where the upper
//!   end of the bracket is not representable and the result is correctly `MAX` itself.
//! - `ln_sub_exp` is `NaN`-free on its documented domain `a ≥ b`, and the only infinities are the
//!   `2,482` pairs with `a == b`, where `ln(eᵃ − eᵃ) = ln 0 = −∞` is the right answer.
//! - the `xlog*` trio honour the absorbing zero at **every** second argument in the grid, plus
//!   `±∞` and `NaN`.
//!
//! The nothing is the result, for the reason given in [`super::log_domain`]: the naive forms
//! break here, so a clean sweep is a statement about these kernels rather than about arithmetic.
//! `(e^{1e300} + e^{1e300}).ln()` is `inf`; `ln_add_exp(1e300, 1e300)` is `1e300 + ln 2`.

use super::*;

/// A named two-argument kernel.
type Binary = (&'static str, fn(f64, f64) -> f64);

/// Magnitudes spanning the format, both signs — the two-argument analogue of
/// [`whole_format_probes`].
///
/// Deliberately coarser than the one-argument list: this is squared into a grid, so the 2,476
/// probes here are already 6,130,576 evaluations per kernel. Two multipliers per decade rather
/// than four keeps the suite inside its runtime while still visiting every exponent, both signs,
/// the subnormals and both extremes.
fn pair_probes() -> Vec<f64> {
    let mut v: Vec<f64> = vec![
        0.0,
        -0.0,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MAX,
        f64::MIN,
        1.0,
        -1.0,
    ];
    let mut e = -323i32;
    while e <= 308 {
        let decade = 10f64.powi(e);
        if decade.is_finite() && decade > 0.0 {
            for scale in [1.0, 3.0] {
                let x = decade * scale;
                if x.is_finite() {
                    v.push(x);
                    v.push(-x);
                }
            }
        }
        e += 1;
    }
    v
}

/// `ln_add_exp` is total, and brackets its own larger argument, across the whole format.
///
/// # Why the bracket is the right assertion and needs no tolerance
///
/// `ln(eᵃ + eᵇ) = max + ln(1 + e^{−|a−b|})`, and the second term lies in `(0, ln 2]` by
/// construction. So `max ≤ ln_add_exp(a,b) ≤ max + ln 2` is a mathematical fact, true at every
/// pair, needing no reference and no tolerance — and false for any implementation that overflows
/// its intermediate or picks the wrong operand as the pivot. `identities::log_domain` asserts it
/// over `[−600, 600]`; this asserts it over the format, which is where an overflow would be.
///
/// # The one pair that looks like a violation and is not
///
/// At `(MAX, MAX)` the true value is `MAX + ln 2`, which is **not representable** — it rounds
/// back to `MAX`. So the upper end of the bracket has to be evaluated as the `f64` sum it is,
/// rather than reasoned about in exact arithmetic, and the test accepts `MAX`. That is correct
/// saturation rather than a broken bound, the same way the non-strict bounds in
/// [`super::range`] are.
#[test]
fn the_log_sum_exp_is_total_and_brackets_its_larger_argument() {
    let probes = pair_probes();
    let ln_two = core::f64::consts::LN_2;
    let mut failures = Vec::new();
    let mut evaluated = 0usize;

    for &a in &probes {
        for &b in &probes {
            if !a.is_finite() || !b.is_finite() {
                continue;
            }
            evaluated += 1;
            let got = Real::ln_add_exp(a, b);
            if !got.is_finite() {
                failures.push(format!(
                    "  ln_add_exp({a:e}, {b:e}) = {got:e}; the log of a sum of exponentials is \
                     finite for every finite pair"
                ));
                continue;
            }
            let larger = a.max(b);
            if got < larger || got > larger + ln_two {
                failures.push(format!(
                    "  ln_add_exp({a:e}, {b:e}) = {got:e}, outside [{larger:e}, {:e}]",
                    larger + ln_two
                ));
            }
        }
    }

    assert!(
        evaluated >= 6_000_000,
        "the grid shrank to {evaluated} pairs"
    );
    assert!(
        failures.is_empty(),
        "ln_add_exp left its bracket or the format:\n{}",
        failures.join("\n")
    );

    // The form it replaces does leave the format, which is why the above is a contract rather
    // than a coincidence about arithmetic.
    assert!(
        (1e300f64.exp() + 1e300f64.exp()).ln().is_infinite(),
        "the naive log-sum-exp no longer overflows; re-derive this module's premise"
    );
    assert!(
        Real::ln_add_exp(1e300f64, 1e300f64).is_finite(),
        "the stable form must be finite exactly where the naive one is not"
    );
}

/// `ln_sub_exp` is `NaN`-free on `a ≥ b`, and `−∞` only where the difference is truly zero.
///
/// The asymmetric sibling of the bracket above, and the interesting half is the boundary rather
/// than the bulk. `ln(eᵃ − eᵇ)` is real only for `a ≥ b`, and at `a == b` the difference is
/// exactly zero, so `−∞` is the correct answer rather than a defect — which makes "infinite" a
/// *conditional* contract here and not a flat prohibition. Every other pair owes a finite value
/// no greater than `a`, since `eᵃ − eᵇ < eᵃ`.
///
/// Measured: of the pairs with `a ≥ b`, exactly the `a == b` diagonal returns `−∞`, and nothing
/// returns `NaN`. `ln_sub_exp(1e300, 0) = 1e300` and `ln_sub_exp(MAX, MIN) = MAX`, where forming
/// either exponential would have overflowed outright.
#[test]
fn the_log_difference_of_exponentials_is_infinite_only_where_it_should_be() {
    let probes = pair_probes();
    let mut failures = Vec::new();
    let (mut diagonal, mut evaluated) = (0usize, 0usize);

    for &a in &probes {
        for &b in &probes {
            if !a.is_finite() || !b.is_finite() || a < b {
                continue;
            }
            evaluated += 1;
            let got = Real::ln_sub_exp(a, b);
            if got.is_nan() {
                failures.push(format!(
                    "  ln_sub_exp({a:e}, {b:e}) = NaN, but a >= b is the documented domain"
                ));
                continue;
            }
            if a == b {
                diagonal += 1;
                if got != f64::NEG_INFINITY {
                    failures.push(format!(
                        "  ln_sub_exp({a:e}, {a:e}) = {got:e}; the difference is exactly zero, so \
                         the logarithm is -inf"
                    ));
                }
                continue;
            }
            if !got.is_finite() {
                failures.push(format!(
                    "  ln_sub_exp({a:e}, {b:e}) = {got:e}, but a > b leaves a positive difference"
                ));
            } else if got > a {
                failures.push(format!(
                    "  ln_sub_exp({a:e}, {b:e}) = {got:e}, above a; e^a - e^b < e^a"
                ));
            }
        }
    }

    assert!(
        evaluated >= 3_000_000,
        "the grid shrank to {evaluated} pairs"
    );
    assert!(
        diagonal >= 2_400,
        "only {diagonal} equal pairs reached; the -inf boundary is barely tested"
    );
    assert!(
        failures.is_empty(),
        "ln_sub_exp broke its domain contract:\n{}",
        failures.join("\n")
    );
}

/// The `xlog*` trio absorb a zero multiplier at every second argument, `NaN` and `±∞` included.
///
/// `xlogy(0, y) = 0` for all `y` is a **convention**, not a theorem — the absorbing zero is
/// chosen over `NaN` propagation so that an entropy or cross-entropy sum stays finite where a
/// plain `x * y.ln()` would evaluate `0 · (−∞)`. `identities::log_domain` pins the convention at
/// the interesting singular points; this sweeps it across the whole format, which matters because
/// the natural way to break it is a branch that tests `y` rather than `x`.
///
/// The three differ only in what they take the logarithm of — `y`, `1 + y`, `1 − y` — so each has
/// its own `y` at which the inner logarithm is singular: `0`, `−1` and `1` respectively. All
/// three are in the probe set, and all three must still return zero.
#[test]
fn the_absorbing_zero_wins_over_every_second_argument() {
    let trio: Vec<Binary> = vec![
        ("xlogy", Real::xlogy),
        ("xlog1py", Real::xlog1py),
        ("xlog1my", Real::xlog1my),
    ];

    let mut arguments = pair_probes();
    // The singular points of the three inner logarithms, plus the non-finite cases that are the
    // whole reason the convention exists.
    arguments.extend_from_slice(&[0.0, -1.0, 1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);

    let mut failures = Vec::new();
    for (name, f) in &trio {
        for &y in &arguments {
            // Both signed zeros: `-0.0` is a distinct bit pattern and an equally valid multiplier.
            for x in [0.0f64, -0.0] {
                let got = f(x, y);
                if got != 0.0 {
                    failures.push(format!(
                        "  {name}({x:e}, {y:e}) = {got:e}; the zero multiplier is absorbing"
                    ));
                }
            }
        }
    }

    assert_eq!(trio.len(), 3, "the sweep lost kernels");
    assert!(
        failures.is_empty(),
        "an xlog* kernel failed to absorb its zero:\n{}",
        failures.join("\n")
    );
}
