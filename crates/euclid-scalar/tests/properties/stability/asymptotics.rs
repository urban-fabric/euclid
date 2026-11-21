//! What every kernel does as its argument grows, against the exact asymptotic expansion.
//!
//! # The mirror of [`super::limits`], and why both ends are needed
//!
//! `limits` asserts the exact *leading term at the origin* for thirty kernels and found real
//! defects doing it. This file asks the same question at the other end of the format, and it is
//! not the same test twice: a kernel's small-argument branch and its large-argument branch are
//! **different code**. Every family here crosses over — `I_ASYMPTOTIC_MIN` at `20`,
//! `EI_ASYMPTOTIC_MIN` at `40`, `HANKEL_MIN` at `18` — so `limits` exercises the series route and
//! this one exercises the asymptotic route, and neither sees the other.
//!
//! Before this file the large-argument regime had almost nothing: `src/tests/stability/tails.rs`
//! spot-checks `erfcx`, `erfc` and the scaled Bessels at a handful of points against `1e-3`
//! tolerances, and `identities::normal` has one `log_ndtr` tail law. A `1e-3` tolerance is three
//! orders of magnitude looser than these kernels actually are, so it would not notice a route
//! losing ten digits.
//!
//! # An asymptotic expansion is only a reference where it has converged
//!
//! This is the whole difficulty, and it is the opposite of [`super::limits`]'s. An asymptotic
//! series **diverges**: its terms shrink, reach a minimum, then grow without bound, so the best
//! accuracy attainable at a given `x` is the size of that minimum term, and no number of extra
//! terms improves on it. At large `x` the minimum is far below an ulp; at the bottom of a window
//! it may be nowhere near. A first pass over `[1e2, 1e300]` reported `104,803` ulp for `I₀ᵉ` and
//! `3.1e8` for `I₁ᵉ` — every bit of it the reference's own truncation, not the kernel.
//!
//! So each reference carries **its own error bar**: [`summed`] returns the partial sum *and* the
//! magnitude of the first omitted term, which is the standard estimate of an asymptotic
//! remainder, and the [`Reference`] combinators propagate that bar through every scaling, shift,
//! logarithm and ratio applied after it. A point whose bar is not comfortably inside the budget is
//! reported as an **unconverged reference**, never as a kernel failure.
//!
//! That distinction is why this file can be trusted over time, and it took two attempts. The
//! first guard compared the series at thirty terms against fifteen, which sounds equivalent and
//! is not: at `x = 100` optimal truncation stops *both* sums at the same term, so the two agreed
//! exactly while their shared truncation error was `1e-11`. The guard passed the very window it
//! existed to reject. Only the omitted-term estimate sees that.
//!
//! # What the sweep found
//!
//! All seventeen agree with their expansions. The spread in *how well* is the interesting part:
//!
//! ```text
//! kernel              window         worst ulp   at x        budget   margin
//! erfcx            [1e1,  1e150]        1.857    1.67e1          4     0.46
//! bessel_i0e       [1e3,  1e300]        8.050    1.00e3         24     0.34
//! bessel_i1e       [1e3,  1e300]       11.151    1.00e3         24     0.46
//! bessel_k0e       [1e3,  1e300]        2.582    3.12e8          8     0.32
//! bessel_k1e       [1e3,  1e300]        2.679    1.31e6          8     0.33
//! i1_over_i0       [1e3,  1e300]       13.002    2.79e3         32     0.41
//! ln_bessel_i0     [1e2,  1e300]        0.954    1.72e4          4     0.24
//! digamma          [1e2,  1e300]        0.850    1.22e4          4     0.21
//! trigamma         [1e2,  1e150]        0.884    1.81e3          4     0.22
//! ln_gamma         [1e2,  1e300]        1.983    1.90e5          4     0.50
//! airy_ai_e        [1e3,   1e80]        4.434    1.00e4          8     0.55
//! airy_bi_e        [1e3,   1e80]        1.980    4.18e8          4     0.49
//! asinh            [1e2,  1e150]        0.999  1.00e111          4     0.25
//! acosh            [1e2,  1e150]        0.999  1.00e111          4     0.25
//! airy_ai_e_prime  [1e1,   1e79]        4.158    2.05e1          8     0.52
//! airy_bi_e_prime  [1e1,   1e79]        1.927    7.50e5          8     0.24
//! zeta             [1e1,  1e1.7]        4.997    1.08e1         12     0.42
//! ```
//!
//! Budgets sit at two to five times measured — the headroom convention `limits` uses — so every
//! margin is between `0.21` and `0.55`, and none is loose enough to hide a doubling.
//!
//! **The three worst used to be worst in the same place, and that is what fixed them.** `I₀ᵉ`,
//! `I₁ᵉ` and `I₁/I₀` peaked at `8–13` ulp between `x = 1e3` and `x = 4e3` and fell to half an ulp
//! above `1e4` — the error sitting at the *top of the series route*, immediately below what was
//! then `I_ASYMPTOTIC_MIN`. Measuring the expansion's own truncation against it showed the series
//! was the worse of the two from `x = 18` upward, by up to `500×`, so **the threshold moved to
//! `20`** and those three are now within `2.11` ulp uniformly from there to `1e5`. The table above
//! predates the change; its `bessel_i0e` and `bessel_i1e` rows are what the *series* used to cost
//! at the bottom of their windows.
//!
//! The shape that made the diagnosis is pinned by
//! [`the_modified_bessel_series_is_worst_just_below_its_crossover`], now measured either side of
//! `20`.
//!
//! Attribution checked rather than assumed: for `I₁/I₀` the ratio-of-two-expansions reference and
//! an independent dedicated expansion for the ratio agree to `0.5` ulp, so the `13` ulp is the
//! kernel's.
//!
//! The two **derivative** rows are the ones most likely to catch something a sibling would not.
//! `Aiᵉ′` and `Biᵉ′` ride the `vₖ` series rather than `uₖ`, so they are not `Aiᵉ`/`Biᵉ`
//! differentiated term by term, and a kernel can be right about a function and wrong about its
//! slope. Before this they had no accuracy test at any large argument at all — only the
//! `NaN`-and-finiteness rows in [`super::range`].
//!
//! # The split
//!
//! This file is the index: it holds the [`Reference`] error-bar machinery and the coefficient
//! generators, which both children share. The tests are split because one file carrying all of it
//! went past the crate's 400-line budget.
//!
//! - [`sweep`] — the seventeen-row table and the pointwise comparison against it.
//! - [`crossover`] — the one finding that is about a *shape* rather than a worst case: where the
//!   modified Bessel error peaks relative to `I_ASYMPTOTIC_MIN`.

use super::*;

/// How many terms every expansion is offered. Optimal truncation stops each one sooner — except
/// at the very bottom of [`crossover`]'s window, which is what sets the figure.
///
/// The modified Bessel expansion's terms are `aₖ/xᵏ` with `aₖ₊₁/aₖ ≈ k/2`, so the term ratio is
/// `k/(2x)` and the *minimum* term — where a divergent series is as close as it will get — sits at
/// `k ≈ 2x`. At `x = 18`, the lowest argument [`crossover`] references, that is `36` terms.
///
/// `30` was ample while every window here started at `1e3`. When [`crossover`] moved down to
/// `[18, 20]` it stopped being, and the symptom was the *reference* reporting `13.92` ulp where
/// the kernel is within `2` — a short reference blaming a correct kernel. `64` covers `x` down to
/// `32`, and the early-exit guards mean the extra terms cost nothing anywhere they are not needed.
const TERMS: usize = 64;

/// Sample points per kernel across its window.
const SAMPLES: usize = 2_000;

const SQRT_PI: f64 = 1.772_453_850_905_516;
const LN_TAU: f64 = 1.837_877_066_409_345_6;

/// A reference value paired with an estimate of its own **absolute** truncation error.
///
/// Carrying the error bar beside the value is what lets this suite tell "the kernel is wrong"
/// from "my reference is wrong here" — see the module header for why that mattered twice.
type Reference = (f64, f64);

/// A kernel, its expansion, the log-window to sweep, and its measured ulp budget.
type Asymptote = (
    &'static str,
    fn(f64) -> f64,
    Box<dyn Fn(f64) -> Reference>,
    f64,
    f64,
    f64,
);

/// Sum `Σ cₖ/xᵏ` at its optimal truncation, returning the sum and the first omitted term.
///
/// The first guard stops at the minimum term, which is as close as a divergent series will ever
/// get. The second stops once a term cannot move the sum. Either way the magnitude returned is
/// the first term *not* included, which is the textbook estimate of the remainder.
fn summed(coefficients: &[f64], x: f64) -> Reference {
    let (mut sum, mut power, mut previous) = (0.0, 1.0, f64::INFINITY);
    for &c in coefficients.iter().take(TERMS) {
        // A zero coefficient is a *gap* in the series, not the end of it: several of these
        // expansions have only even powers, or only odd ones above the first. Letting a zero
        // reach either guard is how `digamma`, `ln_gamma`, `asinh` and `acosh` first reported
        // ten-figure ulp counts against references that were really just `ln x`.
        if c == 0.0 {
            power /= x;
            continue;
        }
        let term = c * power;
        if term.abs() > previous || term.abs() < 0.125 * f64::EPSILON * sum.abs() {
            return (sum, term.abs());
        }
        sum += term;
        previous = term.abs();
        power /= x;
    }
    // Ran out of terms rather than converging. Reporting the last term as the bar is the
    // pessimistic reading, and trips the convergence guard if it matters.
    (sum, previous)
}

/// Multiply a reference by an exactly-known factor.
fn scaled((value, bar): Reference, factor: f64) -> Reference {
    (value * factor, bar * factor.abs())
}

/// Add an exactly-computed quantity, which leaves the absolute error bar untouched.
fn shifted((value, bar): Reference, addend: f64) -> Reference {
    (value + addend, bar)
}

/// `ln` of a reference: a relative error in the argument is an absolute error in the result.
fn log_of((value, bar): Reference) -> Reference {
    (value.ln(), bar / value.abs())
}

/// Ratio of two references, whose relative error bars add.
fn ratio((a, bar_a): Reference, (b, bar_b): Reference) -> Reference {
    let value = a / b;
    (value, value.abs() * (bar_a / a.abs() + bar_b / b.abs()))
}

/// Hankel's coefficients `aₖ(ν) = Π_{j=1..k}(4ν² − (2j−1)²)/(k!·8ᵏ)`.
///
/// From the recurrence rather than a pasted table, so no coefficient can be mistyped.
///
/// `I_ν` takes `(−1)ᵏaₖ` and `K_ν` takes `aₖ` as it stands, where `aₖ` already carries the sign
/// of its product. The two cannot be written as "all positive" and "alternating": that is only
/// true at `ν = 0`, where `aₖ` happens to have sign `(−1)ᵏ`, giving
/// `I₀ᵉ ~ 1 + 1/(8x) + 9/(128x²)` against `K₀ᵉ ~ 1 − 1/(8x) + 9/(128x²)`. At `ν = 1` the product's
/// own signs run `+, −, +`, so `I₁ᵉ ~ 1 − 3/(8x) − 15/(128x²)` — **two consecutive negatives**, and
/// neither family is simply signed. Taking absolute values here instead of multiplying is a bug
/// that passes at `ν = 0` and silently breaks `I₁ᵉ`, which is how it was found.
fn hankel(nu: f64, i_family: bool) -> Vec<f64> {
    let mut out = vec![1.0];
    let mut a = 1.0;
    let four_nu_squared = 4.0 * nu * nu;
    for k in 1..=TERMS {
        let odd = (2 * k - 1) as f64;
        a *= (four_nu_squared - odd * odd) / (k as f64 * 8.0);
        out.push(if i_family && k % 2 == 1 { -a } else { a });
    }
    out
}

/// Airy's `uₖ = uₖ₋₁·(6k−5)(6k−3)(6k−1)/((2k−1)·216·k)`; `u₁ = 5/72`, `u₂ = 385/10368`.
fn airy(alternating: bool) -> Vec<f64> {
    let mut out = vec![1.0];
    let mut u = 1.0;
    for k in 1..=TERMS {
        let kf = k as f64;
        u *= (6.0 * kf - 5.0) * (6.0 * kf - 3.0) * (6.0 * kf - 1.0)
            / ((2.0 * kf - 1.0) * 216.0 * kf);
        out.push(if alternating && k % 2 == 1 { -u } else { u });
    }
    out
}

/// Airy's *derivative* coefficients `vₖ = (6k+1)/(1−6k)·uₖ`, by DLMF 9.7.3.
///
/// A different series from [`airy`], which is the whole reason the derivatives are worth a row of
/// their own: they are not the `u`-series differentiated term by term, so a kernel could be right
/// about `Ai` and wrong about `Ai′`. Derived from `u` rather than pasted, then checked — the
/// relation gives `v₁ = −7/72 = −0.097222`, and `Aiᵉ′` measured `4.61e-6` relative at `x = 1e3`,
/// where `ζ = 21082`, implying a first coefficient of `0.0972`. They agree.
fn airy_derivative(alternating: bool) -> Vec<f64> {
    let u = airy(false);
    let mut out = vec![1.0];
    for (k, &u_k) in u.iter().enumerate().skip(1) {
        let kf = k as f64;
        let v = (6.0 * kf + 1.0) / (1.0 - 6.0 * kf) * u_k;
        out.push(if alternating && k % 2 == 1 { -v } else { v });
    }
    out
}

/// `erfcx`'s `Σ (−1)ⁿ(2n−1)!!/2ⁿ`, to be summed in powers of `x²`.
fn erfcx_coefficients() -> Vec<f64> {
    let mut out = vec![1.0];
    let mut c = 1.0;
    for n in 1..=TERMS {
        c *= -((2 * n - 1) as f64) / 2.0;
        out.push(c);
    }
    out
}

/// Coefficients given by power, zero-filled — for the expansions whose other powers vanish.
fn padded(pairs: &[(usize, f64)]) -> Vec<f64> {
    let mut out = vec![0.0; TERMS + 1];
    for &(power, c) in pairs {
        out[power] = c;
    }
    out
}

/// `ζ(s) = Σ n^{-s}`, its own reference once `s` is large enough to converge.
fn dirichlet(s: f64) -> Reference {
    let mut acc = 0.0;
    for n in 1..100_000u32 {
        let term = f64::from(n).powf(-s);
        acc += term;
        if term < 1e-22 * acc {
            return (acc, term);
        }
    }
    (acc, f64::INFINITY)
}

/// One row of the table, so each expansion reads as a single expression.
fn row(
    name: &'static str,
    kernel: fn(f64) -> f64,
    reference: impl Fn(f64) -> Reference + 'static,
    window: (f64, f64),
    budget: f64,
) -> Asymptote {
    (
        name,
        kernel,
        Box::new(reference),
        window.0,
        window.1,
        budget,
    )
}

#[path = "asymptotics/crossover.rs"]
mod crossover;
#[path = "asymptotics/sweep.rs"]
mod sweep;
