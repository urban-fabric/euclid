//! The beta function family: `ln_beta` / `beta`, and the regularized incomplete
//! beta `beta_reg` `Iₓ(a, b)` (the CDF kernel for the Beta, Student's-t, and F
//! distributions). Continued fraction from Numerical Recipes.

use core::cmp::Ordering;

use super::gamma::{GAMMA_EDGEWORTH_MIN, ln_gamma, ln_rising_factorial};
use crate::Real;
use crate::constants::f64::SQRT_TAU;
use crate::math::continued_fraction::{LENTZ_TINY_F64, converged, lentz_step, nonzero_pivot};

/// `ln B(a, b) = ln Γ(a) + ln Γ(b) − ln Γ(a + b)`.
///
/// # Why it is not written that way
///
/// Written literally, that expression subtracts two nearly-equal large logarithms whenever
/// `a + b ≫ min(a, b)` — and the result is `O(ln)` while the terms are `O(a ln a)`. The damage
/// is total, not gradual. Against the exact `ln B(a, 1) = −ln a`:
///
/// ```text
/// a = 1e2   1.7e-14 relative        a = 1e15   4e-2 relative
/// a = 1e8   1.4e-9  relative        a = 1e16   ln B returned 0, and B returned 1
/// ```
///
/// Past `a = 1/ε` the sum `a + b` rounds back to `a`, so `lnΓ(a + b)` and `lnΓ(a)` become
/// *bit-identical* and the whole answer cancels away. `B(1e300, 1)` came back as `1` where the
/// true value is `1e-300`.
///
/// Rewriting it around the rising factorial removes the subtraction entirely:
///
/// ```text
/// ln B(a, b) = ln Γ(b) − ln (a)_b,     (a)_b = Γ(a+b)/Γ(a)
/// ```
///
/// which is the same identity read differently — and
/// [`ln_rising_factorial`] already computes
/// that ratio without ever forming the two logs apart, by an exact product for small integer
/// `b` and an integrated-digamma asymptotic for large `a`. The larger argument is passed as
/// the base so the increment is the short one.
/// # The `1e8` gate is too high, measured
///
/// Below it this falls to `lnΓ(min) − lnRF(max, min)`, and `ln_rising_factorial` is
/// `lnΓ(base+inc) − lnΓ(base)` — a subtraction of two `O(z ln z)` quantities. It therefore
/// inherits their *absolute* error whatever the result's size. Measured against MPFR at 400
/// bits, worst relative error over 120 random `(a, b)` per band:
///
/// ```text
/// band        a,b in [1e-3,1]  [1,50]  [50,1e4]  [1e4,1e7]
/// ulp                    56.2    22.5      65.5      584.0
/// ```
///
/// At `a = 1.07e4, b = 9.44e6` the three terms are `9.5e4`, `1.4e8` and `1.4e8`, leaving
/// `8.4e4` — so `1.4e8·ε = 1.4e-8` of absolute error lands on a result of `8.4e4`, which is the
/// `1.3e-13` measured. The arithmetic is accounted for exactly.
///
/// The branch below already avoids it, by the standard regrouping
/// `(a−½)ln(a/(a+b)) + (b−½)ln(b/(a+b)) − ½ln(a+b) + ½ln 2π`, in which no two large terms
/// subtract. It is gated at `1e8` because it carries **no Stirling remainder**: `μ(a) + μ(b) −
/// μ(a+b)` is dropped, and that is only below rounding once the arguments are enormous.
///
/// # Fixed by carrying the three `μ` terms, which let the gate come down to `20`
///
/// The branch above the gate is *exactly* the leading Stirling form — expanding
/// `(a−½)ln(a/s) + (b−½)ln(b/s) − ½ln s + ½ln 2π` gives back the expression it computes, term
/// for term — so the only thing it was missing is `μ(a) + μ(b) − μ(a+b)`. With those carried the
/// constraint becomes the remainder series' own range rather than the size of what it omits, and
/// [`STIRLING_MIN`] can be `20` instead of `1e8`.
///
/// Measured on a **fixed** grid of `(a, b)`, so both revisions see identical inputs:
///
/// ```text
/// band        [1, 50]   [50, 1e4]   [1e4, 1e7]   [1e7, 1e12]
/// was             6.9         4.8          8.0           5.0
/// now             6.9         1.0          0.9           1.0
/// ```
///
/// Five to nine times better wherever the gate now reaches, and about one ulp throughout.
///
/// `[1, 50]` is unchanged because most of its pairs have `min(a, b) < 20` and still take the
/// path below. I first recorded that band's `6.9` ulp as "a separate matter"; measured, it is
/// not a matter at all. The direct route inherits the cancellation
/// `max|lnΓ term| / |ln B|`, which reaches `10.6` over that grid, and `lnΓ` is itself `1.5` to
/// `3.6` ulp — so the floor this route can be held to is about `38` ulp and it is delivering
/// `6.9`. The worst point, `a = 1.70, b = 3.40`, has a cancellation ratio of only `1.43`, so the
/// residual is not tracking cancellation either: it is ordinary rounding across three `lnΓ`
/// evaluations and a subtraction.
///
/// So `ln_beta` is at its floor across the whole surface now, and the way to lower it further is
/// to lower `ln_gamma`'s — not to do anything here.
///
/// A caution on how this was measured, because it nearly went the other way: an earlier sweep
/// over *randomly* drawn pairs reported no improvement at all, and the reason was that the two
/// runs were not drawing the same pairs. Spot checks then showed `0.8` ulp where the sweep
/// claimed `586`. A before/after on a numerical change has to hold the inputs fixed, which is
/// obvious in retrospect and was not obvious while the random sweep was saying the fix did
/// nothing.
pub(crate) fn ln_beta(a: f64, b: f64) -> f64 {
    let (base, increment) = if a >= b { (a, b) } else { (b, a) };
    if increment >= STIRLING_MIN && base.is_finite() {
        // Combine Stirling's formulas before forming the large terms. Both
        // leading contributions have the same sign, and base + increment can
        // exceed the format even when ln B is finite.
        let ratio = increment / base;
        let log_sum_ratio = ratio.ln_1p();
        // The three Stirling remainders. Dropping them is what the old `1e8` gate was paying
        // for; carrying them is what lets the gate come down to where the cancellation below
        // actually starts to hurt. `μ(a+b)` is formed from the reciprocal rather than from the
        // sum, so an `a + b` past the format's top contributes `0` — which is its limit.
        let correction = stirling_remainder(increment) + stirling_remainder(base)
            - stirling_remainder_of_sum(base, increment);
        return increment * (ratio.ln() - log_sum_ratio) - base * log_sum_ratio
            + (0.5 * (log_sum_ratio - increment.ln()) + crate::constants::f64::HALF_LN_TAU)
            + correction;
    }
    ln_gamma(increment) - ln_rising_factorial(base, increment)
}

/// At or above this smaller parameter, `ln_beta` takes the Stirling-combined form.
///
/// **`20`, lowered from `1e8`.** The gate was that high because the branch dropped the three
/// Stirling remainders, which are only below rounding once the arguments are enormous. With
/// [`stirling_remainder`] carried the constraint becomes the remainder series' own range
/// instead: its first omitted term is `691/(360360 z¹¹)`, which at `z = 20` is `9.4e-18` against
/// a `μ(20)` of `4.2e-3`, and it is added to a result of magnitude `O(z ln z)`. The smallest of
/// `a`, `b` and `a+b` is the binding one and that is always `increment`.
const STIRLING_MIN: f64 = 20.0;

/// `μ(z) = lnΓ(z) − (z−½)ln z + z − ½ln 2π`, from its asymptotic series.
///
/// The same expansion the `I` family uses, and valid here for the same reason — every argument
/// reaching it is at least [`STIRLING_MIN`]. Formed from `1/z` so that an infinite argument
/// gives `0`, which is `μ`'s limit rather than a special case.
fn stirling_remainder(z: f64) -> f64 {
    let inv = 1.0 / z;
    let inv2 = inv * inv;
    inv * (1.0 / 12.0
        + inv2 * (-1.0 / 360.0 + inv2 * (1.0 / 1260.0 + inv2 * (-1.0 / 1680.0 + inv2 / 1188.0))))
}

/// `μ(z)` for `z ≥ 1`, lifting the argument into [`stirling_remainder`]'s range when it is below.
///
/// The series above is asymptotic and says nothing at small `z` — its first omitted term is
/// `691/(360360 z¹¹)`, `9.4e-18` at `z = 20` and `1.9e-3` at `z = 1`. That, and nothing else, is
/// why the centred prefactor was gated on `min(a, b) ≥ 20`.
///
/// [`stirling_remainder_step`](crate::math::gamma::stirling_remainder_step) is the exact identity
/// `μ(z) − μ(z+1)`, so repeating it costs one step per unit and loses nothing at any of them.
/// Verified against `lnΓ` at 400 bits: `μ(1)` through `μ(19.5)` reconstruct to `2e-16` absolute.
///
/// **It delegates to the copy above rather than to `math::gamma`'s.** The two series are the same
/// expansion written twice and are *not* bit-identical — this one ends `inv2 / 1188.0` where that
/// one ends `inv2 * (1.0/1188.0)`, and `1/1188` is not representable. Taking the other would move
/// every `a, b ≥ 20` result by a last bit of `μ` for no reason.
fn stirling_remainder_from_one(z: f64) -> f64 {
    debug_assert!(z >= 1.0, "the climb is one step per unit; see CENTRED_MIN");
    let mut climbed = 0.0;
    let mut lifted = z;
    for _ in 0..climb_steps(z) {
        climbed += crate::math::gamma::stirling_remainder_step(lifted);
        lifted += 1.0;
    }
    climbed + stirling_remainder(lifted)
}

/// How many unit steps the climb takes before `z` reaches the series' range.
///
/// Lifted out of [`stirling_remainder_from_one`]'s loop because the loop's own condition could
/// not be tested there. `while lifted < STIRLING_MIN` mutated to `<=` runs one extra step, and
/// `μ(20) = step(20) + μ(21)` identically — so the *value* is unchanged and only the arithmetic
/// path differs. No assertion on `μ` can see it.
///
/// As a count it is a value, and `climb_steps(20) == 0` says exactly what the comparison means:
/// a `z` already in range climbs nothing. That is the whole content of the boundary.
fn climb_steps(z: f64) -> u32 {
    let mut lifted = z;
    let mut steps = 0;
    while lifted < STIRLING_MIN {
        lifted += 1.0;
        steps += 1;
    }
    steps
}

/// `μ(base + increment)` without forming the sum, which can leave the format while `ln B` does
/// not — the case the branch's own comment flags.
fn stirling_remainder_of_sum(base: f64, increment: f64) -> f64 {
    let sum = base + increment;
    if sum.is_finite() {
        stirling_remainder(sum)
    } else {
        0.0
    }
}

/// The beta function `B(a, b)`.
pub(crate) fn beta(a: f64, b: f64) -> f64 {
    ln_beta(a, b).exp()
}

pub(crate) mod incomplete;

pub(crate) use incomplete::beta_reg;

#[cfg(test)]
mod stirling_tests {
    use super::*;

    /// A grid of `(a, b)` straddling [`STIRLING_MIN`] in `min(a, b)`, so both arms are covered
    /// and the handover sits inside the sweep rather than beside it.
    const SHAPES: &[f64] = &[0.5, 2.0, 7.0, 15.0, 19.0, 21.0, 50.0, 500.0, 1e5];

    /// `ln B(a, b) = ln B(b, a)`, exactly.
    ///
    /// Not approximately: `ln_beta` orders its arguments before doing anything else, so
    /// symmetry is a property of that ordering and any disagreement means the ordering broke.
    /// Measured `0` over the whole grid, so the assertion is equality.
    #[test]
    fn ln_beta_is_exactly_symmetric_across_the_stirling_handover() {
        for &a in SHAPES {
            for &b in SHAPES {
                assert_eq!(ln_beta(a, b), ln_beta(b, a), "at a = {a}, b = {b}");
            }
        }
    }

    /// `ln B(a+1, b) = ln B(a, b) + ln a − ln(a+b)`, with no reference at all.
    ///
    /// The property that guards the Stirling arm: it holds for every `(a, b)`, it compares two
    /// *different* evaluations of the kernel — one of which may take the other arm — and it
    /// needs nothing but `ln`. A handover that lands in the wrong place shows up as a step.
    ///
    /// **Absolute, not relative.** `ln B` passes near zero for small shapes, and a relative
    /// figure there measures the conditioning of that crossing rather than the kernel: the
    /// worst relative residual is `2.2e-14` at `a = 2, b = 0.5`, where `ln B` is `0.288`, and
    /// the same point is unremarkable in absolute terms. For a logarithm the absolute error is
    /// what the exponential downstream will see.
    ///
    /// One caveat on the bound, since it is easy to read as stronger than it is: the worst
    /// absolute residual is `5.8e-11` at `a = b = 1e5`, where `ln B` is `−138629` — a relative
    /// `4.2e-16`. So a single absolute bound is set by the largest `|ln B|` in the grid and is
    /// correspondingly loose for the small shapes. It catches a gross regression in either arm,
    /// which is what it is for; it is not a tight statement about small `(a, b)`.
    #[test]
    fn ln_beta_satisfies_its_recurrence_across_the_stirling_handover() {
        let mut worst = 0.0_f64;
        let mut worst_at = (0.0, 0.0);
        for &a in SHAPES {
            for &b in SHAPES {
                let want = ln_beta(a, b) + a.ln() - (a + b).ln();
                let residual = (ln_beta(a + 1.0, b) - want).abs();
                if residual > worst {
                    worst = residual;
                    worst_at = (a, b);
                }
            }
        }
        assert!(
            worst < 1.2e-09,
            "the recurrence drifts to {worst:e} at a = {}, b = {}; it measured 5.821e-11 \
             when the Stirling arm was given its remainder terms",
            worst_at.0,
            worst_at.1
        );
    }

    /// The climb's loop bound, as a count.
    ///
    /// `while lifted < STIRLING_MIN` mutated to `<=` runs one extra step, and
    /// `μ(20) = step(20) + μ(21)` identically — so no assertion on `μ` can see it. As a count the
    /// boundary is a value: a `z` already in range climbs nothing.
    #[test]
    fn the_climb_takes_no_step_once_the_series_range_is_reached() {
        assert_eq!(climb_steps(STIRLING_MIN), 0, "already in range");
        assert_eq!(climb_steps(STIRLING_MIN + 5.0), 0, "past it");
        assert_eq!(
            climb_steps(STIRLING_MIN.next_down()),
            1,
            "one ulp short, one step"
        );
        assert_eq!(climb_steps(STIRLING_MIN - 1.0), 1);
        assert_eq!(
            climb_steps(1.0),
            19,
            "the longest climb a caller can ask for"
        );
        // And the climb is what makes `μ` reachable below the series: the reconstruction has to
        // agree with the series' own value at the point where the two meet.
        let direct = stirling_remainder(STIRLING_MIN);
        assert_eq!(stirling_remainder_from_one(STIRLING_MIN), direct);
    }
}
