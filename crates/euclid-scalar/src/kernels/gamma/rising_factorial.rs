//! Gamma ratios using a bounded product or differenced Stirling expansion.

use super::ln_gamma;

/// Lower edge of the differenced-Stirling route in [`ln_rising_factorial`].
///
/// Below it the direct `lnΓ(a+n) − lnΓ(a)` is used, where the cancellation costs about
/// `ε·a·ln a` — `1.9e-13` absolute at the threshold, and shrinking from there. Above it the
/// differenced series is exact to rounding.
///
/// It was `1e4`, with a single correction pair. That left `a = 1e4` itself on the *direct*
/// side, where the two log-gammas are both `82108` and their difference is `4.6`: `1.8e-11` of
/// absolute error, which showed up as `4.7e-13` in `ln B(½, 1e4)`. Three correction pairs
/// bring the threshold down to where the direct route is still harmless.
const STIRLING_RATIO_MIN: f64 = 300.0;

/// `ln Γ(a + n) − ln Γ(a)`, the log rising factorial, computed without the cancellation of
/// subtracting two nearly-equal `ln_gamma` values.
///
/// Three regimes:
///
/// 1. **`n` a small non-negative integer** — sum `ln(a+k)` directly. All terms are the same
///    sign, so there is no cancellation at any `a`, and this is the dominant use (binomial and
///    beta-binomial weights, Dirichlet updates).
/// 2. **large `a`** — integrate the digamma asymptotic, `lnΓ(a+n) − lnΓ(a) = ∫ₐ^{a+n} ψ(t)dt`
///    with `ψ(t) ≈ ln t − 1/(2t) − 1/(12t²)`. Rearranged so every term is formed with `ln_1p`
///    on `n/a` rather than by subtracting two large logarithms:
///    `n·ln a + (a+n)·ln1p(n/a) − n − ½·ln1p(n/a) + 1/(12(a+n)) − 1/(12a)`.
/// 3. **otherwise** — the direct difference, which is accurate when the two terms are not
///    nearly equal.
pub(crate) fn ln_rising_factorial(a: f64, n: f64) -> f64 {
    if n == 0.0 {
        return 0.0;
    }
    if !a.is_finite() || !n.is_finite() {
        return ln_gamma(a + n) - ln_gamma(a);
    }
    // (1) small integer n: an exact product in the log domain.
    if (1.0..=64.0).contains(&n) && n.fract() == 0.0 {
        let mut total = 0.0f64;
        for k in 0..n as u32 {
            total += (a + f64::from(k)).ln();
        }
        return total;
    }
    // (2) large a: integrated digamma asymptotic, cancellation-free in n/a.
    if a.min(a + n) > STIRLING_RATIO_MIN {
        return stirling_ratio(a, n);
    }
    // (3) the terms are not nearly equal; the direct difference is fine.
    ln_gamma(a + n) - ln_gamma(a)
}

/// Stirling ratio when both gamma arguments are large enough for the series.
fn stirling_ratio(a: f64, n: f64) -> f64 {
    let end = a + n;
    let l1p = (n / a).ln_1p();
    n * a.ln() + end * l1p - n - 0.5 * l1p + stirling_correction(end) - stirling_correction(a)
}

/// The first three Bernoulli terms of the log-gamma Stirling expansion.
fn stirling_correction(z: f64) -> f64 {
    let inv = 1.0 / z;
    let inv2 = inv * inv;
    inv * (1.0 / 12.0 + inv2 * (-1.0 / 360.0 + inv2 / 1260.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stirling_correction_matches_the_bernoulli_series() {
        // Independent 70-digit sums through B12. The first omitted term in
        // this kernel is smaller than 3e-21 over these arguments.
        for (z, expected) in [
            (300.0, 0.000_277_777_674_897_445_9),
            (301.0, 0.000_276_854_826_159_367_3),
            (500.0, 0.000_166_666_644_444_469_84),
            (1000.0, 0.000_083_333_330_555_556_35),
        ] {
            let actual = stirling_correction(z);
            assert!(
                (actual - expected).abs() <= 2e-19,
                "{z}: {actual:e} vs {expected:e}"
            );
        }
    }

    #[test]
    fn stirling_ratio_preserves_small_signed_increments() {
        // Independent MPFR lgamma differences, evaluated at 256 bits.
        for (a, n, expected) in [
            (301.0, 0.25, 1.426_466_018_271_043_8),
            (512.0, 0.375, 2.339_142_833_961_544_5),
            (1024.0, -0.625, -4.331_673_786_167_135),
            (1e40, 0.5, 46.051_701_859_880_914),
        ] {
            let actual = stirling_ratio(a, n);
            assert!((actual - expected).abs() <= 2e-15 * expected.abs());
        }
    }

    /// Independently check the empty product and accuracy at the route boundary.
    #[test]
    fn the_rising_factorial_agrees_at_the_stirling_boundary() {
        // The empty product returns before any comparison sees `n`.
        assert_eq!(ln_rising_factorial(3.0, 0.0), 0.0);
        assert_eq!(ln_rising_factorial(1e9, 0.0), 0.0);
        assert_eq!(
            crate::kernels::f32::gamma::ln_rising_factorial(3.0, 0.0),
            0.0
        );

        // At the threshold itself, both branches agree — so moving the comparison by one
        // argument cannot change an answer. Compared against the well-conditioned sum, which is
        // available because an integer count makes the product exact.
        // Across the join by one ulp of the threshold, not by one *unit* of it. A first draft
        // compared `a = 299` with `a = 301` and failed at `n = 250` by `1.2` — correctly, since
        // `d/da ln (a)_n = ψ(a+n) − ψ(a) ≈ ln((a+n)/a)`, which is `0.6` there. That is the
        // function's own slope, not a step at the join, and a continuity test has to step by an
        // ulp to tell the two apart.
        let threshold = STIRLING_RATIO_MIN;
        let gap = threshold * f64::EPSILON;
        for &count in &[70.0f64, 100.5, 250.0] {
            let above = ln_rising_factorial(threshold + gap, count);
            let below = ln_rising_factorial(threshold - gap, count);
            let step = ((above - below) / above).abs();
            assert!(
                above.is_finite() && below.is_finite() && step < 1e-13,
                "ln (a)_n must not step across its threshold at n = {count}: \
                 {below:e} then {above:e}, relative step {step:e}"
            );
        }

        // And the exact-threshold argument itself, against the independent sum for integer
        // counts above 64 — the range where branch (1) does not apply and branch (2) does.
        for &count in &[65u32, 100, 300] {
            let n = f64::from(count);
            let mut reference = 0.0f64;
            for k in 0..count {
                reference += (threshold + f64::from(k)).ln();
            }
            let direct = ln_rising_factorial(threshold, n);
            let relative = ((direct - reference) / reference).abs();
            assert!(
                relative < 1e-13,
                "at the threshold exactly, ln (a)_{count} must match the sum: \
                 {direct:e} against {reference:e}, relative {relative:e}"
            );
        }
    }
}
