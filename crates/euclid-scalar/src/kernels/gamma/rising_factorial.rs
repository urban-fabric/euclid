//! Gamma ratios using a bounded product or differenced Stirling expansion.

use super::ln_gamma;

/// Lower edge of the differenced-Stirling route in [`ln_rising_factorial`].
///
/// **`30`, down from `300`, and the note that stood here was reasoning about the wrong error.**
///
/// It said the direct `lnΓ(a+n) − lnΓ(a)` costs "about `ε·a·ln a` — `1.9e-13` absolute at the
/// threshold, and shrinking from there". The absolute figure is right. But the *answer* is
/// `≈ n·ln a`, so the **relative** error is `ε·a·ln a / (n·ln a) = ε·a/n` — which does not depend
/// on `a` alone and does not shrink with it. It is governed by `a/n`, and for small `n` it is
/// unbounded. Nothing in the harness had ever swept this function, so nobody had seen it:
///
/// ```text
/// a \ n      1e-8     1e-5     1e-3      0.1
/// 10       8.3e7    7.8e5     6635       42   ulp
/// 100      8.9e9    2.7e6    19756     1957
/// 300     2.3e10    1.2e7   335097     1264
/// ```
///
/// Two changes remove it, and both were needed. [`stirling_correction_increment`] fixes the
/// asymptotic route itself, which had the same defect in miniature. Then the threshold can fall
/// to where the three-term series' own truncation *increment* — about `7/(1680·a⁸·ln a)`
/// relative — is still under an ulp, which is near `a = 30`; at `20` it costs 65 to 216 ulp.
/// Below `30`, route (3) walks `a` up rather than differencing.
const STIRLING_RATIO_MIN: f64 = 30.0;

/// Above this `a/|n|` the direct difference is abandoned for the recurrence.
///
/// The direct `lnΓ(a+n) − lnΓ(a)` carries about `ε·a/n` *relative*, so the ratio is the whole
/// decision — not `a`, which is what the old threshold tested. `8` keeps it to a handful of ulp.
///
/// It is bounded below as well as above, and the lower bound was found by a test rather than by
/// reasoning: applying the recurrence unconditionally regressed `B(½, ½) = π` from `2.8` to
/// `3.2` ulp. At `a = n = ½` the direct difference is `lnΓ(1) − lnΓ(½)`, one term of which is
/// *exactly zero* — there is no cancellation to avoid, and thirty steps of recurrence only add
/// their own roundings. Where the ratio is small the naive form is not merely adequate, it is
/// better.
const DIRECT_RATIO_MAX: f64 = 8.0;

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
    // (3) small a with `a` well above `n`: walk `a` up rather than differencing two log-gammas.
    //
    // `lnΓ(a+n) − lnΓ(a) = [lnΓ(a+n+1) − lnΓ(a+1)] − ln1p(n/a)`, from `Γ(z+1) = zΓ(z)` on both
    // sides. Every accumulated term has the same sign as the last, so nothing cancels — the
    // same property that makes the integer-`n` product route exact. At most thirty-one steps,
    // since one addition takes any positive `a` to at least one.
    if a > 0.0 && a + n > 0.0 && a > DIRECT_RATIO_MAX * n.abs() {
        let mut shift = 0.0_f64;
        let mut base = a;
        while base < STIRLING_RATIO_MIN && base + n > 0.0 {
            shift -= (n / base).ln_1p();
            base += 1.0;
        }
        if base >= STIRLING_RATIO_MIN {
            return stirling_ratio(base, n) + shift;
        }
    }
    // (4) off the positive axis, where `lnΓ` is `NaN` anyway and there is nothing to preserve.
    ln_gamma(a + n) - ln_gamma(a)
}

/// Stirling ratio when both gamma arguments are large enough for the series.
fn stirling_ratio(a: f64, n: f64) -> f64 {
    let end = a + n;
    let l1p = (n / a).ln_1p();
    n * a.ln() + end * l1p - n - 0.5 * l1p + stirling_correction_increment(a, n)
}

/// The first three Bernoulli terms of the log-gamma Stirling expansion.
///
/// No longer evaluated in anger — [`stirling_correction_increment`] replaced it, because the
/// difference of two of these is where the small-`n` error lived. Kept as the *reference* that
/// pins the increment form: for a step large enough that subtracting two corrections is sound,
/// the two must agree, and that is a check the increment's own algebra cannot provide itself.
#[cfg(test)]
fn stirling_correction(z: f64) -> f64 {
    let inv = 1.0 / z;
    let inv2 = inv * inv;
    inv * (1.0 / 12.0 + inv2 * (-1.0 / 360.0 + inv2 / 1260.0))
}

/// `stirling_correction(a+n) − stirling_correction(a)`, with the `n` factored out.
///
/// # Taking the difference term by term is the whole point
///
/// `corr` is `O(1/a)` while its *difference* over a step of `n` is `O(n/a²)`, so subtracting two
/// evaluations throws away `log₂(a/n)` bits — and for small `n` that is most of them. At
/// `a = 600`, `n = 1e-8` the two corrections are both `1.389e-4` and their difference is
/// `2.3e-15`: eleven digits gone, which measured as **1746 ulp** of the answer even though every
/// other term in [`stirling_ratio`] was fine.
///
/// Each power difference factors through `uᵏ − vᵏ = (u − v)·Σ uⁱvᵏ⁻¹⁻ⁱ`, with `u = 1/(a+n)` and
/// `v = 1/a`. The leading `u − v` is formed as `−(n/a)·u`, which is exact and carries the whole
/// of the `n` dependence, and every remaining factor is a power of a *reciprocal* — so nothing
/// cancels and nothing grows.
///
/// # Writing it with `(a+n)ᵏ − aᵏ` in the numerator overflows
///
/// The obvious algebra puts the binomial on top: `1/(a+n)⁵ − 1/a⁵ =
/// −n(5a⁴ + 10a³n + ⋯ + n⁴)/(a⁵(a+n)⁵)`. That is correct and unusable. At `a = 30`,
/// `n = 1e200` both `n⁴` and `(a+n)⁵` leave the format, the quotient is `inf/inf`, and
/// `ln_beta(1e-8, 1e200)` came back `NaN` — caught by the symmetry identity rather than by any
/// accuracy check, since `NaN` is not a wrong number so much as an absent one.
fn stirling_correction_increment(a: f64, n: f64) -> f64 {
    let (u, v) = (1.0 / (a + n), 1.0 / a);
    // `u − v`, without the subtraction: `1/(a+n) − 1/a = −(n/a)/(a+n)`.
    let step = -(n / a) * u;
    let (u2, v2) = (u * u, v * v);
    let third = step * (u2 + u * v + v2);
    let fifth = step * (u2 * u2 + u2 * u * v + u2 * v2 + u * v2 * v + v2 * v2);
    step / 12.0 - third / 360.0 + fifth / 1260.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The increment form agrees with the difference of two corrections, where that is sound.
    ///
    /// Only for steps large relative to `a`: the whole reason the increment exists is that the
    /// difference loses `log₂(a/n)` bits, so comparing them at small `n` would be comparing
    /// against the thing being replaced. At `n ≈ a` there is nothing to lose and the two must
    /// agree to rounding.
    #[test]
    fn the_increment_agrees_with_the_difference_where_the_difference_is_sound() {
        let mut worst = 0.0_f64;
        for &a in &[30.0_f64, 55.0, 120.0, 400.0, 2000.0] {
            for &n in &[a * 0.5, a, a * 2.0] {
                let increment = stirling_correction_increment(a, n);
                let differenced = stirling_correction(a + n) - stirling_correction(a);
                let relative = (increment - differenced).abs() / differenced.abs();
                worst = worst.max(relative);
                assert!(
                    relative <= 1e-14,
                    "at a={a}, n={n} the increment is {increment:e} and the difference \
                     {differenced:e}"
                );
            }
        }
        assert!(
            worst > 0.0,
            "the two forms were bit-identical, so this compared nothing"
        );
    }

    /// `lnΓ(a+n) − lnΓ(a) → n·ψ(a)` as `n → 0`, checked against euclid's own digamma.
    ///
    /// # This is the check the function never had
    ///
    /// Its error is `ε·a/n` relative, so it is worst where `n` is smallest — and every existing
    /// test used `n` of order one or an integer, which is where the function is at its best.
    /// Swept against a 400-bit reference the defect reached **6.5e13 ulp**; the identity below
    /// reaches the same corner with no reference at all.
    ///
    /// `n = 2^-70` is small enough that the quadratic term `n²ψ′(a)/2` is far under an ulp of
    /// `n·ψ(a)`, so `ψ` alone is the answer. It is also far below `ulp(a)`, which is the point:
    /// any route that forms `a + n` in `f64` first returns *exactly zero* here, and the correct
    /// answer is an ordinary small number. `digamma` is a different kernel reached by a
    /// different path, so this is not the recurrence checking itself.
    #[test]
    fn the_rising_factorial_tends_to_n_times_digamma() {
        let tiny = f64::from_bits(0x3BB0_0000_0000_0000); // 2^-70, exactly representable
        let mut worst = (0.0_f64, 0.0_f64);
        for &a in &[0.5_f64, 1.0, 2.5, 7.0, 15.0, 29.0, 31.0, 100.0, 5000.0] {
            let got = ln_rising_factorial(a, tiny);
            let expected = tiny * super::super::digamma(a);
            let ulps = (got - expected).abs() / crate::Real::ulp(expected);
            if ulps > worst.0 {
                worst = (ulps, a);
            }
            assert!(
                ulps <= 64.0,
                "ψ({a}) says the rising factorial at n = 2^-70 is {expected:e}, got {got:e} \
                 ({ulps} ulp)"
            );
        }
        assert!(
            worst.0 > 0.0,
            "every argument matched exactly, so this compared nothing"
        );
    }

    /// `rising(a, m+n) = rising(a, m) + rising(a+m, n)`, which crosses the route boundaries.
    ///
    /// The three calls land on different arms — the pairs are chosen so that at least one side
    /// takes the recurrence and the other the asymptotic series — so this constrains the arms
    /// against each other rather than any one of them against itself.
    #[test]
    fn the_rising_factorial_is_additive_across_its_routes() {
        for &(a, m, n) in &[
            (2.0_f64, 0.001_f64, 40.0_f64),
            (10.0, 1e-6, 25.0),
            (28.0, 0.5, 3.0),
            (0.75, 1e-9, 100.0),
            (250.0, 1e-4, 1.5),
        ] {
            let whole = ln_rising_factorial(a, m + n);
            let split = ln_rising_factorial(a, m) + ln_rising_factorial(a + m, n);
            let ulps = (whole - split).abs() / crate::Real::ulp(whole);
            assert!(
                ulps <= 16.0,
                "at a={a}, m={m}, n={n}: whole {whole} against split {split} ({ulps} ulp)"
            );
        }
    }

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
