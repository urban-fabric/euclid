//! Numerical steps shared by the two precisions of the incomplete-beta kernels.
//!
//! The orientation choice and the fraction's prefactor are here; the fraction *itself* is not,
//! because its iteration cap, its convergence tolerance and its Lentz floor are three separate
//! per-format measurements, and `beta_fraction_cdf` exists in each kernel to bind them. That is
//! the split the tree's layout asks for: the arithmetic that is the same number in both formats
//! moves up, the orchestration that names each format's own constants stays down.

use crate::HostReal;

/// Which orientation of the continued fraction converges faster at this argument.
///
/// The fraction converges slowest at the distribution's mean, `a/(a+b)`, so the reflected form
/// `I_x(a,b) = 1 − I_{1−x}(b,a)` is taken on the far side of it. The split point is
/// `(a+1)/(a+b+2)` rather than the mean itself — the mode of a shifted beta, which is where the
/// two orientations cost the same.
///
/// At `(2, 4)` it is exactly `3/8`, a dyadic value, so the switch is exact; the kernels' own
/// tests pin that boundary.
pub(crate) fn use_lower_beta_fraction<T: HostReal>(a: T, b: T, x: T) -> bool {
    x < (a + T::ONE) / (a + b + T::TWO)
}

/// The common factor `x^a(1−x)^b/B(a,b)` of the direct and reflected fractions.
///
/// Formed in the log domain, because all three terms are `O((a+b)ln 2)` while their sum is
/// `O(1)`: at `a = b = 1e3` the exponent carries `1.386(a+b)·ε` of absolute error and `exp` hands
/// it back as relative error. That is the whole reason each format has a handover to an Edgeworth
/// approximation, and why the handover sits far lower in `f32`.
///
/// `ln B(a, b)` is a parameter rather than computed here: the two formats reach it through
/// different arms — a rising-factorial identity below their own Stirling threshold and a combined
/// Stirling expansion above it — and that threshold is a per-format measurement.
pub(crate) fn beta_fraction_prefactor<T: HostReal>(a: T, b: T, x: T, log_beta: T) -> T {
    (a * x.ln() + b * (-x).ln_1p() - log_beta).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_orientation_switches_at_the_shifted_mode() {
        // `(a+1)/(a+b+2)` is `3/8` at `(2,4)`, exactly representable, so the switch is exact.
        let x = 0.375_f64;
        assert!(use_lower_beta_fraction(2.0, 4.0, x.next_down()));
        assert!(!use_lower_beta_fraction(2.0, 4.0, x));
        assert!(!use_lower_beta_fraction(2.0, 4.0, x.next_up()));
        // Swapping the shapes reflects the split point, which is what makes the two orientations
        // cover the whole interval between them.
        assert!(use_lower_beta_fraction(4.0, 2.0, x));
        assert_eq!(
            use_lower_beta_fraction(2.0_f64, 4.0, 0.5),
            !use_lower_beta_fraction(4.0, 2.0, 0.5)
        );
        // Symmetric shapes split at one half.
        assert!(use_lower_beta_fraction(3.0_f64, 3.0, 0.5_f64.next_down()));
        assert!(!use_lower_beta_fraction(3.0_f64, 3.0, 0.5));
        let x = 0.375_f32;
        assert!(use_lower_beta_fraction(2.0, 4.0, x.next_down()));
        assert!(!use_lower_beta_fraction(2.0, 4.0, x));
    }

    #[test]
    fn the_prefactor_is_the_integer_shape_polynomial() {
        // `x^a(1−x)^b/B(a,b)` at `(2,3)` is `12x²(1−x)³`, since `B(2,3) = 1/12`.
        let log_beta = (1.0_f64 / 12.0).ln();
        for x in [2.0_f64.powi(-50), 0.125, 0.25, 0.75, 1.0_f64.next_down()] {
            let expected = 12.0 * x.powi(2) * (1.0 - x).powi(3);
            let actual = beta_fraction_prefactor(2.0, 3.0, x, log_beta);
            assert!(
                (actual / expected - 1.0).abs() <= 3e-14,
                "x={x}, actual={actual}"
            );
        }
        // Zero at both endpoints of the support, where one power vanishes.
        assert_eq!(beta_fraction_prefactor(2.0_f64, 3.0, 0.0, log_beta), 0.0);
        assert_eq!(beta_fraction_prefactor(2.0_f64, 3.0, 1.0, log_beta), 0.0);
        // The normaliser enters as a straight division, so doubling `B` halves the prefactor.
        let doubled = beta_fraction_prefactor(2.0_f64, 3.0, 0.25, log_beta + 2.0_f64.ln());
        let plain = beta_fraction_prefactor(2.0_f64, 3.0, 0.25, log_beta);
        assert!(
            (doubled / plain - 0.5).abs() < 1e-14,
            "{doubled} vs {plain}"
        );
        // And in `f32`, against the same polynomial.
        let log_beta = (1.0_f32 / 12.0).ln();
        let actual = beta_fraction_prefactor(2.0_f32, 3.0, 0.25, log_beta);
        assert!(
            (actual / (12.0 * 0.0625 * 0.421_875) - 1.0).abs() <= 2e-5,
            "{actual}"
        );
    }
}

/// `x^a(1−x)^b/B(a,b)`, formed about the distribution's mode so nothing cancels.
///
/// # The log form sums `±(a+b)ln 2` to get `O(1)`
///
/// [`beta_fraction_prefactor`] is `exp(a ln x + b ln(1−x) − ln B(a,b))`, and at `a = b = 1000`,
/// `x = ½` those three terms are `−693`, `−693` and `+1384` summing to `−1.9`. `exp` turns the
/// surviving absolute error into relative error, so the answer carried about `|largest term|`
/// ulp — **382 ulp measured at the mode, where the floor is 2**. Exactly the defect
/// `log_scaled_gamma` had, in the two-parameter case.
///
/// # The rearrangement, and why the mode is the right centre
///
/// Substituting Stirling for all three gammas and grouping the logarithms,
///
/// ```text
/// ln P = a·ln(1+u) + b·ln(1+v) + ½·ln(ab/2πs) + μ(s) − μ(a) − μ(b)
/// ```
///
/// with `s = a+b`, `u = xs/a − 1` and `v = (1−x)s/b − 1`. Both `u` and `v` vanish at `x = a/s`,
/// so at the mode the whole exponent collapses to `½ln(ab/2πs)` — `2.19` at `a = b = 1000`
/// against the `1384` the old form was handling.
///
/// # And the two logarithms still cancel, which is the part that matters
///
/// `au + bv = 0` identically: `b·v = (1−x)s − b = a − xs = −(xs − a) = −a·u`. So
/// `a·ln(1+u) + b·ln(1+v)` has its *linear* terms cancel exactly, leaving `O(u²)` — which means
/// computing it as written throws away the digits the rearrangement just bought. Written through
/// [`ln_1p_minus_linear`](super::gamma::ln_1p_minus_linear) instead, the subtraction never
/// happens and both terms come out the same sign, so they cannot cancel against each other
/// either.
///
/// `w = xs − a` is formed with one fused multiply-add. It is a difference of nearly equal
/// quantities near the mode — which is the point — and `mul_add` gives it a single rounding
/// where `x * s - a` would give it two and lose the cancellation.
/// Whether the mode-centred form is the better one at this argument.
///
/// Both `u` and `v` have to stay inside [`ln_1p_minus_linear`](super::gamma::ln_1p_minus_linear)'s
/// series, and because
/// `b·v = −a·u` that is not one condition but two: `|v| ≤ ½` means `|u| ≤ ½·b/a`, so for
/// lopsided parameters the band is narrow even though it is still centred on the mode.
///
/// Outside it the rearrangement is **worse than what it replaces**, which is why this is a
/// predicate and not an unconditional substitution. Measured at `a = 500`, `b = 30`: the old
/// form `180` ulp, the mode-centred one `394`. There the dominant term is `b·L(v)` with `v`
/// past the radius, so `L` falls back to its own difference and the centring buys nothing while
/// costing the `ln B` the old form had measured to one ulp.
pub(crate) fn use_beta_power_terms<T: HostReal>(a: T, b: T, x: T) -> bool {
    let sum = a + b;
    let offset = x.mul_add(sum, -a);
    (offset / a).abs() <= T::HALF && (offset / b).abs() <= T::HALF
}

pub(crate) fn beta_power_terms<T: HostReal>(a: T, b: T, x: T, remainders: T) -> T {
    let sum = a + b;
    let offset = x.mul_add(sum, -a);
    let exponent = a * super::gamma::ln_1p_minus_linear(offset / a)
        + b * super::gamma::ln_1p_minus_linear(-offset / b)
        + T::HALF * (a.ln() + b.ln() - sum.ln())
        - T::from_f64(crate::constants::f64::HALF_LN_TAU)
        + remainders;
    exponent.exp()
}

#[cfg(test)]
mod power_term_tests {
    use super::*;

    /// The two prefactor forms are the same quantity, where the log form is still trustworthy.
    ///
    /// They are algebraically identical, so this pins the rearrangement's constants — a wrong
    /// `½ln(ab/2πs)`, a dropped `HALF_LN_TAU`, a sign slip on a Stirling remainder. None of
    /// those makes the function ragged; each shifts it smoothly, which is what a value
    /// assertion on `beta_reg` would straddle without noticing.
    ///
    /// # The bar is the *log* form's error, and that is the whole difficulty with this check
    ///
    /// Asserted at small shapes only, because the log form's error is about `|ln B(a,b)|·ε` and
    /// grows: `32` ulp at `a = b = 20`, `1384` at `a = b = 1000`. Comparing the two up there
    /// would be measuring the form being replaced. Even at these shapes the measured worst is
    /// **498** ulp — all of it the log form's, since the centred form's own terms are `O(1)`
    /// there. So a bar of `1200` is not slack: it is one order of magnitude of headroom over a
    /// disagreement that is dominated by the arm this exists to retire, and it still catches a
    /// mis-grouped constant, which would be wrong by a factor rather than by ulps.
    #[test]
    fn the_two_prefactor_forms_agree_where_the_log_form_is_sound() {
        let mut worst = (0.0_f64, 0.0_f64);
        for &(a, b) in &[
            (20.0_f64, 20.0_f64),
            (20.0, 25.0),
            (31.0, 24.0),
            (40.0, 40.0),
        ] {
            let sum = a + b;
            let remainders = crate::math::gamma::stirling_remainder(sum)
                - crate::math::gamma::stirling_remainder(a)
                - crate::math::gamma::stirling_remainder(b);
            // `ln B` from the same Stirling pieces, so the comparison is of the *grouping*
            // rather than of two different `ln B` implementations.
            let log_beta = (a - 0.5) * a.ln() + (b - 0.5) * b.ln() - (sum - 0.5) * sum.ln()
                + crate::constants::f64::HALF_LN_TAU
                - remainders;
            for step in -4..=4 {
                let x = (a / sum) * (1.0 + f64::from(step) * 0.1);
                if !(0.0..1.0).contains(&x) || !use_beta_power_terms(a, b, x) {
                    continue;
                }
                let centred = beta_power_terms(a, b, x, remainders);
                let logged = beta_fraction_prefactor(a, b, x, log_beta);
                let ulps = (centred - logged).abs() / crate::Real::ulp(centred);
                if ulps > worst.0 {
                    worst = (ulps, x);
                }
                assert!(
                    ulps <= 1200.0,
                    "at a={a}, b={b}, x={x}: centred {centred:e} against logged {logged:e}, \
                     {ulps} ulp apart"
                );
            }
        }
        assert!(
            worst.0 > 0.0,
            "the two forms agreed bit-exactly, so this compared nothing"
        );
    }

    /// The band is centred on the mode and shrinks with the parameters' imbalance.
    #[test]
    fn the_power_term_band_is_centred_and_narrows_with_the_imbalance() {
        // At the mode both offsets are zero, so the band always admits it.
        for &(a, b) in &[(20.0_f64, 20.0_f64), (500.0, 30.0), (30.0, 500.0)] {
            assert!(
                use_beta_power_terms(a, b, a / (a + b)),
                "the mode at a={a}, b={b}"
            );
        }
        // `b·v = −a·u`, so `|v| ≤ ½` is the binding condition when `a > b`: the band is
        // `|u| ≤ ½·b/a`, an eightieth here rather than a half.
        let (a, b) = (800.0_f64, 10.0_f64);
        let mode = a / (a + b);
        assert!(use_beta_power_terms(a, b, mode * (1.0 + 0.005)));
        assert!(!use_beta_power_terms(a, b, mode * (1.0 + 0.05)));
        // And symmetric: swapping the parameters reflects the band about a half.
        assert!(use_beta_power_terms(b, a, 1.0 - mode * (1.0 + 0.005)));
    }
}
