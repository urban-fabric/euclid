//! Where the modified Bessel error peaks relative to its crossover. Part of [`super`].

use super::*;

/// The modified Bessel kernels are least accurate just below their asymptotic crossover.
///
/// This pins the *shape* of the header's finding rather than only its worst case. `I₀ᵉ` and `I₁ᵉ`
/// hand over from the peak-factored series to the asymptotic route at `I_ASYMPTOTIC_MIN`, and
/// against the expansion the series is tens of ulp just below that point and about two ulp above.
///
/// Asserting the contrast is worth more than asserting either number alone, because it says the
/// crossover is placed where the series has genuinely run out: were the series still sharp below
/// it the handover is early and costs speed, and were the error still climbing above it the
/// handover is late and costs accuracy.
///
/// # The windows moved when the threshold did, which is the test earning its place
///
/// This compared `[1e3, 1e4]` against `[1e4, 1e8]` while `I_ASYMPTOTIC_MIN` was `1e4`. Lowering
/// it to `20` put both of those windows on the *same* route and the assertion failed — correctly,
/// because its premise had gone. The windows are now `[18, 20]` and `[20, 100]`.
///
/// `18` rather than lower is forced by the reference, not by taste: the expansion is itself the
/// thing being compared against, and its optimal truncation is `0.125` ulp at `20`, `8.7` at `16`
/// and `27,212` at `12`. Below about `18` it cannot referee anything, so the series' behaviour
/// there is unmeasurable this way — which is also why the threshold sits at `20` and not at the
/// series' true crossing point.
#[test]
fn the_modified_bessel_series_is_worst_just_below_its_crossover() {
    let (i0, i1) = (hankel(0.0, true), hankel(1.0, true));
    let tau = core::f64::consts::TAU;

    let worst_over = |low: f64, high: f64, kernel: fn(f64) -> f64, series: &[f64]| {
        let mut worst = 0.0_f64;
        for i in 0..=400 {
            let x = 10f64.powf(low + (high - low) * f64::from(i) / 400.0);
            let (want, _) = scaled(summed(series, x), 1.0 / (tau * x).sqrt());
            let got = kernel(x);
            if want == 0.0 || !want.is_finite() || !got.is_finite() {
                continue;
            }
            worst = worst.max(((got - want) / want).abs() / f64::EPSILON);
        }
        worst
    };

    for (name, kernel, series) in [
        ("bessel_i0e", Real::bessel_i0e as fn(f64) -> f64, &i0),
        ("bessel_i1e", Real::bessel_i1e as fn(f64) -> f64, &i1),
    ] {
        // `log10(18) = 1.25527`, `log10(20) = 1.301030`, `log10(100) = 2`. The two windows stop
        // a hair either side of the threshold rather than meeting on it: `1.3010` is `19.9986`,
        // which is still the *series*, and including it put a `13.92`-ulp series point inside the
        // "above" window and failed the assertion for the right reason at the wrong place.
        let below = worst_over(1.255_3, 1.300_9, kernel, series);
        let above = worst_over(1.301_1, 2.0, kernel, series);
        assert!(
            below > above,
            "{name} is {below:.2} ulp below the 1e4 crossover and {above:.2} ulp above it; the \
             series route is supposed to be the stretched one"
        );
        assert!(
            above <= 4.0,
            "{name} is {above:.2} ulp above its crossover, where the asymptotic route should be \
             within a couple of ulp"
        );
        assert!(
            below >= 10.0,
            "{name} is only {below:.2} ulp below its crossover. If the series has become that \
             accurate there the threshold could move down again; if the *expansion* has become \
             less accurate, this window is no longer a fair referee — check which before \
             relaxing anything."
        );
    }
}
