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
/// because its premise had gone. The windows became `[18, 20]` and `[20, 100]`, and when the
/// threshold moved again to `17` the same thing happened for the same reason: `[18, 20]` is now
/// the *expansion*, and it reported `2.06` ulp where this test requires at least ten. They are
/// `[16, 17]` and `[17, 100]` now.
///
/// # What let the threshold move, and why this test could not have found it
///
/// The previous note said `18` "rather than lower is forced by the reference, not by taste: the
/// expansion is itself the thing being compared against … below about `18` it cannot referee
/// anything, so the series' behaviour there is unmeasurable this way — which is also why the
/// threshold sits at `20` and not at the series' true crossing point."
///
/// That was an honest account of this instrument's limit and it was the binding constraint on the
/// constant. It is not a limit of the *mathematics*: an MPFR oracle at adaptive precision
/// referees both routes at any argument, and measuring that way put the crossing at `16` and the
/// expansion's floor at `17` — four units below where this test could see. The threshold follows
/// the measurement; this test still pins the shape.
///
/// The reference below is still the expansion, so the "below" window inherits its `8.7`-ulp
/// truncation at `x = 16`. That is tolerable for an inequality between `~50` and `~2` and would
/// not be tolerable for a bound, which is the distinction worth keeping in view.
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
        // `log10(16) = 1.204120`, `log10(17) = 1.230449`, `log10(100) = 2`. The two windows stop
        // a hair either side of the threshold rather than meeting on it, for the reason the
        // previous pair did: a point landing on the boundary takes the other route and puts a
        // series-sized error inside the "above" window, failing the assertion for the right
        // reason at the wrong place.
        let below = worst_over(1.204_2, 1.230_3, kernel, series);
        let above = worst_over(1.230_6, 2.0, kernel, series);
        assert!(
            below > above,
            "{name} is {below:.2} ulp below the crossover and {above:.2} ulp above it; the \
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
