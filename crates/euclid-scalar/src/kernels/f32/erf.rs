//! Native `f32` error-function family — the single-precision peer of
//! [`crate::kernels::erf`]. `erf`/`erfc` reduce to the regularized incomplete gamma `P(½, x²)`;
//! `erf_inv` is Giles's rational approximation alone, **without** the Newton step its `f64` peer
//! takes — see [`erf_inv`], where that difference is measured. Evaluated in `f32` throughout.
//!
//! Every threshold below is the *same derivation* as the `f64` one re-run at this format's
//! epsilon, never the `f64` value inherited; [`ERFCX_CF_MIN`] is the case where inheriting it was
//! the defect. What the two precisions genuinely share is precision-generic and lives in
//! [`crate::math`] — the closing association of the continued fraction is
//! [`crate::math::continued_fraction::reciprocal_of_product`].

/// Below this argument `erf` is its own leading term, `2x/√π`.
///
/// Not an optimisation but a correctness requirement, because `erf` is reached through
/// `gamma_lr(½, x²)` and `x²` leaves the format long before `x` does. At `x = 1e-154` the
/// square is already subnormal and carries only a few digits — `erf(1e-160)` was wrong by
/// `5.6e-6` — and by `x = 1.6e-162` the square is exactly zero, so `erf` returned **zero**
/// where the true value `1.13e-162` is a perfectly ordinary number.
///
/// The series is `erf(x) = (2/√π)(x − x³/3 + ⋯)`, so dropping everything past the linear term
/// costs `x²/3` relative. At the threshold that is ``3.3e-9``, below half an ulp, making the
/// linear form correctly rounded rather than merely adequate.
///
/// Taking `x = 0` through this path as well is what makes `erf(−0.0)` return `−0.0`: the old
/// `if x == 0.0 { 0.0 }` dropped the sign of the zero, which an odd function should not.
const ERF_LINEAR_MAX: f32 = 1e-4;

use super::gamma::gamma_lr;
use core::f32::consts::PI;

/// Which form evaluates `erf` at this magnitude.
///
/// Matched rather than tested as a chain of early returns: the four arms are exhaustive over the
/// line, so a new one cannot be added without the compiler asking what it returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ErfForm {
    /// Not a number.
    Undefined,
    /// `±∞`, where `erf` saturates at exactly `±1` rather than going undefined.
    Saturated,
    /// `|x| < `[`ERF_LINEAR_MAX`]: `erf` is its own leading term `2x/√π`. Taking `x = 0` through
    /// here is what makes `erf(−0.0)` return `−0.0`, which an odd function should.
    Linear,
    /// Everything else: `±P(½, x²)`, the regularized incomplete gamma.
    IncompleteGamma,
}

impl ErfForm {
    /// The non-finite cases first, then the magnitude.
    fn for_argument(x: f32) -> Self {
        if x.is_nan() {
            Self::Undefined
        } else if x.is_infinite() {
            Self::Saturated
        } else if x.abs() < ERF_LINEAR_MAX {
            Self::Linear
        } else {
            Self::IncompleteGamma
        }
    }
}

/// The error function `erf(x) = (2/√π) ∫₀ˣ e^{-t²} dt`.
pub(crate) fn erf(x: f32) -> f32 {
    match ErfForm::for_argument(x) {
        ErfForm::Undefined => f32::NAN,
        ErfForm::Saturated => 1.0_f32.copysign(x),
        ErfForm::Linear => 2.0 / PI.sqrt() * x,
        // `erf` is odd, so the magnitude comes from the incomplete gamma and the sign from the
        // argument — which is the same value the explicit negation produced, to the bit.
        ErfForm::IncompleteGamma => gamma_lr(0.5, x * x).copysign(x),
    }
}

/// Lower edge of the continued-fraction regime.
///
/// This threshold has to be **derived at `f32`'s epsilon rather than inherited**, and inheriting
/// it is exactly what went wrong: the value was `2.0`, carried over from the `f64` peer's
/// reasoning. Both routes behave differently here, in opposite directions.
///
/// The `1 − erf` form is a staircase of relative height `ulp(erf)/erfc ≈ 1.19e-7/erfc(x)` — see
/// the `f64` peer's [`ERFCX_CF_MIN`](crate::kernels::erf::ERFCX_CF_MIN) for why the envelope is the step
/// height — and `f32`'s steps are `2^29` times coarser. The fraction, meanwhile, *converges
/// sooner* in `f32`, because it only has to reach a 24-bit answer: its 60 terms are at the
/// format's limit from well below `1.0`, where `f64` needed `1.7`.
///
/// Measured in `f32` ulps against the `f64` kernel rounded to `f32`:
///
/// | `x` | 0.4 | 0.7 | **1.0** | 1.2 | 1.5 | 1.9 | 2.2 |
/// |---|---|---|---|---|---|---|---|
/// | `1 − erf` | `1` | `1` | **`0`** | `8` | `13` | `45` | `188` |
/// | fraction, 60 terms | `4694` | `12` | **`1`** | `1` | `0` | `3` | `0` |
///
/// So `2.0` stranded the band `[1, 2)`, where the complement reaches 45 ulps and the fraction
/// is at 3 or better — the same defect the `f64` peer had, one decade lower and an order worse.
/// Below `1.0` the fraction has not converged and degrades fast, so this is the join.
const ERFCX_CF_MIN: f32 = 1.0;

/// Which of the two algebraically equal forms of `erfc` is evaluated. The `f32` peer of
/// [`crate::kernels::erf::ErfcForm`]; see that type for what the two routes trade against each other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ErfcForm {
    /// `1 − erf(x)`, quantised at `ulp(erf)/erfc` relative.
    Complement,
    /// `e^{−x²}·erfcx(x)`, no cancellation but needs convergence.
    Fraction,
}

impl ErfcForm {
    fn for_argument(x: f32) -> Self {
        if x >= ERFCX_CF_MIN {
            Self::Fraction
        } else {
            Self::Complement
        }
    }
}

/// The complementary error function `erfc(x) = 1 - erf(x)`.
///
/// Routed by [`ErfcForm`]: the `1 - erf` form cancels in the right tail (in `f32` it underflows
/// to `0` by `x ≈ 4`), so past the crossover the equal, stable `erfc(x) = e^{-x²}·erfcx(x)` runs
/// instead.
pub(crate) fn erfc(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x.is_infinite() {
        return if x.is_sign_positive() { 0.0 } else { 2.0 };
    }
    match ErfcForm::for_argument(x) {
        ErfcForm::Complement => 1.0 - erf(x),
        ErfcForm::Fraction => (-x * x).exp() * erfcx_cf(x),
    }
}

/// Continued-fraction core of `erfcx` for `x ≥ ERFCX_CF_MIN`, depending only on `x`
/// (not on `erfc`), so `erfc` and `erfcx` can both call it without mutual recursion.
/// The depth is a flat `60` here rather than `f64`'s measured ladder, which is a separate
/// measurement and not this one scaled; see the `f64` peer's own ladder. The closing
/// association is shared, and in this format its fallback begins at `f32::MAX/√π = 1.92e38`.
fn erfcx_cf(x: f32) -> f32 {
    let mut t = x;
    for k in (1..=60).rev() {
        t = x + (k as f32) * 0.5 / t;
    }
    crate::math::continued_fraction::reciprocal_of_product(PI.sqrt(), t)
}

/// The scaled complementary error function `erfcx(x) = e^{x²}·erfc(x)`, finite in the right
/// tail where `erfc` underflows. A continued fraction carries the large-`x` regime.
pub(crate) fn erfcx(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x.is_infinite() {
        return if x.is_sign_positive() {
            0.0
        } else {
            f32::INFINITY
        };
    }
    if x < 0.0 {
        return 2.0 * (x * x).exp() - erfcx(-x);
    }
    match ErfcForm::for_argument(x) {
        // Below the crossover `erfc` takes its `Complement` branch, the stable one there.
        ErfcForm::Complement => (x * x).exp() * erfc(x),
        ErfcForm::Fraction => erfcx_cf(x),
    }
}

/// The inverse error function: `erf_inv(erf(x)) = x` for `x ∈ (-1, 1)`.
///
/// Giles' single-precision rational approximation, and **nothing after it**.
///
/// # Why there is no Newton step, where the f64 peer has one
///
/// Because at `f32` it made the result eight times worse. Measured against the true inverse over
/// 4,000 arguments in `(-1, 1)`: the bare seed is accurate to `2.4e-7` — about two `f32` ulps,
/// which is as good as the format allows — and adding one Newton step took that to `1.9e-6`.
///
/// The reason is that the correction is `err / pdf` with `err = erf(result) - x`, and `erf` at
/// `f32` carries its own error of several ulps. Dividing that by `2/√π · e^{-r²}`, which shrinks
/// rapidly as `|r|` grows, amplifies it — so the step injects more noise than the seed had error.
/// It is a refinement computed with an instrument coarser than the thing being refined.
///
/// Giles fitted these coefficients *for* `f32`, so there was nothing left to refine. The f64 peer
/// is the opposite case: its coefficient table is coarser than `f64`, so its Newton step earns
/// its place and stays.
///
/// In the far tail the step was not harmful, merely useless: beyond about `|p| = 0.997`,
/// `e^{-r²}` underflows and the correction term is identically zero, so the two agreed to the
/// last bit.
///
/// Found by mutation testing. Thirty-four mutants survived in this function — every seed
/// coefficient and every operator in the Newton step — because the step scrambled the seed's
/// precision and then contributed nothing reliable of its own, leaving no accuracy test able to
/// tell any version from another. With the step gone the seed is directly observable and
/// `the_f32_inverse_is_as_accurate_as_the_format_allows` pins it at `3e-7`.
pub(crate) fn erf_inv(x: f32) -> f32 {
    match ErfInvRoute::for_argument(x) {
        ErfInvRoute::AtNegativeOne => f32::NEG_INFINITY,
        ErfInvRoute::AtOne => f32::INFINITY,
        // `copysign` rather than a bare `0.0`: `erf_inv` is odd, so `erf_inv(-0.0)` must be
        // `-0.0`. Returning the literal discarded that, which `stability::symmetry_exact`'s
        // signed-zero sweep caught. `ERF_LINEAR_MAX`'s note above records the same mistake being
        // fixed for `erf` itself; `erf_inv` was missed at the time.
        ErfInvRoute::AtOrigin => 0.0_f32.copysign(x),
        ErfInvRoute::Interior => {
            let w = -((1.0 - x) * (1.0 + x)).ln();
            ErfInvSeed::for_log_weight(w).evaluate(w) * x
        }
    }
}

/// Which route the inverse error function takes at this argument.
///
/// The same four arms the `f64` peer in [`crate::kernels::erf`] carries. What differs is what
/// `Interior` does: there is no Newton step here, for the reason [`erf_inv`]'s own documentation
/// gives — it destroyed the seed's precision and hid every one of its coefficients.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ErfInvRoute {
    /// `x ≤ −1`, the left end of `erf`'s range, where the inverse diverges.
    AtNegativeOne,
    /// `x ≥ 1`, the right end.
    AtOne,
    /// `x = ±0`, where the inverse is zero carrying the argument's own sign.
    AtOrigin,
    /// The interior, by a rational seed in `w = −ln((1−x)(1+x))`. A `NaN` reaches this arm —
    /// none of the comparisons above admit it — and propagates through the polynomial unchanged.
    Interior,
}

impl ErfInvRoute {
    fn for_argument(x: f32) -> Self {
        if x <= -1.0 {
            Self::AtNegativeOne
        } else if x >= 1.0 {
            Self::AtOne
        } else if x == 0.0 {
            Self::AtOrigin
        } else {
            Self::Interior
        }
    }
}

/// Which of the seed's two coefficient tables covers this `w`.
///
/// Two fits rather than one, in different variables: the central table is a polynomial in
/// `w − 5/2` and the tail's is in `√w − 3`, so they are not continuations of each other and the
/// choice cannot be folded into the coefficients.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ErfInvSeed {
    /// `w < 5`, the central band.
    Central,
    /// `w ≥ 5`, the tail. A `NaN` `w` lands here and propagates.
    Tail,
}

impl ErfInvSeed {
    fn for_log_weight(w: f32) -> Self {
        if w < 5.0 { Self::Central } else { Self::Tail }
    }

    /// The seed polynomial `p`, which multiplies `x` to give the estimate.
    fn evaluate(self, w: f32) -> f32 {
        let mut p;
        match self {
            Self::Central => {
                let w = w - 2.5;
                p = 2.810_226_4e-8;
                p = 3.432_739_4e-7 + p * w;
                p = -3.523_387_7e-06 + p * w;
                p = -4.391_506_5e-6 + p * w;
                p = 0.000_218_580_87 + p * w;
                p = -0.001_253_725 + p * w;
                p = -0.004_177_681_6 + p * w;
                p = 0.246_640_73 + p * w;
                p = 1.501_409_4 + p * w;
            }
            Self::Tail => {
                let w = w.sqrt() - 3.0;
                p = -0.000_200_214_26;
                p = 0.000_100_950_24 + p * w;
                p = 0.001_349_343_2 + p * w;
                p = -0.003_673_428_4 + p * w;
                p = 0.005_739_507_7 + p * w;
                p = -0.007_622_461_3 + p * w;
                p = 0.009_438_870_5 + p * w;
                p = 1.001_674 + p * w;
                p = 2.832_976_8 + p * w;
            }
        }
        p
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn agrees(a: f32, b: f64, tol: f32) {
        let b = b as f32;
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn erf_matches_f64_reference() {
        for &x in &[-2.0, -1.0, -0.3, 0.0, 0.3, 1.0, 2.0, 3.0] {
            agrees(erf(x), crate::kernels::erf::erf(x as f64), 1e-5);
            agrees(erfc(x), crate::kernels::erf::erfc(x as f64), 1e-5);
        }
        for &x in &[-3.0, -0.5, 0.5, 2.0, 6.0] {
            agrees(erfcx(x), crate::kernels::erf::erfcx(x as f64), 1e-4);
        }
    }

    #[test]
    fn erf_inv_round_trips() {
        for &x in &[-0.9, -0.5, -0.1, 0.1, 0.3, 0.5, 0.8, 0.95] {
            agrees(erf(erf_inv(x)), x as f64, 1e-5);
        }
    }

    /// **The inverse is accurate to about two `f32` ulps, and that is now checkable.**
    ///
    /// Compared against the f64 kernel, which is four orders of magnitude more accurate than
    /// `f32` can represent and so serves as the truth here. Measured maximum over this sweep is
    /// `2.4e-7`; the bound is `3e-7`.
    ///
    /// This is the test the Newton step made impossible. While the step was there the function
    /// was accurate only to `1.9e-6`, and every one of its thirty-four internal mutants sat
    /// inside that — the step destroyed the seed's precision and then added nothing of its own,
    /// so no bound could separate a correct coefficient table from a damaged one. Removing it
    /// improved the function eightfold and made its coefficients observable in the same move.
    ///
    /// A relative bound throughout, which is the right measure for an inverse whose range is
    /// unbounded: near `|p| = 1` the answer is large and a relative claim is the only meaningful
    /// one, and near zero `erf_inv(p) ≈ p·√π/2` stays well away from zero relative to `p`.
    #[test]
    fn the_f32_inverse_is_as_accurate_as_the_format_allows() {
        let mut worst = 0.0_f64;
        let mut at = 0.0_f64;
        for i in 1..4000 {
            #[allow(clippy::cast_precision_loss)]
            let p = -0.999_f32 + 2.0 * 0.999_f32 * (i as f32) / 4000.0;
            if p == 0.0 {
                continue;
            }
            let truth = crate::kernels::erf::erf_inv(f64::from(p));
            let gap = ((f64::from(erf_inv(p)) - truth) / truth).abs();
            if gap > worst {
                worst = gap;
                at = f64::from(p);
            }
        }
        assert!(
            worst <= 3e-7,
            "f32 erf_inv is accurate to {worst:.4e} (worst at p = {at}), budget 3e-7",
        );

        // The far tail, past the `w >= 5` crossover into the second coefficient table. The
        // seed is the whole computation out here, since a Newton correction's `e^{-r²}` factor
        // underflows to zero — which is why removing it changed nothing in this range.
        for &(lo, hi, bound) in &[
            (0.9966_f32, 0.99999_f32, 3e-7),
            (0.99999, 0.999_999_94, 2e-7),
        ] {
            for i in 0..=1000 {
                #[allow(clippy::cast_precision_loss)]
                let p = lo + (hi - lo) * (i as f32) / 1000.0;
                if p >= 1.0 {
                    continue;
                }
                let truth = crate::kernels::erf::erf_inv(f64::from(p));
                let gap = ((f64::from(erf_inv(p)) - truth) / truth).abs();
                assert!(gap <= bound, "f32 erf_inv at p = {p} is off by {gap:.3e}");
            }
        }

        // Both signs, since the seed is applied as `p(w)·x` and the sign rides on `x`.
        for &p in &[0.5_f32, -0.5, 0.9, -0.9, 0.01, -0.01] {
            let truth = crate::kernels::erf::erf_inv(f64::from(p));
            let gap = ((f64::from(erf_inv(p)) - truth) / truth).abs();
            assert!(gap <= 3e-7, "f32 erf_inv at p = {p} is off by {gap:.3e}");
        }
    }

    #[test]
    fn erfcx_tail_is_accurate() {
        // Regression: the old `x < 4.0` threshold (copied from f64) routed [≈3.5, 4)
        // through the catastrophic `1 - erf` cancellation, where the relative error
        // reached 71% at 3.9 and 100% (returned 0.0) at 3.99. The lowered threshold
        // sends that zone through the continued fraction instead.
        for &x in &[1.5f32, 2.0, 2.5, 3.0, 3.5, 3.9, 3.99, 4.0, 5.0, 8.0] {
            let got = erfcx(x);
            let reference = crate::kernels::erf::erfcx(x as f64);
            let rel = ((got as f64) - reference).abs() / reference.abs();
            assert!(
                rel < 1e-5,
                "erfcx({x}) = {got}, ref {reference}, rel {rel:.2e}"
            );
        }
    }

    #[test]
    fn erfc_tail_does_not_underflow() {
        // Regression: `1 - erf` underflowed to exactly 0 by x ≈ 4 in f32; the tail now
        // routes through `e^{-x²}·erfcx`, staying positive and accurate to the
        // underflow of `e^{-x²}` itself.
        for &x in &[3.0f32, 3.5, 4.0, 5.0, 6.0, 8.0] {
            let got = erfc(x);
            let reference = crate::kernels::erf::erfc(x as f64);
            assert!(got > 0.0, "erfc({x}) underflowed to {got}");
            let rel = ((got as f64) - reference).abs() / reference.abs();
            assert!(
                rel < 1e-4,
                "erfc({x}) = {got}, ref {reference}, rel {rel:.2e}"
            );
        }
    }

    /// The four forms, at and either side of the one threshold that is not a non-finite case.
    ///
    /// The linear cutoff is *sited* where the dropped `x²/3` term is below half an ulp, so the
    /// two forms agree across it to well under a rounding — which is what left `<` against `<=`
    /// alive against every value assertion. The decision is what gets pinned.
    #[test]
    fn the_erf_forms_partition_the_line() {
        use ErfForm::{IncompleteGamma, Linear, Saturated, Undefined};
        assert_eq!(ErfForm::for_argument(f32::NAN), Undefined);
        assert_eq!(ErfForm::for_argument(f32::INFINITY), Saturated);
        assert_eq!(ErfForm::for_argument(f32::NEG_INFINITY), Saturated);
        // The boundary belongs to the incomplete gamma, and the magnitude decides, so both
        // signs route alike.
        assert_eq!(ErfForm::for_argument(ERF_LINEAR_MAX), IncompleteGamma);
        assert_eq!(ErfForm::for_argument(-ERF_LINEAR_MAX), IncompleteGamma);
        assert_eq!(ErfForm::for_argument(ERF_LINEAR_MAX.next_down()), Linear);
        assert_eq!(ErfForm::for_argument(-ERF_LINEAR_MAX.next_down()), Linear);
        assert_eq!(
            ErfForm::for_argument(ERF_LINEAR_MAX.next_up()),
            IncompleteGamma
        );
        // Both zeros take the linear arm, which is what preserves the sign of a zero.
        assert_eq!(ErfForm::for_argument(0.0), Linear);
        assert_eq!(ErfForm::for_argument(-0.0), Linear);
        assert_eq!(erf(-0.0).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(erf(0.0).to_bits(), 0.0_f32.to_bits());
        for x in [1.0_f32, -1.0, 10.0] {
            assert_eq!(ErfForm::for_argument(x), IncompleteGamma);
        }
        // And the saturated arm carries the sign rather than a comparison.
        assert_eq!(erf(f32::INFINITY), 1.0);
        assert_eq!(erf(f32::NEG_INFINITY), -1.0);
    }

    /// The inverse's domain routes and its two seed tables.
    ///
    /// `AtNegativeOne` and `AtOne` are the ends of `erf`'s range, where the inverse diverges;
    /// `AtOrigin` is the signed zero the polynomial cannot produce. The seed split is a separate
    /// decision in a separate variable — the central table is a polynomial in `w − 5/2` and the
    /// tail's in `√w − 3` — so neither is a continuation of the other.
    #[test]
    fn the_inverse_routes_split_the_range_and_the_seed_tables() {
        use ErfInvRoute::{AtNegativeOne, AtOne, AtOrigin, Interior};
        use ErfInvSeed::{Central, Tail};

        for x in [-1.0, (-1.0_f32).next_down(), -2.0, f32::NEG_INFINITY] {
            assert_eq!(ErfInvRoute::for_argument(x), AtNegativeOne, "at {x:e}");
            assert_eq!(erf_inv(x), f32::NEG_INFINITY, "at {x:e}");
        }
        for x in [1.0, 1.0_f32.next_up(), 2.0, f32::INFINITY] {
            assert_eq!(ErfInvRoute::for_argument(x), AtOne, "at {x:e}");
            assert_eq!(erf_inv(x), f32::INFINITY, "at {x:e}");
        }
        // Odd at the origin, which is what `copysign` is for.
        for x in [0.0_f32, -0.0] {
            assert_eq!(ErfInvRoute::for_argument(x), AtOrigin);
            assert_eq!(erf_inv(x).to_bits(), x.to_bits(), "signed zero at {x:e}");
        }
        for x in [
            (-1.0_f32).next_up(),
            -0.5,
            -f32::MIN_POSITIVE,
            f32::MIN_POSITIVE,
            0.5,
            1.0_f32.next_down(),
            f32::NAN,
        ] {
            assert_eq!(ErfInvRoute::for_argument(x), Interior, "at {x:e}");
        }
        // `NaN` reaches the interior and propagates rather than being caught by a guard.
        assert!(erf_inv(f32::NAN).is_nan());

        // The seed tables, at and either side of the boundary.
        assert_eq!(ErfInvSeed::for_log_weight(5.0), Tail);
        assert_eq!(ErfInvSeed::for_log_weight(5.0_f32.next_down()), Central);
        assert_eq!(ErfInvSeed::for_log_weight(5.0_f32.next_up()), Tail);
        assert_eq!(ErfInvSeed::for_log_weight(0.0), Central);
        assert_eq!(ErfInvSeed::for_log_weight(f32::NAN), Tail);
        // Both tables are evaluated in different variables, so they are genuinely two fits —
        // yet they agree at the join, which is what makes the split free.
        let join = 5.0_f32;
        let (central, tail) = (Central.evaluate(join), Tail.evaluate(join));
        assert!(
            (central - tail).abs() <= 2e-2 * tail.abs(),
            "the seed tables disagree at the join: {central} vs {tail}"
        );
    }
}
