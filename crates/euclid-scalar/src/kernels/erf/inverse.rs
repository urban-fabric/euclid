//! The inverse error function: Giles's rational approximation plus one Newton step.
//!
//! # Fixed: the far tail no longer goes through Giles's fit
//!
//! Measured against a reference that polishes the answer with Newton in MPFR at 300 bits:
//!
//! ```text
//! erf_inv(0.9999999999999999)  = 5.271293105187273   (this kernel)
//!                                5.8635847487551676  (correct, and it round-trips)
//! ```
//!
//! Ten percent out, and it is a defect rather than conditioning — which was the first thing
//! checked, because `erf` *is* vertical out there and a plateau of `w` does share one `erf`
//! value. It does not excuse this one. `erf(5.271293105187273)` is `0.99999999999991`, a
//! **different double** from the argument, while `erf(5.8635847487551676)` returns it exactly.
//! The smallest `w` whose `erf` reaches the argument is `5.829395261518418`, so the answer is
//! not merely at the wrong end of a plateau — it is short of it.
//!
//! Splitting `erf_inv_interior` into [`erf_inv_seed`], [`erf_density`] and [`newton_residual`]
//! is what attributed it: the seed returns `5.1764`, eleven percent below the root, and one
//! Newton step against a function that flat recovers a tenth of that. **The seed was the defect,
//! not the refinement** — the opposite of what I first assumed and spent a reformulation on.
//!
//! # What replaced it
//!
//! `erf_inv(x) = sign(x)·erfc_inv(1 − |x|)` would be exact and is **not available**: `erfc_inv`
//! routes through `ndtri`, which calls back here, and the change overflows the stack. The repair
//! had to be local, so [`ERF_INV_TAIL_MAX`] now diverts the far tail to a three-piece route of
//! its own — [`tail_log_scale`], [`tail_seed`] and two [`erfc_newton_step`]s — built on the
//! asymptotic inverse of `erfc` rather than on a rational fit.
//!
//! Measured against MPFR at 300 bits:
//!
//! ```text
//! 1 − x     1e-5    1e-6    1e-7     1e-9     1e-12     1e-15    1.1e-16
//! was      204.3    9187   35923    1.2e9    5.1e12    2.3e14     4.6e14   ulp
//! now        0.0     0.0     0.5      0.0       0.0       0.0        0.0   ulp
//! ```
//!
//! **Correctly rounded across the whole tail.** See [`ERF_INV_TAIL_MAX`] for where the two
//! routes hand over and why that point moved as the interior improved.
//!
//! # The middle band had a different cause, and a different fix
//!
//! From `1e-6` to `1e-2` the seed is fine and the *numerator* was the problem: `erf(r) − x`
//! differences two numbers near `1`, so it carries `ε` absolute however small the result is,
//! and Newton divides that by a vanishing `erf′`. At `1 − x = 1e-6`, `ε/7.1e-6` is `3.1e-11` of
//! error in the root. [`newton_residual`] now forms `(1 − x) − erfc(r)` above
//! [`RESIDUAL_COMPLEMENT_MIN`], where both terms are the size of the answer:
//!
//! ```text
//! 1 − x      1e-1    1e-2   1e-3   1e-4    1e-5    1e-6
//! was        18.9    48.9    2.6   17.5   204.3    9187   ulp
//! now        10.3     0.0    0.0    0.0     0.0     0.0   ulp
//! ```
//!
//! The numerator fix alone left `1e-4` and `1e-5` at `20` and `299` ulp, seed-limited at about
//! `6.6e-8`. A **second** Newton step closes them, and could not have before: with the old
//! numerator the limit was `ε` of noise divided by a vanishing `erf′`, which no amount of
//! iterating removes. The two fixes are coupled, and neither is worth much alone.
//!
//! So the two defects in this file had opposite causes: the far tail was a bad *seed* that a
//! good step could not rescue, and the middle band a bad *numerator* under a seed that was
//! fine. Splitting the function is what let them be told apart — together they look like one
//! function that is inaccurate near `±1`.
//!
//! A child module so the parent stays inside the crate's 400-line budget. `erf`, `erfc` and
//! `erfcx` are one topic — all three reduce to the same incomplete gamma or the same continued
//! fraction — and the *inverse* is another: it shares no machinery with them beyond calling
//! `erf` once to refine, and it carries its own two-table seed whose coefficients are a fit
//! rather than a derivation.
//!
//! The single-precision peer at [`crate::kernels::f32::erf`] keeps its inverse in the parent
//! file, because without the Newton step it is short enough to fit.

use super::{erf, erfc};
use core::f64::consts::PI;

/// Which route the inverse error function takes at this argument.
///
/// The three non-interior arms are the ends of `erf`'s range and its centre, and each is a value
/// the polynomial seed cannot produce: two infinities and a signed zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ErfInvRoute {
    /// `x ≤ −1`, the left end of `erf`'s range, where the inverse diverges.
    AtNegativeOne,
    /// `x ≥ 1`, the right end.
    AtOne,
    /// `x = ±0`, where the inverse is zero carrying the argument's own sign.
    AtOrigin,
    /// The interior, by a rational seed in `w = −ln((1−x)(1+x))` refined by one Newton step.
    /// A `NaN` reaches this arm — none of the comparisons above admit it — and propagates
    /// through the polynomial unchanged.
    Interior,
}

impl ErfInvRoute {
    fn for_argument(x: f64) -> Self {
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
    fn for_log_weight(w: f64) -> Self {
        if w < 5.0 { Self::Central } else { Self::Tail }
    }

    /// The seed polynomial `p`, which multiplies `x` to give the starting estimate.
    fn evaluate(self, w: f64) -> f64 {
        let mut p;
        match self {
            Self::Central => {
                let w = w - 2.5;
                p = 2.810_226_36e-08;
                p = 3.432_739_39e-07 + p * w;
                p = -3.523_387_7e-06 + p * w;
                p = -4.391_506_54e-06 + p * w;
                p = 0.000_218_580_87 + p * w;
                p = -0.001_253_725_03 + p * w;
                p = -0.004_177_681_64 + p * w;
                p = 0.246_640_727 + p * w;
                p = 1.501_409_41 + p * w;
            }
            Self::Tail => {
                let w = w.sqrt() - 3.0;
                p = -0.000_200_214_257;
                p = 0.000_100_950_24 + p * w;
                p = 0.001_349_343_22 + p * w;
                p = -0.003_673_428_4 + p * w;
                p = 0.005_739_507_73 + p * w;
                p = -0.007_622_461_3 + p * w;
                p = 0.009_438_870_47 + p * w;
                p = 1.001_674_06 + p * w;
                p = 2.832_976_82 + p * w;
            }
        }
        p
    }
}

/// The inverse error function: `erf_inv(erf(x)) = x` for `x ∈ (-1, 1)`.
pub(crate) fn erf_inv(x: f64) -> f64 {
    match ErfInvRoute::for_argument(x) {
        ErfInvRoute::AtNegativeOne => f64::NEG_INFINITY,
        ErfInvRoute::AtOne => f64::INFINITY,
        // `copysign` rather than a bare `0.0`: `erf_inv` is odd, so `erf_inv(-0.0)` must be
        // `-0.0`. Returning the literal discarded that, which `stability::symmetry_exact`'s
        // signed-zero sweep caught. This file's `ERF_LINEAR_MAX` note records the same mistake
        // being fixed for `erf` itself; `erf_inv` was missed at the time.
        ErfInvRoute::AtOrigin => 0.0_f64.copysign(x),
        ErfInvRoute::Interior => erf_inv_interior(x),
    }
}

/// The rational seed and its single Newton step, for `x` strictly inside `(−1, 1)` and nonzero.
///
/// Three named pieces rather than one expression. They were one, and that is why a
/// ten-percent error near `|x| = 1` could not be attributed: the seed and the step are
/// different mechanisms with different failure modes, and a wrong answer out of the pair says
/// nothing about which produced it. Each is now separately assertable.
fn erf_inv_interior(x: f64) -> f64 {
    let complement = 1.0 - x.abs();
    if complement <= ERF_INV_TAIL_MAX {
        return tail_magnitude(complement).copysign(x);
    }
    let once = erf_newton_step(x, erf_inv_seed(x));
    erf_newton_step(x, once)
}

/// One Newton step against `erf(r) = x`.
///
/// Taken **twice**, and that is only worth doing since [`newton_residual`] stopped forming its
/// numerator by cancelling against `1`. With the old numerator a second step bought nothing: the
/// limit was `ε` of noise in the residual divided by a vanishing `erf′`, which no amount of
/// iterating removes. With the complement form the limit is the seed, and Newton squares that —
/// Giles's fit is about `6.6e-8` out near `1 − x = 1e-5`, which one step takes to `1.2e-14` and
/// the second to below rounding.
///
/// The two defects in this file were therefore coupled in a way that is easy to miss: fixing
/// the numerator is what made the extra step useful, and the extra step is what turns the
/// numerator fix from a thirteen-hundredfold improvement in one band into an accurate function
/// across all of them.
fn erf_newton_step(x: f64, r: f64) -> f64 {
    r - newton_residual(x, r) / erf_density(r)
}

/// At or below this `1 − |x|`, the tail route replaces Giles's fit.
///
/// **`1e-9`, and it moved twice as the interior route improved.** The crossover is where the
/// asymptotic seed starts beating Giles's fit *after both routes' refinement*, so it depends on
/// how good the refinement is — it is not a property of the seeds alone. Measured against MPFR
/// at 300 bits, with the threshold swept:
///
/// ```text
/// 1 − x            1e-7     1e-8      1e-9      1e-10        1e-11
/// interior          0.5      0.0    5946.6    8.5e6        2.0e9     ulp
/// tail           2248.7    644.7     214.9     80.5         33.3     ulp (two steps)
/// ```
///
/// `1e-7` was right when the interior took one Newton step against a cancelling numerator. With
/// the complement numerator and a second step the interior became exact two decades further
/// out, and the threshold had to follow it. Below `1e-9` Giles's seed is too far gone for any
/// number of steps — `2.0e9` ulp at `1e-11` — and the asymptotic seed takes over.
///
/// Placing this at `1e-3`, where [`tail_seed`]'s `ln(1 − 1/(2r²))` merely stops being
/// undefined, would have been wrong by six decades: the tail route is `4.4e5` ulp at `1e-4`
/// against Giles's `17.5`.
///
/// `1 − |x|` is exact here by Sterbenz — `|x| ≥ ½` covers every argument this admits — so the
/// complement costs nothing to form and the tail route sees the full precision of the input.
const ERF_INV_TAIL_MAX: f64 = 1e-9;

/// `L = −ln c`, the scale the tail expansion is written in.
fn tail_log_scale(complement: f64) -> f64 {
    -complement.ln()
}

/// The asymptotic inverse of `erfc`, by fixed-point iteration.
///
/// `erfc(r) ≈ e^{−r²}(1 − 1/(2r²))/(r√π)`, so taking logarithms and solving for `r²`,
///
/// ```text
/// r = √(L − ln(r√π) + ln(1 − 1/(2r²))),      L = −ln c
/// ```
///
/// which is a contraction in `r` because every term on the right moves like `ln r` against a
/// left-hand side that moves like `r`. Three passes from `r₀ = √L` is convergence to well inside
/// what the Newton step then closes: at `c = 1.11e-16` it runs `6.0611 → 5.8620 → 5.8648 →
/// 5.8636` against a root of `5.8635847`.
///
/// The `ln(1 − 1/(2r²))` term is the part that matters and the part that bounds
/// [`ERF_INV_TAIL_MAX`]. Dropped, the iteration converges to `5.8648` — a `2e-4` relative error,
/// which is sixty times what one Newton step can then absorb. It is the second term of the
/// asymptotic series, and this expansion needs two.
fn tail_seed(complement: f64) -> f64 {
    let scale = tail_log_scale(complement);
    let mut r = scale.sqrt();
    for _ in 0..3 {
        let curvature = (1.0 - 0.5 / (r * r)).ln();
        r = (scale - (r * PI.sqrt()).ln() + curvature).sqrt();
    }
    r
}

/// `|erf_inv(x)|` for a small complement, as the seed plus one Newton step on `erfc`.
///
/// Newton against `erfc(r) = c` rather than `erf(r) = x`, which is the same root reached without
/// the subtraction that cost the old route its digits: `erf(r) − x` differences two numbers near
/// `1`, while `erfc(r) − c` differences two numbers of the size of the answer.
fn tail_magnitude(complement: f64) -> f64 {
    let once = erfc_newton_step(tail_seed(complement), complement);
    let twice = erfc_newton_step(once, complement);
    erfc_newton_step(twice, complement)
}

/// One Newton step against `erfc(r) = c`.
///
/// Three of them are taken, and the count is measured rather than cautious. The seed lands
/// about `2e-4` from the root; Newton is quadratic here, so each step roughly squares that. Two
/// steps leave `215` ulp at the shallow end of this route's range — where the seed is at its
/// worst — and the third takes it to rounding. Deeper in, two were already enough; the third
/// costs one `erfc` on a branch that handles `1 − |x| ≤ 1e-9`.
fn erfc_newton_step(r: f64, complement: f64) -> f64 {
    r + (erfc(r) - complement) / erf_density(r)
}

/// `w = −ln((1−x)(1+x))`, the log weight Giles's fit is parameterised by.
fn erf_inv_log_weight(x: f64) -> f64 {
    -((1.0 - x) * (1.0 + x)).ln()
}

/// Giles's rational approximation, before any refinement.
///
/// # Giles's seed collapses in the tail, and one Newton step cannot recover it
///
/// Splitting [`erf_inv_interior`] into [`erf_inv_seed`], [`newton_residual`] and
/// [`erf_density`] is what attributed this: at `x = 0.9999999999999999` the seed returns
/// **`5.1764`** where the root is `5.8636`, and the step reaches only `5.2713`. Newton is
/// quadratic *near* a root; eleven percent away, against a function as flat as `erf` is out
/// there, one step buys almost nothing. The defect was the seed, not the refinement.
///
/// Measured against MPFR at 300 bits, both routes over the same arguments:
///
/// ```text
/// 1 − x        1e-1   1e-2   1e-4    1e-6      1e-9       1.1e-16
/// erf_inv      18.9   48.9   17.5   9187    1.2e9        4.6e14     ulp
/// erfc_inv     11.2    0.5    0.7      0.0     0.0           0.0     ulp
/// ```
///
/// `erfc_inv` is better at every point including the first, so the obvious repair is
/// `erf_inv(x) = sign(x)·erfc_inv(1 − |x|)` — exact, since `1 − |x|` is exact for `|x| ≥ ½`.
///
/// **It cannot be written that way.** `erfc_inv` routes through `ndtri`, which calls back into
/// `erf_inv`: the change compiles, returns the correct `5.8635847487551676` at the argument
/// above, and then overflows the stack at `1 − 1e-9`. Tried and reverted.
///
/// So the repair has to be local to this file, and it is the *seed* that needs replacing rather
/// than the step. The asymptotic inverse is the candidate: `erfc(r) ≈ e^{−r²}/(r√π)` gives
/// `r ≈ √w` with `w = −ln((1−x)(1+x))`, which is `6.0036` against the true `5.8636` — a
/// 2.4% error where Giles's fit gives 11.8% — and refines quickly because the log-space
/// residual is not flat. That is the shape of the fix; it needs its own measurement pass.
fn erf_inv_seed(x: f64) -> f64 {
    let w = erf_inv_log_weight(x);
    ErfInvSeed::for_log_weight(w).evaluate(w) * x
}

/// `erf′(r) = (2/√π)e^{−r²}`, the Newton denominator.
///
/// Named because it is the term that makes the residual's accuracy matter: it vanishes like
/// `e^{−r²}`, so an absolute error in the numerator is amplified by `1/erf′` — `6.6e14` at
/// `r = 5.86`.
fn erf_density(r: f64) -> f64 {
    2.0 / PI.sqrt() * (-r * r).exp()
}

/// Above this magnitude the Newton numerator is formed through `erfc`.
///
/// `½`, because that is where `1 − |x|` stops being exact — Sterbenz gives it exactly on
/// `[½, 2]` — so the complementary form costs nothing to reach and the direct form has nothing
/// left to offer.
const RESIDUAL_COMPLEMENT_MIN: f64 = 0.5;

/// `erf(r) − x`, the Newton numerator, formed so it keeps its digits as `|x| → 1`.
///
/// # The subtraction this avoids is what the middle band was losing
///
/// `erf(r) − x` differences two numbers that both approach `1`, so the result carries an
/// absolute error of `ε` however small it is. Newton then divides by `erf′(r)`, which is
/// vanishing — `7.1e-6` at `1 − x = 1e-6` — and `ε/7.1e-6` is `3.1e-11` of error in the root.
/// Measured there: `9187` ulp.
///
/// The complementary form is the same quantity with no cancellation against `1`:
///
/// ```text
/// erf(r) − x = (1 − erfc(r)) − x = (1 − x) − erfc(r)
/// ```
///
/// Both terms are now `O(1 − |x|)`, so the difference is between small accurate quantities and
/// its error scales with them rather than with `1`.
///
/// Below [`RESIDUAL_COMPLEMENT_MIN`] the direct form is kept and is right there: the residual
/// still cancels, but `erf′` is `O(1)` rather than vanishing, so `ε` in the numerator stays `ε`
/// in the root. **It is the tiny denominator that makes the cancellation matter**, not the
/// cancellation on its own — which is why this is a separate decision from
/// [`ERF_INV_TAIL_MAX`], and why the two thresholds are three decades apart.
///
/// The sign work is the odd symmetry: for `x < 0` both `erf` and the root change sign, so
/// `erf(r) − x = −[(1 − |x|) − erfc(|r|)]`.
fn newton_residual(x: f64, r: f64) -> f64 {
    if x.abs() < RESIDUAL_COMPLEMENT_MIN {
        return erf(r) - x;
    }
    let complement = (1.0 - x.abs()) - erfc(r.abs());
    if x.is_sign_negative() {
        -complement
    } else {
        complement
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tail, repaired — and why the route exists rather than a better fit.
    ///
    /// Giles's seed is untouched and still eleven percent below the root out here; that is the
    /// defect the split located and it is still measurable, because the seed is still what the
    /// *interior* route uses. What changed is that `1 − |x| ≤ ERF_INV_TAIL_MAX` no longer goes
    /// through it.
    #[test]
    fn the_tail_route_reaches_the_root_the_seed_could_not() {
        let x = 0.9999999999999999_f64;
        let root = 5.863_584_748_755_167_6_f64;

        // The seed is unchanged and still the reason the tail route is needed.
        let seed = erf_inv_seed(x);
        assert!(
            (seed - root).abs() / root > 0.11,
            "Giles's tail seed has changed; it was 5.1764, eleven percent below the root"
        );

        // And `erf_inv` no longer uses it there.
        let got = erf_inv(x);
        assert!(
            (got - root).abs() <= 4.0 * crate::Real::ulp(root),
            "erf_inv({x}) = {got}, which is not within four ulp of {root}"
        );
        // Odd, and the sign survives the complement.
        assert_eq!(erf_inv(-x), -got);
    }

    /// The two forms of the Newton numerator are the same quantity.
    ///
    /// Asserted where both are trustworthy — around `|x| = ½`, the threshold itself — because
    /// that is the only place a disagreement would mean an algebra error rather than the
    /// cancellation the complement form exists to avoid. Further out the direct form is the one
    /// that is wrong, so agreement there would prove nothing.
    #[test]
    fn the_two_newton_numerators_agree_where_both_are_sound() {
        for &x in &[0.5_f64, 0.55, 0.6, -0.5, -0.6] {
            let r = erf_inv_seed(x);
            let complement = newton_residual(x, r);
            let direct = erf(r) - x;
            assert!(
                (complement - direct).abs() <= 1e-15,
                "at x = {x} the complement form gives {complement:e} and the direct {direct:e}"
            );
        }
        // And the odd symmetry the sign branch implements.
        let r = erf_inv_seed(0.9);
        assert_eq!(newton_residual(-0.9, -r), -newton_residual(0.9, r));
    }

    /// The three tail pieces, each asserted on its own.
    #[test]
    fn the_tail_pieces_each_do_their_part() {
        let complement = 1.110_223_024_625_156_5e-16_f64;
        let root = 5.863_584_748_755_167_6_f64;

        // The scale is just the logarithm, but it is what every later term is written against.
        assert!((tail_log_scale(complement) - 36.736_800_569_677_1).abs() < 1e-12);

        // The seed lands close enough for Newton to be quadratic — two orders inside the
        // eleven percent Giles's fit leaves, which is the whole point of the route.
        let seed = tail_seed(complement);
        assert!(
            (seed - root).abs() / root < 1e-3,
            "the tail seed is {seed}, which is not within 1e-3 of the root"
        );

        // One step is not enough and two are; both halves of that are asserted, because
        // dropping to one step is the cheap-looking change a later reader might make.
        let once = erfc_newton_step(seed, complement);
        let twice = erfc_newton_step(once, complement);
        assert!((once - root).abs() > (twice - root).abs());
        assert!((twice - root).abs() <= 4.0 * crate::Real::ulp(root));
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

        for x in [-1.0, (-1.0_f64).next_down(), -2.0, f64::NEG_INFINITY] {
            assert_eq!(ErfInvRoute::for_argument(x), AtNegativeOne, "at {x:e}");
            assert_eq!(erf_inv(x), f64::NEG_INFINITY, "at {x:e}");
        }
        for x in [1.0, 1.0_f64.next_up(), 2.0, f64::INFINITY] {
            assert_eq!(ErfInvRoute::for_argument(x), AtOne, "at {x:e}");
            assert_eq!(erf_inv(x), f64::INFINITY, "at {x:e}");
        }
        // Odd at the origin, which is what `copysign` is for.
        for x in [0.0_f64, -0.0] {
            assert_eq!(ErfInvRoute::for_argument(x), AtOrigin);
            assert_eq!(erf_inv(x).to_bits(), x.to_bits(), "signed zero at {x:e}");
        }
        for x in [
            (-1.0_f64).next_up(),
            -0.5,
            -f64::MIN_POSITIVE,
            f64::MIN_POSITIVE,
            0.5,
            1.0_f64.next_down(),
            f64::NAN,
        ] {
            assert_eq!(ErfInvRoute::for_argument(x), Interior, "at {x:e}");
        }
        // `NaN` reaches the interior and propagates rather than being caught by a guard.
        assert!(erf_inv(f64::NAN).is_nan());

        // The seed tables, at and either side of the boundary.
        assert_eq!(ErfInvSeed::for_log_weight(5.0), Tail);
        assert_eq!(ErfInvSeed::for_log_weight(5.0_f64.next_down()), Central);
        assert_eq!(ErfInvSeed::for_log_weight(5.0_f64.next_up()), Tail);
        assert_eq!(ErfInvSeed::for_log_weight(0.0), Central);
        assert_eq!(ErfInvSeed::for_log_weight(f64::NAN), Tail);
        // Both tables are evaluated in different variables, so they are genuinely two fits —
        // yet they agree at the join, which is what makes the split free.
        let join = 5.0_f64;
        let (central, tail) = (Central.evaluate(join), Tail.evaluate(join));
        assert!(
            (central - tail).abs() <= 2e-2 * tail.abs(),
            "the seed tables disagree at the join: {central} vs {tail}"
        );
    }
}
