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
//! 1 − x     1e-6    1e-7      1e-9      1e-12       1e-15      1.1e-16
//! was       9187   35923     1.2e9      5.1e12      2.3e14       4.6e14   ulp
//! now       9187    2249       215        14.3         2.1          0.7   ulp
//! ```
//!
//! Ten bands better, none worse. `1e-6` and above keep Giles's fit, because the asymptotic seed
//! is *worse* there — see [`ERF_INV_TAIL_MAX`] for the table that places the crossover, which is
//! three decades below where the seed merely becomes defined.
//!
//! What is left is the band from `1e-6` to about `1e-2`, where Giles's fit carries `9187` and
//! `48.9` ulp. That is the same seed and the same cause, just milder, and it wants the same
//! treatment: a seed valid over that range, which the asymptotic expansion is not.
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
    let seed = erf_inv_seed(x);
    seed - newton_residual(x, seed) / erf_density(seed)
}

/// At or below this `1 − |x|`, the tail route replaces Giles's fit.
///
/// **`1e-7`, chosen by measurement and not by where the seed is merely defined.** Against MPFR
/// at 300 bits, with the threshold opened wide enough to see both routes everywhere:
///
/// ```text
/// 1 − x       1e-4     1e-5    1e-6    1e-7      1e-9        1.1e-16
/// Giles       17.5    204.3    9187   35923     1.2e9         4.6e14   ulp
/// tail      4.4e5     5.3e4    9544    2249       215            0.7   ulp
/// ```
///
/// The tail route overtakes at `1e-7` and is worse above it — `4.4e5` ulp at `1e-4`, which is
/// twenty-five thousand times Giles's. Two Newton steps are quadratic only once the seed is in
/// the basin, and the asymptotic seed's basin shrinks as `r` does. Placing this at `1e-3`,
/// where [`tail_seed`]'s `ln(1 − 1/(2r²))` merely stops being undefined, would have traded one
/// defect for another over three decades.
///
/// `1 − |x|` is exact here by Sterbenz — `|x| ≥ ½` covers every argument this admits — so the
/// complement costs nothing to form and the tail route sees the full precision of the input.
const ERF_INV_TAIL_MAX: f64 = 1e-7;

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
    erfc_newton_step(once, complement)
}

/// One Newton step against `erfc(r) = c`.
///
/// Two of them are taken, and the reason is measurable rather than cautious: the seed lands
/// about `2e-4` from the root, one step takes that to `2.7e-9`, and the second to rounding.
/// Newton is quadratic here — unlike the `erf`-space step it replaces, which was being asked to
/// cross eleven percent and could not.
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

/// `erf(r) − x`, the Newton numerator.
fn newton_residual(x: f64, r: f64) -> f64 {
    erf(r) - x
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
