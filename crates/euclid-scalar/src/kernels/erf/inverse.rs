//! The inverse error function: Giles's rational approximation plus one Newton step.
//!
//! # Known defect: this loses most of its digits as `|x| -> 1`
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
//! The round trip `erf(erf_inv(x)) == x` is the test that states this cleanly, and it is the one
//! the suite does not make near `±1`.
//!
//! # The fix, and why it is not applied here yet
//!
//! `erfc_inv` is the conditioned route for this region and measures **12 ulp** over the same
//! sweep, so `erf_inv(x) = sign(x)·erfc_inv(1 − |x|)` is the answer for large `|x|`. The
//! subtraction is exact where it matters: `1 − |x|` is exact for `|x| ≥ ½` by Sterbenz, so
//! nothing is lost reaching it.
//!
//! Left unapplied because `erf_inv` feeds `ndtri` and the normal quantile family, the crossover
//! wants its own measured no-regression search, and the round-trip test that should have caught
//! this needs writing first. The diagnosis and the reproduction above are what that work starts
//! from.
//!
//! A child module so the parent stays inside the crate's 400-line budget. `erf`, `erfc` and
//! `erfcx` are one topic — all three reduce to the same incomplete gamma or the same continued
//! fraction — and the *inverse* is another: it shares no machinery with them beyond calling
//! `erf` once to refine, and it carries its own two-table seed whose coefficients are a fit
//! rather than a derivation.
//!
//! The single-precision peer at [`crate::kernels::f32::erf`] keeps its inverse in the parent
//! file, because without the Newton step it is short enough to fit.

use super::erf;
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
fn erf_inv_interior(x: f64) -> f64 {
    let w = -((1.0 - x) * (1.0 + x)).ln();
    let mut result = ErfInvSeed::for_log_weight(w).evaluate(w) * x;
    let err = erf(result) - x;
    result -= err / (2.0 / PI.sqrt() * (-result * result).exp());
    result
}

#[cfg(test)]
mod tests {
    use super::*;

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
