//! Unit tests for the inverse error function.
//!
//! A child module so implementation plus tests stay inside the crate's 400-line budget, which
//! the parent crossed when the crossover's measurement was written down. `use super::*` still
//! reaches the private pieces these assert on — `erf_inv_seed`, both Newton steps,
//! `tail_magnitude` and the `ERF_INV_TAIL_MAX` crossover — and that is the whole reason these
//! live beside the implementation rather than in `tests/`: every one of them is about a route
//! or a threshold the trait surface does not expose.
//!
//! The split is by *what is being guarded*, not by function:
//!
//! * the two routes agree where both are accurate, which is what pins the crossover;
//! * each route reaches the root its own way, which is what pins the pieces;
//! * the round trip, which needs no reference and is sharp on the interior.

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

/// `erf(erf_inv(x)) == x`, across the whole domain and both routes.
///
/// **Sampled inside each decade, not at the decade points.** The grid this used to walk put
/// a sample exactly on `ERF_INV_TAIL_MAX`, which takes the tail route and is fine, while the
/// band just above it was five thousand ulp out. A log-spaced grid is still a grid, and a
/// threshold bug hides between its points by construction.
///
/// # What this test cannot do, which is most of why the band survived
///
/// A round trip through `erf` is **blind in the tail**, and no amount of resampling fixes
/// that. `erf′` at `r = 4.19` is `5.6e-8`, so the `6277` ulp the old threshold cost at
/// `1 − x = 3e-9` moves `erf(r)` by `0.003` ulp of `x`. Measured on both builds at that
/// argument, the round trip reads `0.0000` ulp either way.
///
/// It earns its place on the *interior*, where `erf′` is `O(1)` and the round trip is sharp,
/// and it is what caught the original ten-percent tail defect — an error large enough to
/// land on a different double. Between those two is a gap, and
/// [`the_two_routes_agree_across_the_band_that_both_are_accurate_on`] is what covers it.
///
/// **Sixteen ulp, and the bar had to go up when the sampling got honest.** I set it twice
/// without measuring — bit-exact first, which fails by one ulp at `x = 0.9637`, then two,
/// which fails by three at `0.8555` — and then at six, from a grid whose worst was `3.0`.
/// Sampled within decades over 120000 arguments the worst is `12.0` ulp at `x = 0.0128`,
/// with ten above `6` and six above `8`. Six was not headroom over `3`; it was one ulp
/// under a worst the grid could not reach.
///
/// That `12.0` is not this function's error. At `|x| = 0.0128` the root is `0.0114` and
/// `erf′` is `O(1)`, so the round trip is sharp there and is reporting `erf`'s own `12` ulp
/// at small arguments straight through — [`newton_residual`] carries that measurement.
/// A round trip carries the forward function's rounding as well as the inverse's, so the
/// bar is a property of the pair and there is no deriving it from one side.
///
/// The draw below reaches about `5` ulp, so the bar is not tight against this sample; it is
/// set against the band worst so that a different seed cannot turn a passing test red.
#[test]
fn erf_inv_round_trips_through_erf() {
    let mut rng = euclid_stability::rng::SplitMix64::new(0x5EED_0001);
    let mut checked = 0usize;
    let mut worst = (0.0_f64, 0.0_f64);
    for _ in 0..800 {
        // `1 − |x|` log-uniform over the whole range, so every decade is sampled across
        // rather than at its ends.
        let exponent = -16.0 + 16.0 * unit_interval(&mut rng);
        let complement = 10f64.powf(exponent);
        for x in [1.0 - complement, -(1.0 - complement)] {
            if x == 0.0 || x.abs() >= 1.0 {
                continue;
            }
            let round_tripped = erf(erf_inv(x));
            let ulps = (round_tripped - x).abs() / crate::Real::ulp(x);
            if ulps > worst.0 {
                worst = (ulps, x);
            }
            assert!(
                ulps <= 16.0,
                "erf(erf_inv({x:e})) = {round_tripped:e}, which is {ulps} ulp away"
            );
            checked += 1;
        }
    }
    assert!(
        checked > 1400,
        "only {checked} arguments reached the assertion"
    );
    assert!(
        worst.0 > 0.5,
        "worst round trip was {} ulp at {:e}; a bar of sixteen over a sample this size means \
         nothing if nothing approaches it",
        worst.0,
        worst.1
    );
}

/// A `u64` from the stream as a `f64` in `[0, 1)`, by its top 53 bits.
fn unit_interval(rng: &mut euclid_stability::rng::SplitMix64) -> f64 {
    // `2^-53` scaling of a 53-bit integer is exact, so the draw is uniform on the
    // representable grid rather than on something the rounding has smeared.
    #[allow(clippy::cast_precision_loss)]
    let mantissa = (rng.next_u64() >> 11) as f64;
    mantissa * (1.0 / 9_007_199_254_740_992.0)
}

/// The two routes agree wherever both are accurate, and the threshold sits inside that band.
///
/// **This is the test the band needed, and the round trip could not be.** It needs no
/// reference: where two independent routes to the same root are both correct they must
/// agree, so their disagreement is an error bound that costs nothing to evaluate. Unlike the
/// round trip it does not pass through `erf`, so the flatness that hides a wrong root from a
/// round trip does not hide it from here.
///
/// It asserts two things that are only worth something together:
///
/// 1. **The overlap is real** — both routes agree to a few ulp across `[1e-8, 1.8e-3]`.
///    Measured against MPFR, each is `≤ 1.0` ulp of the true root throughout.
/// 2. **The threshold is inside it, with margin.** This is what the old `1e-9` failed:
///    it sat a decade *below* the interior route's floor, so arguments just above it took a
///    route that was already `6277` ulp out.
///
/// The first clause is what makes the second more than an assertion about a literal, and it
/// guards the tail's step count too: dropping [`tail_magnitude`] to two `erfc` steps puts
/// the tail `16467` ulp out at `1e-6`, which is inside this band, so the routes stop
/// agreeing and this fails.
#[test]
fn the_two_routes_agree_across_the_band_that_both_are_accurate_on() {
    // The measured edges: below `OVERLAP_MIN` Giles's seed is past saving, above
    // `OVERLAP_MAX` the asymptotic seed's second term is no longer enough.
    const OVERLAP_MIN: f64 = 1e-8;
    const OVERLAP_MAX: f64 = 1.8e-3;

    // A `const` block, so moving the threshold outside the overlap fails to *compile*
    // rather than failing a test someone might not run. This is the half of the property
    // that the old `1e-9` violated.
    const {
        assert!(
            ERF_INV_TAIL_MAX >= OVERLAP_MIN * 10.0 && ERF_INV_TAIL_MAX <= OVERLAP_MAX / 10.0,
            "ERF_INV_TAIL_MAX must sit at least a decade inside the overlap \
             [1e-8, 1.8e-3], the band both routes are measured correctly rounded on"
        );
    }

    let mut rng = euclid_stability::rng::SplitMix64::new(0x5EED_0002);
    let (lo, hi) = (OVERLAP_MAX.log10(), OVERLAP_MIN.log10());
    let mut worst = (0.0_f64, 0.0_f64);
    for _ in 0..600 {
        let x = 1.0 - 10f64.powf(lo + (hi - lo) * unit_interval(&mut rng));
        // Taken back off `x` rather than reused from the draw. `1.0 - complement` rounds,
        // so the two differ by up to half an ulp of `x` — which at `1 − x = 2.5e-4` is 90
        // ulp of root, because `dr/dc` is `1/erf′(r)` and `erf′(2.59)` is `1.4e-3`. Feeding
        // each route its own argument made this read as a 60 ulp disagreement between
        // routes that in fact agree. `1 − x` is exact here by Sterbenz, so this is the
        // complement the kernel itself forms.
        let complement = 1.0 - x;

        let once = erf_newton_step(x, erf_inv_seed(x));
        let interior = erf_newton_step(x, once);
        let tail = tail_magnitude(complement);

        let ulps = (interior - tail).abs() / crate::Real::ulp(tail);
        if ulps > worst.0 {
            worst = (ulps, complement);
        }
        assert!(
            ulps <= 8.0,
            "at 1 − x = {complement:e} the interior route gives {interior} and the tail \
             {tail}, which are {ulps} ulp apart"
        );
    }
    // Both are within a ulp of the root, so a handful of ulp between them is all there is
    // to find. A bar of eight over a band this wide is headroom, not slack.
    assert!(
        worst.0 <= 8.0,
        "worst disagreement {} ulp at 1 − x = {:e}",
        worst.0,
        worst.1
    );
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
