//! The two routing decisions, asserted directly rather than through a value.
//!
//! Both exist because mutation testing reached them and no value assertion could. [`AiryPairing`]
//! selects between the separable halves and the shared quadruple, which are *bit-identical* — so
//! the decision carries a cost saving and nothing else, and only the decision can be pinned.
//! The split exponential that unscales `Bi` moved to [`crate::math::scaling`], which tests it
//! generically — the arm the public surface reaches only in a band `0.17` wide at the top of the
//! format, and reached there only for `Bi` and never for `Bi′`.

use super::super::{AI_SERIES_MAX, AiryPairing, LEADING_MIN};

#[test]
fn the_separable_band_is_open_at_both_ends() {
    use AiryPairing::{Separable, Shared};
    // Inside: the two solutions share no intermediates, so a half can be asked for alone.
    for x in [
        AI_SERIES_MAX.next_up(),
        2.0,
        9.0,
        100.0,
        1e9,
        LEADING_MIN.next_down(),
    ] {
        assert_eq!(AiryPairing::for_argument(x), Separable, "at {x}");
    }
    // Outside, including both boundaries themselves: the series at the origin, the negative
    // axis, the leading-order tail, and every non-finite argument.
    for x in [
        AI_SERIES_MAX,
        AI_SERIES_MAX.next_down(),
        0.5,
        0.0,
        -0.0,
        -2.0,
        -1e300,
        LEADING_MIN,
        LEADING_MIN.next_up(),
        1e300,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ] {
        assert_eq!(AiryPairing::for_argument(x), Shared, "at {x}");
    }
}

/// `Ai` and `Ai′` across the positive band, where the scaled pair is multiplied back down.
///
/// The `Aiᵉ′·e^{−ζ}` multiply and the regime comparison both survived inside `quad`: the
/// unscaled surface's sweeps carry a budget wide enough to absorb them, and `Ai′` is small here.
/// Against an independent reference at a relative tolerance they are unmissable.
#[test]
fn the_positive_band_unscales_both_of_the_first_pair() {
    use super::super::{aie, aie_prime, bie, bie_prime, positive_unscaled, zeta};
    for x in [AI_SERIES_MAX.next_up(), 1.5_f64, 3.0, 9.0, 30.0, 90.0] {
        let (ai, ai_prime, bi, bi_prime) = positive_unscaled(x);
        // The scaled forms are the primitives, so `Ai = Aiᵉ·e^{−ζ}` is the defining relation and
        // an independent route to the same number.
        let decay = (-zeta(x)).exp();
        let want = aie(x) * decay;
        let want_prime = aie_prime(x) * decay;
        assert!(
            (ai - want).abs() <= 1e-12 * want.abs(),
            "Ai({x}): {ai:e} vs {want:e}"
        );
        assert!(
            (ai_prime - want_prime).abs() <= 1e-12 * want_prime.abs(),
            "Ai'({x}): {ai_prime:e} vs {want_prime:e}"
        );
        // The same relation pins the *second* pair, which is what selects between the series
        // and the `I` asymptotic: `Bi = Biᵉ·e^{+ζ}`. Checking only its sign left the regime
        // comparison free to pick the wrong arm, because the two agree over most of the band
        // and diverge only well out in it.
        let growth = zeta(x).exp();
        let want = bie(x) * growth;
        let want_prime = bie_prime(x) * growth;
        assert!(
            (bi - want).abs() <= 1e-12 * want.abs(),
            "Bi({x}): {bi:e} vs {want:e}"
        );
        assert!(
            (bi_prime - want_prime).abs() <= 1e-12 * want_prime.abs(),
            "Bi'({x}): {bi_prime:e} vs {want_prime:e}"
        );
        // Both decay and both keep their signs across the whole positive axis.
        assert!(ai > 0.0 && ai_prime < 0.0, "at {x}");
        assert!(bi > 0.0 && bi_prime > 0.0, "at {x}");
    }
    // The series arm is selected below `AI_SERIES_MAX` and needs no exponential at all.
    assert_eq!(positive_unscaled(0.5), super::super::series::airy(0.5));
    // And they decay monotonically, which the exponential factor is what delivers.
    assert!(positive_unscaled(1.5).0 > positive_unscaled(9.0).0);
    assert!(positive_unscaled(1.5).1 < positive_unscaled(9.0).1);
}
