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
    use super::super::regimes::positive_unscaled;
    use super::super::{aie, aie_prime, bie, bie_prime, zeta};
    use crate::math::pair::Solutions;
    for x in [AI_SERIES_MAX.next_up(), 1.5_f64, 3.0, 9.0, 30.0, 90.0] {
        // Destructured by field name: `first_kind` is the `Ai` half whatever order the
        // quadruple happens to be laid out in.
        let Solutions {
            first_kind: ai,
            second_kind: bi,
        } = positive_unscaled(x);
        // The scaled forms are the primitives, so `Ai = Aiᵉ·e^{−ζ}` is the defining relation and
        // an independent route to the same number.
        let decay = (-zeta(x)).exp();
        let want = aie(x) * decay;
        let want_prime = aie_prime(x) * decay;
        assert!(
            (ai.value - want).abs() <= 1e-12 * want.abs(),
            "Ai({x}): {:e} vs {want:e}",
            ai.value
        );
        assert!(
            (ai.derivative - want_prime).abs() <= 1e-12 * want_prime.abs(),
            "Ai'({x}): {:e} vs {want_prime:e}",
            ai.derivative
        );
        // The same relation pins the *second* pair, which is what selects between the series
        // and the `I` asymptotic: `Bi = Biᵉ·e^{+ζ}`. Checking only its sign left the regime
        // comparison free to pick the wrong arm, because the two agree over most of the band
        // and diverge only well out in it.
        let growth = zeta(x).exp();
        let want = bie(x) * growth;
        let want_prime = bie_prime(x) * growth;
        assert!(
            (bi.value - want).abs() <= 1e-12 * want.abs(),
            "Bi({x}): {:e} vs {want:e}",
            bi.value
        );
        assert!(
            (bi.derivative - want_prime).abs() <= 1e-12 * want_prime.abs(),
            "Bi'({x}): {:e} vs {want_prime:e}",
            bi.derivative
        );
        // Both decay and both keep their signs across the whole positive axis.
        assert!(ai.value > 0.0 && ai.derivative < 0.0, "at {x}");
        assert!(bi.value > 0.0 && bi.derivative > 0.0, "at {x}");
    }
    // The series arm is selected below `AI_SERIES_MAX` and needs no exponential at all.
    assert_eq!(positive_unscaled(0.5), super::super::series::airy(0.5));
    // And they decay monotonically, which the exponential factor is what delivers.
    assert!(positive_unscaled(1.5).first_kind.value > positive_unscaled(9.0).first_kind.value);
    assert!(
        positive_unscaled(1.5).first_kind.derivative < positive_unscaled(9.0).first_kind.derivative
    );
}
