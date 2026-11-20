//! The mathematical constants, across both precisions and against their definitions.
//!
//! Split from the delegate checks because the instrument differs: a constant has a *definition*,
//! so it can be checked against arithmetic rather than against another implementation. The
//! derived angle constants are the sharper half — each is a product or quotient of two others,
//! so a transcription slip in any one of them shows up as a broken relation.

use crate::Real;

// --- cross-precision agreement ------------------------------------------

/// The mathematical constants must agree to the last bit of `f32`.
///
/// These are the simplest mutation target in the crate — a constant replaced by a default — and
/// nothing was checking them. There is no accuracy argument available here: the f32 value is the
/// f64 value correctly rounded, or it is wrong.
/// **Every constant on the trait, not a selection.** The list below used to hold four of the
/// twenty-four, and `sqrt_tau` surviving a mutation to `Default::default()` is what that costs:
/// there is no reason for one lifted constant to be checked and its neighbour not, and a
/// partial list gives no signal about which half it is in. A constant added to [`Real`] belongs
/// here the same day.
#[test]
fn the_constants_agree_across_precisions() {
    #[allow(clippy::cast_possible_truncation)]
    let narrowed = |wide: f64| wide as f32;

    // Each entry pairs the two carriers' accessors so the assertion below reads once.
    macro_rules! agree {
        ($($name:ident),* $(,)?) => {{
            $(assert_eq!(
                <f32 as Real>::$name(),
                narrowed(<f64 as Real>::$name()),
                concat!("the lifted constant `", stringify!($name), "`"),
            );)*
        }};
    }

    // The circle and its subdivisions.
    agree!(pi, tau, frac_pi_2, frac_pi_4, sqrt_pi, sqrt_tau);
    // Roots and their reciprocals.
    agree!(sqrt_2, frac_1_sqrt_2);
    // The logarithmic family.
    agree!(ln_2, ln_pi, ln_tau, half_ln_tau, ln_10);
    // The exponential family and the log bases.
    agree!(e, log2_e, log10_e, log2_10, log10_2);
    // The named analytic constants.
    agree!(euler_gamma, golden_ratio, apery, catalan, omega);
    // The small rationals, which are exact at both widths and so are the strictest of all.
    agree!(two, half, neg_one);
    // The degree constants, which exist so a formula can lift them without a literal.
    agree!(
        full_turn_degrees,
        half_turn_degrees,
        quarter_turn_degrees,
        three_half_turn_degrees,
    );

    // The non-finite three, which `narrowed` would compare incorrectly for NaN.
    assert_eq!(<f32 as Real>::infinity(), f32::INFINITY, "infinity");
    assert_eq!(
        <f32 as Real>::neg_infinity(),
        f32::NEG_INFINITY,
        "−infinity"
    );
    assert!(<f32 as Real>::nan().is_nan(), "NaN");
    assert!(<f64 as Real>::nan().is_nan(), "NaN at f64");
}

/// The two **derived** constants, which cannot make the bit-exact claim above — and the
/// distinction is worth stating rather than discovering.
///
/// Every constant in the test above is *lifted*: the body is `Self::scalar(literal)`, so the f32
/// value is the f64 value correctly rounded by construction and anything else is a typo.
/// `radians_per_degree` and `degrees_per_radian` are **computed**, as `pi / 180` and `180 / pi`,
/// so each carrier divides using *its own* `pi` and the results are not related by rounding.
///
/// Extending the list above to cover everything caught this: `degrees_per_radian` at f32 is
/// `57.295_776`, while the f64 value narrowed is `57.295_780` — some thirty-two f32 ulps apart.
/// That is not a defect. `pi` at f32 is already rounded, and putting it in a *denominator*
/// carries that error into the quotient, where rounding the f64 quotient instead keeps it. Its
/// sibling `radians_per_degree` happens to agree bit-for-bit, which is exactly the kind of
/// coincidence that would make a selective list look fine.
///
/// So these two are checked against the true value at each carrier's own resolution, which is
/// the only claim either can support.
#[test]
fn the_derived_angle_constants_match_their_definitions() {
    // 180/π and π/180 from an independent source, written to the digits `f64` can hold —
    // the true values continue `...320_877` and `...295_769`.
    const DEGREES_PER_RADIAN: f64 = 57.295_779_513_082_32;
    const RADIANS_PER_DEGREE: f64 = 0.017_453_292_519_943_295;

    let close = |got: f64, want: f64, tol: f64, what: &str| {
        assert!(
            (got / want - 1.0).abs() <= tol,
            "{what}: got {got}, want {want} (relative gap {:.3e} exceeds {tol:.0e})",
            (got / want - 1.0).abs(),
        );
    };

    // f64: both land within an ulp of the true value.
    close(
        <f64 as Real>::degrees_per_radian(),
        DEGREES_PER_RADIAN,
        1e-15,
        "f64 degrees_per_radian",
    );
    close(
        <f64 as Real>::radians_per_degree(),
        RADIANS_PER_DEGREE,
        1e-15,
        "f64 radians_per_degree",
    );
    // f32: measured at 5.6e-8 and 7.8e-9 respectively, both inside one f32 ulp of the truth.
    close(
        f64::from(<f32 as Real>::degrees_per_radian()),
        DEGREES_PER_RADIAN,
        2e-7,
        "f32 degrees_per_radian",
    );
    close(
        f64::from(<f32 as Real>::radians_per_degree()),
        RADIANS_PER_DEGREE,
        2e-7,
        "f32 radians_per_degree",
    );

    // And they are reciprocals, at each carrier's own precision.
    for (product, what) in [
        (
            <f64 as Real>::degrees_per_radian() * <f64 as Real>::radians_per_degree(),
            "f64",
        ),
        (
            f64::from(<f32 as Real>::degrees_per_radian() * <f32 as Real>::radians_per_degree()),
            "f32",
        ),
    ] {
        close(
            product,
            1.0,
            2e-7,
            &format!("{what}: the two are reciprocals"),
        );
    }
}
