//! `J₀`/`J₁` and `Y₀`/`Y₁`: reference values, the three regimes, and the bands where adjacent
//! methods overlap.
//!
//! Kept as one file because the overlap tests deliberately straddle the two kinds — the
//! Hankel expansion is shared, so checking it against the trapezoid and against the march
//! belongs in one place.

use crate::kernels::bessel::asymptotic::hankel;
use crate::kernels::bessel::ordinary_first::{
    HANKEL_MIN, J_SERIES_MAX, j_aliased, j_series, j0, j1,
};
use crate::kernels::bessel::ordinary_second::{
    Y_SERIES_MAX, y_march, y0, y0_series, y1, y1_series,
};

fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol * b.abs().max(1.0),
        "{a:e} vs {b:e} (tol {tol:e}, abs err {:e})",
        (a - b).abs()
    );
}

/// 600-digit `Decimal` references: `(x, J₀, J₁, Y₀, Y₁)`.
const REFS: &[(f64, f64, f64, f64, f64)] = &[
    (
        0.5,
        9.384_698_072_408_129e-1,
        2.422_684_576_748_739e-1,
        -4.445_187_335_067_065_6e-1,
        -1.471_472_392_670_243e0,
    ),
    (
        1.0,
        7.651_976_865_579_666e-1,
        4.400_505_857_449_335e-1,
        8.825_696_421_567_696e-2,
        -7.812_128_213_002_887e-1,
    ),
    (
        2.0,
        2.238_907_791_412_356_7e-1,
        5.767_248_077_568_734e-1,
        5.103_756_726_497_451e-1,
        -1.070_324_315_409_375_4e-1,
    ),
    (
        5.0,
        -1.775_967_713_143_383e-1,
        -3.275_791_375_914_652_3e-1,
        -3.085_176_252_490_337_6e-1,
        1.478_631_433_912_268_3e-1,
    ),
    (
        8.0,
        1.716_508_071_375_539e-1,
        2.346_363_468_539_146_3e-1,
        2.235_214_893_875_662_2e-1,
        -1.580_604_617_312_475e-1,
    ),
    (
        10.0,
        -2.459_357_644_513_483_5e-1,
        4.347_274_616_886_144e-2,
        5.567_116_728_359_939_5e-2,
        2.490_154_242_069_538_8e-1,
    ),
    (
        15.0,
        -1.422_447_282_678_077_2e-2,
        2.051_040_386_135_227_5e-1,
        2.054_642_960_389_182_8e-1,
        2.107_362_803_687_351e-2,
    ),
    (
        20.0,
        1.670_246_643_405_831_6e-1,
        6.683_312_417_585_005e-2,
        6.264_059_680_938_383e-2,
        -1.655_116_143_625_213e-1,
    ),
    (
        30.0,
        -8.636_798_358_104_021e-2,
        -1.187_510_626_166_229_4e-1,
        -1.172_957_316_866_640_3e-1,
        8.442_557_066_174_723e-2,
    ),
    (
        50.0,
        5.581_232_766_925_181_6e-2,
        -9.751_182_812_517_514e-2,
        -9.806_499_547_007_708e-2,
        -5.679_566_856_201_477e-2,
    ),
    (
        100.0,
        1.998_585_030_422_312_2e-2,
        -7.714_535_201_411_216e-2,
        -7.724_431_336_508_315e-2,
        -2.037_231_200_275_979_2e-2,
    ),
];

#[test]
fn all_four_match_their_reference_values() {
    for &(x, j0_r, j1_r, y0_r, y1_r) in REFS {
        close(j0(x), j0_r, 1e-15);
        close(j1(x), j1_r, 1e-15);
        close(y0(x), y0_r, 1e-15);
        close(y1(x), y1_r, 1e-15);
    }
}

// The J/Y Wronskian `J₀Y₁ − J₁Y₀ = −2/(πx)` is not here: it spans both kinds, so it lives
// in `tests::identities::bessel_family`, where it is checked on the public trait surface
// across every regime.

#[test]
fn the_regimes_join_without_a_step() {
    // Each join is checked against the derivative, which bounds how much the function may
    // legitimately move across the interval: J0' = -J1, Y0' = -Y1, and
    // J1' = J0 - J1/x, Y1' = Y0 - Y1/x.
    let h = 1e-9;
    for &x in &[Y_SERIES_MAX, HANKEL_MIN] {
        for (name, f, derivative) in [
            ("J0", j0 as fn(f64) -> f64, -j1(x)),
            ("J1", j1 as fn(f64) -> f64, j0(x) - j1(x) / x),
            ("Y0", y0 as fn(f64) -> f64, -y1(x)),
            ("Y1", y1 as fn(f64) -> f64, y0(x) - y1(x) / x),
        ] {
            let (below, above) = (f(x - h), f(x + h));
            let implied = derivative.abs() * 2.0 * h;
            assert!(
                (above - below).abs() <= implied * 1.05 + 1e-14,
                "{name} steps at x = {x}: {below:e} then {above:e}"
            );
        }
    }
}

#[test]
fn the_recurrence_relations_hold() {
    // 2n/x * Y_n = Y_{n-1} + Y_{n+1} with n = 1 gives Y2 = 2Y1/x - Y0, and the same for J.
    // Checking the derivative identities instead, which need no J2/Y2:
    //   J0' = -J1,  (x J1)' = x J0
    for i in 1..200 {
        let x = f64::from(i) * 0.15;
        let h = 1e-5 * x;
        // A central difference truncates at order h² times the third derivative, which
        // with h = 1e-5·x is about 1e-8 by x = 30. These tolerances are set by the
        // differencing, not by the functions being differenced.
        let d_j0 = (j0(x + h) - j0(x - h)) / (2.0 * h);
        close(d_j0, -j1(x), 1e-7);
        let d_y0 = (y0(x + h) - y0(x - h)) / (2.0 * h);
        close(d_y0, -y1(x), 1e-7);
        // (x J1)' = x J0
        let d_xj1 = ((x + h) * j1(x + h) - (x - h) * j1(x - h)) / (2.0 * h);
        close(d_xj1, x * j0(x), 1e-6);
    }
}

#[test]
fn j_satisfies_bessels_equation() {
    // x^2 y'' + x y' + x^2 y = 0 for J0, by finite differences.
    for i in 2..120 {
        let x = f64::from(i) * 0.25;
        let h = 1e-4 * x;
        let second = (j0(x + h) - 2.0 * j0(x) + j0(x - h)) / (h * h);
        let first = (j0(x + h) - j0(x - h)) / (2.0 * h);
        let residual = x * x * second + x * first + x * x * j0(x);
        assert!(
            residual.abs() < 1e-5 * x * x,
            "Bessel's equation broken for J0 at {x}: {residual:e}"
        );
    }
}

#[test]
fn adjacent_methods_agree_across_their_whole_overlap() {
    // Stronger than a join test: each pair of methods is run side by side over a band
    // where both should be valid, so agreement constrains them everywhere rather than at
    // a single point. The three are fully independent of one another — a finite
    // trigonometric sum, an ODE march, and a divergent asymptotic expansion — so this is
    // real evidence rather than a tautology.
    for i in 0..=48 {
        let x = 18.0 + f64::from(i) * 0.5; // [18, 42]
        let order0 = hankel(0.0, x);
        let order1 = hankel(1.0, x);
        let (j0_hankel, y0_hankel) = (order0.first_kind, order0.second_kind);
        let (j1_hankel, y1_hankel) = (order1.first_kind, order1.second_kind);
        close(j0_hankel, j_aliased(0, x), 1e-14);
        close(j1_hankel, j_aliased(1, x), 1e-14);
        let marched = y_march(x);
        close(y0_hankel, marched.value, 1e-13);
        close(y1_hankel, marched.derivative, 1e-13);
    }
    // The J series against the quadrature, across the whole band where both hold. The
    // series is exact up to a few, and the quadrature from about 1e-8 upward, so this
    // overlap is wide.
    for i in 1..=60 {
        let x = f64::from(i) * 0.05; // (0, 3]
        close(j_series(0, x), j_aliased(0, x), 1e-15);
        close(j_series(1, x), j_aliased(1, x), 1e-15);
    }
    // And the Y series against the march just above the series cutoff.
    for i in 0..=30 {
        let x = Y_SERIES_MAX + f64::from(i) * 0.05;
        close(y_march(x).value, y0_series(x), 1e-14);
        close(y_march(x).derivative, y1_series(x), 1e-14);
    }
}

#[test]
fn parity_and_non_finite_inputs() {
    for &(x, _, _, _, _) in REFS {
        close(j0(-x), j0(x), 1e-15);
        close(j1(-x), -j1(x), 1e-15);
    }
    assert_eq!(j0(0.0), 1.0);
    assert_eq!(j1(0.0), 0.0);
    assert!(j0(f64::NAN).is_nan());
    assert!(j1(f64::NAN).is_nan());
    assert!(y0(f64::NAN).is_nan());
    assert!(y1(f64::NAN).is_nan());
    assert_eq!(j0(f64::INFINITY), 0.0);
    assert_eq!(j1(f64::INFINITY), 0.0);
    assert_eq!(y0(f64::INFINITY), 0.0);
    assert_eq!(y1(f64::INFINITY), 0.0);
    // Y is real only for positive argument.
    assert!(y0(-1.0).is_nan());
    assert!(y1(-1.0).is_nan());
    assert_eq!(y0(0.0), f64::NEG_INFINITY);
    assert_eq!(y1(0.0), f64::NEG_INFINITY);
}

#[test]
fn no_regime_boundary_leaves_a_step_at_ulp_resolution() {
    // Each of the three crossovers, probed at the two representable values straddling it.
    // `J` switches series -> trapezoid at 2 and trapezoid -> Hankel at 18; `Y` switches
    // series -> march at 4 and march -> Hankel at 18.
    use euclid_utils::no_step_across;
    for &x in &[J_SERIES_MAX, HANKEL_MIN] {
        no_step_across("J0", j0, x, -j1(x), 1e-14);
        no_step_across("J1", j1, x, j0(x) - j1(x) / x, 1e-14);
    }
    for &x in &[Y_SERIES_MAX, HANKEL_MIN] {
        no_step_across("Y0", y0, x, -y1(x), 1e-14);
        no_step_across("Y1", y1, x, y0(x) - y1(x) / x, 1e-14);
    }
    // And the thresholds themselves must be exactly where the code says they are: evaluating
    // one ulp either side must actually cross a branch, which this confirms indirectly by
    // checking that the *other* route is the one being used just across it.
    assert!(
        (j0(J_SERIES_MAX) - j_series(0, J_SERIES_MAX)).abs() < 1e-15,
        "at the threshold J0 should still be the series"
    );
    let just_above = crate::Real::next_after(J_SERIES_MAX, f64::INFINITY);
    assert!(
        (j0(just_above) - j_aliased(0, just_above)).abs() < 1e-15,
        "one ulp above it J0 should be the trapezoid"
    );
}

/// `J`'s six routes, over the whole line, and `Y`'s four over the positive half.
///
/// Both predicates serve two orders apiece — four public functions dispatch on these two enums —
/// so a misroute moves several at once, and every arm returns a plausible oscillating value.
/// `J₀` is even and `J₁` odd, which is why one magnitude ladder covers both.
#[test]
fn the_ordinary_bessel_routes_partition_their_domains() {
    use super::super::ordinary_first::{HANKEL_MIN, J_SERIES_MAX, JRoute};
    use super::super::ordinary_second::{Y_SERIES_MAX, YRoute};
    use JRoute::{Aliased, AtInfinity, AtOrigin, Hankel, Series, Undefined};

    for (x, want) in [
        (f64::NAN, Undefined),
        (0.0, AtOrigin),
        (-0.0, AtOrigin),
        (f64::INFINITY, AtInfinity),
        (f64::NEG_INFINITY, AtInfinity),
        // The ladder is on the magnitude, so both signs route alike.
        (1.0, Series),
        (-1.0, Series),
        (J_SERIES_MAX, Series),
        (-J_SERIES_MAX, Series),
        (J_SERIES_MAX.next_up(), Aliased),
        (10.0, Aliased),
        (HANKEL_MIN.next_down(), Aliased),
        (HANKEL_MIN, Hankel),
        (-HANKEL_MIN, Hankel),
        (1e300, Hankel),
    ] {
        assert_eq!(JRoute::for_argument(x), want, "J at {x:e}");
    }

    // `Y` is defined only for a positive argument, and both orders diverge to `−∞` at zero.
    for (x, want) in [
        (f64::NAN, YRoute::Value(f64::NAN)),
        (-1.0, YRoute::Value(f64::NAN)),
        (0.0, YRoute::Value(f64::NEG_INFINITY)),
        (f64::INFINITY, YRoute::Value(0.0)),
        (1.0, YRoute::Series),
        (Y_SERIES_MAX, YRoute::Series),
        (Y_SERIES_MAX.next_up(), YRoute::March),
        (10.0, YRoute::March),
        (HANKEL_MIN.next_down(), YRoute::March),
        (HANKEL_MIN, YRoute::Hankel),
        (1e300, YRoute::Hankel),
    ] {
        let got = YRoute::for_argument(x);
        match (got, want) {
            // A `NaN` payload is not equal to itself, so that arm compares by shape.
            (YRoute::Value(g), YRoute::Value(w)) if g.is_nan() && w.is_nan() => {}
            _ => assert_eq!(got, want, "Y at {x:e}"),
        }
    }

    // The parities the single ladder depends on, and the signed zero `J₁` carries through it.
    for x in [0.5_f64, 1.5, 3.0, 10.0, 25.0] {
        assert_eq!(j0(-x), j0(x), "J0 even at {x}");
        assert_eq!(j1(-x), -j1(x), "J1 odd at {x}");
    }
    assert_eq!(j0(0.0), 1.0);
    assert_eq!(j1(0.0).to_bits(), 0.0_f64.to_bits());
    assert_eq!(j1(-0.0).to_bits(), (-0.0_f64).to_bits());
    assert!(y0(-1.0).is_nan() && y1(-1.0).is_nan());
    assert_eq!((y0(0.0), y1(0.0)), (f64::NEG_INFINITY, f64::NEG_INFINITY));
}

/// The `Y` series' **term counts**, which are the only observable their stopping rules have.
///
/// `TERM_EPS` is `1e-19`, below `f64`'s resolution, so a term admitted or refused at the margin
/// moves no value — mutation testing finds every comparison in `t < sum·ε` unkillable, including
/// the degenerate `==` that stops the break firing at all and runs the loop to `MAX_TERMS`. The
/// count separates them and nothing else does.
///
/// # The count found a real defect, not just an unkillable mutant
///
/// `y0_series` ran **999 of its 1,000 iterations** for every argument below about `3e-162`.
/// Below there `x²/4` underflows to zero, so `t` and `sum` are both exactly zero and the
/// relative test `0 < 0·ε` is false — the break could never fire. The returned value was right
/// the whole time, which is why nothing else noticed; only the cost was wrong. It is one
/// iteration now, and `y0_series`' exit carries an explicit zero-term case.
///
/// Counts are pinned rather than bounded: a change is exactly what a regression in a stopping
/// rule or in `TERM_EPS` looks like.
#[test]
fn the_y_series_stop_where_their_tolerance_says() {
    use crate::kernels::bessel::ordinary_second::{MAX_TERMS, y0_series_terms, y1_series_terms};

    for (x, y0_want, y1_want) in [
        // The underflow case that used to spin to the cap.
        (1e-308_f64, 1, 3),
        (f64::MIN_POSITIVE, 1, 3),
        (0.01, 5, 5),
        (0.1, 7, 7),
        (0.5, 9, 10),
        (1.0, 11, 12),
        (2.0, 14, 14),
        (Y_SERIES_MAX, 18, 18),
    ] {
        let y0 = y0_series_terms(x);
        assert_eq!(y0.terms, y0_want, "Y₀ at x = {x:e}");
        assert!(y0.terms < MAX_TERMS, "Y₀ at x = {x:e} reached the cap");
        let y1 = y1_series_terms(x);
        assert_eq!(y1.terms, y1_want, "Y₁ at x = {x:e}");
        assert!(y1.terms < MAX_TERMS, "Y₁ at x = {x:e} reached the cap");
    }

    // Monotone in the argument over the band where the series actually sums something, which is
    // what makes the break a convergence test rather than a fixed schedule.
    assert!(y0_series_terms(Y_SERIES_MAX).terms > y0_series_terms(0.01).terms);
    assert!(y1_series_terms(Y_SERIES_MAX).terms > y1_series_terms(0.01).terms);

    // `y1_series`' break carries `&& k > 1`, so it always enters at least two terms however
    // negligible the first is.
    assert!(
        y1_series_terms(f64::MIN_POSITIVE).terms >= 2,
        "the k > 1 guard should force a second term"
    );

    // The counted variants return exactly what the plain ones do.
    for i in 0..=20 {
        let x = 0.2 + f64::from(i) * 0.19;
        assert_eq!(y0_series_terms(x).value, y0_series(x));
        assert_eq!(y1_series_terms(x).value, y1_series(x));
    }
}

/// Stopping early at a vanished term changes the **cost** and not the answer.
///
/// The zero-term exit added to `y0_series` skips iterations that were summing exact zeros, so
/// the value it returns is the same to the bit. Asserted against the pole's leading behaviour,
/// which is the only thing `Y₀` has down there: `Y₀(x) → (2/π)(ln(x/2) + γ)` as `x → 0`.
#[test]
fn the_vanished_term_exit_does_not_move_the_value() {
    use core::f64::consts::PI;

    for &x in &[1e-308_f64, f64::MIN_POSITIVE, 1e-200, 1e-170, 3e-162] {
        let got = y0_series(x);
        // Below the underflow the series contributes exactly nothing, so `Y₀` is its leading
        // term — and that is a closed form, independent of the summation entirely.
        let leading = 2.0 / PI * ((x / 2.0).ln() + crate::constants::f64::EULER_GAMMA);
        assert!(
            (got - leading).abs() <= 1e-14 * leading.abs(),
            "Y₀({x:e}) = {got:e} against its leading term {leading:e}"
        );
        assert!(got.is_finite() && got < 0.0, "Y₀({x:e}) = {got:e}");
    }
    // And it is still monotone increasing towards the origin's pole.
    assert!(y0_series(1e-300) < y0_series(1e-200));
    assert!(y0_series(1e-200) < y0_series(1e-100));
}

/// The Taylor march reproduces a *different* solution of the same equation, at a nonzero order.
///
/// `y_march` only ever asks for order zero, which left `taylor_step`'s `order_sq` term multiplied
/// by zero at every call site — so flipping its sign was invisible and mutation testing said so.
/// `J₁` satisfies Bessel's equation with `ν = 1`, so marching it with `order_sq = 1` and landing
/// on `J₁` again exercises that term and pins its sign.
///
/// It is also a genuinely independent check of the recurrence: `J₁` shares no code with the `Y`
/// series the march is normally seeded from.
#[test]
fn the_march_recurrence_carries_its_order_term() {
    use crate::kernels::bessel::ordinary_second::taylor_step;
    use crate::math::pair::Slope;

    for &x0 in &[2.0_f64, 3.5, 5.0, 8.0] {
        for &delta in &[0.1_f64, 0.25, 0.5] {
            // `J₁′(x) = J₀(x) − J₁(x)/x`.
            let seeded = Slope::new(j1(x0), j0(x0) - j1(x0) / x0);
            let marched = taylor_step(x0, seeded, delta, 1.0);
            let truth = j1(x0 + delta);
            assert!(
                (marched.value - truth).abs() <= 1e-12 * truth.abs().max(1e-3),
                "order-1 march from {x0} by {delta}: {} vs J₁ = {truth}",
                marched.value
            );
        }
    }
    // And order zero still reproduces `J₀`, so the term is carried rather than ignored: the
    // same seed with the wrong `order_sq` misses by far more than the tolerance above.
    let x0 = 5.0_f64;
    let seeded = Slope::new(j1(x0), j0(x0) - j1(x0) / x0);
    let wrong_order = taylor_step(x0, seeded, 0.5, 0.0);
    let right_order = taylor_step(x0, seeded, 0.5, 1.0);
    assert!(
        (wrong_order.value - right_order.value).abs() > 1e-6,
        "the order term must change the answer: {} vs {}",
        wrong_order.value,
        right_order.value
    );
}

/// The Hankel expansion's **term count**, which is all its tolerance exit changes.
///
/// `hankel` itself cannot see it: the growth check stops the divergent expansion at the same
/// `P` and `Q` whatever the tolerance does, so mutation testing finds every spelling of
/// `magnitude < TERM_EPS` unkillable through the public value. `hankel_sums` returns the count,
/// which is why it is a separate function.
#[test]
fn the_hankel_expansion_truncates_sooner_as_the_argument_grows() {
    use crate::kernels::bessel::asymptotic::{MAX_TERMS as ASYMPTOTIC_MAX, hankel_sums};

    // Further out, fewer terms: `a_k/x^k` reaches the floor earlier, so the tolerance exit wins
    // and wins sooner. That monotone relation is what the second exit exists to produce.
    let near = hankel_sums(0.0, 18.0);
    let far = hankel_sums(0.0, 1.0e6);
    assert!(
        far.terms < near.terms,
        "x = 1e6 should truncate sooner than x = 18: {} vs {}",
        far.terms,
        near.terms
    );
    for &x in &[18.0_f64, 50.0, 1.0e3, 1.0e6] {
        let counted = hankel_sums(0.0, x);
        assert!(
            counted.terms < ASYMPTOTIC_MAX,
            "x = {x:e} took {} terms",
            counted.terms
        );
        // `P → 1` and `Q → 0` as the argument grows, which is the expansion's leading behaviour
        // and an independent check that the sums are the right way round.
        assert!(
            (counted.value.first_kind - 1.0).abs() < 1e-2,
            "P({x:e}) = {}",
            counted.value.first_kind
        );
        assert!(counted.value.second_kind.abs() < 1e-2);
    }
}

/// The march's **step count**, which is what makes its cap a backstop rather than a schedule.
///
/// `MARCH_MAX_STEPS` exists because three mutations of the march's advance made the suite *hang*
/// rather than fail — a timeout says nothing about which change caused it, and in a library a
/// non-terminating loop is worse than a wrong number. The cap converts those into bounded wrong
/// answers; this asserts that no real argument comes near it, which is the other half of the
/// bargain.
#[test]
fn the_march_reaches_every_target_well_inside_its_cap() {
    use crate::kernels::bessel::ordinary_second::{MARCH_MAX_STEPS, y_march_steps};

    let mut worst = 0;
    // The whole band the march serves: from the series cutoff to the Hankel handover.
    for i in 0..=280 {
        let target = Y_SERIES_MAX + f64::from(i) * 0.05;
        let marched = y_march_steps(target);
        assert!(
            marched.terms < MARCH_MAX_STEPS,
            "march to {target} took {} steps against a cap of {MARCH_MAX_STEPS}",
            marched.terms
        );
        assert!(marched.value.value.is_finite());
        worst = worst.max(marched.terms);
    }
    // Measured: **seven**. The constant's own estimate is `log(18/4)/log(1.25) ≈ 6.7`, so seven
    // is that rounded up — the final step is short, being clamped to `target - x`. Pinned
    // because a change here is what a regression in `MARCH_RATIO` looks like.
    assert_eq!(worst, 7, "worst-case march length moved to {worst}");
    // Monotone in the target: further to go, more steps.
    assert!(y_march_steps(18.0).terms > y_march_steps(5.0).terms);
}
