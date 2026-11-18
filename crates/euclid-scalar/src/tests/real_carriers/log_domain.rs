//! The log-domain composites: `ln_sub_exp`, `ln_add_exp`, `logit`, `xlogy` and their relatives.
//!
//! See the [parent module](super) for why these need values rather than cross-precision budgets.
//!
//! These exist so that a probability can be carried as its logarithm without ever being
//! exponentiated back, which is what keeps a likelihood from underflowing to zero. The
//! consequence for testing is that the interesting arguments are the ones where the naive form
//! *fails* — `ln(e^a − e^b)` where `a` and `b` are close, `ln(1 − p)` where `p` is tiny — so a
//! sample of comfortable mid-range values proves very little.

use super::{F32_TOL, F64_TOL, agrees};
use crate::tests::*;
use core::fmt::Debug;

/// `ln_sub_exp(a, b) = ln(e^a − e^b)`, and its edge behaviour.
///
/// The body is six guards and then one line, and the guards are the part worth testing: the
/// ordering requirement, the two infinities, and the equal-argument case that must give `−∞`
/// rather than `ln 0 = NaN`.
fn ln_sub_exp_laws<T>(tol: f64, what: &str)
where
    T: Real<Mask = bool> + Debug,
{
    let s = T::scalar;

    // The defining identity, against logarithms of exact values: `ln(e^ln u − e^ln v)` is
    // `ln(u − v)`. Sourced from ordinary arithmetic on `u` and `v`, not from the kernel.
    //
    // `u` and `v` are kept well separated on purpose. `(100, 99)` is excluded because its
    // answer is `ln 1 = 0`: the inputs `ln 100` and `ln 99` differ by `0.01`, so at `f32` the
    // subtraction of two values near `4.6` leaves barely two significant digits and the result
    // is `-3.8e-6` rather than zero. That is the conditioning of the question, not a defect —
    // and a tolerance wide enough to admit it would be wide enough to admit anything.
    for &(u, v) in &[(2.0_f64, 1.0_f64), (10.0, 3.0), (1.0, 0.5), (100.0, 50.0)] {
        let got = s(u).ln().ln_sub_exp(s(v).ln());
        agrees(
            &got,
            (u - v).ln(),
            tol,
            what,
            "ln(e^ln u − e^ln v) = ln(u − v)",
        );
    }

    // Inverted by its own sibling: `ln_add_exp(ln_sub_exp(a, b), b) = a`. A round trip, so it
    // needs no reference value at all — and it cross-checks the two composites against each
    // other rather than restating either.
    for &(a, b) in &[(1.0_f64, 0.0_f64), (5.0, 2.0), (0.5, -3.0), (-1.0, -4.0)] {
        let back = s(a).ln_sub_exp(s(b)).ln_add_exp(s(b));
        agrees(&back, a, tol, what, "ln_add_exp(ln_sub_exp(a, b), b) = a");
    }

    // `a < b` has no real logarithm: `e^a − e^b` is negative.
    assert!(
        s(1.0).ln_sub_exp(s(2.0)).is_nan(),
        "{what}: ln_sub_exp below the diagonal must be NaN",
    );
    // Equal arguments give `ln 0`, which is `−∞` and not NaN.
    assert!(
        s(3.0).ln_sub_exp(s(3.0)).eq(&T::neg_infinity()),
        "{what}: ln_sub_exp(a, a) = −∞",
    );
    // `e^{−∞}` is zero, so subtracting it is the identity.
    assert!(
        s(3.0).ln_sub_exp(T::neg_infinity()).eq(&s(3.0)),
        "{what}: ln_sub_exp(a, −∞) = a",
    );
    // `∞ − ∞` is indeterminate.
    assert!(
        T::infinity().ln_sub_exp(T::infinity()).is_nan(),
        "{what}: ln_sub_exp(∞, ∞) must be NaN",
    );
    // But `∞ − e^b` for *finite* `b` is still `∞`, and that distinction is the whole content of
    // the `&&` in the guard above it: widened to `||`, a single infinity on the left would
    // return NaN here instead. Both of these are therefore needed — the pair, not either alone.
    assert!(
        T::infinity().ln_sub_exp(s(5.0)).eq(&T::infinity()),
        "{what}: ln_sub_exp(∞, b) = ∞ for finite b",
    );
    assert!(
        T::infinity()
            .ln_sub_exp(T::neg_infinity())
            .eq(&T::infinity()),
        "{what}: ln_sub_exp(∞, −∞) = ∞",
    );
    // NaN propagates from either side.
    assert!(
        T::nan().ln_sub_exp(s(1.0)).is_nan(),
        "{what}: NaN propagates from the left"
    );
    assert!(
        s(1.0).ln_sub_exp(T::nan()).is_nan(),
        "{what}: NaN propagates from the right"
    );
}

/// `ln_add_exp(a, b) = ln(e^a + e^b)` — the log-sum-exp of two terms.
fn ln_add_exp_laws<T>(tol: f64, what: &str)
where
    T: Real<Mask = bool> + Debug,
{
    let s = T::scalar;

    for &(u, v) in &[(2.0_f64, 1.0_f64), (10.0, 3.0), (0.25, 0.75), (1e-3, 1.0)] {
        let got = s(u).ln().ln_add_exp(s(v).ln());
        agrees(
            &got,
            (u + v).ln(),
            tol,
            what,
            "ln(e^ln u + e^ln v) = ln(u + v)",
        );
    }
    // Commutative, which the `max`/`gap` formulation must preserve.
    for &(a, b) in &[(1.0_f64, 5.0_f64), (-3.0, 2.0), (0.0, 0.0)] {
        let forward = s(a).ln_add_exp(s(b));
        let backward = s(b).ln_add_exp(s(a));
        assert!(
            forward.eq(&backward),
            "{what}: ln_add_exp commutes at ({a}, {b})"
        );
    }
    // The absorbing and dominating identities.
    assert!(
        s(3.0).ln_add_exp(T::neg_infinity()).eq(&s(3.0)),
        "{what}: ln_add_exp(a, −∞) = a",
    );
    assert!(
        T::neg_infinity().ln_add_exp(s(3.0)).eq(&s(3.0)),
        "{what}: ln_add_exp(−∞, a) = a",
    );
    assert!(
        s(3.0).ln_add_exp(T::infinity()).eq(&T::infinity()),
        "{what}: ln_add_exp(a, ∞) = ∞",
    );
    assert!(
        T::nan().ln_add_exp(s(1.0)).is_nan(),
        "{what}: NaN propagates"
    );
    // **NaN against an infinity**, which is where the opening `is_nan() || is_nan()` guard earns
    // its `||`. Weakened to `&&`, a lone NaN falls past it, past both `−∞` tests, and into
    // `self == ∞ || other == ∞` — which fires on the infinity and returns `+∞` for an input
    // that contains a NaN. Every other NaN argument reaches the arithmetic at the bottom and
    // comes back NaN by itself, so this pair is the only thing that distinguishes the guard.
    assert!(
        T::nan().ln_add_exp(T::infinity()).is_nan(),
        "{what}: ln_add_exp(NaN, ∞) must be NaN, not ∞",
    );
    assert!(
        T::infinity().ln_add_exp(T::nan()).is_nan(),
        "{what}: ln_add_exp(∞, NaN) must be NaN, not ∞",
    );
    assert!(
        T::nan().ln_add_exp(T::neg_infinity()).is_nan(),
        "{what}: ln_add_exp(NaN, −∞) must be NaN",
    );

    // Far-apart arguments: the larger must dominate to within the resolution of the format,
    // which is what the `max`-plus-`ln_1p_exp` rearrangement buys over the naive form.
    let dominated = s(100.0).ln_add_exp(s(-100.0));
    agrees(
        &dominated,
        100.0,
        tol,
        what,
        "a dominant term absorbs the other",
    );
}

/// The sigmoid/logit pair, and the log-domain forms around them.
///
/// `logit` and `sigmoid` are inverses by definition, so the round trip needs no reference —
/// and it is a genuinely strong check, since `logit` is `ln(p) − ln(1 − p)` and an error in
/// either term shows up immediately.
fn sigmoid_and_logit_laws<T>(tol: f64, what: &str)
where
    T: Real<Mask = bool> + Debug,
{
    let s = T::scalar;

    // The round trip, over the range where it is well conditioned.
    //
    // `|x| <= 3` is not timidity. `logit` differentiates as `1/(p(1−p))`, so recovering `x` from
    // `σ(x)` amplifies `p`'s rounding by that factor: at `x = 6`, `p` is `0.99752`, `1 − p` is
    // `0.00248`, the amplification is about 400, and `f32` returns `6.0000176`. The information
    // is destroyed when `p` saturates towards one, not by either function. The log-domain
    // `ln_1m_sigmoid` below is the surface that *is* accurate out there, and it is checked there.
    for &x in &[-3.0_f64, -2.0, -0.5, 0.0, 0.5, 2.0, 3.0] {
        let round_trip = s(x).sigmoid().logit();
        agrees(&round_trip, x, tol, what, "logit(sigmoid(x)) = x");
    }
    // The fixed point, which is exact in both directions.
    agrees(&s(0.5).logit(), 0.0, tol, what, "logit(1/2) = 0");
    agrees(&s(0.0).sigmoid(), 0.5, tol, what, "sigmoid(0) = 1/2");
    // `logit` against its closed form at values whose ratio is exact.
    for &p in &[0.25_f64, 0.75, 0.1, 0.9] {
        agrees(
            &s(p).logit(),
            (p / (1.0 - p)).ln(),
            tol,
            what,
            "logit(p) = ln(p/(1−p))",
        );
    }

    // `ln_sigmoid` and `ln_1m_sigmoid` are the two log-domain halves, and they are reflections:
    // `ln(1 − σ(x)) = ln σ(−x)`. An identity between the two bodies, needing no reference.
    for &x in &[-5.0_f64, -1.0, 0.0, 1.0, 5.0] {
        let complement = s(x).ln_1m_sigmoid();
        let reflected = (-s(x)).ln_sigmoid();
        assert!(
            (complement - reflected).abs().le(&T::scalar(tol)),
            "{what}: ln(1 − σ(x)) = ln σ(−x) at {x}",
        );
    }
    // They must also sum the way probabilities do: `σ(x) + (1 − σ(x)) = 1`.
    for &x in &[-5.0_f64, -1.0, 0.0, 1.0, 5.0] {
        let total = s(x).ln_sigmoid().exp() + s(x).ln_1m_sigmoid().exp();
        agrees(&total, 1.0, tol, what, "σ(x) + (1 − σ(x)) = 1");
    }
}

/// `ln_1m`, `ln_1m_exp`, `ln_1p_exp` and `ln_exp_m1` — the four one-argument log-domain forms.
fn log_domain_elementary_laws<T>(tol: f64, what: &str)
where
    T: Real<Mask = bool> + Debug,
{
    let s = T::scalar;

    // `ln_1m(x) = ln(1 − x)`, accurate for small `x` where `1 − x` would cancel.
    for &x in &[0.5_f64, 0.25, 1e-3, 1e-6] {
        agrees(
            &s(x).ln_1m(),
            (1.0 - x).ln(),
            tol,
            what,
            "ln_1m(x) = ln(1 − x)",
        );
    }
    // `ln_1m_exp(x) = ln(1 − e^x)` for `x < 0`.
    for &x in &[-0.5_f64, -2.0, -1e-3, -10.0] {
        agrees(
            &s(x).ln_1m_exp(),
            (1.0 - x.exp()).ln(),
            tol,
            what,
            "ln_1m_exp(x) = ln(1 − e^x)",
        );
    }
    // `ln_1p_exp(x) = ln(1 + e^x)`, the softplus.
    for &x in &[-5.0_f64, -1.0, 0.0, 1.0, 5.0] {
        agrees(
            &s(x).ln_1p_exp(),
            (1.0 + x.exp()).ln(),
            tol,
            what,
            "ln_1p_exp(x) = ln(1 + e^x)",
        );
    }
    // The softplus must not overflow where the naive form would: `ln(1 + e^100)` is 100, but
    // `e^100` alone is `2.7e43` and in `f32` it is already `+∞`. This is the branch the `x > 0`
    // guard exists for, and the assertion that catches it being removed.
    agrees(
        &s(100.0).ln_1p_exp(),
        100.0,
        tol,
        what,
        "ln_1p_exp is stable at large x",
    );
    agrees(
        &s(50.0).ln_1p_exp(),
        50.0,
        tol,
        what,
        "ln_1p_exp is stable at moderate x",
    );

    // `ln_exp_m1` inverts `ln_1p_exp`: both are bijections on their domains, so the round trip
    // pins the pair without a reference value.
    for &x in &[-2.0_f64, 0.5, 1.0, 5.0] {
        let round_trip = s(x).ln_1p_exp().ln_exp_m1();
        agrees(&round_trip, x, tol, what, "ln_exp_m1(ln_1p_exp(x)) = x");
    }
}

/// `xlogy`, `xlog1py` and `xlog1my` — the absorbing-zero products.
///
/// These exist for one reason: `0 · ln 0` must be `0` and not `NaN`, because an entropy or a
/// log-likelihood sums over categories of which some have zero probability. The zero cases are
/// therefore the whole point, and they are *exact*.
fn absorbing_zero_laws<T>(tol: f64, what: &str)
where
    T: Real<Mask = bool> + Debug,
{
    let s = T::scalar;

    // The absorbing zeros, asserted exactly. `0 · (−∞)` would be NaN without the guard.
    assert!(
        s(0.0).xlogy(s(0.0)).eq(&T::zero()),
        "{what}: xlogy(0, 0) = 0, not NaN",
    );
    assert!(
        s(0.0).xlogy(s(5.0)).eq(&T::zero()),
        "{what}: xlogy(0, y) = 0"
    );
    assert!(
        s(0.0).xlog1py(s(-1.0)).eq(&T::zero()),
        "{what}: xlog1py(0, −1) = 0, not NaN",
    );
    assert!(
        s(0.0).xlog1my(s(1.0)).eq(&T::zero()),
        "{what}: xlog1my(0, 1) = 0, not NaN",
    );

    // Away from zero they are ordinary products, so a closed form is available.
    for &(x, y) in &[(2.0_f64, 3.0_f64), (0.5, 10.0), (-1.0, 2.0)] {
        agrees(
            &s(x).xlogy(s(y)),
            x * y.ln(),
            tol,
            what,
            "xlogy(x, y) = x·ln y",
        );
        agrees(
            &s(x).xlog1py(s(y)),
            x * y.ln_1p(),
            tol,
            what,
            "xlog1py(x, y) = x·ln(1 + y)",
        );
    }
    // `xlog1my` needs `y < 1`, since `ln(1 − y)` is otherwise not real — the `y > 1` values
    // above give NaN on both sides, which is agreement but not evidence.
    //
    // It also needs `y` to stay clear of `1`. At `y = 0.999` the comparison stops being about
    // the kernel: `0.999` is not an `f32`, and the rounding of the *input* perturbs `1 − y` by
    // six parts in `10^5`, which is four times the gap this tolerance allows. The reference
    // would have to be evaluated at the narrowed input to say anything — which is exactly what
    // `precision_agrees` does, and why the near-boundary behaviour is checked there instead.
    for &(x, y) in &[(2.0_f64, 0.25_f64), (0.5, 0.75), (-1.0, -3.0), (3.0, 0.9)] {
        agrees(
            &s(x).xlog1my(s(y)),
            x * (-y).ln_1p(),
            tol,
            what,
            "xlog1my(x, y) = x·ln(1 − y)",
        );
    }
    // And it is `xlog1py(x, −y)` by definition, which is a pure identity between two bodies.
    for &(x, y) in &[(2.0_f64, 0.25_f64), (0.5, 0.75), (-1.0, -3.0)] {
        let direct = s(x).xlog1my(s(y));
        let negated = s(x).xlog1py(-s(y));
        assert!(
            direct.eq(&negated),
            "{what}: xlog1my(x, y) = xlog1py(x, −y) at ({x}, {y})"
        );
    }
}

#[test]
fn the_f64_carrier_computes_the_log_domain_composites() {
    ln_sub_exp_laws::<f64>(F64_TOL, "f64");
    ln_add_exp_laws::<f64>(F64_TOL, "f64");
    sigmoid_and_logit_laws::<f64>(F64_TOL, "f64");
    log_domain_elementary_laws::<f64>(F64_TOL, "f64");
    absorbing_zero_laws::<f64>(F64_TOL, "f64");
}

#[test]
fn the_f32_carrier_computes_the_log_domain_composites() {
    ln_sub_exp_laws::<f32>(F32_TOL, "f32");
    ln_add_exp_laws::<f32>(F32_TOL, "f32");
    sigmoid_and_logit_laws::<f32>(F32_TOL, "f32");
    log_domain_elementary_laws::<f32>(F32_TOL, "f32");
    absorbing_zero_laws::<f32>(F32_TOL, "f32");
}
