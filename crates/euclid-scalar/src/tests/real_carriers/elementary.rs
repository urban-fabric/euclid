//! `sinc`, `cos_m1` and `two_product` — the composites that **recover what the naive expression
//! discards**.
//!
//! See the [parent module](super) for why these need values rather than cross-precision budgets,
//! and [`super::reduction`] for the argument-reducing pair this file was split from.
//!
//! All three exist because a subtraction or a division throws away the answer: `sin(x)/x` is
//! `0/0` at the origin, `cos(x) − 1` cancels to nothing near zero, and `a·b` silently drops the
//! low half of its own product. Each replacement keeps what the naive form loses — a series, an
//! identity, an error-free transform — so the test has to visit the argument where the naive
//! form fails. A mid-range sample exercises only the branch that was never in doubt.

use super::{F32_TOL, F64_TOL, agrees};
use crate::tests::*;
use core::fmt::Debug;

/// `sinc(x) = sin(x)/x`, with `sinc(0) = 1`.
///
/// The largest single survivor cluster in `impls/real_f32.rs` at 22 mutants, all in the Taylor
/// branch below `|x| < 1e-2` — the series `1 − x²/6 + x⁴/120 − x⁶/5040`, whose coefficients are
/// `1/3!`, `1/5!` and `1/7!`.
///
/// # Why the branch boundary is the test
///
/// A cross-precision budget cannot see a wrong coefficient here. At `x = 1e-3` the `x⁴/120` term
/// contributes about `8e-15` of a result near `1`, which is below `f32`'s resolution entirely —
/// so dropping it, or getting its sign wrong, changes nothing an `f32` comparison can detect.
/// What *does* detect it is **continuity**: the series and `sin(x)/x` must agree where they meet,
/// and they only do if the coefficients are right. Evaluated just inside and just outside the
/// crossover, where both branches are well-conditioned and the terms still matter.
fn sinc_laws<T>(tol: f64, what: &str)
where
    T: Real<Mask = bool> + Debug,
{
    let s = T::scalar;

    // The removable singularity, exactly.
    assert!(s(0.0).sinc().eq(&T::one()), "{what}: sinc(0) = 1");

    // Away from zero, the definition itself.
    for &x in &[0.1_f64, 0.5, 1.0, 2.0, 3.0, 10.0, -0.5, -2.0] {
        agrees(&s(x).sinc(), x.sin() / x, tol, what, "sinc(x) = sin(x)/x");
    }

    // The zeros: `sinc(nπ) = 0` for every non-zero integer `n`. Not exact, because `π` is not a
    // float, but the value must be of the order of the rounding of `π` rather than of `1`.
    for n in 1..=4 {
        let x = f64::from(n) * core::f64::consts::PI;
        let got = s(x).sinc().abs();
        assert!(
            got.le(&s(tol * 10.0)),
            "{what}: sinc({n}π) should vanish, got {got:?}",
        );
    }

    // Continuity across the crossover at `1e-2`. The two branches are different algorithms, so
    // this is a real cross-check of one against the other — and it is what pins the series
    // coefficients, which no ulp budget against `f64` can reach.
    for &x in &[9.0e-3_f64, 9.9e-3, 1.0e-2, 1.01e-2, 1.1e-2] {
        agrees(
            &s(x).sinc(),
            x.sin() / x,
            tol,
            what,
            "sinc is continuous at its crossover",
        );
    }

    // Even: `sinc(−x) = sinc(x)`, which the series satisfies only if every power is even.
    for &x in &[1e-3_f64, 5e-3, 0.5, 2.0] {
        let plus = s(x).sinc();
        let minus = s(-x).sinc();
        assert!(plus.eq(&minus), "{what}: sinc is even at {x}");
    }

    // Deep in the series, against a reference carrying more terms than the implementation.
    // Computed here in `f64` from the same mathematical series, extended far enough that its
    // own truncation is irrelevant at either precision.
    for &x in &[1e-4_f64, 1e-3, 5e-3] {
        let x2 = x * x;
        let reference = 1.0 - x2 / 6.0 + x2 * x2 / 120.0 - x2 * x2 * x2 / 5040.0
            + x2 * x2 * x2 * x2 / 362_880.0;
        agrees(
            &s(x).sinc(),
            reference,
            tol,
            what,
            "sinc matches its series",
        );
    }
}

/// `two_product(a, b) = (p, err)` with `p + err = a·b` **exactly**.
///
/// An error-free transform: the whole contract is that nothing is lost, so the test is an exact
/// reconstruction rather than a tolerance. Each carrier supplies its own `reconstruct`, and in
/// both cases it is a *different algorithm* from the one under test:
///
/// - **`f32`** reconstructs in `f64`, which holds the product of two 24-bit significands exactly.
/// - **`f64`** reconstructs with Dekker's splitting algorithm, which reaches the same remainder
///   through splitting and six multiplications where the implementation uses one FMA. Checking
///   an FMA against `mul_add(a, b, -p)` would merely restate the implementation.
fn two_product_laws<T>(what: &str, reconstruct: impl Fn(T, T, T, T) -> bool)
where
    T: Real<Mask = bool> + Debug,
{
    let s = T::scalar;

    for &(a, b) in &[
        (3.0_f64, 7.0_f64),
        (0.1, 0.3),
        (1.0 / 3.0, 7.0 / 11.0),
        (1e10, 1e-10),
        (123.456, 789.012),
        (-2.5, 1.0 / 7.0),
    ] {
        let (p, err) = s(a).two_product(s(b));
        assert!(
            reconstruct(s(a), s(b), p.clone(), err.clone()),
            "{what}: two_product({a}, {b}) = ({p:?}, {err:?}) must reconstruct a·b exactly",
        );
        // The remainder is the *small* half by construction.
        assert!(
            err.abs().le(&p.abs()),
            "{what}: the remainder must not exceed the product at ({a}, {b})",
        );
    }

    // An exact product has no remainder at all — powers of two multiply exactly.
    let (p, err) = s(0.5).two_product(s(4.0));
    assert!(
        p.eq(&s(2.0)) && err.eq(&T::zero()),
        "{what}: an exact product has zero remainder"
    );

    // A non-finite product yields a zero remainder rather than a NaN one, as documented.
    let (p, err) = T::infinity().two_product(s(2.0));
    assert!(
        p.eq(&T::infinity()) && err.eq(&T::zero()),
        "{what}: a non-finite product carries no remainder",
    );
}

/// `cos_m1(x) = cos(x) − 1`, computed as `−2 sin²(x/2)` to avoid cancellation near zero.
fn cos_m1_laws<T>(tol: f64, what: &str)
where
    T: Real<Mask = bool> + Debug,
{
    let s = T::scalar;

    assert!(s(0.0).cos_m1().eq(&T::zero()), "{what}: cos_m1(0) = 0");
    // Away from zero, the naive form is fine and serves as the reference.
    for &x in &[0.5_f64, 1.0, 2.0, 3.0, -1.0] {
        agrees(
            &s(x).cos_m1(),
            x.cos() - 1.0,
            tol,
            what,
            "cos_m1(x) = cos(x) − 1",
        );
    }
    // Near zero, `cos(x) − 1` loses everything but `cos_m1` must not: the true value is
    // `−x²/2 + x⁴/24`, which is where the identity earns its place. A relative check, which the
    // naive form would fail outright.
    for &x in &[1e-3_f64, 1e-4, 1e-5] {
        let reference = -x * x / 2.0 + x * x * x * x / 24.0;
        agrees(
            &s(x).cos_m1(),
            reference,
            tol,
            what,
            "cos_m1 is accurate near zero",
        );
    }
    // Even, like the cosine.
    for &x in &[1e-4_f64, 0.5, 2.0] {
        assert!(
            s(x).cos_m1().eq(&s(-x).cos_m1()),
            "{what}: cos_m1 is even at {x}"
        );
    }
}
#[test]
fn the_f64_carrier_computes_the_elementary_composites() {
    sinc_laws::<f64>(F64_TOL, "f64");
    cos_m1_laws::<f64>(F64_TOL, "f64");
    // Dekker's splitting algorithm as the independent reference. `2^27 + 1` is the `f64`
    // splitting constant: it cuts each 53-bit significand into two halves of at most 26 bits,
    // whose pairwise products are therefore exact.
    two_product_laws::<f64>("f64", |a, b, p, err| {
        let split = |x: f64| {
            let c = 134_217_729.0_f64 * x; // (2^27 + 1)·x
            let high = c - (c - x);
            (high, x - high)
        };
        let (a_hi, a_lo) = split(a);
        let (b_hi, b_lo) = split(b);
        let expected_err = a_lo * b_lo - (((p - a_hi * b_hi) - a_lo * b_hi) - a_hi * b_lo);
        p == a * b && err == expected_err
    });
}

#[test]
fn the_f32_carrier_computes_the_elementary_composites() {
    sinc_laws::<f32>(F32_TOL, "f32");
    cos_m1_laws::<f32>(F32_TOL, "f32");
    // For `f32` the reconstruction is done in `f64`, which holds the product of two `f32`
    // significands exactly — an independent check rather than one using the same FMA.
    two_product_laws::<f32>("f32", |a, b, p, err| {
        f64::from(a) * f64::from(b) == f64::from(p) + f64::from(err)
    });
}
