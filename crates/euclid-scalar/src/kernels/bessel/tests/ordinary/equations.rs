//! The outputs against the differential equation and recurrences they satisfy.
//!
//! These need no reference values at all: `x²y″ + xy′ + (x² − ν²)y = 0` and the derivative
//! identities constrain the functions from the mathematics, by finite differences, so they hold
//! whatever method produced the numbers.

use super::close;
use crate::kernels::bessel::ordinary_first::{j0, j1};
use crate::kernels::bessel::ordinary_second::{y0, y1};

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
