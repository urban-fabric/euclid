//! **Does the Airy family satisfy the relations it must?**
//!
//! Written entirely against the trait surface, so none of it can see which of the six routes
//! produced a value — which is the point. Every test here constrains all four functions at
//! once, and the implementation shares no code between the branches they land in.
//!
//! # Why these particular relations
//!
//! The Airy functions are unusually well served by identities, because two of them are *exact*
//! and hold at every argument:
//!
//! - **The Wronskian** `Ai·Bi′ − Ai′·Bi = 1/π`. This is the strongest single check available.
//!   It is a constant — no reference table, no truncation, nothing transcribed — and it ties
//!   all four functions together, so no one of them can be wrong on its own and pass.
//! - **The equation** `y″ = xy` itself, which both solutions satisfy. Checked through the
//!   derivative of `Ai′` rather than by finite-differencing `Ai` twice, since the trait hands
//!   out the first derivative directly.
//!
//! A relation that is exact and parameter-free is worth more than any number of sampled values,
//! and between them these two pin the whole family across every branch.

use crate::Real;
use euclid_utils::{assert_close, derivative_with_step, rel, sweep};

#[test]
fn the_wronskian_is_one_over_pi_everywhere() {
    // Ai·Bi' - Ai'·Bi = 1/pi, exactly, at every real argument. The sweep is deliberately dense
    // and crosses every crossover in the module: the negative series/march join at -2, the
    // march/Hankel join at -9, and the two positive joins at 1 and 9.
    let target = core::f64::consts::FRAC_1_PI;
    let mut worst = 0.0f64;
    for x in sweep(-30.0, 30.0, 4000) {
        let w =
            Real::airy_ai(x) * Real::airy_bi_prime(x) - Real::airy_ai_prime(x) * Real::airy_bi(x);
        // Unscaled Bi has overflowed by x = 104 but is fine across this sweep; above x ~ 50 the
        // `ζ·ε` term documented on the trait enters, hence the tolerance rather than exactness.
        let tol = if x > 20.0 { 1e-12 } else { 1e-14 };
        assert_close(w, target, tol, &format!("Wronskian at x = {x}"));
        worst = worst.max(rel(w, target));
    }
    // Measured worst across the sweep: ~4e-15, dominated by the far positive end.
    assert!(worst < 1e-12, "Wronskian worst relative error {worst:e}");
}

#[test]
fn the_wronskian_holds_in_the_scaled_forms_too() {
    // The scaled Wronskian is Aie·Bie' - Aie'·Bie = 1/pi as well, because the e^{+zeta} on one
    // factor cancels the e^{-zeta} on the other. That makes this the *only* form of the check
    // available past x = 104, where the unscaled values have left f64's range entirely -- and
    // it reaches as far as the arguments go.
    let target = core::f64::consts::FRAC_1_PI;
    for x in [
        0.5, 1.0, 5.0, 9.0, 20.0, 50.0, 104.0, 200.0, 1e4, 1e8, 1e9, 1e10, 1e12, 1e30, 1e100,
    ] {
        let w = Real::airy_ai_e(x) * Real::airy_bi_e_prime(x)
            - Real::airy_ai_e_prime(x) * Real::airy_bi_e(x);
        assert_close(w, target, 1e-14, &format!("scaled Wronskian at x = {x:e}"));
    }
}

#[test]
fn both_solutions_satisfy_the_airy_equation() {
    // y'' = x·y, reached as d/dx[y'] = x·y. Differentiating numerically costs most of the
    // precision -- the five-point rule is good to ~1e-8 relative at this step -- so this is a
    // structural check rather than a sharp one. It is still worth having: it is the *defining*
    // property, and it would catch a derivative that was accurate but belonged to the wrong
    // function, which the Wronskian alone could miss by compensating between the two.
    for x in sweep(-8.0, 8.0, 160) {
        let step = 1e-3;
        for (name, value, prime) in [
            (
                "Ai",
                Real::airy_ai(x),
                Real::airy_ai_prime as fn(f64) -> f64,
            ),
            (
                "Bi",
                Real::airy_bi(x),
                Real::airy_bi_prime as fn(f64) -> f64,
            ),
        ] {
            let second = derivative_with_step(prime, x, step);
            assert_close(
                second,
                x * value,
                2e-8,
                &format!("{name}'' = x·{name} at x = {x}"),
            );
        }
    }
}

#[test]
fn the_derivatives_are_the_derivatives() {
    // The trait's Ai' must be the derivative of its Ai, not merely a plausible companion
    // function. Checked against a numerical derivative of Ai itself, which is independent of
    // however Ai' is computed -- and on the positive axis the two take *different routes*
    // (orders 1/3 and 2/3 of K), so agreement is a real constraint.
    for x in sweep(-8.0, 6.0, 140) {
        let step = 1e-3;
        for (name, f, prime) in [
            (
                "Ai",
                Real::airy_ai as fn(f64) -> f64,
                Real::airy_ai_prime(x),
            ),
            (
                "Bi",
                Real::airy_bi as fn(f64) -> f64,
                Real::airy_bi_prime(x),
            ),
        ] {
            let numeric = derivative_with_step(f, x, step);
            assert_close(numeric, prime, 2e-9, &format!("d{name}/dx at x = {x}"));
        }
    }
}

#[test]
fn the_scaled_and_unscaled_forms_differ_by_exactly_the_exponential() {
    // Aie = e^{zeta}·Ai and Bie = e^{-zeta}·Bi for x > 0, with zeta = (2/3)x^{3/2}. Checked only
    // where the unscaled value is still in range; the point of the scaled form is that it keeps
    // going after that, which the scaled Wronskian test covers.
    for x in sweep(0.05, 40.0, 500) {
        let zeta = (2.0 / 3.0) * x * x.sqrt();
        let growth = zeta.exp();
        // The reconstruction costs `ζ·ε`, the same term the trait documents on the unscaled
        // functions; at x = 40 that is 5.6e-14.
        let tol = 4.0 * zeta * f64::EPSILON + 1e-15;
        assert_close(
            Real::airy_ai_e(x) / growth,
            Real::airy_ai(x),
            tol,
            &format!("Aie/e^zeta = Ai at x = {x}"),
        );
        assert_close(
            Real::airy_bi_e(x) * growth,
            Real::airy_bi(x),
            tol,
            &format!("Bie·e^zeta = Bi at x = {x}"),
        );
    }
    // On the negative axis the two are defined to be the same function -- bit for bit, since
    // nothing is multiplied at all. That convention is worth pinning exactly.
    for x in sweep(-30.0, 0.0, 400) {
        assert_eq!(Real::airy_ai_e(x), Real::airy_ai(x), "Aie = Ai for x <= 0");
        assert_eq!(Real::airy_bi_e(x), Real::airy_bi(x), "Bie = Bi for x <= 0");
        assert_eq!(Real::airy_ai_e_prime(x), Real::airy_ai_prime(x));
        assert_eq!(Real::airy_bi_e_prime(x), Real::airy_bi_prime(x));
    }
}

#[test]
fn the_product_ai_bi_approaches_its_asymptote() {
    // Ai(x)Bi(x) -> 1/(2·pi·sqrt(x)) as x -> infinity, because the exponentials cancel in the
    // product. An independent check on the *envelope* of both functions at once, and one that
    // survives into the range where neither factor is representable on its own -- the product
    // of the scaled forms equals the product of the unscaled ones, exactly as above.
    for x in [10.0, 30.0, 100.0, 1e3, 1e5, 1e8, 1e10, 1e14] {
        let product = Real::airy_ai_e(x) * Real::airy_bi_e(x);
        let asymptote = 1.0 / (2.0 * core::f64::consts::PI * x.sqrt());
        // The leading correction to the product is O(1/zeta^2) = O(x^{-3}), so the tolerance
        // has to loosen at the small end: at x = 10 the true gap is ~1e-4.
        let tol = 20.0 / (x * x * x) + 1e-15;
        assert_close(
            product,
            asymptote,
            tol,
            &format!("Ai·Bi -> 1/(2pi sqrt x) at x = {x:e}"),
        );
    }
}
