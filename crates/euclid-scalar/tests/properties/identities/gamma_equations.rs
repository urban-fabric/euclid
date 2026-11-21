//! The gamma family's functional equations: reflection, duplication and recurrence.
//!
//! Part of the [`identities`](super) suite.
//!
//! These are the strongest laws the gamma family has, and they are the right instrument for the
//! same reason the Airy Wronskians were: each one constrains the function at **two or three
//! different arguments at once**, so a wrong branch, a wrong coefficient or a misplaced crossover
//! has nowhere to hide. A systematic error in the Lanczos coefficients does not survive
//! duplication.
//!
//! # What these add over the fixed tests
//!
//! `src/tests/relationships/functional_equations.rs` already asserts four of these, and the
//! generated versions are not duplicates of it. The fixed tests sample *round arguments in narrow
//! ranges* — `digamma` reflection at twenty points in `(0, 1)` at multiples of `0.05`, duplication
//! only at quarter-integers below twenty — which is precisely the gap that let `ulp`'s direction
//! bug survive a thousand-decade sweep: the law was right and the arguments could not reach the
//! failure. What is new here:
//!
//! - **`Γ(x)Γ(1−x) = π/sin(πx)` on the negative axis**, which nothing tested. This is the law
//!   that pins `Γ` between its poles, where the measured spread is `387` ulp
//!   across the pole structure. The reference compensates for rounding of the angle before
//!   applying the sine addition identity.
//! - **`polygamma` at orders above one.** The fixed suite covers `digamma` and `trigamma`; the
//!   general recurrence covers every order, and `polygamma.rs` plus its f32 peer held 24
//!   survivors.
//! - **The f32 peers**, where 43 of the family's survivors live and no functional equation had
//!   ever been applied.
//! - **`rgamma`**, including the claim its docstring makes and nothing checked: `1/Γ(175)` is a
//!   representable `1.56e-316` while `Γ(175).recip()` is `0`, because the intermediate overflows.
//!
//! # Four tolerance traps, all found by measuring first
//!
//! - **`π·cot(πx)` is zero at `x = ½`**, so `digamma`'s reflection has a reference that crosses
//!   zero and a relative comparison there reports an error of `1.0` against a correct kernel.
//!   This is the **fourth** suite in a row to hit a zero-crossing reference; it is now the first
//!   thing to check when a law involves a trigonometric factor.
//! - **The `polygamma` recurrence's reference cancels, and badly.** `ψ⁽ⁿ⁾(x+1) = ψ⁽ⁿ⁾(x) +
//!   (−1)ⁿn!/x^{n+1}` is exact, but at `x = 0.05` and `n = 5` the correction term is `7.7e9`
//!   while the result is `O(1)`: two huge quantities cancelling to a small one. Measured relative
//!   error climbs `6.5e-14, 1.8e-14, 8.5e-12, 4.4e-12, 6.6e-9, 2.2e-7` for orders zero to five —
//!   that is the *reference* losing digits, not the kernel, so the bound carries the
//!   amplification explicitly.
//! - **The recurrence degrades near a pole.** `Γ(x+1) = xΓ(x)` is `2.3e-10` at `x = −15 + 5e-7`,
//!   which is `ε` divided by the distance to the pole. The bound carries that distance.
//! - **f32 cannot support the high-order recurrence at all**: orders zero to three reach `15%`
//!   relative. The f32 property keeps the orders and the range where the cancellation is
//!   survivable and says so, rather than loosening a bound until it passes.

use super::*;

/// Distance from `x` to the nearest integer — where `Γ`'s poles and the reflection's zeros are.
fn pole_distance(x: f64) -> f64 {
    (x - x.round()).abs()
}

/// Recover the low part of πx and apply the sine addition identity.
fn reflection_sine_reference(x: f64) -> f64 {
    const PI_LOW: f64 = 1.224_646_799_147_353_2e-16;
    let pi = core::f64::consts::PI;
    let angle = pi * x;
    let residual = x.mul_add(pi, -angle) + x * PI_LOW;
    let (sin, cos) = angle.sin_cos();
    sin * residual.cos() + cos * residual.sin()
}

#[path = "gamma_equations/incomplete_gamma.rs"]
mod incomplete_gamma;
#[path = "gamma_equations/polygamma.rs"]
mod polygamma;
#[path = "gamma_equations/reflection.rs"]
mod reflection;
#[path = "gamma_equations/rising_factorial.rs"]
mod rising_factorial;
#[path = "gamma_equations/single_precision.rs"]
mod single_precision;
