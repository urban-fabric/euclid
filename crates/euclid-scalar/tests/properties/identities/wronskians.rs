//! Wronskian and scaling identities for the Airy and Bessel families.
//!
//! Part of the [`identities`](super) suite, and the sharpest instrument in it.
//!
//! # Why these are worth more than an accuracy band
//!
//! A Wronskian is an identity between a pair of solutions to the same second-order equation,
//! and its value is a **closed form** — `1/π`, `1/x`, `−2/(πx)`. So unlike almost everything
//! else about a special function, it can be checked without a reference value, at any argument,
//! to the precision of the format.
//!
//! That buys two things no pointwise accuracy test does:
//!
//! - **One identity constrains four separately-implemented functions at once.** `Ai`, `Ai′`,
//!   `Bi` and `Bi′` are four different code paths — series, asymptotic, quadrature, each with
//!   its own crossovers — and the Wronskian is false unless all four agree with each other
//!   *and* with the equation they solve. A fixed-point test checks one function at one argument
//!   against one transcribed constant.
//! - **It reaches arguments no reference table does.** The scaled forms satisfy the same
//!   identity (see below), so the check runs out to `1e12` and `1e15` where nothing has ever
//!   been transcribed.
//!
//! # The exponentials cancel, which is what makes the scaled forms usable here
//!
//! The scaled variants are defined with reciprocal factors —
//! `Aiᵉ = e^{ζ}Ai` against `Biᵉ = e^{−ζ}Bi`, and `I₀ᵉ = e^{−|x|}I₀` against `K₀ᵉ = e^{x}K₀` —
//! so every product in a Wronskian pairs one of each and the factor disappears exactly:
//!
//! ```text
//! Aiᵉ·Biᵉ′ − Aiᵉ′·Biᵉ = e^{ζ}Ai·e^{−ζ}Bi′ − e^{ζ}Ai′·e^{−ζ}Bi = Ai·Bi′ − Ai′·Bi = 1/π
//! ```
//!
//! The exponential factors cancel, so the same identity exercises the separate scaled
//! implementation without overflow across the tested range.
//!
//! # Where the tolerances come from
//!
//! Each was measured over 2,000-point sweeps per decade before being written here, and each is
//! a claim about a mechanism rather than a round number:
//!
//! - **`I₀K₁ + I₁K₀ = 1/x` is a sum of positive terms** for `x > 0`, so nothing cancels and the
//!   residual is just the inputs' own error. `bessel_i0e` documents a measured worst case of
//!   `1.7e-14`; the sweep's worst is `1.6e-14`, at `x ≈ 38`. The bound is `1e-13`.
//! - **The Airy Wronskian on the positive axis** holds to `5.1e-15` from `0` to `1e12`, because
//!   the scaled forms never build the exponential.
//! - **On the negative axis it degrades as `ζ·ε`**, with `ζ = ⅔|x|^{3/2}`, which is the error
//!   term `airy_ai`'s own docstring names. At `x = −8.96e4` that predicts `3.9e-9` and the
//!   sweep measures `1.3e-9`. A fixed tolerance here would be either wrong near zero or
//!   vacuous far from it, so the bound carries the `ζ` term explicitly — the same reasoning
//!   `exp_and_ln_round_trip` uses for its conditioning factor.
//! - **`J₀Y₁ − J₁Y₀`** does cancel, and its phase stops being resolvable once `ulp(x)` passes
//!   the spacing of a zero; the trait documents that limit as `|x| ≈ 1e11`. The sweep confirms
//!   it: `5.9e-16` to `1e8`, then `9.9e-12` by `1.2e11`. The property stops at `1e8` and says
//!   so rather than loosening the bound to cover a regime the kernel does not claim.

use super::*;
use euclid_stability::target;

/// `ζ = ⅔|x|^{3/2}`, the Airy phase, and the error term both Airy branches are documented
/// against.
fn airy_zeta(x: f64) -> f64 {
    2.0 / 3.0 * x.abs().powf(1.5)
}

/// Safety factor for the `I`/`K` Wronskian's forward-stability bound; see the property.
const MODIFIED_SAFETY: f64 = 512.0;

/// Safety factor for the `J`/`Y` Wronskian's forward-stability bound; see the property.
const OSCILLATING_SAFETY: f64 = 16.0;

/// Relative residual against `want`, as a fraction of `bound`. `1.0` is exactly at the bound.
///
/// The score to hand a budget where a condition number cannot serve — see the scaled Airy
/// Wronskian, whose exact value is a constant.
fn bound_fraction(got: f64, want: f64, bound: f64) -> f64 {
    if !got.is_finite() || want == 0.0 || bound <= 0.0 {
        return 0.0;
    }
    ((got - want) / want).abs() / bound
}

/// The relative bound for an Airy Wronskian residual at `x`.
///
/// Two terms, matching the two regimes in the kernel's own precision note: a floor for the four
/// kernel evaluations and the difference between their products, plus `ζ·ε` on the negative axis
/// where the phase carries the error. On the positive axis the scaled forms never form `e^{ζ}`,
/// so only the floor applies.
fn airy_bound(x: f64) -> f64 {
    let floor = 64.0 * f64::EPSILON;
    if x < 0.0 {
        floor + airy_zeta(x) * f64::EPSILON
    } else {
        floor
    }
}

#[path = "wronskians/pairs.rs"]
mod pairs;
#[path = "wronskians/scaling.rs"]
mod scaling;
#[path = "wronskians/single_precision.rs"]
mod single_precision;
