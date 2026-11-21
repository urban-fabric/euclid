//! The Bessel Wronskians at single precision.
//!
//! Part of the [`wronskians`](super) suite. The `f32` kernels are native rather than widened, so
//! the identity has to hold in a format whose scaled forms have their own crossovers.

use super::*;

property! {
    // Floors on all four labelled regions, at roughly half the thinnest coverage measured over
    // six seeds — `J/Y quadrature or march` is the scarce one at 4.8–7.0%, the others at 10% or
    // better. Without these the labels only *report* the crossovers; a generator that drifted
    // away from a join would test one branch twice and still pass, which is exactly how the
    // `beta_reg` hole survived a suite that was reaching the branch.
    #![config(settings("f32_native_bessel")
        .cases(600)
        .require("J/Y series", 5.0)
        .require("J/Y quadrature or march", 2.5)
        .require("J/Y Hankel", 60.0)
        .require("K series", 5.0))]

    /// The `f32` Bessel family's Wronskians, now that all four kinds are native.
    ///
    /// These had no generated coverage while `J`, `Y` and `K` narrowed the `f64` kernels — and
    /// could not have had any worth running, because a Wronskian between four narrowed values is
    /// a statement about `f64`'s kernels with two roundings on top. Now each of the four is a
    /// separate single-precision implementation with its own crossovers, and the identity is
    /// false unless all of them agree with each other *and* with the equation they solve.
    ///
    /// Four code paths per kind makes sixteen regime combinations across this range, and the
    /// generated argument is what reaches them: `J`/`Y` switch at `2`, `7` and (for `Y`) `2`,
    /// `K` at `1.5`, so a fixed sweep can straddle a join without ever landing beside it.
    ///
    /// Both identities are checked, because they fail differently. `J₀Y₁ − J₁Y₀ = −2/(πx)`
    /// constrains the oscillating pair, where the shared Hankel phase is the common mode — it
    /// would catch the `J` and `Y` halves of that expansion being swapped, which no value check
    /// on either alone would. `I₀ᵉK₁ᵉ + I₁ᵉK₀ᵉ = 1/x` constrains the monotone pair, and the
    /// scaled forms make it checkable across the whole axis rather than only where `K` is
    /// representable.
    fn the_native_f32_bessel_wronskians_hold(x in generate::float32::in_range(0.05, 60.0)) {
        if !x.is_finite() || x <= 0.0 {
            return Ok(());
        }
        classify!("J/Y series", x < 2.0);
        classify!("J/Y quadrature or march", (2.0..7.0).contains(&x));
        classify!("J/Y Hankel", x >= 7.0);
        classify!("K series", x <= 1.5);

        // The oscillating pair. Compared against `2/(πx)`, the natural scale of the identity,
        // rather than relatively against zero.
        let scale = 2.0 / (core::f32::consts::PI * x);
        let oscillating =
            Real::bessel_j0(x) * Real::bessel_y1(x) - Real::bessel_j1(x) * Real::bessel_y0(x);
        relation::agree(
            "f32: J0 Y1 - J1 Y0 = -2/(pi x)",
            (),
            |()| f64::from(oscillating),
            |()| -f64::from(scale),
            &Absolute(32.0 * f64::from(f32::EPSILON) * f64::from(scale).max(1e-2)),
        )?;

        // The monotone pair, scaled so it holds everywhere rather than only where K survives.
        let monotone = Real::bessel_i0e(x) * Real::bessel_k1e(x)
            + Real::bessel_i1e(x) * Real::bessel_k0e(x);
        relation::agree(
            "f32: I0e K1e + I1e K0e = 1/x",
            (),
            |()| f64::from(monotone * x),
            |()| 1.0,
            &Absolute(64.0 * f64::from(f32::EPSILON)),
        )?;

        // `Y₀′ = −Y₁` and `J₀′ = −J₁`: the march carries the derivative, so a sign or an
        // off-by-one in it shows up here and in nothing else.
        let step = (x * 1e-3).max(1e-4);
        for (name, f, minus_derivative) in [
            (
                "J0' = -J1",
                Real::bessel_j0 as fn(f32) -> f32,
                Real::bessel_j1 as fn(f32) -> f32,
            ),
            (
                "Y0' = -Y1",
                Real::bessel_y0 as fn(f32) -> f32,
                Real::bessel_y1 as fn(f32) -> f32,
            ),
        ] {
            let central = (f(x + step) - f(x - step)) / (2.0 * step);
            let expected = -minus_derivative(x);
            // A central difference carries `O(step²)` truncation plus `ε/step` of rounding, so
            // the bound is the sum of the two rather than the format's resolution.
            let bound = f64::from(step * step) + f64::from(f32::EPSILON / step);
            check!(
                (f64::from(central) - f64::from(expected)).abs() <= 8.0 * bound,
                "f32 {name} broken at x = {x:e}: {central:e} vs {expected:e}"
            );
        }
    }
}
