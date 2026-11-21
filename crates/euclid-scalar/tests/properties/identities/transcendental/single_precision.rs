//! The same classical identities at single precision.
//!
//! Part of the [`transcendental`](super) suite. Both kernels are native in `f32`, with their own
//! rederived term counts — the eta accelerator drops from 24 terms to 10 — so the laws have to
//! hold against those rather than against the `f64` schedule.

use super::*;

property! {
    // Floors on all four labelled regions, at roughly half the thinnest coverage measured over
    // six seeds. `trig series` is the scarce one at 12.8–15.8%; the `Ei` split is near even.
    // These are the two crossovers this property exists to straddle — `TRIG_SERIES_MAX` at 4 and
    // `EI_ASYMPTOTIC_MIN` at 20 — so a run that stopped reaching either side should fail rather
    // than report a clean pass over half the routes.
    #![config(settings("f32_native_classical")
        .cases(600)
        .require("trig series", 6.0)
        .require("trig fraction", 60.0)
        .require("Ei series", 20.0)
        .require("Ei asymptotic", 40.0))]

    /// The native `f32` integrals and Lambert `W`, through identities rather than values.
    ///
    /// `Si`, `Ci`, `E₁`, `Ei` and `W` all became native in this tree, and none had generated
    /// coverage before: while they narrowed, every identity they satisfied was `f64`'s identity
    /// one rounding away. These are the laws that reach their branch structure — `Si`/`Ci` switch
    /// series to continued fraction at `4`, `E₁` at `1`, `Ei` series to asymptotic at `20`, and
    /// `W` between three seeds at `−1/e + 0.01` and `1.5`.
    ///
    /// Each law is chosen to fail in a way a value check would not:
    ///
    /// - **`W(x)·e^{W(x)} = x`** is the defining equation, so it holds at every argument with no
    ///   reference value, and it is the only check that would catch a seed landing in the wrong
    ///   basin — the result would still be *a* solution of a nearby problem, just not this one.
    /// - **`Ei(x) = −E₁(−x)`** ties the two exponential integrals together across the origin,
    ///   which is the one place their four routes meet.
    /// - **`Si` odd, `Ci` real only for `x > 0`**, and both bounded: shape constraints the
    ///   continued fraction would violate first if its Lentz recurrence stalled.
    fn the_native_f32_classical_identities_hold(x in generate::float32::in_range(0.01, 60.0)) {
        if !x.is_finite() || x <= 0.0 {
            return Ok(());
        }
        classify!("trig series", x < 4.0);
        classify!("trig fraction", x >= 4.0);
        classify!("Ei series", x < 20.0);
        classify!("Ei asymptotic", x >= 20.0);

        // W's defining equation. Relative, because W spans decades over this range.
        let w = Real::lambert_w(x);
        check!(w.is_finite() && w > 0.0, "W({x:e}) = {w:e} off the branch");
        let back = w * w.exp();
        relation::agree(
            "f32: W(x) e^W(x) = x",
            (),
            |()| f64::from(back),
            |()| f64::from(x),
            &Relative(64.0 * f64::from(f32::EPSILON)),
        )?;

        // Ei(-x) = -E1(x), across the origin where all four routes meet.
        let reflected = Real::exp_int_ei(-x);
        let negated = -Real::exp_int_e1(x);
        if reflected.is_finite() && negated.is_finite() && negated != 0.0 {
            relation::agree(
                "f32: Ei(-x) = -E1(x)",
                (),
                |()| f64::from(reflected),
                |()| f64::from(negated),
                &Relative(64.0 * f64::from(f32::EPSILON)),
            )?;
        }

        // Si is odd, and bounded by its limit plus the first lobe.
        let si = Real::sin_int(x);
        check!(
            Real::sin_int(-x) == -si,
            "f32 Si must be odd: Si({x:e}) = {si:e}, Si(-x) = {:e}",
            Real::sin_int(-x)
        );
        check!(
            si.abs() <= 1.86,
            "f32 Si({x:e}) = {si:e} exceeds its global bound"
        );
        // Ci is real for positive argument and bounded above by its maximum.
        let ci = Real::cos_int(x);
        check!(
            ci.is_finite() && ci <= 0.473,
            "f32 Ci({x:e}) = {ci:e} is not a bounded real"
        );
        // E1 is positive and strictly decreasing.
        let e1 = Real::exp_int_e1(x);
        let further = Real::exp_int_e1(x * 1.5);
        check!(
            e1 > 0.0 && further <= e1,
            "f32 E1 must be positive and decreasing: E1({x:e}) = {e1:e}, E1(1.5x) = {further:e}"
        );
    }
}
