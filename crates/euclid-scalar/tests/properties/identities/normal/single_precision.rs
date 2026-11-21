//! The single-precision inverse complementary error function.
//!
//! Part of the [`normal`](super) suite. Native here rather than widened, so the subnormal
//! halving that the `f64` route gets for free has to be done in the log domain.

use super::*;

property! {
    #![config(settings("f32_native_erfc_inv").cases(500))]

    /// The native `f32` `erfc_inv`, through the round trip and the branch it now owns.
    ///
    /// This had no generated property while it widened to `f64`: the round trip was then a
    /// statement about the `f64` kernel, and the only `f32` content was one rounding. It is
    /// native now, and the thing worth a law is the branch — `HalvingCoordinate` splits on
    /// whether `y/2` is exact, which is a property of the *representation* rather than of the
    /// magnitude, so no threshold constant marks it and no fixed test case lands on it reliably.
    /// A generated `y` ranging over the subnormals does.
    ///
    /// The bound is the conditioning of the round trip, not a flat tolerance. Going out and back
    /// amplifies by `|x·erfc'(x)/erfc(x)|`, which grows with `x` — about `8.9` at `x = 2` and
    /// past `20` by `x = 10` — so a budget that does not scale with it either passes everything
    /// in the centre or fails everything in the tail. `erfc` carries its own `7` ulp on top; see
    /// `kernels::erf::f32`'s `ERFCX_CF_MIN`.
    fn the_native_f32_erfc_inverse_round_trips(y in generate::float::magnitude(-44, -1)) {
        // Drawn by decade in `f64` and narrowed, because `generate::float32` has no `magnitude`
        // and the point is to reach the `f32` subnormals, which start at `1.18e-38` — the whole
        // band where `HalvingCoordinate::Logarithmic` is the branch taken.
        let y = (y as f32).abs();
        if !(y > 0.0 && y < 2.0) {
            return Ok(());
        }
        let half = 0.5f32 * y;
        classify!("halving is exact", half + half == y);
        classify!("halving rounds", half + half != y);
        classify!("deep tail", y < 1e-30);

        let x = Real::erfc_inv(y);
        if !x.is_finite() {
            return Ok(());
        }
        // erfc is decreasing, so the inverse must be too: a monotonicity check that costs
        // nothing and catches a branch returning a value from the wrong coordinate.
        let lower = Real::erfc_inv(y * 0.5f32);
        if lower.is_finite() {
            check!(
                lower >= x,
                "f32 erfc_inv must decrease in y: erfc_inv({y:e}) = {x:e} but \
                 erfc_inv({:e}) = {lower:e}",
                y * 0.5f32
            );
        }

        let back = Real::erfc(x);
        let condition = 1.0 + f64::from(x * x);
        relation::agree(
            "f32: erfc(erfc_inv(y)) = y",
            (),
            |()| f64::from(back),
            |()| f64::from(y),
            &Relative(64.0 * f64::from(f32::EPSILON) * condition),
        )?;
    }
}
