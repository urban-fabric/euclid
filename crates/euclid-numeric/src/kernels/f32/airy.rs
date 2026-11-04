//! The `f32` peer of the Airy family: `Ai`, `Bi`, their derivatives, and the scaled forms.
//!
//! Evaluated in `f64` and narrowed once; see the [module header](crate::kernels::f32). Airy is
//! the strongest case in the crate for narrowing rather than porting, because it reaches its
//! values through *all three* of the mechanisms whose error floor is set by the working
//! precision — a cancelling Maclaurin series, an outward march of the equation, and a divergent
//! asymptotic expansion. In `f32` the series' `ε·e^{ζ}` cancellation alone would leave `Ai(6)`
//! with no correct digits; narrowing gives a correctly-rounded `f32` instead.

use crate::kernels::airy;

/// The Airy function of the first kind `Ai(x)`.
pub(crate) fn ai(x: f32) -> f32 {
    airy::ai(f64::from(x)) as f32
}

/// The derivative `Ai′(x)`.
pub(crate) fn ai_prime(x: f32) -> f32 {
    airy::ai_prime(f64::from(x)) as f32
}

/// The Airy function of the second kind `Bi(x)`.
pub(crate) fn bi(x: f32) -> f32 {
    airy::bi(f64::from(x)) as f32
}

/// The derivative `Bi′(x)`.
pub(crate) fn bi_prime(x: f32) -> f32 {
    airy::bi_prime(f64::from(x)) as f32
}

/// The scaled `Aiᵉ(x) = e^{ζ}Ai(x)` for `x > 0`, and `Ai(x)` itself for `x ≤ 0`.
pub(crate) fn aie(x: f32) -> f32 {
    airy::aie(f64::from(x)) as f32
}

/// The scaled `Aiᵉ′(x) = e^{ζ}Ai′(x)` for `x > 0`, and `Ai′(x)` itself for `x ≤ 0`.
pub(crate) fn aie_prime(x: f32) -> f32 {
    airy::aie_prime(f64::from(x)) as f32
}

/// The scaled `Biᵉ(x) = e^{−ζ}Bi(x)` for `x > 0`, and `Bi(x)` itself for `x ≤ 0`.
pub(crate) fn bie(x: f32) -> f32 {
    airy::bie(f64::from(x)) as f32
}

/// The scaled `Biᵉ′(x) = e^{−ζ}Bi′(x)` for `x > 0`, and `Bi′(x)` itself for `x ≤ 0`.
pub(crate) fn bie_prime(x: f32) -> f32 {
    airy::bie_prime(f64::from(x)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use euclid_utils::assert_close_f32;

    /// `true` when `want` is a magnitude `f32` can actually hold.
    ///
    /// The unscaled pair leaves `f32`'s range around `|x| = 26` — four times earlier than
    /// `f64`'s `104` — so beyond that the narrowed value is correctly `inf` or `0` and there is
    /// nothing to compare. Skipping those points keeps this test about *agreement*; the range
    /// behavior itself is asserted below.
    fn representable(want: f64) -> bool {
        want == 0.0 || (want.abs() <= f64::from(f32::MAX) && want.abs() >= f64::from(f32::MIN))
    }

    #[test]
    fn the_narrowed_values_track_the_f64_reference() {
        // One ulp of `f32` is the whole budget: narrowing is a single rounding of a value the
        // `f64` path already has to ~1e-15, so anything larger means a branch disagrees.
        for i in -60..60 {
            let x = (i as f32) * 0.5;
            let wide = f64::from(x);
            for (name, got, want) in [
                ("Ai", ai(x), airy::ai(wide)),
                ("Ai'", ai_prime(x), airy::ai_prime(wide)),
                ("Bi", bi(x), airy::bi(wide)),
                ("Bi'", bi_prime(x), airy::bi_prime(wide)),
                ("Aie", aie(x), airy::aie(wide)),
                ("Aie'", aie_prime(x), airy::aie_prime(wide)),
                ("Bie", bie(x), airy::bie(wide)),
                ("Bie'", bie_prime(x), airy::bie_prime(wide)),
            ] {
                if representable(want) {
                    assert_close_f32(got, want, 1.0, name);
                }
            }
        }
    }

    #[test]
    fn the_scaled_forms_stay_representable_where_the_unscaled_ones_cannot() {
        // The complement of the skip above: across the whole band where `Bi` has left `f32`'s
        // range, the scaled pair is still an ordinary number on both carriers.
        //
        // `Bi` overflows just past x = 26.5; `Ai` does *not* reach an exact zero until about
        // 28.4, spending the interval between as an f32 subnormal. Two different boundaries,
        // so only the `Bi` one is asserted here.
        for i in 54..200 {
            let x = (i as f32) * 0.5;
            assert!(
                bi(x).is_infinite(),
                "Bi should have overflowed f32 by x = {x}"
            );
            assert_close_f32(aie(x), airy::aie(f64::from(x)), 1.0, "Aie past the range");
            assert_close_f32(bie(x), airy::bie(f64::from(x)), 1.0, "Bie past the range");
        }
    }

    #[test]
    fn the_scaled_forms_carry_the_range_f32_cannot() {
        // `f32` overflows at 3.4e38, which `Bi` reaches by x = 36 -- far earlier than `f64`'s
        // x = 104. That makes the scaled forms more necessary here, not less.
        assert!(bi(40.0).is_infinite(), "Bi(40) should overflow in f32");
        assert_eq!(ai(40.0), 0.0, "Ai(40) should underflow in f32");
        for &x in &[40.0f32, 100.0, 1e4, 1e10, 1e30] {
            assert!(
                bie(x).is_finite() && bie(x) > 0.0,
                "Bie({x}) left the range"
            );
            assert!(
                aie(x).is_finite() && aie(x) > 0.0,
                "Aie({x}) left the range"
            );
        }
    }
}
