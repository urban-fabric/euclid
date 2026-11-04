//! Floating-point representation primitives: mantissa/exponent decomposition, the
//! representable-neighbour walk, and the spacing between neighbours.
//!
//! Rust's standard library does not expose `frexp`/`ldexp` (they were removed before 1.0) and
//! stabilized `next_up`/`next_down` only recently, so these are implemented here over
//! `to_bits`/`from_bits`. That is entirely safe code — no transmute and no `unsafe` — which
//! matters because the workspace denies `unsafe_code`.
//!
//! The arithmetic is deliberately exact: every operation here either reinterprets bits or
//! multiplies by a power of two, so nothing introduces a rounding error of its own.

/// Bit layout of an `f64`.
const F64_EXP_BITS: u64 = 0x7ff0_0000_0000_0000;
const F64_EXP_SHIFT: u32 = 52;
const F64_EXP_MASK: u64 = 0x7ff;
/// Exponent field value that puts the mantissa in `[0.5, 1)` — the `frexp` convention.
const F64_HALF_BIAS: u64 = 1022;
/// Scaling applied to bring a subnormal into the normal range before decomposing.
const F64_SUBNORMAL_SHIFT: i32 = 54;

/// Decompose `x` into `(mantissa, exponent)` with `x == mantissa · 2^exponent` and
/// `|mantissa| ∈ [0.5, 1)`.
///
/// The exponent is returned as an `f64` holding an integral value; see the note on
/// [`Real::frexp`](crate::Real::frexp) for why it is not an integer type.
pub(crate) fn frexp(x: f64) -> (f64, f64) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0.0);
    }
    let bits = x.to_bits();
    let raw = (bits >> F64_EXP_SHIFT) & F64_EXP_MASK;
    if raw == 0 {
        // Subnormal: scale into the normal range, decompose, then undo the scaling. Exact,
        // since the scale is a power of two.
        let (m, e) = frexp(x * 2f64.powi(F64_SUBNORMAL_SHIFT));
        return (m, e - f64::from(F64_SUBNORMAL_SHIFT));
    }
    let exponent = raw as i64 - F64_HALF_BIAS as i64;
    // Replace the exponent field with the one that normalizes the mantissa into [0.5, 1).
    let mantissa = f64::from_bits((bits & !F64_EXP_BITS) | (F64_HALF_BIAS << F64_EXP_SHIFT));
    (mantissa, exponent as f64)
}

/// `x · 2^n`, exact and without forming `2^n` on its own.
///
/// Applying the scale in bounded chunks is what lets `1e-300.ldexp(1000)` succeed: the result
/// is representable even though `2^1000` is not.
pub(crate) fn ldexp(x: f64, n: f64) -> f64 {
    if x == 0.0 || !x.is_finite() || !n.is_finite() {
        return x * 2f64.powi(0); // preserves NaN/inf and signed zero
    }
    let mut result = x;
    let mut remaining = n.trunc();
    // Chunks of 512 keep every intermediate power of two comfortably representable.
    const CHUNK: f64 = 512.0;
    while remaining > CHUNK {
        result *= 2f64.powi(CHUNK as i32);
        remaining -= CHUNK;
        if !result.is_finite() {
            return result;
        }
    }
    while remaining < -CHUNK {
        result *= 2f64.powi(-(CHUNK as i32));
        remaining += CHUNK;
        if result == 0.0 {
            return result;
        }
    }
    result * 2f64.powi(remaining as i32)
}

/// The next representable value after `x` toward `y`.
pub(crate) fn next_after(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x == y {
        return y;
    }
    if x == 0.0 {
        // Step off zero into the smallest subnormal with the right sign.
        return if y > 0.0 {
            f64::from_bits(1)
        } else {
            -f64::from_bits(1)
        };
    }
    let bits = x.to_bits();
    // Moving away from zero increases the magnitude bits; moving toward zero decreases them.
    let moving_away = (y > x) == (x > 0.0);
    f64::from_bits(if moving_away { bits + 1 } else { bits - 1 })
}

/// The distance from `x` to the next representable value away from zero.
pub(crate) fn ulp(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }
    let away = if x < 0.0 {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    (next_after(x, away) - x).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frexp_normalizes_the_mantissa_and_round_trips() {
        for &x in &[
            1.0f64,
            0.5,
            2.0,
            3.0,
            1e-300,
            1e300,
            -7.25,
            f64::MIN_POSITIVE,
        ] {
            let (m, e) = frexp(x);
            assert!(
                (0.5..1.0).contains(&m.abs()),
                "frexp({x}) mantissa {m} out of [0.5, 1)"
            );
            assert_eq!(ldexp(m, e), x, "frexp/ldexp round trip at {x}");
        }
    }

    #[test]
    fn frexp_handles_subnormals_zero_and_non_finite() {
        let sub = f64::MIN_POSITIVE / 4.0;
        let (m, e) = frexp(sub);
        assert!((0.5..1.0).contains(&m.abs()), "subnormal mantissa {m}");
        assert_eq!(ldexp(m, e), sub);

        assert_eq!(frexp(0.0), (0.0, 0.0));
        assert_eq!(frexp(f64::INFINITY), (f64::INFINITY, 0.0));
        assert!(frexp(f64::NAN).0.is_nan());
    }

    #[test]
    fn frexp_exponents_are_the_expected_integers() {
        assert_eq!(frexp(1.0), (0.5, 1.0));
        assert_eq!(frexp(2.0), (0.5, 2.0));
        assert_eq!(frexp(0.5), (0.5, 0.0));
        assert_eq!(frexp(-1.0), (-0.5, 1.0));
        assert_eq!(frexp(0.75), (0.75, 0.0));
    }

    #[test]
    fn ldexp_survives_exponents_whose_power_of_two_would_overflow() {
        // 2^1000 is not representable, but the result is.
        assert!(2f64.powi(1000).is_infinite() || 2f64.powi(1000) > 1e300);
        let v = ldexp(1e-300, 1000.0);
        assert!(v.is_finite() && v > 0.0, "ldexp = {v}");
        // 2^1000 ~ 1.07e301, so 1e-300 * 2^1000 ~ 10.7
        assert!((v - 1.071_508_607_186_267_3e1).abs() < 1e-9, "got {v}");
        // And the other direction.
        let w = ldexp(1e300, -1000.0);
        assert!(w > 0.0 && w.is_finite());
    }

    #[test]
    fn ldexp_is_exact_at_small_exponents() {
        for n in -60..60 {
            assert_eq!(ldexp(3.0, f64::from(n)), 3.0 * 2f64.powi(n));
        }
    }

    #[test]
    fn next_after_steps_exactly_one_representable_value() {
        let x = 1.0f64;
        let up = next_after(x, f64::INFINITY);
        assert_eq!(up, 1.0 + f64::EPSILON, "one step above 1.0");
        let down = next_after(x, f64::NEG_INFINITY);
        assert!(
            down < x && next_after(down, f64::INFINITY) == x,
            "reversible"
        );
        // Equal arguments return the target.
        assert_eq!(next_after(2.0, 2.0), 2.0);
        // Stepping off zero reaches the smallest subnormal.
        assert_eq!(next_after(0.0, 1.0), f64::from_bits(1));
        assert_eq!(next_after(0.0, -1.0), -f64::from_bits(1));
        assert!(next_after(f64::NAN, 1.0).is_nan());
    }

    #[test]
    fn next_after_is_reversible_across_magnitudes() {
        for &x in &[1e-300f64, 1e-10, 0.5, 1.0, 3.0, 1e10, 1e300, -2.5] {
            let up = next_after(x, f64::INFINITY);
            let back = next_after(up, f64::NEG_INFINITY);
            assert_eq!(back, x, "round trip at {x}");
        }
    }

    #[test]
    fn ulp_is_the_spacing_and_scales_with_magnitude() {
        assert_eq!(ulp(1.0), f64::EPSILON);
        // Doubling the value doubles the spacing, except across a power-of-two boundary.
        assert_eq!(ulp(2.0), 2.0 * f64::EPSILON);
        assert_eq!(ulp(4.0), 4.0 * f64::EPSILON);
        // Always positive, and the step really lands one value away.
        for &x in &[1e-300f64, 1e-5, 1.0, 1e5, 1e300, -3.0] {
            let u = ulp(x);
            assert!(u > 0.0, "ulp({x}) = {u}");
            let stepped = if x < 0.0 { x - u } else { x + u };
            assert_eq!(
                stepped,
                next_after(
                    x,
                    if x < 0.0 {
                        f64::NEG_INFINITY
                    } else {
                        f64::INFINITY
                    }
                )
            );
        }
        assert!(ulp(f64::NAN).is_nan());
    }
}
