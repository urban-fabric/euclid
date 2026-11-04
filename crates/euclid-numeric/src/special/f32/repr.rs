//! Single-precision peer of [`super::super::repr`]: the same bit-level decomposition, with
//! `f32`'s 8-bit exponent field and 23-bit mantissa, and a narrower chunk size for `ldexp`
//! because the exponent range is much smaller.

/// Bit layout of an `f32`: 1 sign bit, 8 exponent bits, 23 mantissa bits, bias 127.
const F32_EXP_BITS: u32 = 0x7f80_0000;
const F32_EXP_SHIFT: u32 = 23;
const F32_EXP_MASK: u32 = 0xff;
/// Exponent field value that puts the mantissa in `[0.5, 1)` — the `frexp` convention.
const F32_HALF_BIAS: u32 = 126;
/// Scaling applied to bring a subnormal into the normal range before decomposing.
const F32_SUBNORMAL_SHIFT: i32 = 25;

/// Decompose `x` into `(mantissa, exponent)` with `x == mantissa · 2^exponent` and
/// `|mantissa| ∈ [0.5, 1)`.
///
/// The exponent is returned as an `f32` holding an integral value; see the note on
/// [`Real::frexp`](crate::Real::frexp) for why it is not an integer type.
pub(crate) fn frexp(x: f32) -> (f32, f32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0.0);
    }
    let bits = x.to_bits();
    let raw = (bits >> F32_EXP_SHIFT) & F32_EXP_MASK;
    if raw == 0 {
        // Subnormal: scale into the normal range, decompose, then undo the scaling. Exact,
        // since the scale is a power of two.
        let (m, e) = frexp(x * 2f32.powi(F32_SUBNORMAL_SHIFT));
        return (m, e - (F32_SUBNORMAL_SHIFT as f32));
    }
    let exponent = raw as i32 - F32_HALF_BIAS as i32;
    // Replace the exponent field with the one that normalizes the mantissa into [0.5, 1).
    let mantissa = f32::from_bits((bits & !F32_EXP_BITS) | (F32_HALF_BIAS << F32_EXP_SHIFT));
    (mantissa, exponent as f32)
}

/// `x · 2^n`, exact and without forming `2^n` on its own.
///
/// Applying the scale in bounded chunks is what lets `1e-300.ldexp(1000)` succeed: the result
/// is representable even though `2^1000` is not.
pub(crate) fn ldexp(x: f32, n: f32) -> f32 {
    if x == 0.0 || !x.is_finite() || !n.is_finite() {
        return x * 2f32.powi(0); // preserves NaN/inf and signed zero
    }
    let mut result = x;
    let mut remaining = n.trunc();
    // Chunks of 512 keep every intermediate power of two comfortably representable.
    const CHUNK: f32 = 60.0;
    while remaining > CHUNK {
        result *= 2f32.powi(CHUNK as i32);
        remaining -= CHUNK;
        if !result.is_finite() {
            return result;
        }
    }
    while remaining < -CHUNK {
        result *= 2f32.powi(-(CHUNK as i32));
        remaining += CHUNK;
        if result == 0.0 {
            return result;
        }
    }
    result * 2f32.powi(remaining as i32)
}

/// The next representable value after `x` toward `y`.
pub(crate) fn next_after(x: f32, y: f32) -> f32 {
    if x.is_nan() || y.is_nan() {
        return f32::NAN;
    }
    if x == y {
        return y;
    }
    if x == 0.0 {
        // Step off zero into the smallest subnormal with the right sign.
        return if y > 0.0 {
            f32::from_bits(1)
        } else {
            -f32::from_bits(1)
        };
    }
    let bits = x.to_bits();
    // Moving away from zero increases the magnitude bits; moving toward zero decreases them.
    let moving_away = (y > x) == (x > 0.0);
    f32::from_bits(if moving_away { bits + 1 } else { bits - 1 })
}

/// The distance from `x` to the next representable value away from zero.
pub(crate) fn ulp(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x.is_infinite() {
        return f32::INFINITY;
    }
    let away = if x < 0.0 {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    };
    (next_after(x, away) - x).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frexp_and_ldexp_round_trip_in_f32() {
        for &x in &[1.0f32, 0.5, 3.0, -7.25, 1e-30, 1e30, f32::MIN_POSITIVE] {
            let (m, e) = frexp(x);
            assert!(
                (0.5..1.0).contains(&m.abs()),
                "frexp({x}) mantissa {m} out of range"
            );
            assert_eq!(ldexp(m, e), x, "round trip at {x}");
        }
        assert_eq!(frexp(1.0f32), (0.5, 1.0));
        assert_eq!(frexp(2.0f32), (0.5, 2.0));
        // Subnormal.
        let sub = f32::MIN_POSITIVE / 4.0;
        let (m, e) = frexp(sub);
        assert!((0.5..1.0).contains(&m.abs()));
        assert_eq!(ldexp(m, e), sub);
    }

    #[test]
    fn ldexp_survives_exponents_whose_power_of_two_would_overflow_f32() {
        // f32 overflows at 2^128, so this is a much easier threshold to hit than in f64.
        let v = ldexp(1e-30f32, 120.0);
        assert!(v.is_finite() && v > 0.0, "ldexp = {v}");
    }

    #[test]
    fn next_after_and_ulp_in_f32() {
        assert_eq!(next_after(1.0f32, f32::INFINITY), 1.0 + f32::EPSILON);
        assert_eq!(ulp(1.0f32), f32::EPSILON);
        assert_eq!(next_after(0.0f32, 1.0), f32::from_bits(1));
        for &x in &[1e-30f32, 1.0, 1e30, -2.5] {
            let up = next_after(x, f32::INFINITY);
            assert_eq!(next_after(up, f32::NEG_INFINITY), x, "reversible at {x}");
        }
    }
}
