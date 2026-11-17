//! Single-precision peer of [`super::super::repr`], with binary32 exponent encoding.

use crate::math::representation;

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
    let mantissa = f32::from_bits((bits & !F32_EXP_BITS) + (F32_HALF_BIAS << F32_EXP_SHIFT));
    (mantissa, exponent as f32)
}

/// Scale by 2 raised to the truncated exponent, rounding only the final result.
pub(crate) fn ldexp(x: f32, n: f32) -> f32 {
    if x == 0.0 || !x.is_finite() || !n.is_finite() {
        return x;
    }
    let (mantissa, current) = frexp(x);
    let exponent = (current as i32).saturating_add(n as i32);
    representation::scale_mantissa(
        mantissa,
        exponent,
        f32::MIN_EXP - 1,
        f32::MAX_EXP,
        23,
        |power| f32::from_bits(((power + 127) as u32) << 23),
    )
}

/// Step from x toward y by one representable value.
pub(crate) fn next_after(x: f32, y: f32) -> f32 {
    representation::next_after(x, y, f32::next_up, f32::next_down)
}

/// The distance to the next representable value away from zero.
pub(crate) fn ulp(x: f32) -> f32 {
    representation::ulp(x, f32::next_up)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ldexp_rounds_subnormal_results_directly_in_binary32() {
        let tie = 2.0_f32.powi(-89);
        assert_eq!(ldexp(tie, -61.0).to_bits(), 0);
        assert_eq!(ldexp(tie.next_up(), -61.0).to_bits(), 1);
        assert_eq!(ldexp(-tie.next_up(), -61.0).to_bits(), (1_u32 << 31) | 1);
        assert_eq!(ldexp(f32::from_bits(1), 149.0), 1.0);
    }

    #[test]
    fn ldexp_preserves_binary32_zeros_and_nonfinite_inputs() {
        for zero in [0.0_f32, -0.0] {
            assert_eq!(ldexp(zero, f32::MAX).to_bits(), zero.to_bits());
        }
        for infinite in [f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(ldexp(infinite, -f32::MAX), infinite);
        }
        assert!(ldexp(f32::NAN, -1.0).is_nan());
        for exponent in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(ldexp(-3.0, exponent), -3.0);
        }
    }

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
        // 2^120 is near the upper end of the representable range.
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
