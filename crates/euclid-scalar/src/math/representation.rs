//! Scaling and neighbouring-value decisions for binary floating-point formats.

use core::cmp::Ordering;

use crate::HostReal;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Scale {
    Overflow,
    Normal(i32),
    Subnormal(i32),
    Underflow,
}

/// Select a scale for a mantissa with magnitude in `[0.5, 1)`.
fn mantissa_scale(exponent: i32, min_normal: i32, max_exp: i32, fraction_bits: i32) -> Scale {
    if exponent > max_exp {
        Scale::Overflow
    } else if exponent > min_normal {
        Scale::Normal(exponent - 1)
    } else if exponent >= min_normal - fraction_bits {
        Scale::Subnormal(exponent - min_normal)
    } else {
        Scale::Underflow
    }
}

/// Reconstruct a scaled value from the normalized mantissa. The supplied
/// power-of-two constructor only receives exponents that encode normal values.
pub(crate) fn scale_mantissa<T: HostReal>(
    mantissa: T,
    exponent: i32,
    min_normal: i32,
    max_exp: i32,
    fraction_bits: i32,
    normal_power: impl FnOnce(i32) -> T,
) -> T {
    match mantissa_scale(exponent, min_normal, max_exp, fraction_bits) {
        Scale::Overflow => T::INFINITY.copysign(mantissa),
        Scale::Normal(power) => (T::TWO * mantissa) * normal_power(power),
        // The first multiplication stays normal and is exact. Round into the
        // subnormal range only at the final multiplication.
        Scale::Subnormal(power) => (mantissa * normal_power(power)) * T::MIN_POSITIVE,
        Scale::Underflow => T::ZERO.copysign(mantissa),
    }
}

/// Compare before stepping, so equal operands return the target's signed zero.
pub(crate) fn next_after<T: HostReal>(
    x: T,
    target: T,
    up: impl FnOnce(T) -> T,
    down: impl FnOnce(T) -> T,
) -> T {
    match x.partial_cmp(&target) {
        Some(Ordering::Less) => up(x),
        Some(Ordering::Greater) => down(x),
        Some(Ordering::Equal) => target,
        None => T::NAN,
    }
}

/// The spacing away from zero equals the ascending spacing of the magnitude.
pub(crate) fn ulp<T: HostReal>(x: T, up: impl FnOnce(T) -> T) -> T {
    if HostReal::is_nan(x) {
        return T::NAN;
    }
    if !HostReal::is_finite(x) {
        return T::INFINITY;
    }
    let magnitude = x.abs();
    up(magnitude) - magnitude
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_route_distinguishes_every_format_boundary() {
        for (exponent, expected) in [
            (1025, Scale::Overflow),
            (1024, Scale::Normal(1023)),
            (-1021, Scale::Normal(-1022)),
            (-1022, Scale::Subnormal(0)),
            (-1074, Scale::Subnormal(-52)),
            (-1075, Scale::Underflow),
        ] {
            assert_eq!(mantissa_scale(exponent, -1022, 1024, 52), expected);
        }
        assert_eq!(mantissa_scale(-149, -126, 128, 23), Scale::Subnormal(-23));
    }

    #[test]
    fn mantissa_reconstruction_rounds_subnormals_once() {
        let scale = |mantissa, exponent| {
            scale_mantissa(mantissa, exponent, -1022, 1024, 52, |power| {
                f64::from_bits(((power + 1023) as u64) << 52)
            })
        };
        assert_eq!(scale(0.75, 1), 1.5);
        assert_eq!(scale(0.5, -1074).to_bits(), 0);
        assert_eq!(scale(0.5_f64.next_up(), -1074).to_bits(), 1);
        assert_eq!(scale(0.75, -1075).to_bits(), 0);
        assert_eq!(scale(-0.75, -1075).to_bits(), (-0.0_f64).to_bits());
        assert_eq!(scale(-0.75, 1025), f64::NEG_INFINITY);
    }
}
