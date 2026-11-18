//! Logarithms of sums and differences of exponentials.

use crate::{HostReal, Real};

/// Low limbs of ln(2), after rounding the high limb to the input format.
pub(crate) const LN_2_LOW_F64: f64 = 2.319_046_813_846_299_6e-17;
pub(crate) const LN_2_LOW_F32: f32 = -1.904_654_3e-9;

/// `ln(1 + exp(x)) = max(x, 0) + ln(1 + exp(-|x|))`.
/// The exponent is nonpositive, even when `exp(x)` would overflow.
pub(crate) fn ln_1p_exp<T: HostReal>(x: T) -> T {
    Real::max(x, T::ZERO) + (-x.abs()).exp().ln_1p()
}

/// Choose `expm1` when `exp(x)` exceeds one half and may round to one.
fn use_expm1<T: HostReal>(x: T) -> bool {
    x > -T::ln_2()
}

fn ln_1m_exp_near_zero<T: Real>(x: T) -> T {
    (-x.exp_m1()).ln()
}

fn ln_1m_exp_tail<T: Real>(x: T) -> T {
    (-x.exp()).ln_1p()
}

/// `ln(1 - exp(x))` on `x <= 0`. The logarithm gives `-∞` at zero and
/// `NaN` above zero.
pub(crate) fn ln_1m_exp<T: HostReal>(x: T) -> T {
    if use_expm1(x) {
        ln_1m_exp_near_zero(x)
    } else {
        ln_1m_exp_tail(x)
    }
}

/// Near ln(2), `exp(x) - 1 = 1 + 2 * expm1(x - ln(2))`.
/// In `[0.5, 1]`, subtracting the high limb is exact by Sterbenz's lemma.
fn ln_exp_m1_near_ln_2<T: HostReal>(x: T, ln_2_low: T) -> T {
    let offset = (x - T::ln_2()) - ln_2_low;
    (T::TWO * offset.exp_m1()).ln_1p()
}

/// `ln(exp(x) - 1)` on `x >= 0`. Around ln(2), keep the small output in
/// `ln1p`. Elsewhere factor out `exp(x)`.
pub(crate) fn ln_exp_m1<T: HostReal>(x: T, ln_2_low: T) -> T {
    if (T::HALF..=T::ONE).contains(&x) {
        ln_exp_m1_near_ln_2(x, ln_2_low)
    } else {
        x + ln_1m_exp(-x)
    }
}

/// `ln(exp(a) + exp(b))`, factoring out the larger exponential.
pub(crate) fn ln_add_exp<T: HostReal>(a: T, b: T) -> T {
    if HostReal::is_nan(a) || HostReal::is_nan(b) {
        return T::NAN;
    }
    let high = Real::max(a, b);
    if high == T::INFINITY || Real::min(a, b) == T::NEG_INFINITY {
        return high;
    }
    high + ln_1p_exp(-(a - b).abs())
}

/// `ln(exp(a) - exp(b))`, factoring out `exp(a)`. The shared complement
/// handles equal finite operands and rejects a positive gap.
pub(crate) fn ln_sub_exp<T: HostReal>(a: T, b: T) -> T {
    if b == T::NEG_INFINITY {
        return a;
    }
    a + ln_1m_exp(b - a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complement_uses_expm1_strictly_above_the_switch() {
        let x = -core::f64::consts::LN_2;
        assert!(!use_expm1(x.next_down()));
        assert!(!use_expm1(x));
        assert!(use_expm1(x.next_up()));
        assert!(use_expm1(-0.0_f64));

        let x = -core::f32::consts::LN_2;
        assert!(!use_expm1(x));
        assert!(use_expm1(x.next_up()));
    }

    #[test]
    fn complement_steps_preserve_subnormal_inputs_and_outputs() {
        let tiny = f64::from_bits(1);
        assert_eq!(
            ln_1m_exp_near_zero(-tiny),
            -1074.0 * core::f64::consts::LN_2
        );
        assert!(ln_1m_exp_near_zero(tiny).is_nan());
        assert_eq!(
            ln_1m_exp_tail(-740.0_f64).to_bits(),
            (-f64::from_bits(85)).to_bits()
        );
        assert_eq!(
            ln_1m_exp_tail(f64::NEG_INFINITY).to_bits(),
            (-0.0_f64).to_bits()
        );
        assert_eq!(ln_1m_exp(-0.0_f32), f32::NEG_INFINITY);
        assert_eq!(ln_1m_exp(0.0_f32), f32::NEG_INFINITY);
        assert!(ln_1m_exp(f32::from_bits(1)).is_nan());
    }

    #[test]
    fn ln_exp_m1_preserves_the_residual_at_rounded_ln_2() {
        // Computed directly from exp(x) - 1 with 500-digit Decimal arithmetic.
        let ln_2 = core::f64::consts::LN_2.to_bits();
        for (input, expected) in [
            (ln_2 - 1, -2.684_255_412_019_573e-16_f64),
            (ln_2, -4.638_093_627_692_599e-17),
            (ln_2 + 1, 1.756_636_686_481_053e-16),
        ] {
            let x = f64::from_bits(input);
            for value in [ln_exp_m1_near_ln_2(x, LN_2_LOW_F64), Real::ln_exp_m1(&x)] {
                assert!((value / expected - 1.0).abs() <= 4.0 * f64::EPSILON);
            }
        }
        let ln_2 = core::f32::consts::LN_2.to_bits();
        for (input, expected) in [
            (ln_2 - 1, 0xb3f7_d1d0),
            (ln_2, 0x3182_e308),
            (ln_2 + 1, 0x3404_1718),
        ] {
            let x = f32::from_bits(input);
            for value in [ln_exp_m1_near_ln_2(x, LN_2_LOW_F32), Real::ln_exp_m1(&x)] {
                assert!(
                    value.to_bits().abs_diff(expected) <= 2,
                    "input={x}, value={value}"
                );
            }
        }
    }

    #[test]
    fn logarithmic_arithmetic_preserves_extended_real_boundaries() {
        let n = f64::NEG_INFINITY;
        let p = f64::INFINITY;
        for [a, b, sum, difference] in [
            [n, n, n, n],
            [p, p, p, f64::NAN],
            [p, 1.0, p, p],
            [1.0, p, p, f64::NAN],
            [p, n, p, p],
            [n, p, p, f64::NAN],
            [p, f64::NAN, f64::NAN, f64::NAN],
            [f64::NAN, p, f64::NAN, f64::NAN],
            [-0.0, n, -0.0, -0.0],
            [-2.0, -2.0, -2.0 + core::f64::consts::LN_2, n],
        ] {
            for (value, expected) in [(ln_add_exp(a, b), sum), (ln_sub_exp(a, b), difference)] {
                if expected.is_nan() {
                    assert!(value.is_nan(), "a={a}, b={b}, value={value}");
                } else {
                    assert_eq!(value.to_bits(), expected.to_bits(), "a={a}, b={b}");
                }
            }
        }

        let a = 1000.0_f64;
        assert_eq!(ln_1p_exp(a), a);
        assert_eq!(ln_1p_exp(f64::NEG_INFINITY), 0.0);
        assert!(ln_1p_exp(f64::NAN).is_nan());
        assert_eq!(ln_add_exp(a, 999.0), 1_000.313_261_687_518_2);
        assert_eq!(ln_sub_exp(a, 999.0), 999.541_324_854_612_9);
        assert_eq!(ln_add_exp(0.0_f64, -740.0).to_bits(), 85);
        assert_eq!(
            ln_sub_exp(0.0_f64, -740.0).to_bits(),
            (-f64::from_bits(85)).to_bits()
        );
        assert!(ln_sub_exp(1.0_f64, 2.0).is_nan());
    }
}
