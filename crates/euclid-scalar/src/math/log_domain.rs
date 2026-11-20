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

/// Which form of `ln(2πk)` keeps its intermediate inside the format.
///
/// `2πk` leaves the format once `k > MAX/2π`, and the modified-`I` kernels reach it with
/// `k = |x|/2`, so forming the product directly made `I₁(x)` return **zero** for
/// `|x| ≥ MAX/π` — `5.7e307` in `f64`, `1.0831524e38` in `f32`. The logarithm of an infinity is
/// an infinity, the peak core came back `−∞`, and `exp(−∞)` is `0` where the true value is
/// unbounded. `I₀` was correct at every one of those arguments, which is what made it look like a
/// property of `I₁` rather than of the helper both share.
///
/// **The product is kept on the path where it works, rather than replaced outright.** `ln(2π) +
/// ln k` is one rounding longer, and replacing unconditionally cost 20% of the margin on
/// `the_scaled_modified_bessel_wronskian_is_one_over_x` — a real accuracy loss across the whole
/// range, traded for a fix that only matters at the very top of it. The budget instrument caught
/// that; this form gives up nothing.
///
/// # Mutation testing is what asked for this
///
/// Both operators in the sum arm survived in both precisions. The arm is selected only above
/// `MAX/2π`, and the two kernels that call it reach that only at the very top of the range, where
/// the result is dominated by terms many orders larger — so no assertion on `I₀` or `I₁` could
/// see `ln(2π) + ln k` become `ln(2π) − ln k`. Named and generic, the decision is one table and
/// the arithmetic is reachable at any `k`. One component also replaces the two identical copies
/// the kernels each carried, which is what let a single fix serve both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LnTauForm {
    /// `2πk` fits: `ln(2πk)`, one rounding shorter and correct through the normal range.
    Product,
    /// `2πk` has left the format: `ln(2π) + ln k` reaches the same value without forming it.
    Sum,
}

impl LnTauForm {
    /// The product is formed once by [`ln_tau_times`], and its range *is* the branch.
    pub(crate) fn for_product<T: HostReal>(product: T) -> Self {
        if HostReal::is_finite(product) {
            Self::Product
        } else {
            Self::Sum
        }
    }
}

/// `ln(2πk)`, as the product where that fits and as a sum of logarithms where it does not.
pub(crate) fn ln_tau_times<T: HostReal>(k: T) -> T {
    let product = T::TAU * k;
    match LnTauForm::for_product(product) {
        LnTauForm::Product => product.ln(),
        LnTauForm::Sum => T::TAU.ln() + k.ln(),
    }
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

    #[test]
    fn the_log_of_tau_times_switches_to_a_sum_of_logarithms() {
        use LnTauForm::{Product, Sum};
        assert_eq!(LnTauForm::for_product(1.0_f64), Product);
        assert_eq!(LnTauForm::for_product(f64::MAX), Product);
        assert_eq!(LnTauForm::for_product(f64::INFINITY), Sum);
        assert_eq!(LnTauForm::for_product(f64::NAN), Sum);
        assert_eq!(LnTauForm::for_product(f32::MAX), Product);
        assert_eq!(LnTauForm::for_product(f32::INFINITY), Sum);

        // Where the product fits, that is the form, and it is one rounding shorter.
        let tau = core::f64::consts::TAU;
        assert_eq!(ln_tau_times(3.0_f64), (tau * 3.0).ln());
        assert_eq!(ln_tau_times(1.0_f64), tau.ln());

        // Past `MAX/2π = 2.86e307` the product is gone and the sum carries the value. `k = 1e308`
        // has `2πk = inf`, and `ln(2π) + ln k` is `711.034` — the answer the product threw away.
        let big = 1e308_f64;
        assert!(!(tau * big).is_finite());
        assert_eq!(ln_tau_times(big), tau.ln() + big.ln());
        assert!((ln_tau_times(big) - 711.034_085_708_575_4).abs() < 1e-9);
        // The two agree either side of the crossing, which is what makes the guard free.
        let edge = f64::MAX / tau;
        for k in [edge.next_down(), edge, edge.next_up()] {
            let sum = tau.ln() + k.ln();
            assert!((ln_tau_times(k) - sum).abs() <= 2.0 * f64::EPSILON * sum);
        }

        // And the same in `f32`, whose product leaves the format at `f32::MAX/2π = 5.4e37`.
        let tau = core::f32::consts::TAU;
        assert_eq!(ln_tau_times(3.0_f32), (tau * 3.0).ln());
        let big = 1e38_f32;
        assert!(!(tau * big).is_finite());
        assert_eq!(ln_tau_times(big), tau.ln() + big.ln());
    }
}
