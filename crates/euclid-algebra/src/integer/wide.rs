//! The unsigned intermediates needed to add fixed-width, signed fractions.

use super::gcd_abs;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Wide {
    high: u128,
    low: u128,
}

impl Wide {
    fn product(left: u128, right: u128) -> Self {
        if let Some(low) = left.checked_mul(right) {
            return Self { high: 0, low };
        }
        // Each limb product fits in u128. Propagate the carry from the middle column.
        let mask = u128::from(u64::MAX);
        let (left_hi, left_lo) = (left >> 64, left & mask);
        let (right_hi, right_lo) = (right >> 64, right & mask);
        let low = left_lo * right_lo;
        let mid_a = left_lo * right_hi;
        let mid_b = left_hi * right_lo;
        let middle = (low >> 64) + (mid_a & mask) + (mid_b & mask);
        Self {
            high: left_hi * right_hi + (mid_a >> 64) + (mid_b >> 64) + (middle >> 64),
            low: (middle << 64) + (low & mask),
        }
    }

    fn add(self, other: Self) -> Self {
        let (low, carry) = self.low.overflowing_add(other.low);
        Self {
            high: self.high + other.high + u128::from(carry),
            low,
        }
    }

    fn sub(self, other: Self) -> Self {
        let (low, borrow) = self.low.overflowing_sub(other.low);
        Self {
            high: self.high - other.high - u128::from(borrow),
            low,
        }
    }

    // All divisors here fit in a positive i128, so doubling a remainder fits in u128.
    fn div_rem(self, divisor: u128) -> (Self, u128) {
        if self.high == 0 {
            return (
                Self {
                    high: 0,
                    low: self.low / divisor,
                },
                self.low % divisor,
            );
        }
        let mut rem = self.high % divisor;
        let mut low = 0;
        for bit in (0..128).rev() {
            rem = (rem << 1) + ((self.low >> bit) & 1);
            if rem >= divisor {
                rem -= divisor;
                low |= 1 << bit;
            }
        }
        (
            Self {
                high: self.high / divisor,
                low,
            },
            rem,
        )
    }
}

pub(super) fn fraction_sum(
    left: i128,
    left_den: i128,
    right: i128,
    right_den: i128,
    subtract: bool,
) -> Option<(i128, i128)> {
    let g = gcd_abs(left_den, right_den, 0);
    let left_scale = right_den / g;
    let right_scale = left_den / g;
    if let Some(num) = left
        .checked_mul(left_scale)
        .zip(right.checked_mul(right_scale))
        .and_then(|(left, right)| {
            if subtract {
                left.checked_sub(right)
            } else {
                left.checked_add(right)
            }
        })
    {
        let common = gcd_abs(num.unsigned_abs(), g as u128, 0) as i128;
        return Some((num / common, right_scale.checked_mul(right_den / common)?));
    }

    let left_negative = left.is_negative();
    let right_negative = right.is_negative() ^ subtract;
    let left = Wide::product(left.unsigned_abs(), left_scale as u128);
    let right = Wide::product(right.unsigned_abs(), right_scale as u128);
    let (magnitude, negative) = if left_negative == right_negative {
        (left.add(right), left_negative)
    } else if left >= right {
        (left.sub(right), left_negative)
    } else {
        (right.sub(left), right_negative)
    };

    // For reduced inputs all cancellation in the numerator is with g.
    let (_, remainder) = magnitude.div_rem(g as u128);
    let common = gcd_abs(remainder, g as u128, 0);
    let (quotient, _) = magnitude.div_rem(common);
    if quotient.high != 0 {
        return None;
    }
    let num = if negative {
        let num = quotient.low.wrapping_neg() as i128;
        if num.is_positive() {
            return None;
        }
        num
    } else {
        i128::try_from(quotient.low).ok()?
    };
    Some((num, right_scale.checked_mul(right_den / common as i128)?))
}

#[cfg(test)]
mod tests {
    use super::Wide;

    #[test]
    fn limb_products_reconstruct_the_full_unsigned_product() {
        let max = u128::MAX;
        assert_eq!(
            Wide::product(max, max),
            Wide {
                high: max - 1,
                low: 1
            }
        );
        assert_eq!(
            Wide::product(max, 2),
            Wide {
                high: 1,
                low: max - 1
            }
        );
        assert_eq!(
            Wide::product(2, max),
            Wide {
                high: 1,
                low: max - 1
            }
        );
        assert_eq!(Wide::product(21, 17), Wide { high: 0, low: 357 });
    }

    #[test]
    fn low_limbs_carry_and_borrow_from_the_high_limb() {
        let left = Wide {
            high: 7,
            low: u128::MAX - 2,
        };
        let right = Wide { high: 5, low: 4 };
        let sum = Wide { high: 13, low: 1 };
        assert_eq!(left.add(right), sum);
        assert_eq!(sum.sub(left), right);
        assert_eq!(sum.sub(right), left);
        assert_eq!(
            left.sub(right),
            Wide {
                high: 2,
                low: u128::MAX - 6
            }
        );
        assert_eq!(right.add(right), Wide { high: 10, low: 8 });
    }

    #[test]
    fn division_returns_both_limbs_and_the_remainder() {
        assert_eq!(
            Wide { high: 0, low: 123 }.div_rem(7),
            (Wide { high: 0, low: 17 }, 4)
        );
        // 2^128 is 1 modulo 3, and u128::MAX is a multiple of 3.
        assert_eq!(
            Wide { high: 7, low: 11 }.div_rem(3),
            (
                Wide {
                    high: 2,
                    low: u128::MAX / 3 + 4
                },
                0
            )
        );
        assert_eq!(
            Wide { high: 7, low: 12 }.div_rem(3),
            (
                Wide {
                    high: 2,
                    low: u128::MAX / 3 + 4
                },
                1
            )
        );
    }
}
