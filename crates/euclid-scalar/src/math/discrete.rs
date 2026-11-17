//! Euclidean arithmetic for the primitive integer carriers.

use crate::Element;
use core::ops::{Div, Mul, Rem};

fn euclidean_step<T: Copy + Ord + Rem<Output = T>>(dividend: T, divisor: T) -> (T, T) {
    let remainder = dividend % divisor;
    debug_assert!(remainder < divisor, "Euclid's remainder must shrink");
    (divisor, remainder)
}

/// Greatest common divisor of two magnitudes.
pub(crate) fn gcd<T: Element + Copy + Ord + Rem<Output = T>>(mut a: T, mut b: T) -> T {
    while b != T::zero() {
        (a, b) = euclidean_step(a, b);
    }
    a
}

/// Divide before multiplying. A zero product can skip GCD and its magnitude conversions.
pub(crate) fn lcm<T, G>(a: T, b: T, gcd: G) -> T
where
    T: Element + Copy + Eq + Div<Output = T> + Mul<Output = T>,
    G: FnOnce(T, T) -> T,
{
    if a == T::zero() || b == T::zero() {
        return T::zero();
    }
    (a / gcd(a, b)) * b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euclidean_step_follows_the_division_identity() {
        for b in 1..=32_u32 {
            for remainder in 0..b {
                for quotient in [0, 1, 2, 127] {
                    let a = quotient * b + remainder;
                    assert_eq!(euclidean_step(a, b), (b, remainder));
                }
            }
        }
    }

    #[test]
    fn zero_lcm_skips_the_gcd() {
        // Taking the magnitude of MIN can overflow. Check both operand orders.
        for (a, b) in [(0, 0), (0, i64::MIN), (i64::MIN, 0)] {
            let mut called = false;
            assert_eq!(
                lcm(a, b, |_, _| {
                    called = true;
                    1
                }),
                0
            );
            assert!(!called);
        }
        assert_eq!(lcm(12_u32, 42, gcd), 84);
    }
}
