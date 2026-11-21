//! Bit-exact checks for delegates, and the eight-way agreement check.
//!
//! See the [parent module](super::super) for why this file is split from
//! `super::super::parameterised`. The constants and the native-kernel budgets moved into
//! [`constants`] and [`budgets`] when this file outgrew the crate's 400-line limit.

mod budgets;
mod constants;

use crate::Real;

/// Arguments swept per kernel. Fixed, because the budgets below are the maximum gap observed
/// over *this* sweep — changing it changes which arguments are visited and therefore the
/// measurement.
const SWEEP: usize = 400;

// --- cross-precision agreement ------------------------------------------

#[test]
fn f32_and_f64_agree_to_single_precision() {
    // The same formula on both carriers must land within f32's own resolution. This is the
    // check that catches a divergent f32 code path rather than a mere precision difference.
    for i in 1..60 {
        let x64 = f64::from(i) * 0.1;
        let x32 = x64 as f32;
        let pairs: [(f64, f32); 8] = [
            (Real::erf(x64), Real::erf(x32)),
            (Real::erfc(x64), Real::erfc(x32)),
            (Real::ln_gamma(x64), Real::ln_gamma(x32)),
            (Real::digamma(x64), Real::digamma(x32)),
            (Real::trigamma(x64), Real::trigamma(x32)),
            (Real::sigmoid(&x64), Real::sigmoid(&x32)),
            (Real::ln_1p_exp(&x64), Real::ln_1p_exp(&x32)),
            (Real::bessel_i0e(x64), Real::bessel_i0e(x32)),
        ];
        for (idx, (a, b)) in pairs.iter().enumerate() {
            // Mixed criterion: `ln_gamma(1) = 0`, where a relative comparison between two
            // different roundings of zero is meaningless. f32 carries ~1.2e-7 resolution, so
            // the absolute floor has to accommodate it.
            let err = (f64::from(*b) - *a).abs();
            let bound = 1e-5 * a.abs().max(1.0);
            assert!(
                err <= bound,
                "f32/f64 diverged at x={x64}, fn index {idx}: {a} vs {b} \
                 (abs err {err:.3e} > {bound:.3e})"
            );
        }
    }
}

/// The primitive-backed methods delegate to Rust's f32 implementations.
#[test]
fn the_f32_real_methods_match_the_native_primitives() {
    // Rust 1.95's native asinh(-7.88_f32) differs by one ulp from rounding f64 asinh.
    // Compare against the native delegate to check dispatch on every supported toolchain.
    macro_rules! matches_native {
        ($name:ident, $lo:expr, $hi:expr, $call:expr) => {{
            for i in 0..=SWEEP {
                let x = ($lo + ($hi - $lo) * (i as f64) / (SWEEP as f64)) as f32;
                let actual: f32 = ($call)(x);
                assert_eq!(
                    actual.to_bits(),
                    f32::$name(x).to_bits(),
                    "{}({x})",
                    stringify!($name)
                );
            }
        }};
    }
    matches_native!(asin, -0.99, 0.99, Real::asin);
    matches_native!(acos, -0.99, 0.99, Real::acos);
    matches_native!(atan, -8.0, 8.0, Real::atan);
    matches_native!(asinh, -8.0, 8.0, Real::asinh);
    matches_native!(acosh, 1.01, 10.0, Real::acosh);
    matches_native!(sinh, -5.0, 5.0, Real::sinh);
    matches_native!(cosh, -5.0, 5.0, Real::cosh);
    matches_native!(tanh, -5.0, 5.0, Real::tanh);
    matches_native!(cbrt, -20.0, 20.0, Real::cbrt);
    matches_native!(exp2, -10.0, 10.0, Real::exp2);
    matches_native!(exp_m1, -2.0, 2.0, |x: f32| Real::exp_m1(&x));
}
