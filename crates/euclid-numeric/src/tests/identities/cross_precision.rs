//! Agreement between the f64 and f32 carriers.

use crate::Real;

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
