//! Numerical steps shared by the two precisions of the Bessel kernels.
//!
//! What is here and what is not follows the rule in [`crate::kernels`]: a step whose arithmetic
//! carries no format-dependent constant is written once, over `T: HostReal`; a *threshold* or a
//! measured term count stays beside the code it governs. So the `I₀` series below is here — its
//! only tolerance is `ε` itself, which the type supplies — while `stirling_remainder` is not,
//! because `f64` sums five Bernoulli terms there and `f32` sums three, and that difference is the
//! measurement rather than an accident.

use crate::HostReal;

/// `I₀(x) − 1` for a half-argument `h = x/2`, with the number of series terms it summed.
///
/// Summing the difference directly avoids forming `I₀` and subtracting one from it, which would
/// discard the whole answer near the origin: `I₀(x) − 1 ≈ x²/4`, so at `x = 1e-6` the subtraction
/// loses every significant digit while this series keeps them all.
///
/// # Why the term count is returned
///
/// The loop's guard is a convergence test and a cap, and neither is visible in the sum: the terms
/// it stops after are by construction below a quarter ulp of the running total, so widening `&&`
/// to `||` — which runs every call to the cap — leaves the returned value bit-identical and costs
/// sixty iterations. Mutation testing found all three of its operators unkillable through the
/// value. The count separates the one that matters; the other two differ only at an exact
/// tolerance hit, or at a cap convergence never reaches, and are equivalent mutants.
pub(crate) fn i0_minus_one_terms<T: HostReal>(h: T) -> (T, u32) {
    let t = h * h;
    let (mut term, mut sum) = (t, t);
    let mut k = T::ONE;
    let mut terms = 1;
    while term > T::from_f64(0.25) * T::EPSILON * sum && k < T::from_f64(64.0) {
        k += T::ONE;
        term *= t / (k * k);
        sum += term;
        terms += 1;
    }
    (sum, terms)
}

/// The integrand of the scaled `K_ν` quadrature, `e^{−2x sinh²(t/2)}·cosh(νt)`.
///
/// `√x` is applied *before* squaring so `cosh(t)` can leave the format while the decay stays
/// finite — the integrand is evaluated far out in `t`, where the unscaled `cosh` has long since
/// overflowed but the product has not.
pub(crate) fn k_scaled_integrand<T: HostReal>(nu: T, root_x: T, t: T) -> T {
    let scaled_sinh = root_x * (t * T::HALF).sinh();
    let decay = T::TWO * scaled_sinh * scaled_sinh;
    (-decay).exp() * (nu * t).cosh()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_i0_series_leads_with_a_quarter_of_the_square() {
        // `I₀(x) − 1 = x²/4 + x⁴/64 + ⋯`, so the leading term is exact to a rounding for a tiny
        // argument — which is the whole reason this is summed rather than differenced.
        for x in [1e-6_f64, 1e-4, 1e-2] {
            let (sum, _) = i0_minus_one_terms(x * 0.5);
            assert!((sum / (0.25 * x * x) - 1.0).abs() < 1e-4, "{sum:e}");
        }
        let (sum, _) = i0_minus_one_terms(1e-4_f32 * 0.5);
        assert!((sum / (0.25 * 1e-8) - 1.0).abs() < 1e-4, "{sum:e}");
        // And against the kernel's own `I₀`, where the subtraction is still safe.
        for x in [0.5_f64, 1.0, 2.0] {
            let (sum, _) = i0_minus_one_terms(x * 0.5);
            let differenced = crate::kernels::bessel::modified_first::i0(x) - 1.0;
            assert!((sum - differenced).abs() <= 1e-14 * differenced, "at {x}");
        }
    }

    #[test]
    fn the_i0_series_term_count_is_bounded_in_both_formats() {
        // Measured, and monotone in the argument: a convergence test, not a fixed schedule. The
        // two formats differ because `ε` does, which is the point of taking it from the type.
        for (x, want) in [
            (1e-8_f64, 2),
            (1e-6, 3),
            (1e-4, 3),
            (0.01, 4),
            (0.05, 5),
            (0.099, 6),
            (0.5, 8),
            (1.0, 10),
        ] {
            let (_, terms) = i0_minus_one_terms(x * 0.5);
            assert_eq!(terms, want, "f64 at x = {x}");
            assert!(terms < 64, "the cap would bind at x = {x}");
        }
        for (x, want) in [
            (1e-6_f32, 2),
            (1e-4, 2),
            (0.01, 3),
            (0.05, 3),
            (0.099, 4),
            (0.5, 5),
            (1.0, 6),
        ] {
            let (_, terms) = i0_minus_one_terms(x * 0.5);
            assert_eq!(terms, want, "f32 at x = {x}");
            assert!(terms < 64, "the cap would bind at x = {x}");
        }
    }

    #[test]
    fn the_k_integrand_is_one_at_the_origin_and_decays_in_both_arguments() {
        // At `t = 0` the decay vanishes and `cosh(0) = 1`, whatever `ν` and `x` are — which pins
        // the whole expression's shape without any reference.
        for (nu, root_x) in [(0.0_f64, 0.0_f64), (0.0, 3.0), (2.0, 1.0), (0.5, 10.0)] {
            assert_eq!(k_scaled_integrand(nu, root_x, 0.0), 1.0);
        }
        // A zero argument leaves only the `cosh(νt)` factor, undamped.
        assert_eq!(k_scaled_integrand(1.0_f64, 0.0, 2.0), 2.0_f64.cosh());
        assert_eq!(k_scaled_integrand(0.0_f64, 0.0, 5.0), 1.0);
        // `√x` before squaring: the decay is `2x sinh²(t/2)`, so it grows with both and the
        // integrand falls off faster than `cosh(νt)` rises.
        let near = k_scaled_integrand(0.0_f64, 1.0, 1.0);
        let far = k_scaled_integrand(0.0_f64, 1.0, 4.0);
        assert!(near > far && far > 0.0, "{near} vs {far}");
        assert!(k_scaled_integrand(0.0_f64, 3.0, 1.0) < near);
        // Applying `√x` *before* squaring is what keeps the **decay** finite where `sinh²(t/2)`
        // alone would not: at `t = 1420`, `sinh(710)²` overflows, while `(√x·sinh(710))²` with
        // `√x = 1e-160` is an ordinary `2.5e296` and the integrand is a clean zero.
        assert!(!(710.0_f64.sinh() * 710.0_f64.sinh()).is_finite());
        let scaled = 1e-160_f64 * 710.0_f64.sinh();
        assert!((2.0 * scaled * scaled).is_finite());
        assert_eq!(k_scaled_integrand(0.0_f64, 1e-160, 1420.0), 0.0);
        // And the same in `f32`, whose range stops thirty decades sooner.
        assert_eq!(k_scaled_integrand(0.0_f32, 2.0, 0.0), 1.0);
        assert!(!(88.0_f32.sinh() * 88.0_f32.sinh()).is_finite());
        assert_eq!(k_scaled_integrand(0.0_f32, 1e-20, 176.0), 0.0);
    }
}
