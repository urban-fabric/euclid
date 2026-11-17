//! Independent references near Lambert W's branch point and zeta's eta/reflection seam.

use crate::Real;
use euclid_utils::assert_close;

#[test]
fn lambert_w_selects_the_principal_root_near_the_branch_point() {
    // 90-digit Decimal bisection of w*exp(w) = x on [-1, 0], using
    // Decimal.from_float(x) so the reference uses the exact binary input.
    // Substitution alone cannot resolve w here: its derivative vanishes at -1.
    // Measured errors are 5.3e-9, 3.0e-11, and below 1e-14 for the last two.
    for (x, expected, tolerance) in [
        (-0.367_879_441_171_442_3, -0.999_999_984_695_745_9, 6e-9),
        (-0.367_879_441_17, -0.999_997_199_763_411_1, 4e-11),
        (-0.367_8, -0.979_360_714_957_830_5, 2e-14),
        (-0.367, -0.932_399_184_747_928_3, 2e-14),
    ] {
        let w = Real::lambert_w(x);
        assert!(w >= -1.0, "W({x}) = {w} is outside the principal branch");
        assert_close(w, expected, tolerance, &format!("principal W({x})"));
    }
}

#[test]
fn zeta_eta_agrees_with_independent_euler_transformation_left_of_zero() {
    // 160-digit Decimal evaluation with 500 Euler-transformed eta terms:
    // eta(s) = sum_k (Delta^k a_0) / 2^(k+1), a_n = (n+1)^(-s),
    // Delta a_n = a_n - a_(n+1); zeta = eta/(1 - 2^(1-s)).
    // This is independent of the production Chebyshev/CVZ acceleration.
    // Measured absolute error is below 2e-15; allow 3e-15.
    for (s, expected) in [
        (-0.75, -0.133_642_774_436_584_57),
        (-0.5, -0.207_886_224_977_354_57),
        (-0.25, -0.320_451_264_228_577_3),
    ] {
        assert_close(
            Real::zeta(s),
            expected,
            3e-15,
            &format!("zeta({s}) via Euler eta"),
        );
    }
    // Both sides of the eta/reflection crossover approach the exact Bernoulli
    // value zeta(-1) = -1/12. Measured error is below 3e-15.
    for s in [(-1.0f64).next_down(), -1.0, (-1.0f64).next_up()] {
        assert_close(
            Real::zeta(s),
            -1.0 / 12.0,
            4e-15,
            "zeta at the eta/reflection seam",
        );
    }
}
