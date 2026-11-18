//! **Do the count tiers carry a formula across every width, and bridge into the reals?**
//!
//! The semiring and ring tiers' portability suite. [`Natural`] (`u32`/`u64`/`u128`) has `+` and
//! `*` but no negation; [`Integer`] (`i32`/`i64`/`i128`) adds `-`. A formula bounded at the
//! narrower tier must therefore run at the wider one too, and the same source must compile at
//! all three widths of each.
//!
//! The *bridges* are the part worth testing. Discrete distribution work starts in exact integer
//! arithmetic — a binomial coefficient, a factorial — and has to reach the real surface without
//! silently losing the exactness on the way. These tests walk both steps: natural into the
//! integer ring, then integer into [`Real`], checking the value survives.
//!
//! See [`super::real_surface`] for the analysis tier these feed into.

use super::*;

/// A write-once ring formula (`+ - *`, no division). The binomial step `(n − k)·(k + 1)` is
/// the count arithmetic a discrete distribution does before reaching the real surface.
fn ring_step<I: Integer>(n: I, k: I) -> I {
    (n - k.clone()) * (k + I::one())
}

#[test]
fn lazy_ring_formula_runs_on_i32_i64_and_i128() {
    // (10 − 3)·(3 + 1) = 28, on every signed integer carrier.
    assert_eq!(ring_step(10_i32, 3), 28);
    assert_eq!(ring_step(10_i64, 3), 28);
    assert_eq!(ring_step(10_i128, 3), 28);
    // The width-matched cast into the real surface.
    assert_eq!(3_i32.to_real(), 3.0_f32);
    assert_eq!(3_i64.to_real(), 3.0_f64);
    // Lift, sign, abs, Euclidean div/rem.
    assert_eq!(i64::from_i64(7), 7);
    assert_eq!(Integer::signum(-7_i64), -1);
    assert_eq!(Integer::abs(-7_i64), 7);
    assert_eq!(7_i64.div_rem(3), (2, 1));
    assert_eq!((-7_i64).div_rem(3), (-3, 2));
    assert_eq!(7_i64.div_rem(-3), (-2, 1));
    assert_eq!((-7_i64).div_rem(-3), (3, 2));
    assert_eq!((-7_i32).div_rem(3), (-3, 2));
    assert_eq!((-7_i128).div_rem(3), (-3, 2));
}

/// The integer→real bridge: a Poisson log-mass kernel with an integer count and real
/// parameters, crossing into the real surface once via [`Integer::to_real`].
fn poisson_log_pmf<I, T>(k: I, lambda: T, ln_factorial_k: T) -> T
where
    I: Integer<Real = T>,
    T: Real,
{
    k.to_real() * lambda.clone().ln() - lambda - ln_factorial_k
}

#[test]
fn lazy_integer_bridges_into_the_real_surface() {
    // Poisson(λ=2) log-pmf at k=3: 3·ln2 − 2 − ln(3!) = 3·ln2 − 2 − ln6.
    let got = poisson_log_pmf(3_i64, 2.0_f64, 6.0_f64.ln());
    let want = 3.0 * 2.0_f64.ln() - 2.0 - 6.0_f64.ln();
    assert!((got - want).abs() < 1e-12);
}

/// A write-once semiring formula (`+ *`, no subtraction): `n·n + n` — the count arithmetic a
/// formula does staying inside the naturals, before any cast leaves the semiring.
fn semiring_step<N: Natural>(n: N) -> N {
    n.clone() * n.clone() + n
}

#[test]
fn lazy_semiring_formula_runs_on_u32_u64_and_u128() {
    // 6·6 + 6 = 42, on every natural carrier.
    assert_eq!(semiring_step(6_u32), 42);
    assert_eq!(semiring_step(6_u64), 42);
    assert_eq!(semiring_step(6_u128), 42);
    // The width-matched cast into the real surface.
    assert_eq!(3_u32.to_real(), 3.0_f32);
    assert_eq!(3_u64.to_real(), 3.0_f64);
    // Lift, the widen into the signed ring, Euclidean div/rem (floor division, no sign).
    assert_eq!(u64::from_u64(7), 7);
    assert_eq!(7_u32.to_integer(), 7_i64);
    assert_eq!(7_u64.to_integer(), 7_i128);
    assert_eq!(7_u64.div_rem(3), (2, 1));
    assert_eq!(5_u64.factorial(), 120.0);
    assert_eq!(5_u64.binomial(&2), 10.0);
    assert_eq!(5_u32.factorial(), 120.0);
    assert_eq!(5_u32.binomial(&2), 10.0);
    assert_eq!(5_u128.factorial(), 120.0);
    assert_eq!(5_u128.binomial(&2), 10.0);
    assert_eq!(3_u128.binomial(&4), 0.0);
    assert_eq!(3_u128.ln_binomial(&4), f64::NEG_INFINITY);
    assert!((5_u32.ln_factorial() - 120.0_f32.ln()).abs() < 1e-6);
}

/// The natural→integer bridge: a signed difference of counts crosses into the ring via
/// [`Natural::to_integer`], where the subtraction the semiring withholds is available —
/// the result is unrepresentable in the naturals themselves.
fn count_difference<N: Natural>(a: N, b: N) -> N::Integer {
    a.to_integer() - b.to_integer()
}

#[test]
fn lazy_natural_bridges_into_the_integer_ring() {
    // 3 − 7 = −4 in the signed ring; the naturals have no such value.
    assert_eq!(count_difference(3_u32, 7_u32), -4_i64);
    assert_eq!(count_difference(3_u64, 7_u64), -4_i128);
}
