//! The elementary gaps: log10/exp2, and exact integer arithmetic.
//!
//! Part of the [`normal_family`](super) suite.

#![allow(clippy::excessive_precision)]

// Reference digits are written out in full as provenance; see the note in `accuracy`.

use crate::{Integer, Natural, Real};
use euclid_utils::{assert_close, assert_close_f32, assert_exact, sweep};

// --- log10 and exp2 -----------------------------------------------------

#[test]
fn log10_and_exp2_are_exact_at_their_powers() {
    for k in -300..301 {
        let p = f64::from(k);
        // log10 is exact at exact powers of ten within range.
        if (-300..=300).contains(&k) {
            assert_exact(Real::log10(10f64.powi(k)), p, &format!("log10(1e{k})"));
        }
    }
    for k in -1000..1001 {
        let p = f64::from(k);
        assert_exact(Real::exp2(p), 2f64.powi(k), &format!("exp2({k})"));
        assert_exact(Real::log2(Real::exp2(p)), p, &format!("log2(exp2({k}))"));
    }
}

#[test]
fn log10_and_exp2_relate_to_the_natural_forms() {
    for x in sweep(0.1, 100.0, 200) {
        assert_close(
            Real::log10(x),
            Real::ln(&x) * <f64 as Real>::log10_e(),
            1e-14,
            &format!("log10({x})"),
        );
        assert_close(
            Real::log10(x) * <f64 as Real>::log2_10(),
            Real::log2(x),
            1e-14,
            &format!("log10 to log2 at {x}"),
        );
    }
    for x in sweep(-20.0, 20.0, 200) {
        assert_close(
            Real::exp2(x),
            Real::exp(&(x * <f64 as Real>::ln_2())),
            1e-14,
            &format!("exp2({x})"),
        );
    }
    // Round trip both ways.
    for x in sweep(0.01, 1000.0, 100) {
        assert_close(Real::exp2(Real::log2(x)), x, 1e-14, "exp2(log2(x)) = x");
    }
}

// --- exact integer arithmetic -------------------------------------------

#[test]
fn gcd_satisfies_its_defining_properties() {
    for a in 0..60u64 {
        for b in 0..60u64 {
            let g = Natural::gcd(a, b);
            if a == 0 && b == 0 {
                assert_eq!(g, 0, "gcd(0,0) = 0");
                continue;
            }
            assert!(g > 0, "gcd({a},{b}) must be positive");
            assert_eq!(a % g, 0, "gcd({a},{b}) must divide {a}");
            assert_eq!(b % g, 0, "gcd({a},{b}) must divide {b}");
            assert_eq!(g, Natural::gcd(b, a), "gcd is commutative");
            // Maximality: nothing larger divides both.
            for d in (g + 1)..=a.max(b) {
                assert!(
                    !(a % d == 0 && b % d == 0),
                    "{d} divides both {a} and {b} but exceeds gcd {g}"
                );
            }
        }
    }
    assert_eq!(Natural::gcd(12u64, 0), 12, "gcd(n,0) = n");
    assert_eq!(Natural::gcd(1_071u64, 462), 21, "the classic example");
}

#[test]
fn lcm_satisfies_the_product_identity() {
    // gcd(a,b) * lcm(a,b) = a * b
    for a in 1..50u64 {
        for b in 1..50u64 {
            let (g, l) = (Natural::gcd(a, b), Natural::lcm(a, b));
            assert_eq!(g * l, a * b, "gcd*lcm = a*b at ({a},{b})");
            assert_eq!(l % a, 0, "lcm must be a multiple of {a}");
            assert_eq!(l % b, 0, "lcm must be a multiple of {b}");
        }
    }
    assert_eq!(Natural::lcm(0u64, 5), 0, "lcm with zero is zero");
    assert_eq!(Natural::lcm(4u64, 6), 12);
}

#[test]
fn lcm_divides_before_multiplying_so_large_operands_survive() {
    // (2^62) and 4 share a factor of 4, so the lcm is 2^62 and fits — but a*b would not.
    let a = 1u64 << 62;
    assert_eq!(
        Natural::lcm(a, 4),
        a,
        "no overflow on the way to a small answer"
    );
    assert_eq!(Natural::lcm(a, 2), a);
}

#[test]
fn isqrt_is_exact_where_the_float_round_trip_is_not() {
    for n in 0..1_000u64 {
        let r = Natural::isqrt(n);
        assert!(r * r <= n, "isqrt({n}) = {r} too large");
        assert!((r + 1) * (r + 1) > n, "isqrt({n}) = {r} too small");
    }
    // Perfect squares.
    for r in 0..1_000u64 {
        assert_eq!(Natural::isqrt(r * r), r, "isqrt of a perfect square");
        if r > 0 {
            assert_eq!(Natural::isqrt(r * r - 1), r - 1, "just below");
        }
    }
    // Above 2^52 an f64 can no longer represent every integer, so a `sqrt` round trip stops
    // being exact. A root near 2^30 puts the square near 2^60 — past that threshold, and
    // still inside u64.
    for k in [30u32, 31] {
        let root = (1u64 << k) + 1;
        let n = root * root;
        assert!(
            n > (1u64 << 52),
            "n must exceed 2^52 for this to be the hard case"
        );
        assert_eq!(Natural::isqrt(n), root, "isqrt is exact at n = {n}");
        assert_eq!(Natural::isqrt(n - 1), root - 1, "and just below it");
    }
}

#[test]
fn signed_gcd_and_lcm_return_non_negative_representatives() {
    for a in -30..30i64 {
        for b in -30..30i64 {
            let g = Integer::gcd(a, b);
            assert!(g >= 0, "gcd({a},{b}) = {g} must be non-negative");
            if a != 0 || b != 0 {
                assert_eq!(a % g, 0);
                assert_eq!(b % g, 0);
            }
            let l = Integer::lcm(a, b);
            assert!(l >= 0, "lcm({a},{b}) = {l} must be non-negative");
        }
    }
    assert_eq!(Integer::gcd(-12i64, 18), 6);
    assert_eq!(Integer::gcd(12i64, -18), 6);
    assert_eq!(Integer::gcd(-12i64, -18), 6);
    assert_eq!(Integer::lcm(-4i64, 6), 12);
}

// --- the f32 carrier ----------------------------------------------------

#[test]
fn the_normal_family_works_on_f32() {
    assert_close_f32(Real::ndtr(0.0f32), 0.5, 8.0, "ndtr(0)");
    assert_close_f32(Real::ndtr(1.0f32), 0.841_344_746_068_542_9, 64.0, "ndtr(1)");
    assert_close_f32(Real::ndtri(0.975f32), 1.959_963_984_540_054, 512.0, "ndtri");
    // The f32 exponent range is far narrower, so log_ndtr matters sooner.
    assert_eq!(Real::ndtr(-20.0f32), 0.0, "Phi underflows in f32 here");
    let v = Real::log_ndtr(-20.0f32);
    assert!(v.is_finite() && v < 0.0, "log_ndtr(-20) = {v}");
    // Round trip in the tail.
    for p in [1e-10f32, 1e-20, 1e-30] {
        let x = Real::ndtri(p);
        assert!(x.is_finite() && x < 0.0, "ndtri({p}) = {x}");
        assert!((Real::log_ndtr(x) - p.ln()).abs() < 1e-3 * p.ln().abs());
    }
    // log10 / exp2 on f32.
    assert_eq!(Real::log10(1000.0f32), 3.0);
    assert_eq!(Real::exp2(10.0f32), 1024.0);
    // And the exact integer surface on the narrow widths.
    assert_eq!(Natural::gcd(12u32, 18), 6);
    assert_eq!(Natural::lcm(4u32, 6), 12);
    assert_eq!(Natural::isqrt(99u32), 9);
    assert_eq!(Integer::gcd(-12i32, 18), 6);
}
