//! The [`Natural`] tier: `u32`, `u64`, `u128`.
//!
//! See the [parent module](super) for why this exists, where the expectations come from, and
//! which two mutants here are unkillable.

use super::{agrees, pascal};
use crate::tests::*;
use core::fmt::Debug;

/// The whole of [`Natural`] at one carrier.
///
/// `tol` is the carrier's real surface: `u32` casts into `f32` and gets `1e-5`, the two wider
/// carriers cast into `f64` and get `1e-12`. The combinatorial arguments stay inside `f32`'s
/// exact-integer range (`10!` is `3_628_800`, under `2^24`) so the same assertions can run at
/// all three widths; [`super::super::kernels::host::factorial`] carries the wide cases.
fn natural_carrier_laws<N>(tol: f64, what: &str)
where
    N: Natural + PartialEq + Debug + Copy + Ord,
    N::Real: Real<Mask = bool> + Debug,
    N::Integer: PartialEq + Debug,
{
    let n = |x: u64| N::from_u64(x);

    // --- the identities, from their algebraic definitions ------------------------------
    //
    // `one` is checked as a *multiplicative identity* rather than against the literal `1`,
    // which is what makes `one() -> Default::default()` fail: zero is not one. The
    // corresponding `zero() -> Default::default()` is unkillable and excluded in
    // `.cargo/mutants.toml` — `Default` for an unsigned integer *is* zero.
    assert_eq!(n(7) + N::zero(), n(7), "{what}: x + 0 = x");
    assert_eq!(n(7) * N::one(), n(7), "{what}: x · 1 = x");
    assert_eq!(n(7) * N::zero(), N::zero(), "{what}: x · 0 = 0");
    assert_ne!(N::zero(), N::one(), "{what}: 0 ≠ 1");

    // --- the casts, against integer literals -------------------------------------------
    assert_eq!(
        n(7).to_integer(),
        N::Integer::from_i64(7),
        "{what}: to_integer"
    );
    assert_eq!(
        n(0).to_integer(),
        N::Integer::zero(),
        "{what}: to_integer(0)"
    );
    agrees(&n(7).to_real(), 7.0, tol, what, "to_real");
    agrees(&n(0).to_real(), 0.0, tol, what, "to_real(0)");

    // `from_real` rounds rather than truncates, so the half-way cases are the test: 7.6 is
    // nearer 8 and 7.4 is nearer 7. Truncation would send both to 7.
    assert_eq!(
        N::from_real(N::Real::scalar(7.6)),
        n(8),
        "{what}: from_real rounds up"
    );
    assert_eq!(
        N::from_real(N::Real::scalar(7.4)),
        n(7),
        "{what}: from_real rounds down"
    );
    assert_eq!(
        N::from_real(N::Real::scalar(0.0)),
        N::zero(),
        "{what}: from_real(0)"
    );

    // --- div_rem, against its own definition -------------------------------------------
    //
    // Checked structurally — `q · rhs + r == self` with `r < rhs` — so the law holds for
    // every pair rather than for the one triple a literal would pin.
    for a in [0_u64, 1, 7, 100, 1023] {
        for b in [1_u64, 2, 3, 7, 64] {
            let (q, r) = n(a).div_rem(n(b));
            assert_eq!(q * n(b) + r, n(a), "{what}: {a} = q·{b} + r");
            assert!(r < n(b), "{what}: remainder {r:?} < {b}");
        }
    }
    assert_eq!(n(7).div_rem(n(3)), (n(2), n(1)), "{what}: 7 = 2·3 + 1");

    // --- gcd, from Euclid's worked example ---------------------------------------------
    assert_eq!(
        N::gcd(n(1071), n(462)),
        n(21),
        "{what}: gcd(1071, 462) = 21"
    );
    assert_eq!(N::gcd(n(462), n(1071)), n(21), "{what}: gcd commutes");
    // The documented edges.
    assert_eq!(N::gcd(n(0), n(0)), N::zero(), "{what}: gcd(0, 0) = 0");
    assert_eq!(N::gcd(n(13), n(0)), n(13), "{what}: gcd(n, 0) = n");
    assert_eq!(N::gcd(n(0), n(13)), n(13), "{what}: gcd(0, n) = n");
    // Coprime, and a prime against a multiple of itself.
    assert_eq!(N::gcd(n(9), n(28)), N::one(), "{what}: gcd(9, 28) = 1");
    assert_eq!(N::gcd(n(17), n(51)), n(17), "{what}: gcd(17, 51) = 17");

    // --- lcm, and the theorem tying it to gcd ------------------------------------------
    assert_eq!(N::lcm(n(4), n(6)), n(12), "{what}: lcm(4, 6) = 12");
    assert_eq!(N::lcm(n(0), n(5)), N::zero(), "{what}: lcm(0, n) = 0");
    assert_eq!(N::lcm(n(5), n(0)), N::zero(), "{what}: lcm(n, 0) = 0");
    assert_eq!(N::lcm(n(0), n(0)), N::zero(), "{what}: lcm(0, 0) = 0");

    // `gcd(a, b) · lcm(a, b) = a · b`. A statement about both functions at once, provable
    // without reference to either implementation — so it cross-checks them rather than
    // restating them. It is also what catches a `lcm` that divides where it should multiply.
    for (a, b) in [
        (4_u64, 6_u64),
        (1071, 462),
        (9, 28),
        (17, 51),
        (36, 60),
        (1, 1),
    ] {
        assert_eq!(
            N::gcd(n(a), n(b)) * N::lcm(n(a), n(b)),
            n(a) * n(b),
            "{what}: gcd·lcm = a·b at ({a}, {b})",
        );
    }
    // Divisibility, the other half of what `gcd` means: it divides both, and `lcm` is
    // divisible by both.
    for (a, b) in [(1071_u64, 462_u64), (36, 60), (9, 28)] {
        let g = N::gcd(n(a), n(b));
        assert_eq!(n(a).div_rem(g).1, N::zero(), "{what}: gcd divides {a}");
        assert_eq!(n(b).div_rem(g).1, N::zero(), "{what}: gcd divides {b}");
        let l = N::lcm(n(a), n(b));
        assert_eq!(l.div_rem(n(a)).1, N::zero(), "{what}: {a} divides lcm");
        assert_eq!(l.div_rem(n(b)).1, N::zero(), "{what}: {b} divides lcm");
    }

    // --- isqrt, from its definition ----------------------------------------------------
    //
    // `r² ≤ n < (r+1)²` rather than a table of roots: the definition is the specification,
    // and checking it sweeps the perfect squares and their neighbours together.
    for x in [
        0_u64, 1, 2, 3, 4, 8, 9, 10, 15, 16, 17, 99, 100, 101, 1023, 1024, 1025,
    ] {
        let r = N::isqrt(n(x));
        assert!(r * r <= n(x), "{what}: isqrt({x})² ≤ {x}");
        assert!(
            n(x) < (r + N::one()) * (r + N::one()),
            "{what}: {x} < (isqrt+1)²"
        );
    }

    // --- the combinatorial four --------------------------------------------------------
    //
    // Values chosen to stay inside `f32`'s exact-integer range so all three carriers run
    // the same assertions.
    agrees(&n(0).factorial(), 1.0, tol, what, "0! = 1");
    agrees(&n(1).factorial(), 1.0, tol, what, "1! = 1");
    agrees(&n(5).factorial(), 120.0, tol, what, "5! = 120");
    agrees(&n(10).factorial(), 3_628_800.0, tol, what, "10! = 3628800");

    // `ln(n!) − ln((n−1)!) = ln n`, independent of how either is computed, and the only
    // assertion here that crosses the `n = 21` threshold where the exact sum gives way to
    // `ln_gamma`. A threshold nothing steps over is cause 3's "asymptotic branch".
    for m in [2_u64, 5, 20, 21, 22, 30] {
        let step = n(m).ln_factorial() - n(m - 1).ln_factorial();
        agrees(
            &step,
            (m as f64).ln(),
            tol,
            what,
            "ln(n!) − ln((n−1)!) = ln n",
        );
    }
    agrees(&n(0).ln_factorial(), 0.0, tol, what, "ln(0!) = 0");
    agrees(&n(1).ln_factorial(), 0.0, tol, what, "ln(1!) = 0");

    // Pascal's rule as the reference for `binomial`, plus the two edges and the symmetry.
    let triangle = pascal(20);
    for (row_index, row) in triangle.iter().enumerate() {
        for (k, &exact) in row.iter().enumerate() {
            agrees(
                &n(row_index as u64).binomial(&n(k as u64)),
                exact as f64,
                tol,
                what,
                "Pascal's rule",
            );
        }
    }
    agrees(&n(20).binomial(&n(0)), 1.0, tol, what, "C(n, 0) = 1");
    agrees(&n(20).binomial(&n(20)), 1.0, tol, what, "C(n, n) = 1");
    // Off the support: `k > n` is zero, and its logarithm is −∞.
    agrees(&n(3).binomial(&n(4)), 0.0, tol, what, "C(3, 4) = 0");
    assert!(
        n(3).ln_binomial(&n(4)).eq(&N::Real::neg_infinity()),
        "{what}: ln C(3, 4) = −∞",
    );

    // `ln C(n, k) = ln C(n, k)` via the coefficient itself, so the two agree where both are
    // defined. `C(n, n) = 1` makes `ln C(n, n) = 0`, which is what catches a `k > n` guard
    // relaxed to `k >= n`.
    agrees(&n(20).ln_binomial(&n(20)), 0.0, tol, what, "ln C(n, n) = 0");
    agrees(&n(20).ln_binomial(&n(0)), 0.0, tol, what, "ln C(n, 0) = 0");
    for (a, k) in [(5_u64, 2_u64), (10, 3), (10, 6), (20, 10), (17, 8)] {
        let exact = triangle[a as usize][k as usize] as f64;
        agrees(
            &n(a).ln_binomial(&n(k)),
            exact.ln(),
            tol,
            what,
            "ln C(n, k)",
        );
    }
}

#[test]
fn the_u32_carrier_honours_the_natural_contract() {
    // `f32`'s 24-bit significand sets the tolerance; the arguments are all inside its
    // exact-integer range, so this is headroom rather than a concession.
    natural_carrier_laws::<u32>(1e-5, "u32");
}

#[test]
fn the_u64_carrier_honours_the_natural_contract() {
    natural_carrier_laws::<u64>(1e-12, "u64");
}

#[test]
fn the_u128_carrier_honours_the_natural_contract() {
    natural_carrier_laws::<u128>(1e-12, "u128");
}

/// The three carriers must not merely each be *self-consistent* — they must agree with one
/// another, since [`Natural`] is one contract and a formula bounded on it is expected to give
/// the same answer whichever width a caller picks.
///
/// This is the assertion `u128`'s hand-written combinatorial copies failed: it differenced
/// `ln_binomial` straight off `ln_gamma` where `u64` used the exact small sum, so the two
/// disagreed in the last digits across the whole small-argument range.
#[test]
fn the_natural_carriers_agree_with_one_another() {
    for m in [0_u64, 1, 2, 5, 10, 20, 21, 25, 40] {
        assert_eq!(
            m.factorial(),
            (m as u128).factorial(),
            "u64 and u128 disagree on {m}!",
        );
        assert_eq!(
            m.ln_factorial(),
            (m as u128).ln_factorial(),
            "u64 and u128 disagree on ln({m}!)",
        );
    }
    for (a, k) in [
        (5_u64, 2_u64),
        (20, 10),
        (48, 25),
        (56, 29),
        (60, 30),
        (3, 4),
    ] {
        assert_eq!(
            a.binomial(&k),
            (a as u128).binomial(&(k as u128)),
            "u64 and u128 disagree on C({a}, {k})",
        );
        assert_eq!(
            a.ln_binomial(&k),
            (a as u128).ln_binomial(&(k as u128)),
            "u64 and u128 disagree on ln C({a}, {k})",
        );
    }
    // The `u32 -> f32` carrier is a different precision, so it is held to agreement after
    // rounding rather than bit-for-bit.
    for (a, k) in [(5_u32, 2_u32), (20, 10), (17, 8)] {
        assert_eq!(
            a.binomial(&k),
            (u64::from(a)).binomial(&u64::from(k)) as f32,
            "u32 and u64 disagree on C({a}, {k})",
        );
    }
}

/// `isqrt` is exact where the obvious `to_real().sqrt()` round trip is not — the claim its
/// doc comment makes, and the reason it exists at all rather than being left to the caller.
///
/// Above `2^52` consecutive integers are no longer distinguishable in `f64`, so the round trip
/// cannot tell `n` from `n+1` and lands on the wrong side of a perfect square. Only the two
/// wide carriers can hold these arguments.
#[test]
fn isqrt_is_exact_where_the_float_round_trip_is_not() {
    for r in [67_108_864_u64, 94_906_265, 1_000_000_007] {
        // `r²` and `r² − 1` straddle a perfect square far above `2^52`.
        let square = r * r;
        assert_eq!(Natural::isqrt(square), r, "isqrt({square}) = {r}");
        assert_eq!(
            Natural::isqrt(square - 1),
            r - 1,
            "isqrt({square} − 1) = {}",
            r - 1
        );
        assert_eq!(
            Natural::isqrt(square as u128),
            r as u128,
            "u128: isqrt({square})"
        );
        assert_eq!(
            Natural::isqrt(square as u128 - 1),
            r as u128 - 1,
            "u128: isqrt({square} − 1)",
        );
    }
    // The `u128` range proper, past anything `f64` can enumerate.
    let wide = 1_u128 << 100;
    assert_eq!(Natural::isqrt(wide), 1_u128 << 50, "isqrt(2^100) = 2^50");
    assert_eq!(
        Natural::isqrt(wide - 1),
        (1_u128 << 50) - 1,
        "isqrt(2^100 − 1)"
    );
}

/// `lcm` overflows for a genuinely unrepresentable answer, and *only* then — the
/// divide-before-multiply in its body exists so that a representable answer never overflows
/// on the way to itself.
#[test]
fn lcm_reaches_answers_a_naive_product_would_overflow() {
    // `lcm(2^63, 2^62) = 2^63`, representable; `a · b` is not.
    let a = 1_u64 << 63;
    let b = 1_u64 << 62;
    assert_eq!(Natural::lcm(a, b), a, "lcm(2^63, 2^62) = 2^63");
    assert_eq!(a.checked_mul(b), None, "the naive product overflows");

    let a = 1_u128 << 127;
    let b = 1_u128 << 126;
    assert_eq!(Natural::lcm(a, b), a, "u128: lcm(2^127, 2^126) = 2^127");
    assert_eq!(a.checked_mul(b), None, "u128: the naive product overflows");
}
