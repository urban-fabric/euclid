//! Unit tests for the combinatorial host kernels.
//!
//! A child module in its own file so implementation plus tests stay inside the crate's
//! 400-line budget, while `use super::*` still reaches `binomial_exact`,
//! `binomial_saturating` and the `SMALL` crossover.
//!
//! Two of these exist because of a defect rather than a gap.
//! `exactness_holds_wherever_the_answer_is_representable` pins the off-by-one that an `f64`
//! accumulator produced inside the range the docs promised was exact, and
//! `the_f32_companion_switches_branches_at_small` pins which side of `SMALL` each argument
//! falls on — see MUTANTS.md.

use super::*;

#[test]
fn factorials_exact() {
    assert_eq!(factorial(0), 1.0);
    assert_eq!(factorial(5), 120.0);
    assert_eq!(factorial(10), 3_628_800.0);
}

#[test]
fn ln_factorial_matches() {
    for n in [0u64, 1, 5, 20, 25, 100] {
        let expect = (1..=n).map(|k| (k as f64).ln()).sum::<f64>();
        assert!((ln_factorial(n) - expect).abs() <= 1e-7 * expect.abs().max(1.0));
    }
}

#[test]
fn binomials() {
    assert_eq!(binomial(5, 2), 10.0);
    assert_eq!(binomial(10, 5), 252.0);
    assert_eq!(binomial(3, 4), 0.0);
    // Regression: the old `exp(lnΓ)` path returned 30957699535775 here, one short of
    // the true value (which is < 2^53, so the multiplicative form is exact).
    assert_eq!(binomial(48, 25), 30_957_699_535_776.0);
    assert_eq!(binomial(52, 5), 2_598_960.0);
    // Symmetry C(n, k) == C(n, n-k).
    assert_eq!(binomial(40, 3), binomial(40, 37));
}

/// **Which side of `SMALL` each argument falls on**, for the `u32 -> f32` pair.
///
/// `SMALL` is `21`, so `n = 20` takes the exact loop and `n = 21` takes `ln_gamma`. Nothing
/// was checking that, and four mutants of the two comparisons survived: moving the
/// crossover, or inverting it, went unnoticed because both branches are approximately
/// right everywhere.
///
/// Pinned by value rather than by an accuracy bound, and that is forced rather than lazy:
/// at `n = 21` the loop is *more* accurate than the branch actually taken, so any tolerance
/// loose enough to admit the real behaviour admits the mutant too. The only thing that
/// separates them is which value comes back.
///
/// # An accuracy observation, deliberately not acted on
///
/// Measured relative error against the true `n!`:
///
/// | `n` | branch taken | error | the other branch |
/// |---|---|---|---|
/// | 20 | exact loop | `6.2e-9` | — |
/// | 21 | `exp(ln_gamma)` | `4.3e-6` | loop: exact in `f32` |
/// | 22 | `exp(ln_gamma)` | `2.4e-6` | — |
///
/// `4.3e-6` is about thirty-six `f32` ulps for a value `f32` can hold to within one, so the
/// `f32` companion would be more accurate with a higher crossover. It is left alone on
/// purpose: `SMALL` is one constant shared with the `f64` path, where `20` is the documented
/// and correct cutoff, and splitting it is a design change rather than a test. Recorded here
/// so the next person to touch `SMALL` knows which direction the evidence points.
#[test]
fn the_f32_companion_switches_branches_at_small() {
    // `n = 20`: the exact loop. Both of these are the true value rounded to `f32`.
    assert_eq!(ln_factorial_u32(20), 42.335_617);
    assert_eq!(factorial_u32(20), 2.432_902e18);
    // `n = 21`: `ln_gamma`. The loop would give `45.380_138` and `5.109_094e19`.
    assert_eq!(ln_factorial_u32(21), 45.380_135);
    assert_eq!(factorial_u32(21), 5.109_072e19);
    // `n = 22`, well clear of the crossover.
    assert_eq!(ln_factorial_u32(22), 48.471_184);
    assert_eq!(factorial_u32(22), 1.124_003_5e21);
    // And the small end, where the loop is genuinely exact.
    assert_eq!(factorial_u32(0), 1.0);
    assert_eq!(factorial_u32(1), 1.0);
    assert_eq!(factorial_u32(12), 479_001_600.0);
    assert_eq!(ln_factorial_u32(0), 0.0);
    assert_eq!(ln_factorial_u32(1), 0.0);
}

/// Pascal's triangle up to row `rows`, in exact integers — the reference for the
/// exactness sweep below.
///
/// Addition only, where the kernel multiplies and divides. That independence is the
/// point: the off-by-one this test exists for came from the multiplicative recurrence,
/// and a reference sharing that recurrence agreed with it.
fn pascal(rows: usize) -> Vec<Vec<u128>> {
    let mut triangle = vec![vec![1_u128]];
    for n in 1..=rows {
        let previous = &triangle[n - 1];
        let mut row = vec![1_u128];
        for k in 1..n {
            row.push(previous[k - 1] + previous[k]);
        }
        row.push(1);
        triangle.push(row);
    }
    triangle
}

/// **The promise is exactness wherever the coefficient is representable, and it now
/// holds — it did not before.**
///
/// With an `f64` accumulator, twelve of the 1,840 coefficients at `n <= 60` whose true
/// value `f64` can hold exactly came back wrong, every one of them low by one. The
/// smallest is `C(56, 23) = 3_167_295_784_216_200`; the most quotable is
/// `C(56, 29) = 7_384_942_649_010_080`, which is only `2^52.7`.
///
/// The cause was not the range of the answer but the range of the *intermediate*:
/// `acc·(n-k+i)` is `i` times the partial it produces, so it crosses `2^53` a factor of
/// roughly `k` earlier than the result does. The old doc comment reasoned from the
/// partial being an integer — true, and not sufficient.
#[test]
fn exactness_holds_wherever_the_answer_is_representable() {
    let triangle = pascal(120);
    let mut checked = 0_u32;
    for (n, row) in triangle.iter().enumerate() {
        for (k, &exact) in row.iter().enumerate() {
            // The promise covers exactly those coefficients `f64` can hold; past that
            // it is "nearest representable", checked separately below.
            if (exact as f64) as u128 != exact {
                continue;
            }
            checked += 1;
            assert_eq!(binomial(n as u64, k as u64), exact as f64, "C({n}, {k})");
            assert_eq!(
                binomial_u128(n as u128, k as u128),
                exact as f64,
                "u128 C({n}, {k})"
            );
        }
    }
    // Guard the guard: a representability test that accidentally rejected everything
    // would leave this test green and vacuous.
    assert!(
        checked > 3_000,
        "only {checked} coefficients were actually exact"
    );

    // The specific values that regressed, as literals a reader can check by hand.
    assert_eq!(binomial(56, 29), 7_384_942_649_010_080.0);
    assert_eq!(binomial(56, 23), 3_167_295_784_216_200.0);
    assert_eq!(binomial(57, 22), 3_489_348_548_829_780.0);
}

/// Beyond `u128` the kernel falls back to the `f64` recurrence, and beyond `f64` it
/// saturates — the true behaviour of a coefficient that large, not an artefact.
#[test]
fn beyond_the_exact_range_it_degrades_rather_than_breaking() {
    // `C(200, 100)` is about `9.1e58`: past `u128`, comfortably inside `f64`. The true
    // value is 90548514656103281165404177077484163874504589675413336841320, whose
    // nearest `f64` is `9.054851465610328e58` — held to 1e-15 relative, which is the
    // most the `f64` recurrence can promise this far out.
    let wide = binomial(200, 100);
    assert!(wide.is_finite(), "C(200, 100) should be finite, got {wide}");
    assert!(
        (wide / 9.054_851_465_610_328e58 - 1.0).abs() < 1e-15,
        "C(200, 100) = {wide}",
    );
    // Far enough out that no `f64` holds it.
    assert_eq!(binomial(2_000, 1_000), f64::INFINITY);
    // The `f32` surface saturates correspondingly earlier.
    assert_eq!(binomial_u32(2_000, 1_000), f32::INFINITY);
}

/// The very top of the `u128` range, where `n` itself is `u128::MAX`.
///
/// `C(u128::MAX, 2)` is `MAX·(MAX−1)/2`, about `5.79e76` — unrepresentable as an integer in
/// the carrier that produced it, but perfectly ordinary as an `f64`. So this exercises the
/// path where [`binomial_exact`] gives up partway and [`binomial_saturating`] finishes the
/// job, at the one argument where `n` has no headroom at all.
///
/// It is also the only test that can distinguish `k.min(n - k)` from `k.min(n + k)`. The two
/// are the symmetric halves of the same coefficient and the integer recurrence returns the
/// same exact value for either, so no ordinary argument tells them apart — but at `n = MAX`
/// the addition overflows, and the mutant takes a different `k` into the recurrence and
/// returns `MAX` rather than `C(MAX, 2)`.
#[test]
fn the_widest_u128_arguments_fall_back_rather_than_wrapping() {
    // True value: 57896044618658097711785492504343953926124568782438874324533730092808912502785
    assert_eq!(binomial_u128(u128::MAX, 2), 5.789_604_461_865_81e76);
    // `C(n, 1) = n` and `C(n, 0) = 1` must hold even here.
    assert_eq!(binomial_u128(u128::MAX, 1), u128::MAX as f64);
    assert_eq!(binomial_u128(u128::MAX, 0), 1.0);
    assert_eq!(binomial_u128(u128::MAX, u128::MAX), 1.0);
    // And a shape that leaves `f64` entirely.
    assert_eq!(binomial_u128(u128::MAX, 100), f64::INFINITY);
}

/// The `u32 -> f32` companion makes the same promise in its own range, and it is a much
/// nearer bound: an `f32` accumulator rounds from `C(13, 6)` upward.
#[test]
fn the_f32_companion_is_exact_within_its_own_range() {
    let triangle = pascal(40);
    let mut checked = 0_u32;
    for (n, row) in triangle.iter().enumerate() {
        for (k, &exact) in row.iter().enumerate() {
            if (exact as f32) as u128 != exact {
                continue;
            }
            checked += 1;
            assert_eq!(
                binomial_u32(n as u32, k as u32),
                exact as f32,
                "C({n}, {k})"
            );
        }
    }
    assert!(
        checked > 300,
        "only {checked} coefficients were exact in f32"
    );
    // `C(26, 13) = 10_400_600` is where an `f32` accumulator first goes wrong, returning
    // 10_400_601 — the `f32` analogue of `C(56, 29)`, and well inside `2^24`.
    assert_eq!(binomial_u32(26, 13), 10_400_600.0);
    assert_eq!(binomial_u32(30, 15), 155_117_520.0);
}

#[test]
fn u32_factorials_are_native_f32() {
    assert_eq!(factorial_u32(0), 1.0);
    assert_eq!(factorial_u32(5), 120.0);
    assert_eq!(factorial_u32(10), 3_628_800.0);
    assert!((ln_factorial_u32(25) - ln_factorial(25) as f32).abs() < 1e-4);
}

#[test]
fn u32_binomials_are_native_f32() {
    assert_eq!(binomial_u32(5, 2), 10.0);
    assert_eq!(binomial_u32(10, 5), 252.0);
    assert_eq!(binomial_u32(3, 4), 0.0);
}
