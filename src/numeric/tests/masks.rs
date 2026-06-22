use super::*;

/// A write-once piecewise kernel over the mask tier: `relu(x) = select(x > 0, x, 0)`, a
/// comparison to a [`Compare::Mask`] driving a branchless `select` with no scalar `bool`.
fn relu<T: Compare>(x: T, zero: T) -> T {
    let positive = x.gt(&zero);
    T::select(&positive, x, zero)
}

#[test]
fn lazy_bool_mask_drives_select_and_logic() {
    // relu on both a real and an integer carrier, write-once.
    assert_eq!(relu(3.0_f64, 0.0), 3.0);
    assert_eq!(relu(-2.0_f64, 0.0), 0.0);
    assert_eq!(relu(5_i32, 0), 5);
    assert_eq!(relu(-7_i32, 0), 0);
    assert_eq!(relu(5_i64, 0), 5);
    assert_eq!(relu(-7_i64, 0), 0);

    // Naturals are ordered carriers too: min(4, 7) = 4 via comparison + select.
    let lt = Compare::lt(&4_u64, &7_u64);
    assert_eq!(u64::select(&lt, 4, 7), 4);

    // The Boolean algebra on the masks (trait-qualified to avoid std's `PartialOrd`).
    let t = Compare::lt(&2.0_f64, &3.0); // true
    let f = Compare::gt(&2.0_f64, &3.0); // false
    assert!(Boolean::or(&t, &f));
    assert!(!Boolean::and(&t, &f));
    assert!(Boolean::xor(&t, &f));
    assert!(Boolean::not(&f));
}

#[test]
fn eager_integer_masks_drive_host_while_loop() {
    let signed = <i32 as Compare>::while_loop(0_i32, |n| Compare::lt(n, &5), |n| n + 1);
    assert_eq!(signed, 5);

    let natural = <u64 as Compare>::while_loop(1_u64, |n| Compare::lt(n, &16), |n| n * 2);
    assert_eq!(natural, 16);
}

/// `if`/`then`/`else` over the real surface *alone*. Because `Real: Compare`, a
/// piecewise kernel needs no separate `+ Compare` bound: `signed_clip(x) = if |x| > 1
/// { sign(x) } else { x }` — the branchless saturation a truncated density reaches for — is
/// write-once over [`Real`], using `abs`, comparison, and `select`.
fn signed_clip<T: Real>(x: T) -> T {
    let outside = x.abs().gt(&T::one());
    T::if_then_else(&outside, x.clone().signum(), x)
}

#[test]
fn lazy_real_carries_comparison_and_if_then_else() {
    // Control flow reached through the `Real` bound alone, on every real tier.
    assert_eq!(signed_clip(0.4_f64), 0.4);
    assert_eq!(signed_clip(2.5_f64), 1.0);
    assert_eq!(signed_clip(-3.0_f32), -1.0);

    // The six comparisons return the mask tier; `select` / `if_then_else` consume it.
    assert!(Compare::ge(&2.0_f64, &1.0));
    assert_eq!(
        f64::if_then_else(&Compare::lt(&1.0_f64, &2.0), 10.0, 20.0),
        10.0
    );
    assert_eq!(
        <f32 as Compare>::select(&Compare::gt(&1.0_f32, &2.0), 10.0, 20.0),
        20.0
    );
}

/// Scrub `NaN` to a fallback — a `Real`-generic kernel composing the `is_nan` predicate
/// with `if_then_else`, the shape a guarded log-density uses outside its support.
fn scrub_nan<T: Real>(x: T, fallback: T) -> T {
    T::if_then_else(&x.is_nan(), fallback, x)
}

#[test]
fn lazy_real_non_finite_sentinels_and_predicates() {
    // The sentinels lift through `scalar` on every real tier.
    assert_eq!(f64::infinity(), f64::INFINITY);
    assert_eq!(f64::neg_infinity(), f64::NEG_INFINITY);
    assert_eq!(f32::infinity(), f32::INFINITY);
    assert!(f64::nan().is_nan());

    // Trait-qualified calls bypass the inherent `f64::is_nan` etc. and exercise the
    // `Real` classifier surface.
    assert!(Real::is_infinite(&f64::infinity()));
    assert!(Real::is_infinite(&f64::neg_infinity()));
    assert!(Real::is_nan(&f64::nan()));
    assert!(!Real::is_finite(&f64::nan()));
    assert!(!Real::is_finite(&f64::infinity()));
    assert!(Real::is_finite(&3.5_f64));
    assert!(!Real::is_nan(&3.5_f64));
    assert!(Real::is_nan(&f32::nan()));

    // The predicates compose with the control-flow surface, write-once over `Real`.
    assert_eq!(scrub_nan(f64::nan(), 0.0), 0.0);
    assert_eq!(scrub_nan(7.0_f64, 0.0), 7.0);
    assert_eq!(scrub_nan(f32::nan(), -1.0), -1.0);
}

#[test]
fn lazy_real_non_finite_recovery_composites() {
    // `nan_to_num` scrubs NaN but preserves an intended ±∞.
    assert_eq!(f64::nan().nan_to_num(0.0), 0.0);
    assert_eq!(5.0_f64.nan_to_num(0.0), 5.0);
    assert_eq!(f64::infinity().nan_to_num(0.0), f64::INFINITY);

    // `finite_or` forces finiteness, replacing NaN *and* ±∞.
    assert_eq!(f64::nan().finite_or(-1.0), -1.0);
    assert_eq!(f64::infinity().finite_or(-1.0), -1.0);
    assert_eq!(f64::neg_infinity().finite_or(-1.0), -1.0);
    assert_eq!(3.0_f64.finite_or(-1.0), 3.0);

    // `mask_invalid` is the out-of-support output guard: keep where valid, else fill.
    assert_eq!(
        7.0_f64.mask_invalid(false, f64::neg_infinity()),
        f64::NEG_INFINITY
    );
    assert_eq!(7.0_f64.mask_invalid(true, f64::neg_infinity()), 7.0);

    // The checkify hook is the identity on a finite value (and is erased in release).
    assert_eq!(2.0_f64.debug_assert_finite(), 2.0);
    assert_eq!(2.0_f32.debug_assert_finite(), 2.0);
}
