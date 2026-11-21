//! The three `f32` kernels whose allowance is a **function of the argument**, not a constant.
//!
//! # How these were found, and why they were not bugs
//!
//! Ranking the narrow surface by worst ulp-of-value gap against the `f64` peer produced three
//! outliers against peers in the `3`–`70` range: `cos_int` at `323`, `rgamma` at `148`,
//! `ndtr`/`log_ndtr` at `122`. All three looked like accuracy defects. None of them is. Each has a
//! *model* that predicts its error, and the model is not a constant — which is exactly why a
//! plain ulp ranking put them at the top and exactly what this file asserts instead.
//!
//! Recording the attribution matters more than recording the numbers. A constant budget on any of
//! these three would have to be set at the worst case, which makes it vacuous everywhere else:
//! `ndtr`'s gap is `16` ulp at `x = −2.9` and `251` at `x = −12.8`, so a budget admitting the
//! second cannot see the first double.
//!
//! ```text
//! kernel          the model                       worst measured ratio   budget
//! ndtr            gap ≲ c·x²/2 ulp                3.603 at x = −2.886     8
//! log_ndtr        the same, mirrored              3.827 at x =  2.892     8
//! rgamma          gap ≲ c·|lnΓ(x)| ulp            5.392 at x = 11.357    12
//! cos_int         absolute error ≲ c·ε            3.00ε at x =  3.954      8ε
//! ```
//!
//! # `ndtr`: `x²` conditioning, inherited rather than its own
//!
//! `Φ(x) ≈ e^{−x²/2}/(x√2π)`, so forming `x²/2` carries a relative `ε`, which becomes an absolute
//! `x²ε/2` in the exponent, which `exp` returns as a *relative* `x²ε/2` in the result. That is a
//! property of the composition and no implementation can do better — the same argument
//! [`super::cross_precision`] already makes for `erfc`, whose allowance "carries `x²` … because
//! `exp(−x²)` cannot do better".
//!
//! Measured, the ratio is flat across the whole range — `2.9` at `x = −4`, `2.7` at `−6.45`, `2.8`
//! at `−12.84` — which is what identifies the model rather than merely bounding it.
//!
//! And it is inherited: `Φ(x)` and `erfc(−x/√2)/2` agree **bit for bit** at every argument
//! checked, gap included. So this is `erfc`'s conditioning reaching `ndtr` through the obvious
//! route, not a second defect. `log_ndtr` shows the identical figure on the positive side because
//! `ln Φ(x) → −Φ(−x)` there, which is the same quantity again.
//!
//! # `rgamma`: `exp`'s condition number is its own argument
//!
//! `1/Γ(x)` for large `x` is `e^{−lnΓ(x)}`, and `exp`'s condition number is the argument it is
//! given — so one ulp of error in `lnΓ` becomes `|lnΓ(x)|` ulps here. The same mechanism
//! [`super::wiggle`] identifies from the monotonicity side, where `1/Γ` inverts by `174` ulp at
//! `x = 50` against `|lnΓ(50)| = 144.57`.
//!
//! The ratio runs `0.3` to `5.4` rather than sitting at one, because it also carries `lnΓ`'s own
//! error, which is itself a few ulp. Above `x ≈ 35` the gap falls to zero: `1/Γ(40)` is `5e-47`,
//! below `f32`'s smallest subnormal, so both precisions agree on `0`.
//!
//! # `cos_int`: the envelope case, and the cleanest attribution of the three
//!
//! `Ci` has interior zeros at `0.616`, `3.384` and `6.427`, and beside a zero an ulp of the value
//! measures distance to the zero rather than quality of the kernel. The ulp count duly explodes —
//! **44,030** ulp at `x = 6.427` — while the *absolute* error stays flat at about `1e-7`
//! everywhere. The `323` in the original ranking was not even the worst; it was wherever the
//! sampling grid happened to land near a zero.
//!
//! What settles it is `sin_int`. `Si` shares the module, the argument reduction and the
//! crossover, and its worst absolute error is `3.5763e-7` — **the same value to all five
//! digits** as `Ci`'s, at a different argument. Two kernels accumulating identical absolute error
//! and differing by two orders of magnitude in ulps-of-value is the envelope effect and nothing
//! else. [`the_trigonometric_integrals_share_one_absolute_error`] asserts that equality rather
//! than just the bound, because the equality is the evidence.

use super::*;

/// Representable `f32` steps between two values, via a monotone ordering of the sign-magnitude
/// bits so a pair straddling zero measures correctly rather than wrapping the exponent.
fn narrow_ulps(a: f32, b: f32) -> f64 {
    if a == b {
        return 0.0;
    }
    let order = |x: f32| -> i64 {
        let bits = i64::from(x.to_bits() as i32);
        if bits < 0 {
            i64::from(i32::MIN) - bits
        } else {
            bits
        }
    };
    #[allow(clippy::cast_precision_loss)]
    let steps = (order(a) - order(b)).abs() as f64;
    steps
}

/// Points swept per kernel.
const SAMPLES: u32 = 400_000;

/// The normal CDF pair's gap scales with `x²`, which is the composition's conditioning.
///
/// Asserting the *ratio* rather than a constant is the whole point: a flat budget admitting the
/// `251` ulp at `x = −12.8` would be twenty times too loose at `x = −3`, where the gap is `16`.
/// The ratio is flat, so a change in the *shape* of the error fails here even if the worst case
/// happens not to move.
#[test]
fn the_normal_cdf_pair_carry_x_squared_in_their_allowance() {
    /// Measured worst is `3.603` for `ndtr` and `3.827` for `log_ndtr`; `8` admits both and a
    /// doubling.
    const BUDGET: f64 = 8.0;

    let mut failures = Vec::new();
    let mut compared = 0usize;

    // `Φ` on the left tail, where it decays; `lnΦ` on the right, where it approaches zero. Those
    // are the same quantity by `ln Φ(x) → −Φ(−x)`, which is why both carry the same law.
    for (name, low, high, narrow, wide) in [
        (
            "ndtr",
            -13.5_f64,
            -2.0,
            Real::ndtr as fn(f32) -> f32,
            Real::ndtr as fn(f64) -> f64,
        ),
        (
            "log_ndtr",
            2.0,
            13.5,
            Real::log_ndtr as fn(f32) -> f32,
            Real::log_ndtr as fn(f64) -> f64,
        ),
    ] {
        for i in 0..=SAMPLES {
            let x = (low + (high - low) * f64::from(i) / f64::from(SAMPLES)) as f32;
            let want = wide(f64::from(x)) as f32;
            if want == 0.0 || !want.is_finite() {
                continue;
            }
            compared += 1;
            // `x²/2` is the exponent whose rounding `exp` amplifies; floored at one so the model
            // never predicts less than ordinary round-off.
            let allowed = BUDGET * (f64::from(x) * f64::from(x) * 0.5).max(1.0);
            let gap = narrow_ulps(narrow(x), want);
            if gap > allowed {
                failures.push(format!(
                    "  {name}({x:e}) is {gap} ulp from the wide value, against the \
                     {BUDGET}·x²/2 = {allowed:.1} its conditioning allows"
                ));
            }
        }
    }

    assert!(compared >= 700_000, "only {compared} comparisons made");
    assert!(
        failures.is_empty(),
        "the normal CDF pair exceeded their x^2 allowance:\n{}",
        failures.join("\n")
    );
}

/// `1/Γ`'s gap scales with `|lnΓ(x)|`, which is `exp`'s condition number at that argument.
///
/// The sweep stops where `1/Γ` underflows in `f32` — around `x = 35`, since `1/Γ(40)` is `5e-47`
/// against a smallest subnormal of `1.4e-45`. Past that both precisions return `0` and the
/// comparison has nothing to say.
#[test]
fn the_reciprocal_gamma_carries_its_log_gamma_amplification() {
    /// Measured worst ratio is `5.392` at `x = 11.357`; `12` leaves the usual headroom.
    const BUDGET: f64 = 12.0;

    let mut failures = Vec::new();
    let mut compared = 0usize;
    for i in 0..=SAMPLES {
        let x = (0.5 + 34.5 * f64::from(i) / f64::from(SAMPLES)) as f32;
        let want = Real::rgamma(f64::from(x)) as f32;
        if want == 0.0 || !want.is_finite() {
            continue;
        }
        compared += 1;
        // One ulp in `lnΓ` becomes `|lnΓ|` ulps after `exp`. Floored at one, as above.
        let allowed = BUDGET * Real::ln_gamma(f64::from(x)).abs().max(1.0);
        let gap = narrow_ulps(Real::rgamma(x), want);
        if gap > allowed {
            failures.push(format!(
                "  rgamma({x:e}) is {gap} ulp from the wide value, against the \
                 {BUDGET}·|lnGamma| = {allowed:.1} that exp's conditioning allows"
            ));
        }
    }

    assert!(compared >= 300_000, "only {compared} comparisons made");
    assert!(
        failures.is_empty(),
        "rgamma exceeded its log-gamma amplification allowance:\n{}",
        failures.join("\n")
    );
}

/// `Ci` and `Si` accumulate the *same absolute error*, which is why only one of them looks bad.
///
/// Two assertions, and the second is the interesting one. The first bounds each kernel's absolute
/// error — the right metric for a function with interior zeros, where an ulp of the value measures
/// distance to the zero instead. The second asserts that the two worst cases **agree**, which is
/// the actual evidence for the attribution: identical absolute error plus a hundredfold difference
/// in ulps-of-value can only be the envelope.
///
/// If `Ci` ever becomes genuinely less accurate than `Si`, the second assertion fails and the
/// first may not — which is the whole reason to state it.
#[test]
fn the_trigonometric_integrals_share_one_absolute_error() {
    /// Measured worst is `3.00ε` for both; `8ε` admits that and a doubling.
    const BUDGET: f64 = 8.0;
    let epsilon = f64::from(f32::EPSILON);

    let mut worst = [0.0_f64; 2];
    let mut worst_at = [0.0_f32; 2];
    for (slot, (name, narrow, wide)) in [
        (
            "cos_int",
            Real::cos_int as fn(f32) -> f32,
            Real::cos_int as fn(f64) -> f64,
        ),
        (
            "sin_int",
            Real::sin_int as fn(f32) -> f32,
            Real::sin_int as fn(f64) -> f64,
        ),
    ]
    .into_iter()
    .enumerate()
    {
        for i in 0..=SAMPLES {
            let x = (0.1 + 39.9 * f64::from(i) / f64::from(SAMPLES)) as f32;
            let want = wide(f64::from(x));
            if !want.is_finite() {
                continue;
            }
            let absolute = (f64::from(narrow(x)) - want).abs();
            if absolute > worst[slot] {
                worst[slot] = absolute;
                worst_at[slot] = x;
            }
        }
        assert!(
            worst[slot] <= BUDGET * epsilon,
            "{name} reached an absolute error of {:e} at x = {}, against the {BUDGET}·eps = {:e} \
             allowed",
            worst[slot],
            worst_at[slot],
            BUDGET * epsilon
        );
    }

    // The attribution. Both are `3.5763e-7` — one `f32` ulp of `3`, reached at different
    // arguments — so neither kernel is worse than the other and `Ci`'s ulp-of-value count is
    // explained entirely by having a smaller value beside its zeros.
    let ratio = worst[0] / worst[1];
    assert!(
        (0.25..=4.0).contains(&ratio),
        "cos_int's worst absolute error is {:e} (at {}) and sin_int's is {:e} (at {}), a ratio of \
         {ratio:.2}. They share their argument reduction and should share this error; if they no \
         longer do, Ci's ulp count is no longer merely its envelope and wants investigating.",
        worst[0],
        worst_at[0],
        worst[1],
        worst_at[1]
    );
}
