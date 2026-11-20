//! The Maclaurin series about the origin, and the two constants that scale them.
//!
//! `Ai` and `Bi` are both combinations of the same pair of entire functions — the two solutions
//! of `y″ = xy` picked out by the initial conditions at the origin:
//!
//! ```text
//! f(x) = Σ 3^k (⅓)_k x^{3k}/(3k)!       f(0) = 1,  f′(0) = 0
//! g(x) = Σ 3^k (⅔)_k x^{3k+1}/(3k+1)!   g(0) = 0,  g′(0) = 1
//! ```
//!
//! so that
//!
//! ```text
//! Ai  = c₁f − c₂g        Bi  = √3(c₁f + c₂g)
//! Ai′ = c₁f′ − c₂g′      Bi′ = √3(c₁f′ + c₂g′)
//! ```
//!
//! with `c₁ = Ai(0) = 3^{−2/3}/Γ(⅔)` and `c₂ = −Ai′(0) = 3^{−1/3}/Γ(⅓)`.
//!
//! Neither Pochhammer symbol nor factorial is ever formed: dividing consecutive terms collapses
//! both into a two-factor denominator, leaving four plain recurrences in `x³`.
//!
//! ```text
//! a_k = a_{k−1}·x³/(3k(3k−1))         f  = Σ a_k,  a₀ = 1
//! b_k = b_{k−1}·x³/(3k(3k+1))         g  = Σ b_k,  b₀ = x
//! p_k = p_{k−1}·x³/((3k−1)(3k−3))     f′ = Σ p_k,  p₁ = x²/2   (k ≥ 2)
//! q_k = q_{k−1}·x³/(3k(3k−2))         g′ = Σ q_k,  q₀ = 1
//! ```
//!
//! # Why this is the small-argument branch only
//!
//! The series is *convergent everywhere*, which makes it tempting to use far out. For `Bi` on
//! the positive axis that is legitimate — `f`, `g`, `c₁` and `c₂` are all positive there, so
//! `c₁f + c₂g` adds without cancelling and the result stays at machine precision as far as the
//! terms stay in range (measured exact to `x = 40`).
//!
//! `Ai` is the opposite case, and it is the reason this module stops where it does. On the
//! positive axis `f` and `g` both grow like `e^{ζ}` with `ζ = ⅔x^{3/2}`, while their difference
//! `c₁f − c₂g` *decays* like `e^{−ζ}`. The subtraction therefore discards about `2ζ/ln 10`
//! digits, and measurement follows that prediction exactly:
//!
//! | `x` | 2 | 2.5 | 3 | 4 | 6 |
//! |-----|---|-----|---|---|---|
//! | relative error in `Ai` | `6e-16` | `1.8e-15` | `1e-13` | `5.9e-12` | `5.7e-8` |
//!
//! Summing the terms *exactly* does not help — tried, and it moves `x = 7` from `4.4e-12` to
//! `1.8e-12`. The loss is not in the accumulation but in the terms: each `a_k` carries the
//! roundings of the `k` multiplications that built it, and those absolute errors are already
//! `e^{ζ}`-sized relative to the answer. That is what makes this a hard floor rather than
//! something a compensated sum could lift, and why the positive axis switches to the `K`
//! route at [`super::SERIES_MAX`] instead.
//!
//! On the negative axis the same cancellation appears — the terms alternate — and
//! [`super::march`] takes over there.

use crate::math::arithmetic::term_is_negligible;

/// Iteration cap. The ratio of successive terms is `x³/(9k²)`, so at the largest argument this
/// module is asked for the loop exits in well under 20 passes; the cap exists only so a `NaN`
/// cannot spin forever.
const MAX_TERMS: u32 = 400;

/// Relative size at which a term stops contributing.
///
/// **`1e-19`, which is deliberately only a little below `f64`'s resolution.** `ε = 2.22e-16`, so
/// a term at `1e-19` of the sum moves the result by about a thousandth of an ulp, and every term
/// after it moves it by less. The value was `1e-20`, which asked the Maclaurin series to converge
/// one order past the point where the format can represent the difference —
/// iterations bought and then discarded.
///
/// Raising it to `1e-19` is **free rather than a trade.** The outputs are bit-identical across a
/// 306,280-point sweep covering every kernel in this tree at both signs, while contributing about **2%** to `Ai` and `Bi`, whose larger win came from sharing quadrature nodes instead.2%** and `Bi` by **2.0%**.
///
/// Timings are the minimum of six interleaved A/B runs of two prebuilt binaries pinned to idle
/// cores, which cancels drift — this machine is shared, and a naive before/after on it reported
/// swings of 2x on *identical* code.
///
/// `1e-18` was measured too, and is another 1-3% faster, but it is no longer bit-identical: it
/// differs by up to 3 ulp on `Bi`. So `1e-19` is the last value that costs nothing at all, and
/// the margin over `ε` is what pays for a series whose tail is amplified by `largest/sum` where
/// it cancels.
const TERM_EPS: f64 = 1e-19;

/// `(f, g, f′, g′)` at `x`, by the four recurrences above.
///
/// `√3`, narrowed from the single definition in [`crate::math::airy::SQRT_3`] so this format
/// cannot drift from it.
pub(super) const SQRT_3: f64 = crate::math::airy::SQRT_3;

/// Returned as the raw pair of solutions rather than as `Ai`/`Bi` so that [`super::march`] can
/// use them as its starting values: the march needs `(f, f′)` and `(g, g′)` as two independent
/// solutions of the same equation, not their Airy combinations.
pub(super) fn fg(x: f64) -> (f64, f64, f64, f64) {
    let x3 = x * x * x;
    let (mut a, mut b, mut p, mut q) = (1.0f64, x, x * x * 0.5, 1.0f64);
    let (mut f, mut g, mut f_prime, mut g_prime) = (a, b, p, q);
    for k in 1..MAX_TERMS {
        let k3 = 3.0 * f64::from(k);
        a *= x3 / (k3 * (k3 - 1.0));
        b *= x3 / (k3 * (k3 + 1.0));
        q *= x3 / (k3 * (k3 - 2.0));
        f += a;
        g += b;
        g_prime += q;
        // `p` starts at k = 1 with x²/2 rather than at k = 0, so its recurrence begins a step
        // later than the other three.
        if k >= 2 {
            p *= x3 / ((k3 - 1.0) * (k3 - 3.0));
            f_prime += p;
        }
        let largest = a.abs().max(b.abs()).max(p.abs()).max(q.abs());
        let scale = f.abs().max(g.abs()).max(f_prime.abs()).max(g_prime.abs());
        if term_is_negligible(largest, scale, TERM_EPS) {
            break;
        }
    }
    (f, g, f_prime, g_prime)
}

/// `(Ai, Ai′, Bi, Bi′)` at `x`, from the series. Accurate for `|x| ≤ `[`super::SERIES_MAX`];
/// see the module header for what happens beyond it.
pub(super) fn airy(x: f64) -> (f64, f64, f64, f64) {
    let (f, g, f_prime, g_prime) = fg(x);
    crate::math::airy::combine(f, g, f_prime, g_prime)
}
