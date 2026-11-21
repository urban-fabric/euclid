//! Does each kernel stay as accurate as it was yesterday, against the best bound available?
//!
//! Two halves, which share the machinery and differ in what they can compare against. The `f32`
//! kernels have a wider peer, so their bound is the wide result rounded to `f32`. The `f64`
//! **inverses** have no wider peer and no closed form, so their bound is the *backward* error —
//! substitute the answer back through the forward function. Both are ratios against an
//! admissible bound, so a budget can hold either.
//!
//! This is the searched, budgeted complement to the fixed sweeps in
//! `src/tests/identities/cross_precision`. Those are the right place for a published per-kernel
//! ulp figure and they have two blind spots, both of which this file exists to close.
//!
//! **A grid finds the worst case on the grid.** `agrees_to!` walks 400 evenly-spaced arguments
//! and asserts each one. A narrow kernel's worst input is characteristically a thin region — one
//! side of a crossover, the last argument before a series is abandoned, the place where a
//! coefficient set fitted at `f64` stops being the right polynomial — and 400 linear steps step
//! over thin regions. Every property here hill-climbs instead, on the gap itself, so the search
//! is pulled towards the bad argument rather than hoping to land beside it.
//!
//! **A grid discards its answer.** `agrees_to!` computes a gap, compares it to a budget, and
//! drops it. So the suite can see a kernel breaching its stated ulp figure and cannot see it
//! using nine tenths of it where it used to use a tenth. That is the regression that matters
//! most here, because the `f32` tree was rewritten from narrowing `f64` to native evaluation: a
//! native kernel that quietly gets worse still satisfies every identity and every ulp contract
//! written with headroom. [`target`] plus `.budget()` is the only instrument in the crate that
//! notices.
//!
//! # The bound, and why it is not a tolerance
//!
//! Comparing an `f32` kernel against `f64` needs a bound that does not charge it for being
//! narrow. The wide result **rounded to `f32`** is the best any `f32` implementation could
//! return, so the gap is measured in `f32` ulps against that, which is what
//! [`relation::precision_agrees`] asserts. The wide kernel is evaluated at the *narrowed* input
//! so the input's own rounding is not counted against the narrow kernel — in a steep region that
//! term dominates everything else.
//!
//! The score fed to the climb is that gap as a **fraction of the kernel's allowed ulps**, so
//! `1.0` is exactly at the contract and the number is comparable across kernels of wildly
//! different conditioning. `ALLOWED_*` is the contract, written with headroom; the budget beside
//! it is the measurement. Neither subsumes the other, and a breach of the first is a bug while a
//! rise in the second is a regression.
//!
//! # How the numbers here were measured
//!
//! Every `ALLOWED_*` records a worst gap found over **ten seeds at `EUCLID_STABILITY_SOAK=100`**,
//! which is six million cases per kernel, with the budget temporarily detached so a baseline
//! trip could not mask a later seed's measurement. The contract is then set to roughly four
//! times that, so it admits today's error and a worse one — which is what makes a breach of it
//! news rather than noise.
//!
//! Six kernels do not use a plain ulp-of-value count, and each says why at its own definition
//! rather than here. `erfc` carries `x²` in its allowance, because `exp(−x²)` cannot do better.
//! `J₀`, `Ai`, `lnΓ`, `ψ` and `Ei` count against a stated envelope, because each has an interior
//! zero where an ulp of the value measures distance to the zero instead of quality of the kernel.
//! Only two of those five were caught by a failing run; `lnΓ` and `ψ` were latent flakes, passing
//! only because this seed's climb had not yet reached their zero.
//!
//! The baselines in `tests/budgets.txt` are recorded at the default seed and `SOAK=100`, per the
//! convention in CONTRIBUTING. A different seed or a deeper soak will legitimately trip them;
//! what must not move is the contract.
//!
//! Set `EUCLID_STABILITY_STATS=1` to see the worst fraction each climb reached.

use euclid_scalar::Real;
use euclid_stability::{Config, check, classify, generate, property, relation, target};

/// Representable `f32` steps between `a` and `b`, or `None` if either is non-finite.
///
/// The engine keeps this private to `precision_agrees`, which asserts rather than scores — so
/// the measurement has to be reconstructed here to be hill-climbed on. Sign-magnitude bits are
/// mapped to a monotone ordering so a pair straddling zero measures correctly rather than
/// wrapping through the exponent field.
#[allow(clippy::cast_possible_wrap)]
fn f32_ulps_between(a: f32, b: f32) -> Option<u64> {
    if !a.is_finite() || !b.is_finite() {
        return None;
    }
    let order = |x: f32| -> i64 {
        let bits = i64::from(x.to_bits() as i32);
        if bits < 0 {
            i64::from(i32::MIN) - bits
        } else {
            bits
        }
    };
    Some(order(a).abs_diff(order(b)))
}

/// Whether an ulp count against this value means anything.
///
/// **An ulp is a relative unit only in the normal range.** Below `f32::MIN_POSITIVE` the spacing
/// stops halving and becomes the fixed absolute `2^-149`, so an ulp count against a subnormal
/// measures how far into the subnormals the value is rather than how good the kernel is.
///
/// `erfc` over `-3..10` is the case that made this explicit: `erfc(9.2) = 1.1e-38` is already
/// subnormal in `f32` and `erfc(10) = 2.1e-45` has barely two significant bits left, so an ulp
/// count out there is counting subnormal granularity.
///
/// Excluding them is necessary and was **not** sufficient — the same kernel still scored 63 ulp
/// afterwards, from `x ≈ 9.15`, which is in the last normal binade rather than the subnormals.
/// That residue is the conditioning of `exp(−x²)` and is handled separately; see
/// [`the_f32_erfc_tracks_f64`]. Recording the distinction because the first draft of this file
/// assumed the subnormals explained the whole number, and they did not.
///
/// Zero is excluded on the same grounds, which costs nothing: the exact zeros of these kernels
/// are pinned by the endpoint and symmetry suites.
fn ulp_count_is_meaningful(best: f32) -> bool {
    best.is_finite() && best.abs() >= f32::MIN_POSITIVE
}

/// How much of its allowed narrowing error the `f32` kernel used at `x`, as a fraction.
///
/// `None` where the comparison is undefined or meaningless: either result non-finite, which
/// happens at the ends of several of these domains and is the wide kernel's business rather than
/// a narrowing failure, or a subnormal reference — see [`ulp_count_is_meaningful`].
#[allow(clippy::cast_possible_truncation)]
fn narrowing_margin(
    x: f32,
    wide: impl Fn(f64) -> f64,
    narrow: impl Fn(f32) -> f32,
    allowed: u32,
) -> Option<f64> {
    debug_assert!(allowed > 0);
    // The wide kernel at the *narrowed* input, rounded back: the best `f32` could do.
    let best = wide(f64::from(x)) as f32;
    if !ulp_count_is_meaningful(best) {
        return None;
    }
    let got = narrow(x);
    let gap = f32_ulps_between(got, best)?;
    Some(gap as f64 / f64::from(allowed))
}

/// The `f32` ulp at `v`: the distance to the next representable magnitude.
fn ulp_of(v: f32) -> f32 {
    let v = v.abs();
    if v == 0.0 || !v.is_finite() {
        return f32::from_bits(1);
    }
    f32::from_bits(v.to_bits() + 1) - v
}

/// The same measurement, but in ulps of a stated **scale** rather than of the value.
///
/// An ulp comparison against the value is the wrong instrument beside a zero, and an oscillating
/// kernel is nothing but zeros. `Ai` has a simple zero at `−9.0227`; at `x = −9.0533` the value
/// has fallen to `3.0e-2` while the quantities the kernel actually combines are still the size of
/// the local amplitude, about `0.33`. So the absolute error stays at `ε·amplitude` while
/// `ulp(value)` shrinks with the value, and the quotient diverges — the first run of this suite
/// reported `263 ulp` there for an absolute error of `4.9e-7`, which is **17 ulps of the
/// amplitude** and entirely ordinary.
///
/// Counting against the amplitude is therefore not a looser bound, it is the bound that does not
/// change meaning as the function passes through zero. The envelope is the standard asymptotic
/// one for each kernel, stated at the call site, and the value itself is used wherever it
/// exceeds the envelope so this can never be the weaker of the two.
fn narrowing_margin_at_scale(
    x: f32,
    wide: impl Fn(f64) -> f64,
    narrow: impl Fn(f32) -> f32,
    allowed: u32,
    envelope: f32,
) -> Option<f64> {
    debug_assert!(allowed > 0);
    #[allow(clippy::cast_possible_truncation)]
    let best = wide(f64::from(x)) as f32;
    let got = narrow(x);
    if !best.is_finite() || !got.is_finite() {
        return None;
    }
    // The *scale* is what the ulp is taken at here, so it is the scale that has to be normal —
    // `best` itself is allowed to be subnormal, or zero, which is the whole point beside a zero.
    let scale = envelope.abs().max(best.abs());
    if !ulp_count_is_meaningful(scale) {
        return None;
    }
    let gap = (f64::from(got) - f64::from(best)).abs();
    let unit = f64::from(allowed) * f64::from(ulp_of(scale));
    (unit > 0.0).then(|| gap / unit)
}

fn settings(suite: &str) -> Config {
    Config::default()
        .cases(600)
        .regressions(format!("tests/regressions/{suite}.txt"))
        .budget("tests/budgets.txt")
}

/// Assert the contract and score the measurement, for a one-argument kernel.
///
/// Both halves every time: `precision_agrees` is what a caller may rely on, `target` is what
/// notices the kernel drifting inside it.
macro_rules! tracks_f64 {
    ($name:literal, $x:expr, $allowed:expr, $wide:expr, $narrow:expr) => {{
        let Some(margin) = narrowing_margin($x, $wide, $narrow, $allowed) else {
            return Ok(());
        };
        target(margin);
        relation::precision_agrees($name, f64::from($x), $wide, $narrow, $allowed)?;
    }};
}

/// The conditioned peer of [`tracks_f64`]: an allowance that grows with the problem.
///
/// For a kernel whose achievable accuracy provably degrades with the argument, a flat ulp
/// contract is the wrong shape — it is either too tight where the problem is hard or vacuous
/// where it is easy. `allowed` is evaluated per argument, so the claim becomes "no worse than
/// the problem itself" rather than "within a number I picked", which is the same move
/// `relation::forward_stable` makes for the `f64` kernels.
macro_rules! tracks_f64_conditioned {
    ($name:literal, $x:expr, $allowed:expr, $wide:expr, $narrow:expr) => {{
        let allowed: f64 = $allowed;
        debug_assert!(allowed > 0.0);
        #[allow(clippy::cast_possible_truncation)]
        let best = $wide(f64::from($x)) as f32;
        if !ulp_count_is_meaningful(best) {
            return Ok(());
        }
        let got = $narrow($x);
        let Some(gap) = f32_ulps_between(got, best) else {
            return Ok(());
        };
        #[allow(clippy::cast_precision_loss)]
        let margin = gap as f64 / allowed;
        target(margin);
        check!(
            margin <= 1.0,
            "{}: the f32 kernel used {:.2}x its conditioned allowance of {:.1} ulp at x = {:e} \
             — f32 {:e} against the f64 result rounded to {:e}, a gap of {} ulp",
            $name,
            margin,
            allowed,
            $x,
            got,
            best,
            gap
        );
    }};
}

/// The oscillating peer of [`tracks_f64`], counting against an amplitude envelope.
///
/// Asserts with `check!` rather than `precision_agrees`, because the engine's relation counts
/// ulps of the value by construction and that is the quantity this is deliberately not using.
/// See [`narrowing_margin_at_scale`].
macro_rules! tracks_f64_at_scale {
    ($name:literal, $x:expr, $allowed:expr, $envelope:expr, $wide:expr, $narrow:expr) => {{
        let Some(margin) = narrowing_margin_at_scale($x, $wide, $narrow, $allowed, $envelope)
        else {
            return Ok(());
        };
        target(margin);
        check!(
            margin <= 1.0,
            "{}: the f32 kernel used {:.2}x its allowed {} ulp of the local amplitude at \
             x = {:e} — f32 {:e} against the f64 result rounded",
            $name,
            margin,
            $allowed,
            $x,
            $narrow($x)
        );
    }};
}

// ---- the gamma family ----------------------------------------------------

/// Measured worst: **67 ulp** — the Lanczos core and the recurrence over a domain four times the grid's.
///
/// Contract set at `512`, about 8x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_GAMMA: u32 = 512;

// --- the per-family ulp budgets -------------------------------------------
// Kept together rather than beside each law: what matters about a budget is how it
// compares to its neighbours, and a family whose allowance is an order above the rest
// is the thing a reader should notice. Each is measured, not chosen; see the header.

/// Measured worst: **4 ulp** — in ulps of the `ulp(1)` envelope, which is the unit beside its two zeros.
///
/// Contract set at `16`, about 4x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_LN_GAMMA: u32 = 16;
/// Measured worst: **2 ulp** — in ulps of the `ulp(1)` envelope, beside the zero at `1.4616`.
///
/// Contract set at `8`, about 4x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_DIGAMMA: u32 = 8;
/// Measured worst: **6 ulp** — no zeros, so this is an ordinary ulp count.
///
/// Contract set at `32`, about 5x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_TRIGAMMA: u32 = 32;
/// Measured worst: **14 ulp** — worst near the inflection, not at the tails.
///
/// Contract set at `64`, about 5x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_ERF: u32 = 64;
/// Measured worst: **32 ulp** — this is the *base*; the allowance adds `x²` per argument.
///
/// Contract set at `128`, about 4x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_ERFC: u32 = 128;
/// Measured worst: **26 ulp** — worst just above the continued fraction's envelope.
///
/// Contract set at `128`, about 5x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_ERFCX: u32 = 128;
/// Measured worst: **3 ulp** — Giles fitted these coefficients for `f32`, and it shows.
///
/// Contract set at `16`, about 5x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_ERF_INV: u32 = 16;
/// Measured worst: **47 ulp** — the peak-factored sum walks `O(√x)` terms and drifts like `√n·ε`.
///
/// Contract set at `256`, about 5x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_I0E: u32 = 256;
/// Measured worst: **49 ulp** — as `I₀ᵉ`, plus the `h/(k+1)` factor.
///
/// Contract set at `256`, about 5x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_I1E: u32 = 256;
/// Measured worst: **18 ulp** — worst in the quadrature rather than the series.
///
/// Contract set at `128`, about 7x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_K0E: u32 = 128;
/// Measured worst: **7 ulp** — in ulps of the `√(2/πx)` amplitude envelope.
///
/// Contract set at `32`, about 5x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_J0: u32 = 32;
/// Measured worst: **91 ulp** — in ulps of the `π^{−1/2}|x|^{−1/4}` envelope; the loosest kernel here.
///
/// Contract set at `512`, about 6x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_AIRY_AI: u32 = 512;
/// Measured worst: **5 ulp** — Halley converges cubically, so the seed is the only risk.
///
/// Contract set at `32`, about 6x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_LAMBERT_W: u32 = 32;
/// Measured worst: **3 ulp** — the compensated series and the fraction agree closely.
///
/// Contract set at `16`, about 5x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_SIN_INT: u32 = 16;
/// Measured worst: **8 ulp** — in ulps of the `ulp(1)` envelope, beside the zero at `0.3725`.
///
/// Contract set at `32`, about 4x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_EXP_INT_EI: u32 = 32;

#[path = "cross_precision/bessel.rs"]
mod bessel;
#[path = "cross_precision/erf.rs"]
mod erf;
#[path = "cross_precision/gamma.rs"]
mod gamma;
#[path = "cross_precision/transcendental.rs"]
mod transcendental;
