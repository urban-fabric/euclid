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

property! {
    #![config(settings("f32_precision_gamma"))]

    /// `Γ` at `f32`, against the `f64` kernel rounded.
    ///
    /// The domain crosses the small-shape route, the Lanczos core and the recurrence, and stops
    /// below the overflow at `Γ(35)` so the comparison stays defined.
    fn the_f32_gamma_tracks_f64(x in generate::float32::in_range(0.02, 34.0)) {
        if !x.is_finite() || x <= 0.0 {
            return Ok(());
        }
        tracks_f64!("gamma", x, ALLOWED_GAMMA, Real::gamma, Real::gamma);
    }
}

/// Measured worst: **4 ulp** — in ulps of the `ulp(1)` envelope, which is the unit beside its two zeros.
///
/// Contract set at `16`, about 4x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_LN_GAMMA: u32 = 16;

property! {
    #![config(settings("f32_precision_ln_gamma")
        .require("ln_gamma recurrence", 2.5)
        .require("ln_gamma asymptotic", 45.0))]

    /// `lnΓ` at `f32`, across the recurrence/asymptotic handover at `10`.
    ///
    /// Both sides are required: the recurrence walks up to the threshold and the asymptotic
    /// takes over above it, and a climb that stopped reaching one of them would be reporting a
    /// clean pass over half the kernel.
    fn the_f32_ln_gamma_tracks_f64(x in generate::float32::in_range(0.02, 400.0)) {
        if !x.is_finite() || x <= 0.0 {
            return Ok(());
        }
        classify!("ln_gamma recurrence", x < 10.0);
        classify!("ln_gamma asymptotic", x >= 10.0);
        // `lnΓ` has two zeros in this domain, at `x = 1` and `x = 2`, and they are the reason the
        // fixed grid quotes 72 ulp for this kernel while the climb measured 4: the grid happens
        // to land near one of them and a hill climb on a thin feature does not have to. Both are
        // cancellation zeros — `Γ` is near 1 and `ln` of a value near 1 has absolute accuracy,
        // not relative — so `ulp(1)` is the honest unit. Above `x ≈ 2.4` the value exceeds the
        // envelope and this reverts to an ordinary ulp count.
        tracks_f64_at_scale!(
            "ln_gamma",
            x,
            ALLOWED_LN_GAMMA,
            1.0,
            Real::ln_gamma,
            Real::ln_gamma
        );
    }
}

/// Measured worst: **2 ulp** — in ulps of the `ulp(1)` envelope, beside the zero at `1.4616`.
///
/// Contract set at `8`, about 4x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_DIGAMMA: u32 = 8;

property! {
    #![config(settings("f32_precision_digamma")
        .require("psi recurrence", 4.0)
        .require("psi asymptotic", 22.0))]

    /// `ψ` at `f32`, across `PSI_ASYMPTOTIC_MIN`, which the port rederived at `6` rather than
    /// inheriting the `f64` value.
    fn the_f32_digamma_tracks_f64(x in generate::float32::in_range(0.02, 60.0)) {
        if !x.is_finite() || x <= 0.0 {
            return Ok(());
        }
        classify!("psi recurrence", x < 6.0);
        classify!("psi asymptotic", x >= 6.0);
        // `ψ` has a zero at `1.4616`, and like `Ei`'s it is a cancellation of `O(1)` terms rather
        // than an analytic one, so the envelope is `ulp(1)`. This was a latent flake rather than
        // an observed failure: the measured worst was 4 ulp because *this* seed's climb never
        // got close enough to the zero. A kernel whose test passes only while the search misses
        // a thin region is a kernel with no test there.
        tracks_f64_at_scale!(
            "digamma",
            x,
            ALLOWED_DIGAMMA,
            1.0,
            Real::digamma,
            Real::digamma
        );
    }
}

/// Measured worst: **6 ulp** — no zeros, so this is an ordinary ulp count.
///
/// Contract set at `32`, about 5x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_TRIGAMMA: u32 = 32;

property! {
    #![config(settings("f32_precision_trigamma"))]

    /// `ψ'` at `f32`. Steeper than `ψ` near the origin, where it diverges like `1/x²`.
    fn the_f32_trigamma_tracks_f64(x in generate::float32::in_range(0.02, 60.0)) {
        if !x.is_finite() || x <= 0.0 {
            return Ok(());
        }
        tracks_f64!("trigamma", x, ALLOWED_TRIGAMMA, Real::trigamma, Real::trigamma);
    }
}

// ---- the error function family ------------------------------------------

/// Measured worst: **14 ulp** — worst near the inflection, not at the tails.
///
/// Contract set at `64`, about 5x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_ERF: u32 = 64;

property! {
    #![config(settings("f32_precision_erf"))]

    /// `erf` at `f32`, including the linear region near the origin.
    fn the_f32_erf_tracks_f64(x in generate::float32::in_range(-6.0, 6.0)) {
        if !x.is_finite() {
            return Ok(());
        }
        tracks_f64!("erf", x, ALLOWED_ERF, Real::erf, Real::erf);
    }
}

/// Measured worst: **32 ulp** — this is the *base*; the allowance adds `x²` per argument.
///
/// Contract set at `128`, about 4x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_ERFC: u32 = 128;

property! {
    #![config(settings("f32_precision_erfc")
        .require("erfc complement form", 9.0)
        .require("erfc fraction form", 38.0))]

    /// `erfc` at `f32`, across the form crossover.
    ///
    /// This is the threshold the erf work moved — `1 − erf` below it, the continued fraction
    /// above — so both sides are labelled and required. The `f32` peer sites its own crossover
    /// independently of the `f64` one, which is exactly the kind of divergence a narrowing
    /// comparison catches and an identity does not.
    fn the_f32_erfc_tracks_f64(x in generate::float32::in_range(-3.0, 10.0)) {
        if !x.is_finite() {
            return Ok(());
        }
        classify!("erfc complement form", x < 1.6875);
        classify!("erfc fraction form", x >= 1.6875);
        // The allowance carries `x²`, which is the conditioning rather than slack. The tail
        // route is `e^{−x²}·erfcx(x)`, and forming `−x²` in `f32` costs `ε·x²` *absolute* in the
        // exponent; `exp` turns an absolute error in its argument into a relative error in its
        // result, so the output cannot be better than `ε·x²` relative however the kernel is
        // written. At `x = 9.15` that is `1.0e-5`, about 84 ulp — and the measured gap there is
        // 28, so the kernel is comfortably inside a bound it cannot escape. A flat contract
        // would have had to be 84+ ulp everywhere to admit this one argument, which would have
        // made it vacuous at `x = 1`.
        tracks_f64_conditioned!(
            "erfc",
            x,
            f64::from(ALLOWED_ERFC) + f64::from(x) * f64::from(x),
            Real::erfc,
            Real::erfc
        );
    }
}

/// Measured worst: **26 ulp** — worst just above the continued fraction's envelope.
///
/// Contract set at `128`, about 5x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_ERFCX: u32 = 128;

property! {
    #![config(settings("f32_precision_erfcx")
        .require("erfcx below the envelope", 2.5)
        .require("erfcx fraction", 23.0))]

    /// `erfcx` at `f32`. The scaled form is where the continued fraction's convergence shows,
    /// and the `f32` port had to widen its own envelope rather than inherit `2.0`.
    fn the_f32_erfcx_tracks_f64(x in generate::float32::in_range(0.01, 40.0)) {
        if !x.is_finite() || x <= 0.0 {
            return Ok(());
        }
        classify!("erfcx below the envelope", x < 1.6875);
        classify!("erfcx fraction", x >= 1.6875);
        tracks_f64!("erfcx", x, ALLOWED_ERFCX, Real::erfcx, Real::erfcx);
    }
}

/// Measured worst: **3 ulp** — Giles fitted these coefficients for `f32`, and it shows.
///
/// Contract set at `16`, about 5x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_ERF_INV: u32 = 16;

property! {
    #![config(settings("f32_precision_erf_inv"))]

    /// `erf⁻¹` at `f32`, which takes no Newton step because Giles fitted its coefficients *for*
    /// `f32` — so this is the one kernel here whose narrow version is not a narrowed wide one.
    fn the_f32_erf_inv_tracks_f64(x in generate::float32::in_range(-0.999, 0.999)) {
        if !x.is_finite() {
            return Ok(());
        }
        tracks_f64!("erf_inv", x, ALLOWED_ERF_INV, Real::erf_inv, Real::erf_inv);
    }
}

// ---- the modified Bessel functions --------------------------------------

/// Measured worst: **47 ulp** — the peak-factored sum walks `O(√x)` terms and drifts like `√n·ε`.
///
/// Contract set at `256`, about 5x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_I0E: u32 = 256;

property! {
    #![config(settings("f32_precision_i0e")
        .require("i0e no-log", 2.0)
        .require("i0e peak-factored", 27.0)
        .require("i0e asymptotic", 18.0))]

    /// `I₀ᵉ` at `f32`, across all three forms the regime split selects.
    ///
    /// The domain deliberately spans `I1_NO_LOG_MAX` at `1` and `I_ASYMPTOTIC_MIN` at `100`, and
    /// all three are required. The asymptotic side is the one worth insisting on: the series is
    /// wrong by a factor of 15 by `x = 1e8` while staying finite and plausible, so the handover
    /// is load bearing rather than an optimisation.
    fn the_f32_i0e_tracks_f64(x in generate::float32::in_range(0.0, 400.0)) {
        if !x.is_finite() || x < 0.0 {
            return Ok(());
        }
        classify!("i0e no-log", x < 1.0);
        classify!("i0e peak-factored", (1.0..100.0).contains(&x));
        classify!("i0e asymptotic", x >= 100.0);
        tracks_f64!("bessel_i0e", x, ALLOWED_I0E, Real::bessel_i0e, Real::bessel_i0e);
    }
}

/// Measured worst: **49 ulp** — as `I₀ᵉ`, plus the `h/(k+1)` factor.
///
/// Contract set at `256`, about 5x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_I1E: u32 = 256;

property! {
    #![config(settings("f32_precision_i1e")
        .require("i1e no-log", 2.0)
        .require("i1e peak-factored", 27.0)
        .require("i1e asymptotic", 18.0))]

    /// `I₁ᵉ` at `f32`. Unlike `I₀ᵉ` this one really does take the no-logarithm route, where
    /// `I₁ ~ x/2` must come back exact.
    ///
    /// Swept on the positive axis only, deliberately: `I₁ᵉ` is odd, so the narrowing gap at `−x`
    /// is the mirror of the gap at `x` and a two-sided sweep would measure each argument twice.
    /// The *sign* is what a one-sided sweep cannot see, and `cargo mutants` duly found that it
    /// was unchecked here — it is now pinned by `the_first_kind_is_odd_in_every_form` beside the
    /// kernel, which is the stronger assertion since it is exact rather than budgeted.
    fn the_f32_i1e_tracks_f64(x in generate::float32::in_range(0.0, 400.0)) {
        if !x.is_finite() || x < 0.0 {
            return Ok(());
        }
        classify!("i1e no-log", x < 1.0);
        classify!("i1e peak-factored", (1.0..100.0).contains(&x));
        classify!("i1e asymptotic", x >= 100.0);
        tracks_f64!("bessel_i1e", x, ALLOWED_I1E, Real::bessel_i1e, Real::bessel_i1e);
    }
}

/// Measured worst: **18 ulp** — worst in the quadrature rather than the series.
///
/// Contract set at `128`, about 7x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_K0E: u32 = 128;

property! {
    #![config(settings("f32_precision_k0e")
        .require("k0e series", 2.0)
        .require("k0e quadrature", 46.0))]

    /// `K₀ᵉ` at `f32`, across the series/quadrature handover the port moved to `1.5`.
    fn the_f32_k0e_tracks_f64(x in generate::float32::in_range(0.01, 200.0)) {
        if !x.is_finite() || x <= 0.0 {
            return Ok(());
        }
        classify!("k0e series", x <= 1.5);
        classify!("k0e quadrature", x > 1.5);
        tracks_f64!("bessel_k0e", x, ALLOWED_K0E, Real::bessel_k0e, Real::bessel_k0e);
    }
}

// ---- oscillating, Airy, and the transcendentals -------------------------

/// Measured worst: **7 ulp** — in ulps of the `√(2/πx)` amplitude envelope.
///
/// Contract set at `32`, about 5x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_J0: u32 = 32;

property! {
    #![config(settings("f32_precision_j0")
        .require("j0 series", 3.0)
        .require("j0 march", 24.0)
        .require("j0 Hankel", 20.0))]

    /// `J₀` at `f32`, counted against its amplitude envelope `√(2/πx)`.
    ///
    /// `J₀` has infinitely many zeros in this domain, so an ulp count against the value measures
    /// distance to the nearest zero rather than the quality of the kernel — see
    /// [`narrowing_margin_at_scale`]. The envelope is the Hankel amplitude, capped at `1` because
    /// it diverges as `x → 0` while `J₀(0) = 1`.
    fn the_f32_j0_tracks_f64(x in generate::float32::in_range(0.0, 40.0)) {
        if !x.is_finite() || x < 0.0 {
            return Ok(());
        }
        classify!("j0 series", x < 2.0);
        classify!("j0 march", (2.0..7.0).contains(&x));
        classify!("j0 Hankel", x >= 7.0);
        let envelope = if x > 0.0 {
            (2.0 / (core::f32::consts::PI * x)).sqrt().min(1.0)
        } else {
            1.0
        };
        tracks_f64_at_scale!(
            "bessel_j0",
            x,
            ALLOWED_J0,
            envelope,
            Real::bessel_j0,
            Real::bessel_j0
        );
    }
}

/// Measured worst: **91 ulp** — in ulps of the `π^{−1/2}|x|^{−1/4}` envelope; the loosest kernel here.
///
/// Contract set at `512`, about 6x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_AIRY_AI: u32 = 512;

property! {
    #![config(settings("f32_precision_airy_ai")
        .require("airy oscillatory", 12.0)
        .require("airy decaying", 35.0))]

    /// `Ai` at `f32`, over both signs, counted against its amplitude envelope.
    ///
    /// The oscillatory left half and the decaying right half are different algorithms, and the
    /// port rederived three of their four thresholds. On the left the envelope is the standard
    /// `π^{−1/2}|x|^{−1/4}`; on the right `Ai` decays monotonically with no zeros, so the value
    /// is its own scale and the criterion reduces to an ordinary ulp count.
    fn the_f32_airy_ai_tracks_f64(x in generate::float32::in_range(-15.0, 15.0)) {
        if !x.is_finite() {
            return Ok(());
        }
        classify!("airy oscillatory", x < 0.0);
        classify!("airy decaying", x >= 0.0);
        let envelope = if x < 0.0 {
            x.abs().powf(-0.25) / core::f32::consts::PI.sqrt()
        } else {
            0.0
        };
        tracks_f64_at_scale!(
            "airy_ai",
            x,
            ALLOWED_AIRY_AI,
            envelope,
            Real::airy_ai,
            Real::airy_ai
        );
    }
}

/// Measured worst: **5 ulp** — Halley converges cubically, so the seed is the only risk.
///
/// Contract set at `32`, about 6x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_LAMBERT_W: u32 = 32;

property! {
    #![config(settings("f32_precision_lambert_w")
        .require("W near the branch point", 3.0)
        .require("W ordinary", 25.0))]

    /// `W` at `f32`, from just above the branch point outwards.
    ///
    /// The branch point is `−1/e`, where `W` has a square-root singularity and the Halley
    /// iteration is seeded from a series rather than from the logarithm. `f32` leaves that
    /// neighbourhood about `1e4` times further out than `f64` does, so the window is labelled.
    fn the_f32_lambert_w_tracks_f64(x in generate::float32::in_range(-0.36, 300.0)) {
        if !x.is_finite() {
            return Ok(());
        }
        classify!("W near the branch point", x < -0.3);
        classify!("W ordinary", x >= -0.3);
        tracks_f64!("lambert_w", x, ALLOWED_LAMBERT_W, Real::lambert_w, Real::lambert_w);
    }
}

/// Measured worst: **3 ulp** — the compensated series and the fraction agree closely.
///
/// Contract set at `16`, about 5x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_SIN_INT: u32 = 16;

property! {
    #![config(settings("f32_precision_sin_int")
        .require("Si series", 3.5)
        .require("Si fraction", 45.0))]

    /// `Si` at `f32`, across `TRIG_SERIES_MAX`, which the measurement left at the `f64` value.
    fn the_f32_sin_int_tracks_f64(x in generate::float32::in_range(0.0, 60.0)) {
        if !x.is_finite() || x < 0.0 {
            return Ok(());
        }
        classify!("Si series", x < 4.0);
        classify!("Si fraction", x >= 4.0);
        tracks_f64!("sin_int", x, ALLOWED_SIN_INT, Real::sin_int, Real::sin_int);
    }
}

/// Measured worst: **8 ulp** — in ulps of the `ulp(1)` envelope, beside the zero at `0.3725`.
///
/// Contract set at `32`, about 4x that, per the headroom
/// convention in the module header. The budget beside it pins the measurement.
const ALLOWED_EXP_INT_EI: u32 = 32;

property! {
    #![config(settings("f32_precision_exp_int_ei")
        .require("Ei series", 8.0)
        .require("Ei asymptotic", 18.0))]

    /// `Ei` at `f32`, across `EI_ASYMPTOTIC_MIN`. Every term of the series is positive for
    /// `x > 0`, so the series cannot cancel — it can only be stopped too early, which is what
    /// the lower cap risks and this comparison would see.
    fn the_f32_exp_int_ei_tracks_f64(x in generate::float32::in_range(0.01, 60.0)) {
        if !x.is_finite() || x <= 0.0 {
            return Ok(());
        }
        classify!("Ei series", x < 20.0);
        classify!("Ei asymptotic", x >= 20.0);
        // `Ei` has a zero at `0.3725074`, reached by cancellation rather than analytically:
        // `Ei = γ + ln x + Σ`, all `O(1)` there, summing to `1e-3`. So the achievable absolute
        // error beside it is `ε·1`, not `ε·|Ei|`, and the fair unit is `ulp(1)` — the suite found
        // `x = 0.37225` at 310 ulp of the *value*, which is `1.85e-8` absolute, or a sixth of one
        // ulp of 1. Away from the zero `|Ei|` exceeds the envelope and this is an ordinary ulp
        // count.
        tracks_f64_at_scale!(
            "exp_int_ei",
            x,
            ALLOWED_EXP_INT_EI,
            1.0,
            Real::exp_int_ei,
            Real::exp_int_ei
        );
    }
}
