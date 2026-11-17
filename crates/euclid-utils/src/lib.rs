//! Shared numerical helpers for the euclid workspace: comparison, calculus, sweeps, and
//! shape assertions.
//!
//! Every euclid crate depends on this under `[dev-dependencies]`, so today it is compiled for
//! tests only and never becomes a runtime dependency of anything a consumer sees.
//! `publish = false` keeps it out of releases. It pulls in no third-party code, so the
//! workspace stays dependency-free in the sense that matters.
//!
//! It exists because the same handful of helpers kept being rewritten — `rel` had four copies
//! and `close`/`agrees` five each, with slightly different tolerance conventions. Divergent
//! comparison helpers are worse than duplicated ones: a test can look strict and be loose.
//!
//! The name is deliberately broader than "test helpers". If a genuinely general-purpose
//! utility turns up later that does not belong in a contracts crate, this is where it goes —
//! at which point some consumer would move it out of `[dev-dependencies]`.
//!
//! ## The comparison convention
//!
//! Everything here compares with a **mixed absolute/relative** criterion:
//!
//! ```text
//! |got − want| ≤ tol · max(1, |want|)
//! ```
//!
//! A purely relative test is wrong wherever the quantity passes through zero — `ln Γ` at 1 and
//! 2, `digamma` near 1.46, `erf` at 0 — because it compares two different roundings of zero
//! and reports a meaningless ratio. A purely absolute test is wrong wherever the quantity is
//! large. Scaling by `max(1, |want|)` degrades gracefully to absolute near zero and to
//! relative away from it, which is the behavior these suites want everywhere.

/// The Euler–Mascheroni constant γ, to `f64` precision.
///
/// Here rather than in each suite because it had been retyped in seven files. Note this is an
/// *independent* copy of the digits: a test checking `Real::euler_gamma` against this is
/// comparing a lifted constant to a literal, not to itself.
///
/// `core` gained `f64::consts::EULER_GAMMA` in Rust 1.99, and `euclid-scalar`'s catalogue now
/// takes its value from there. This copy deliberately does **not** — the independence is the
/// whole point. Routing both sides to `core` would turn every test that compares them into a
/// tautology that passes no matter what the trait does, which is exactly the failure this
/// constant exists to prevent. Hence the `allow` rather than a fix.
#[allow(clippy::approx_constant)]
pub const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// Relative error, falling back to absolute error when the reference is exactly zero.
///
/// Prefer [`assert_close`] in tests; this is for when a test needs the number itself (to
/// report a measured worst case, or to assert that an error is *large*).
#[must_use]
pub fn rel(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        got.abs()
    } else {
        ((got - want) / want).abs()
    }
}

/// `true` when `got` and `want` agree to `tol` under the mixed criterion.
#[must_use]
pub fn is_close(got: f64, want: f64, tol: f64) -> bool {
    (got - want).abs() <= tol * want.abs().max(1.0)
}

/// Assert `got ≈ want` to `tol` under the mixed criterion, reporting both values and the
/// bound that was exceeded.
#[track_caller]
pub fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
    let err = (got - want).abs();
    let bound = tol * want.abs().max(1.0);
    assert!(
        err <= bound,
        "{what}: got {got:.17e}, want {want:.17e}, abs err {err:.3e} > bound {bound:.3e}"
    );
}

/// Assert bit-exact equality. The right bar for a constant, an integral-valued result, or an
/// operation documented as exact.
#[track_caller]
pub fn assert_exact(got: f64, want: f64, what: &str) {
    assert!(
        got.to_bits() == want.to_bits(),
        "{what}: got {got:.17e}, want {want:.17e} (exact equality required)"
    );
}

/// One `f32` ulp, relative — `f32::EPSILON` as an `f64`.
pub const F32_ULP: f64 = 1.192_092_895_507_812_5e-7;

/// Assert an `f32` result agrees with an `f64` reference to within `ulps` of `f32` resolution.
///
/// Quoting `f32` tolerances in ulp rather than as a bare number makes them legible as "how
/// many trailing bits does single precision lose here", which is the question actually being
/// asked of a narrowed implementation.
#[track_caller]
pub fn assert_close_f32(got: f32, want: f64, ulps: f64, what: &str) {
    let err = (f64::from(got) - want).abs();
    let bound = ulps * F32_ULP * want.abs().max(1.0);
    assert!(
        err <= bound,
        "{what}: f32 {got:e} vs f64 {want:e} (abs err {err:.3e} > {bound:.3e}, {ulps} ulp)"
    );
}

// --- numerical calculus --------------------------------------------------

/// The default relative step for [`derivative`], chosen to balance the two error terms.
pub const DERIV_STEP: f64 = 1e-4;

/// Five-point central derivative `(8Δ₁ − Δ₂) / 12h`, with the step scaled to `x`.
///
/// Truncation is `O(h⁴)`; roundoff is `O(ε/h)`. At the default step the two land around
/// 1e-16 and 2e-12 respectively, so derivative comparisons are good to roughly 1e-8 — not to
/// machine precision. Where the function itself carries error `δ`, differentiation *amplifies*
/// it to `δ/h`, so use [`derivative_with_step`] with a wider step for a function known to be
/// inaccurate (the Bessel family, for instance).
#[must_use]
pub fn derivative(f: impl Fn(f64) -> f64, x: f64) -> f64 {
    derivative_with_step(f, x, DERIV_STEP * x.abs().max(1.0))
}

/// [`derivative`] with an explicit absolute step.
#[must_use]
pub fn derivative_with_step(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    let d1 = f(x + h) - f(x - h);
    let d2 = f(x + 2.0 * h) - f(x - 2.0 * h);
    (8.0 * d1 - d2) / (12.0 * h)
}

/// Composite Simpson's rule over `[a, b]` with `panels` subintervals (rounded down to even).
///
/// Error is `O(h⁴)` **only for an integrand smooth on the closed interval, endpoints
/// included**. A density behaving like `xᵏ` at an endpoint with non-integer `k` has an
/// unbounded higher derivative there and convergence degrades badly, so check the integrand
/// before trusting a tight tolerance.
#[must_use]
pub fn integrate(f: impl Fn(f64) -> f64, a: f64, b: f64, panels: usize) -> f64 {
    let n = panels.max(2) / 2 * 2;
    let h = (b - a) / n as f64;
    let mut total = f(a) + f(b);
    for i in 1..n {
        let x = a + h * i as f64;
        total += f(x) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    total * h / 3.0
}

// --- sweeps --------------------------------------------------------------

/// `n` evenly spaced points across `[lo, hi]` inclusive.
///
/// Spelling sweeps this way keeps a test's domain explicit, instead of leaving the reader to
/// decode `for i in -60..60 { let x = f64::from(i) * 0.25; }`.
///
/// The endpoints are returned verbatim, including the sign of zero. Interior points use a
/// convex combination, so subtracting endpoints of opposite sign cannot overflow.
pub fn sweep(lo: f64, hi: f64, n: usize) -> impl Iterator<Item = f64> {
    let last = n.max(2) - 1;
    let interior = (1..last).map(move |i| {
        let t = i as f64 / last as f64;
        lo * (1.0 - t) + hi * t
    });
    [lo].into_iter().chain(interior).chain([hi])
}

/// `n` points spaced evenly in `log10` across `[lo, hi]`, both of which must be positive.
///
/// The right sweep for a function whose interesting behavior spans orders of magnitude —
/// tails, tiny arguments, large shape parameters.
pub fn log_sweep(lo: f64, hi: f64, n: usize) -> impl Iterator<Item = f64> {
    let n = n.max(2);
    let (l, h) = (lo.log10(), hi.log10());
    let step = (h - l) / (n - 1) as f64;
    (0..n).map(move |i| 10f64.powf(l + step * i as f64))
}

// --- shape assertions ----------------------------------------------------

/// Assert a sequence never decreases, allowing `slack` for rounding.
///
/// The common check for a CDF or any monotone transform. Takes `slack` because an
/// implementation accurate to `δ` may step backward by up to `δ` without being wrong.
#[track_caller]
pub fn assert_nondecreasing<I: IntoIterator<Item = f64>>(values: I, slack: f64, what: &str) {
    let mut prev = f64::NEG_INFINITY;
    for (i, v) in values.into_iter().enumerate() {
        assert!(
            v >= prev - slack,
            "{what}: not non-decreasing at index {i}: {prev:.17e} then {v:.17e}"
        );
        prev = v;
    }
}

/// Assert a sequence never increases, allowing `slack` for rounding.
#[track_caller]
pub fn assert_nonincreasing<I: IntoIterator<Item = f64>>(values: I, slack: f64, what: &str) {
    let mut prev = f64::INFINITY;
    for (i, v) in values.into_iter().enumerate() {
        assert!(
            v <= prev + slack,
            "{what}: not non-increasing at index {i}: {prev:.17e} then {v:.17e}"
        );
        prev = v;
    }
}

/// Assert every value lies in `[lo, hi]`, and that none is `NaN`.
#[track_caller]
pub fn assert_in_range<I: IntoIterator<Item = f64>>(values: I, lo: f64, hi: f64, what: &str) {
    for (i, v) in values.into_iter().enumerate() {
        assert!(!v.is_nan(), "{what}: NaN at index {i}");
        assert!(
            v >= lo && v <= hi,
            "{what}: index {i} escaped [{lo}, {hi}]: {v:.17e}"
        );
    }
}

/// Assert every value lies in `[0, 1]` — the probability/CDF check.
#[track_caller]
pub fn assert_probability<I: IntoIterator<Item = f64>>(values: I, what: &str) {
    assert_in_range(values, 0.0, 1.0, what);
}

/// Assert every value is finite (neither `NaN` nor `±∞`).
#[track_caller]
pub fn assert_all_finite<I: IntoIterator<Item = f64>>(values: I, what: &str) {
    for (i, v) in values.into_iter().enumerate() {
        assert!(v.is_finite(), "{what}: index {i} is not finite: {v:.17e}");
    }
}

// --- sample statistics ---------------------------------------------------

/// Arithmetic mean, `0.0` for an empty slice.
#[must_use]
pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

/// Population variance (divisor `n`), computed in two passes.
///
/// Two-pass rather than the textbook `E[x²] − E[x]²`: that form subtracts two nearly-equal
/// large numbers whenever the mean dominates the spread, and can even return a negative
/// "variance". Centering first costs one extra pass and removes the cancellation.
#[must_use]
pub fn variance(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let m = mean(xs);
    xs.iter()
        .map(|x| {
            let delta = x - m;
            delta * delta
        })
        .sum::<f64>()
        / xs.len() as f64
}

/// Population standard deviation.
#[must_use]
pub fn std_dev(xs: &[f64]) -> f64 {
    variance(xs).sqrt()
}

/// Assert that `f` has no step at `x`, by evaluating it at the two representable values that
/// *straddle* `x`.
///
/// This is the tool for checking that a piecewise implementation's branches agree where they
/// meet. Probing at ulp resolution rather than with a hand-chosen `h` is what makes it sharp:
/// the interval is as small as the format allows, so the function's own slope contributes
/// almost nothing and what remains is the disagreement between the two methods joining at `x`.
///
/// "Almost nothing" is not nothing, though, and the bound has to carry it. `Ei` changes by 42
/// across one ulp of `x = 40` — not a defect but exactly `Ei′(40)·ulp(40)` — so a bound built
/// only from `f` would reject a perfectly continuous function. Hence both terms:
///
/// - `derivative · (above − below)`, the change the function is *entitled* to make, and
/// - `relative · |f(x)|`, the accuracy to which the two methods are documented to agree.
///
/// A genuine branch mismatch is orders larger than either and still fails.
///
/// # Panics
/// If `f` steps at `x` by more than that bound.
pub fn no_step_across(name: &str, f: impl Fn(f64) -> f64, x: f64, derivative: f64, relative: f64) {
    let below = x.next_down();
    let above = x.next_up();
    let (lo, at, hi) = (f(below), f(x), f(above));
    let bound = derivative.abs() * (above - below) + relative * at.abs() + f64::MIN_POSITIVE;
    assert!(
        (at - lo).abs() <= bound && (hi - at).abs() <= bound,
        "{name} steps at the {x} threshold: {lo:e} / {at:e} / {hi:e} \
         (changes {:e} and {:e}, bound {bound:e})",
        at - lo,
        hi - at
    );
}

#[cfg(test)]
mod tests;

// The README's examples, compiled.
//
// `cfg(doctest)` means this item exists only while rustdoc is collecting doctests: it is absent
// from every real build, adds nothing to the public surface, and never appears in the rendered
// docs. What it buys is that `cargo test --doc` compiles the code blocks in README.md itself.
//
// A crate whose deliverable is its trait surface has a README whose example *is* part of that
// surface, and a README example that no longer compiles is worse than none — it documents a
// signature the crate no longer has. Two crates used to check this by hand-copying the example
// into `tests/readme_check.rs`, which compiles the copy and lets the original drift. This checks
// the original.
//
// Illustrative fragments in the README are marked ```rust,ignore, which is the honest marking:
// they reference names that are deliberately undefined.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeExamples;
