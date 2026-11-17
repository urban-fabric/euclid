//! Lambert `W` and the Riemann zeta function.
//!
//! Part of the [`identities`](super) suite.
//!
//! Both had no generated property before this file, and `kernels/transcendental.rs` was measured
//! at **43 survivors of 194 mutants** — 35 of them inside `lambert_w`.
//!
//! # `W` has the best law in the crate: its own definition
//!
//! `W` is *defined* as the inverse of `w ↦ w e^w`, so `W(x)·e^{W(x)} = x` is an identity at every
//! argument in the domain, needs no reference value, and pins the function rather than a sample
//! of it. Nothing else here comes that cheap.
//!
//! It is checked in two forms, each chosen to avoid an overflow rather than a tolerance:
//!
//! - For `x > 0`, in logs: `ln W(x) + W(x) = ln x`. The direct form would need `e^{684}` at
//!   `x = 1e300` — representable, but only just, and `W(f64::MAX)·e^{W(f64::MAX)}` is within a
//!   factor of three of overflowing.
//! - For `−1/e ≤ x < 0`, directly: `W e^W = x`. Here `|x| ≤ 1/e`, so nothing is near a limit,
//!   and the log form is unavailable because `W` is negative.
//!
//! Measured at **one ulp** in both forms — `2.2e-16` across `[1e-300, 1e-1]`, `[1e1, 1e100]` and
//! `[1e100, 1e300]`, and `2.2e-16` on the negative branch.
//!
//! # Where the identity stops being a test, and why that is not a defect
//!
//! At the branch point the residual check goes **blind**, and it is worth saying so rather than
//! discovering it later. `d/dw(w e^w) = e^w(1 + w)` vanishes at `w = −1`, so an error in `W`
//! maps to a *smaller* error in the recovered `x`: at `1e-16` past the branch point the residual
//! is exactly zero while `1 + W` is `1.9e-8`. The function is not Lipschitz there — its own
//! docstring says accuracy is "inherently halved" — so the identity cannot see the error the
//! format cannot avoid. The property asserts the asymptotic shape there instead.
//!
//! # `ln x` passes through zero at `x = 1`
//!
//! The log form's reference is `ln x`, which is `0` at `x = 1`, so a relative comparison there is
//! undefined — measured as a relative error of `1.0` at exactly `x = 1`, which says nothing about
//! the kernel. The bound is mixed, the same correction the log-domain and normal suites both
//! needed. Three suites in a row have now hit this: **a reference that crosses zero needs an
//! absolute term, and the sweep will find the crossing.**

use super::*;

/// `−1/e`, the branch point, below which the real branches end.
fn branch_point() -> f64 {
    -1.0 / core::f64::consts::E
}

property! {
    #![config(settings("lambert_w").cases(600))]

    /// `ln W(x) + W(x) = ln x` for `x > 0` — the defining identity, overflow-free.
    ///
    /// One ulp across three hundred decades. The amplification is `1 + W`, because
    /// `d ln x/d ln w = 1 + w`, and that reaches `685` at `x = 1e300` — so the bound carries it
    /// rather than being a constant that is slack at one end and wrong at the other.
    fn the_lambert_identity_holds_in_logs_on_the_positive_axis(
        x in generate::float::magnitude(-1000, 1000),
    ) {
        let x = x.abs();
        if x <= 0.0 || !x.is_finite() || x.is_nan() {
            return Ok(());
        }
        classify!("below one", x < 1.0);
        classify!("above one", x >= 1.0);
        classify!("very large", x > 1e100);

        let w = Real::lambert_w(x);
        check!(
            w.is_finite() && w > 0.0,
            "W({x:e}) = {w:e} must be finite and positive for positive x"
        );

        // The two sides are quantities of size `|ln x|` — `690` at `x = 1e300` — so the bound
        // scales with that, not with `1`. A first draft used `8·eps·(1 + w)` and failed at
        // `x = 1.8e-15`, where `1 + w` is still `1` while `ln x` is already `-34`: the
        // amplification factor and the *magnitude* of the compared values are different things,
        // and it is the magnitude that sets an absolute bound.
        let bound = 8.0 * f64::EPSILON * x.ln().abs().max(1.0);
        relation::agree(
            "ln W(x) + W(x) = ln x",
            (),
            |()| w.ln() + w,
            |()| x.ln(),
            // Absolute, because `ln x` is zero at `x = 1` and a relative bound there is
            // undefined — the sweep does reach exactly 1.0.
            &Absolute(bound),
        )?;
    }

    /// `W(x)·e^{W(x)} = x` on `[−1/e, 0)`, where the direct form is the safe one.
    ///
    /// Also the branch: the principal `W` is `≥ −1` everywhere, which is what *principal* means,
    /// and a kernel that wandered onto `W₋₁` would return values below `−1` while still
    /// satisfying the identity.
    fn the_lambert_identity_holds_directly_on_the_negative_branch(
        x in generate::float::in_range(-0.367_879_441_171_442_33, -1e-300),
    ) {
        if x >= 0.0 || x < branch_point() {
            return Ok(());
        }
        classify!("near the branch point", x < branch_point() + 1e-3);
        classify!("near zero", x > -1e-3);

        let w = Real::lambert_w(x);
        check!(
            w.is_finite() && (-1.0..0.0).contains(&w),
            "W({x:e}) = {w:e} must lie in [-1, 0) on the negative branch"
        );
        relation::agree(
            "W e^W = x",
            (),
            |()| w * w.exp(),
            |()| x,
            &Relative(16.0 * f64::EPSILON),
        )?;
    }

    /// `W(x·e^x) = x` for `x ≥ −1`, with the bound the inversion is entitled to.
    ///
    /// The round trip the other way, and the one that *can* see the branch point: inverting
    /// `x ↦ x e^x` amplifies by `1/(1 + x)` because that map's derivative is `e^x(1 + x)`. At
    /// `x = −0.983` the measured error is `9.3e-15` against a predicted `ε/(1 + x) = 1.3e-14`,
    /// which is the agreement that makes this a bound rather than a guess.
    fn the_lambert_function_inverts_its_own_argument(
        x in generate::float::in_range(-1.0, 50.0),
    ) {
        let argument = x * x.exp();
        if !argument.is_finite() || argument < branch_point() {
            return Ok(());
        }
        classify!("near the branch point", x < -0.9);
        classify!("ordinary", x >= -0.9);

        // `1/(1 + x)` is the amplification of inverting `x e^x`; it diverges at `x = -1`, which
        // is the branch point, and the `max` keeps the bound finite exactly there.
        let amplification = (1.0 / (1.0 + x)).abs().min(1e9);
        relation::agree(
            "W(x e^x) = x",
            (),
            |()| Real::lambert_w(argument),
            |()| x,
            &Absolute(32.0 * f64::EPSILON * amplification.max(1.0)),
        )?;
    }

    /// The endpoints, the domain, and the signed zero.
    ///
    /// `W(−0.0)` is `−0.0`, which the kernel goes out of its way to preserve — it returns `x`
    /// rather than a literal zero, with the comment that `W` is `x − x² + …` near the origin and
    /// so odd there. That is exactly the kind of claim a refactor erases and no comparison can
    /// see, so it is asserted on the bits.
    fn the_lambert_endpoints_are_exact(x in generate::float::in_range(-10.0, -0.368)) {
        check_ulps!(Real::lambert_w(0.0f64), 0.0, 0, "W(0) is zero");
        let negative_zero = Real::lambert_w(-0.0f64);
        check!(
            negative_zero == 0.0 && Real::signbit(&negative_zero),
            "W(-0.0) must be -0.0, since W is odd near the origin; got {negative_zero:e}"
        );
        check_ulps!(
            Real::lambert_w(core::f64::consts::E),
            1.0,
            0,
            "W(e) is exactly 1"
        );
        check_ulps!(
            Real::lambert_w(branch_point()),
            -1.0,
            0,
            "W(-1/e) is exactly -1"
        );
        check!(
            Real::lambert_w(f64::INFINITY) == f64::INFINITY,
            "W(inf) is inf"
        );
        check!(Real::lambert_w(f64::NAN).is_nan(), "W(NaN) is NaN");

        // Below the branch point there is no real value, and NaN is the honest answer rather
        // than the nearest real one.
        if x < branch_point() {
            check!(
                Real::lambert_w(x).is_nan(),
                "W({x:e}) is below the branch point and must be NaN"
            );
        }
    }

    /// `1 + W(x) ≈ √(2(1 + ex))` just above the branch point.
    ///
    /// The square-root behaviour the docstring names, asserted because it is the reason the
    /// accuracy claim there is halved — and because the residual identity is blind to it. The
    /// next term of the series is `−p²/3`, which is `2.3%` of `p` at the far end of this range,
    /// so five percent is the honest bound for the leading shape.
    fn the_lambert_function_has_a_square_root_branch_point(
        offset in generate::float::in_range(1e-18, 1e-3),
    ) {
        if offset <= 0.0 || offset.is_nan() {
            return Ok(());
        }
        let x = branch_point() + offset;
        if x < branch_point() {
            return Ok(());
        }
        let w = Real::lambert_w(x);
        check!(
            w >= -1.0,
            "W must stay on the principal branch at the branch point, got {w:e}"
        );

        // `1 + e·x` is the offset in the series' own variable; computed from `x` so it survives
        // the rounding of `-1/e`.
        let p = (2.0 * (1.0 + core::f64::consts::E * x)).max(0.0).sqrt();
        if p == 0.0 {
            check_ulps!(w, -1.0, 0, "at the branch point itself W is exactly -1");
            return Ok(());
        }
        relation::agree(
            "1 + W ~ sqrt(2(1 + e x))",
            (),
            |()| 1.0 + w,
            |()| p,
            &Relative(0.05),
        )?;
    }

    /// `W` is increasing on its whole domain, which no residual check implies.
    fn the_lambert_function_increases(x in generate::float::in_range(-0.367_879_4, 1e6)) {
        if x < branch_point() {
            return Ok(());
        }
        let w = Real::lambert_w(x);
        // A step large enough to clear the branch point's square-root flatness.
        let further = x + (x.abs() * 1e-6).max(1e-9);
        let w_further = Real::lambert_w(further);
        if !w.is_finite() || !w_further.is_finite() {
            return Ok(());
        }
        check!(
            w_further >= w,
            "W must not decrease: W({x:e}) = {w:e} then W({further:e}) = {w_further:e}"
        );
    }
}

property! {
    #![config(settings("zeta").cases(400))]

    /// `ζ(s) = Σ_{k≥1} k^{−s}` for `s ≥ 8`, summed directly.
    ///
    /// The defining Dirichlet series, and a genuinely **independent** reference: the kernel
    /// reaches `s > 1` by Euler–Maclaurin summation, which shares no code with a plain sum. The
    /// range starts at `8` so the truncation is not the thing being measured — the tail past
    /// `k = 200` is `200^{−7}/7 ≈ 1.1e-17`, below an ulp of the answer, where at `s = 3` it
    /// would be `7.7e-12` and would dominate.
    fn zeta_agrees_with_its_dirichlet_series(s in generate::float::in_range(8.0, 40.0)) {
        let mut series = 0.0f64;
        for k in 1..=200u32 {
            series += f64::from(k).powf(-s);
        }
        relation::agree(
            "zeta = sum k^-s",
            (),
            |()| Real::zeta(s),
            |()| series,
            &Relative(1e-14),
        )?;
    }

    /// The functional equation, inside the strip where it is not how the kernel works.
    ///
    /// ```text
    /// ζ(s) = 2^s π^{s−1} sin(πs/2) Γ(1−s) ζ(1−s)
    /// ```
    ///
    /// For `s ≤ −1` the kernel *evaluates* this equation, so asserting it there would be
    /// circular. On `(−1, 1)` it does not: both `ζ(s)` and `ζ(1−s)` come from the accelerated
    /// eta series, so the identity ties two independent evaluations together through `Γ` and
    /// `sin` — and it is the only check available inside the critical strip, which the
    /// Euler–Maclaurin route cannot reach and reflection cannot escape.
    ///
    /// Measured `4.1e-15` on `(0, 1)` and `5.4e-14` on `(−1, 0)`.
    fn zeta_satisfies_its_functional_equation_inside_the_strip(
        s in generate::float::in_range(-0.99, 0.99),
    ) {
        if s == 0.0 {
            return Ok(());
        }
        classify!("negative", s < 0.0);
        classify!("critical strip", s > 0.0);

        let reflected = 2f64.powf(s)
            * core::f64::consts::PI.powf(s - 1.0)
            * (core::f64::consts::FRAC_PI_2 * s).sin()
            * Real::gamma(1.0 - s)
            * Real::zeta(1.0 - s);
        if !reflected.is_finite() {
            return Ok(());
        }
        relation::agree(
            "zeta(s) = 2^s pi^(s-1) sin(pi s/2) Gamma(1-s) zeta(1-s)",
            (),
            |()| Real::zeta(s),
            |()| reflected,
            &Relative(1e-12),
        )?;
    }

    /// The closed forms, the trivial zeros, the pole and the limits — all exact or one ulp.
    ///
    /// `ζ(2) = π²/6` and `ζ(4) = π⁴/90` come out **bit-identical** to the closed form, and
    /// `ζ(−1) = −1/12` and `ζ(−3) = 1/120` likewise. The trivial zeros are returned as exact
    /// zeros rather than left to cancellation, which the kernel's docstring calls out and which
    /// a `1e-15` tolerance would not distinguish from a near-miss.
    fn zeta_hits_its_closed_forms_exactly(s in generate::float::in_range(2.0, 100.0)) {
        let pi = core::f64::consts::PI;
        // `zeta(2)` is bit-identical to `pi^2/6`; `zeta(4)` and `zeta(6)` are one ulp away,
        // which is the closed form's own rounding through `powi` rather than the kernel's.
        //
        // The ulp counts here were each measured, and one of them had to be corrected after an
        // exploration *printed* `zeta(4)` and `pi^4/90` as the same number. They are not the
        // same number: `Debug` emits the shortest string that round-trips, so two neighbours
        // print identically. Comparing the printed form is not comparing the value.
        check_ulps!(Real::zeta(2.0f64), pi * pi / 6.0, 0, "zeta(2) = pi^2/6");
        check_ulps!(Real::zeta(4.0f64), pi.powi(4) / 90.0, 1, "zeta(4) = pi^4/90");
        check_ulps!(Real::zeta(6.0f64), pi.powi(6) / 945.0, 1, "zeta(6) = pi^6/945");
        check_ulps!(Real::zeta(0.0f64), -0.5, 0, "zeta(0) = -1/2");
        check_ulps!(Real::zeta(-1.0f64), -1.0 / 12.0, 0, "zeta(-1) = -1/12");
        check_ulps!(Real::zeta(-3.0f64), 1.0 / 120.0, 0, "zeta(-3) = 1/120");

        // The trivial zeros, exactly.
        for even in [-2.0f64, -4.0, -6.0, -20.0, -100.0] {
            check!(
                Real::zeta(even) == 0.0,
                "zeta({even}) must be exactly zero, got {:e}",
                Real::zeta(even)
            );
        }

        // The pole and the limits.
        check!(Real::zeta(1.0f64) == f64::INFINITY, "zeta(1) is the pole");
        check!(
            Real::zeta(f64::INFINITY) == 1.0,
            "zeta(inf) is 1, the first term alone"
        );
        check!(Real::zeta(f64::NAN).is_nan(), "zeta(NaN) is NaN");

        // And on `s > 1` it decreases monotonically towards one from above — reaching exactly
        // one once `2^{-s}` falls below `ulp(1)/2`, which is near `s = 53`. The bound is `>= 1`
        // rather than `> 1` for that reason: `zeta(100)` is `1.0` on the nose, and a strict
        // inequality there would be asserting more resolution than the format has.
        if s > 1.0 {
            let value = Real::zeta(s);
            check!(
                value >= 1.0 && value <= Real::zeta(2.0f64) + f64::EPSILON,
                "zeta({s}) = {value:e} must lie in [1, zeta(2)]"
            );
            check!(
                Real::zeta(s + 1.0) <= value,
                "zeta must decrease on s > 1, at s = {s}"
            );
            classify!("saturated at one", value == 1.0);
        }
        // The saturation itself, stated so a rewrite that returned `1 + something` there would
        // have to justify it.
        check!(
            Real::zeta(60.0f64) == 1.0,
            "zeta(60) saturates to exactly one"
        );
    }
}
