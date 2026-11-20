//! The Lambert `W`: its defining identity, both branches, and the branch point.
//!
//! Part of the [`transcendental`](super) suite. The identity is asserted *in logs* on the
//! positive axis and *directly* on the negative branch, because `w·e^w` leaves the format at one
//! end and cancels at the other — one spelling cannot cover both.

use super::*;

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
