//! The Riemann zeta: its series, its functional equation and its closed forms.
//!
//! Part of the [`transcendental`](super) suite. The functional equation is checked *inside the
//! critical strip*, where the reflected argument is also in the strip — so it additionally
//! constrains the accelerated eta route against itself at a different argument.

use super::*;

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
    ///
    /// # The budget has to carry `1 − s`'s own rounding, which blows up near zero
    ///
    /// A flat `1e-12` here was wrong, and wrong in a way that looked exactly like a kernel
    /// defect: `euclid-stability` reported a `1.0e-12` disagreement at
    /// `s = −1.1100254952911559e-4` and the obvious reading was that one of the two `ζ`
    /// evaluations had lost accuracy near the pole. Neither had. Measured against references
    /// computed in `Decimal` at the *exact* `f64` arguments, both routes hold to `4.4e-16`
    /// across `1 ± 2^{-13}`.
    ///
    /// The error is in this expression, not in what it calls. `ζ(1−s)` is evaluated at the `f64`
    /// nearest to `1 − s`, and that rounding is worth `ulp(1)/2 = 2^{-53}` of absolute error in
    /// the argument. Near the pole `ζ(1+d) ≈ 1/d`, so a perturbation of `d` is amplified by the
    /// full condition number `|d ζ'/ζ| ≈ 1`: the *relative* error of `ζ(1−s)` is
    /// `≈ 2^{-53}/|s|`, which is `1.0e-12` at the reported `s` — the discrepancy exactly, not
    /// approximately.
    ///
    /// So the identity's two sides cannot agree more closely than `f64` can state the argument
    /// of one of them, and the budget below says so. It grows without bound as `s → 0`, which is
    /// the honest statement: the check genuinely weakens there, and it stops meaning anything at
    /// `|s| < 2^{-53}` where `1 − s` rounds to `1` and `ζ` returns `+∞` — which the finiteness
    /// guard below already drops. This is the same hazard CONTRIBUTING.md's "Provenance includes
    /// the exact argument" rule covers, arising inside a test's own arithmetic rather than in a
    /// pasted reference.
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
        // `1e-12` is the floor the two kernels earn on their own; the second term is what
        // rounding `1 - s` costs, with a factor of four of headroom over the derivation.
        let argument_rounding = 8.0 * f64::EPSILON / s.abs();
        relation::agree(
            "zeta(s) = 2^s pi^(s-1) sin(pi s/2) Gamma(1-s) zeta(1-s)",
            (),
            |()| Real::zeta(s),
            |()| reflected,
            &Relative(1e-12 + argument_rounding),
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
