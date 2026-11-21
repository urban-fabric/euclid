//! Recurrences as metamorphic references, with the budget set by the conditioning.
//!
//! Every function here satisfies a shift identity — `Γ(x+1) = x·Γ(x)`, `ψ(x+1) = ψ(x) + 1/x`,
//! and so on. Those make a **metamorphic** test in the strict sense: the reference is the same
//! kernel at a *different argument*, combined by exact arithmetic, so no external oracle is
//! needed and no tolerance has to be invented.
//!
//! # Why this is worth more than a value table
//!
//! A recurrence constrains the function against itself across its whole domain, and it crosses
//! route boundaries without naming them: `Γ` reduces to `[1, 2)` below `STIRLING_MIN` and takes
//! Stirling above, so `Γ(x+1) = x·Γ(x)` straddles that handover at every `x` near it. A table of
//! values has to be told where the boundaries are; this does not. A wrong coefficient shifts a
//! smooth function by a smooth amount, which a value assertion straddles and a recurrence
//! catches.
//!
//! # The multiplier is what the budget has to allow for
//!
//! `x·Γ(x)` is a product, so it carries `Γ`'s relative error plus one rounding, and
//! `ψ(x) + 1/x` is a sum whose terms diverge as `x → 0` — at `x = 1e-8` the `1/x` is `1e8` and
//! `ψ(x)` is `−1e8`, so the sum keeps almost nothing. That is the problem's conditioning rather
//! than the kernel's error, and [`forward_stable`] is the relation that can tell those apart.
//!
//! What it cannot tell apart is a reference that cancels: it measures `κ` of the *left* side, so
//! a reference formed by subtracting two large terms reports its own shortfall against the
//! kernel. Both digamma recurrences below are therefore oriented to compute the **larger** of
//! the two sides, which is the same rule the complement pairs use and is what makes a
//! conditioning budget meaningful rather than something to widen.

use euclid_scalar::Real;
use euclid_stability::relation::{Relative, agree, forward_stable};
use euclid_stability::{Config, generate, property};

/// Safety factor on `κ·u`, measured rather than inherited.
///
/// `64` came from the round-trip property; these pass at `8` and the digamma recurrence fails at
/// `4`, so `8` is where the budget binds. It is `8` rather than the `16`
/// [`super::conditioned`] settles at because both digamma recurrences are oriented to compute
/// the larger side — the reverse orientation cancels in the *reference*, which cost a factor of
/// two and read as the kernel's fault.
const SAFETY: f64 = 8.0;

/// Relative floor of a central difference at a near-optimal step, measured.
///
/// `(f(x+h) − f(x−h))/2h` carries `h²f‴/6` of truncation and `ε|f|/h` of amplified rounding, so
/// the best a step can do is about `ε^{2/3} ≈ 4e-11`. Measured worst over the `I`/`K` pairs on
/// `[0.5, 50]` is `2.8e-10`; this is that with an order of margin. It bounds the *instrument*,
/// which is why these two properties use [`agree`] rather than [`forward_stable`].
const DIFFERENCE_FLOOR: f64 = 3e-9;

mod gamma_family {
    use super::*;

    property! {
        #![config(Config::default().cases(500))]

        /// `Γ(x+1) = x·Γ(x)`, across the Lanczos/Stirling handover.
        ///
        /// The one property that constrains `Γ` at every argument without a reference value.
        /// `STIRLING_MIN` is `20`, so the generated range straddles it and the identity holds on
        /// both sides and across — a misplaced threshold shows up as a step, which no value
        /// table at round arguments would sit on.
        fn gamma_satisfies_its_recurrence_stably(
            x in generate::float::in_range(0.5, 60.0)
        ) {
            forward_stable(
                "gamma(x+1) = x gamma(x)",
                x,
                |v| Real::gamma(v + 1.0),
                |v| v * Real::gamma(v),
                SAFETY,
            )?;
        }
    }

    property! {
        #![config(Config::default().cases(500))]

        /// `lnΓ(x+1) = lnΓ(x) + ln x`, which is the additive form and far better conditioned.
        ///
        /// Worth having beside the multiplicative one because it reaches arguments `Γ` cannot:
        /// `lnΓ` is finite to `1e308` where `Γ` overflows at `171`. The `+ ln x` is a sum of two
        /// terms of the same sign once `x > 1`, so nothing cancels and this is a statement about
        /// `lnΓ` alone.
        fn log_gamma_satisfies_its_recurrence_stably(
            x in generate::float::in_range(1.0, 1e6)
        ) {
            forward_stable(
                "ln_gamma(x+1) = ln_gamma(x) + ln x",
                x,
                |v| Real::ln_gamma(v + 1.0),
                |v| Real::ln_gamma(v) + v.ln(),
                SAFETY,
            )?;
        }
    }

    property! {
        #![config(Config::default().cases(500))]

        /// `ψ(x) = ψ(x+1) − 1/x`, which is the orientation that does not cancel.
        ///
        /// # The identity is symmetric and the test is not
        ///
        /// Written the other way — `ψ(x+1) = ψ(x) + 1/x` — the reference is a *difference* of two
        /// `1/x`-sized terms: at `x = 0.5` it is `−1.9635 + 2 = 0.0365`, a 54-fold cancellation,
        /// and towards zero it is total because `ψ(x) ~ −1/x`. That version exceeded its budget
        /// by `1.5×` and the error was the reference's, not `digamma`'s — `forward_stable`
        /// measures `κ` of the left side, so it cannot see a reference that cancels and reports
        /// the shortfall against the kernel.
        ///
        /// Read in this direction the same identity is benign: `ψ(1.5) − 2 = −1.9635`, where the
        /// result is *larger* than either term and nothing is lost. So the fix is to compute the
        /// bigger of the two sides, which is the same rule the complement pairs below follow,
        /// and it needs no change to the relation and no widened tolerance.
        fn digamma_satisfies_its_recurrence_stably(
            x in generate::float::in_range(0.5, 1e4)
        ) {
            forward_stable(
                "digamma(x) = digamma(x+1) - 1/x",
                x,
                Real::digamma,
                |v| Real::digamma(v + 1.0) - v.recip(),
                SAFETY,
            )?;
        }
    }

    property! {
        #![config(Config::default().cases(500))]

        /// `ψ′(x) = ψ′(x+1) + 1/x²`, oriented the same way and for the same reason.
        ///
        /// Here both terms of the reference are *positive* — `ψ′` and `1/x²` are both positive on
        /// the positive axis — so this direction cannot cancel at all, at any argument. The other
        /// way round, `ψ′(x+1) = ψ′(x) − 1/x²`, is a subtraction that loses a factor of `2.5` at
        /// `x = 1` and everything as `x → 0`, where `ψ′(x) ~ 1/x²`.
        ///
        /// Which makes this the better of the two recurrences to hold `trigamma` to: it
        /// constrains the asymptotic series directly — the arm found short of its recorded floor
        /// earlier this quarter — with no cancellation anywhere in the comparison.
        fn trigamma_satisfies_its_recurrence_stably(
            x in generate::float::in_range(1.0, 1e4)
        ) {
            forward_stable(
                "trigamma(x) = trigamma(x+1) + 1/x^2",
                x,
                Real::trigamma,
                |v| Real::trigamma(v + 1.0) + (v * v).recip(),
                SAFETY,
            )?;
        }
    }

    property! {
        #![config(Config::default().cases(400))]

        /// `1/Γ(x)` agrees with `rgamma`, which is not the tautology it looks like.
        ///
        /// `rgamma` exists as its own kernel precisely so it stays finite where `Γ` overflows,
        /// so the two are separate implementations wherever the range allows both. Restricted to
        /// where `Γ` itself is finite, which is the only place the comparison is defined.
        fn the_reciprocal_gamma_agrees_with_the_quotient_stably(
            x in generate::float::in_range(0.5, 170.0)
        ) {
            forward_stable(
                "rgamma = 1/gamma",
                x,
                Real::rgamma,
                |v| Real::gamma(v).recip(),
                SAFETY,
            )?;
        }
    }
}

mod bessel_family {
    use super::*;

    property! {
        #![config(Config::default().cases(500))]

        /// `K₀ᵉ` and `K₁ᵉ` satisfy the three-term recurrence `K₂ = K₀ + (2/x)K₁`.
        ///
        /// Written in the scaled forms so the `e^{−x}` cancels and the identity is checkable at
        /// any argument. `K₂` is not on the public surface, so this is asserted in the direction
        /// that only needs the two that are: the recurrence rearranged to
        /// `K₀ᵉ = K₂ᵉ − (2/x)K₁ᵉ` would need `K₂`, while the derivative form below does not.
        ///
        /// # This one cannot use `forward_stable`, and the failure message said so backwards
        ///
        /// `K₀′ = −K₁` is the usable direction, by central difference with step `x·2^{-17}` —
        /// near the optimum, where the `h²` truncation and the `ε/h` amplification balance. That
        /// balance floors the *comparison* at about `ε^{2/3} ≈ 4e-11` relative, and that floor
        /// belongs to the difference formula rather than to either kernel.
        ///
        /// `forward_stable` bounds by `κ·u`, so it cannot express that: run on the `I` pair it
        /// reported `2.8e-10` against an admissible `1.1e-10` and concluded "the condition number
        /// is small, so this is an unstable kernel rather than a hard input" — a correct
        /// inference from a premise that did not hold, because the error was the instrument's.
        /// Absorbing it into the safety factor instead would have made the number meaningless.
        ///
        /// So these two use [`agree`] at the measured differencing floor. They still catch what a
        /// derivative check is for — a wrong order, a sign, a factor — and they no longer claim
        /// to be a stability statement about `K₁`.
        fn the_second_kind_derivative_matches_its_order_one_stably(
            x in generate::float::in_range(0.5, 50.0)
        ) {
            let step = x * f64::from(2.0f32).powi(-17);
            agree(
                "K0' = -K1",
                x,
                |v| {
                    let (lo, hi) = (Real::bessel_k0e(v - step), Real::bessel_k0e(v + step));
                    // Differencing the *scaled* form measures `(e^x K0)'`, which is
                    // `e^x (K0 + K0')`, so the exponential's own derivative comes back out.
                    (hi - lo) / (2.0 * step) - Real::bessel_k0e(v)
                },
                |v| -Real::bessel_k1e(v),
                &Relative(DIFFERENCE_FLOOR),
            )?;
        }
    }

    property! {
        #![config(Config::default().cases(500))]

        /// `I₀` and `I₁` satisfy `I₀′ = I₁`, in the scaled forms.
        ///
        /// The companion to the `K` derivative and the other half of what the Wronskian
        /// constrains jointly: the Wronskian can be satisfied by a pair that is wrong in a
        /// correlated way, while this pins `I₁` against `I₀`'s slope alone. Same differencing
        /// floor, and it is this one that measured `2.8e-10` and so set it.
        fn the_first_kind_derivative_matches_its_order_one_stably(
            x in generate::float::in_range(0.5, 50.0)
        ) {
            let step = x * f64::from(2.0f32).powi(-17);
            agree(
                "I0' = I1",
                x,
                |v| {
                    let (lo, hi) = (Real::bessel_i0e(v - step), Real::bessel_i0e(v + step));
                    (hi - lo) / (2.0 * step) + Real::bessel_i0e(v)
                },
                Real::bessel_i1e,
                &Relative(DIFFERENCE_FLOOR),
            )?;
        }
    }
}

mod complementary_pairs {
    use super::*;

    property! {
        #![config(Config::default().cases(500))]

        /// `P(a,x) + Q(a,x) = 1`, in the orientation that does not cancel.
        ///
        /// A complement is only a real check in the direction where the small tail is the
        /// subject: comparing `Q` against `1 − P` where `Q` is `1e-20` asserts nothing, because
        /// `1 − P` has already lost it. So each side is checked against the complement of the
        /// other only where it is the **larger** of the two, which is where the subtraction is
        /// benign — and the kernel reaches one through the other internally, so what this really
        /// pins is that the shared routing agrees with itself across the `x = a + 1` seam.
        fn the_incomplete_gamma_pair_complements_stably(
            x in generate::float::in_range(0.1, 200.0),
            a in generate::float::in_range(0.5, 100.0)
        ) {
            let (lower, upper) = (Real::gamma_lr(x, a), Real::gamma_ur(x, a));
            if lower >= 0.5 {
                forward_stable(
                    "P = 1 - Q where P is the larger",
                    x,
                    |v| Real::gamma_lr(v, a),
                    |v| 1.0 - Real::gamma_ur(v, a),
                    SAFETY,
                )?;
            }
            if upper >= 0.5 {
                forward_stable(
                    "Q = 1 - P where Q is the larger",
                    x,
                    |v| Real::gamma_ur(v, a),
                    |v| 1.0 - Real::gamma_lr(v, a),
                    SAFETY,
                )?;
            }
        }
    }

    property! {
        #![config(Config::default().cases(500))]

        /// `erfc(x) = 1 − erf(x)` where `erfc` is the larger, which is `x ≤ 0`.
        ///
        /// The same restriction and the same reason: above zero `erfc` decays and `1 − erf` has
        /// nothing left of it, which is why `erfc` is a separate kernel at all. Below zero
        /// `erfc ≥ 1` and the subtraction is harmless, so the identity becomes a statement about
        /// the two kernels agreeing rather than about the subtraction.
        fn the_error_function_pair_complements_stably(
            x in generate::float::in_range(-6.0, 0.0)
        ) {
            forward_stable(
                "erfc = 1 - erf on the negative axis",
                x,
                Real::erfc,
                |v| 1.0 - Real::erf(v),
                SAFETY,
            )?;
        }
    }
}
