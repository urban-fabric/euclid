//! Error budgets set by the problem's conditioning rather than by a chosen literal.
//!
//! euclid-stability's header says the numerical instruments are the reason for the crate's name,
//! and across this suite they were the least-used part of it: `agree` appeared 91 times and
//! `forward_stable` three. The identity laws were wired thoroughly and the *stability* laws
//! almost not at all.
//!
//! The distinction matters for the reason [`forward_stable`] states: an error of `1e-9` is a
//! scandal at `κ = 1` and excellent at `κ = 1e7`, so a fixed tolerance is either too loose to
//! catch an unstable kernel in the benign region or too tight to pass a good one near a
//! singularity — and the usual resolution is to loosen it until the suite is green, which
//! discards the finding. Three fixed tolerances written by hand this quarter had exactly that
//! problem; two of them had to be renormalised by an amplification factor after the fact, which
//! is this instrument reinvented badly.
//!
//! # What can serve as the reference
//!
//! [`forward_stable`] needs "a more accurate computation of the same quantity". The kernels'
//! *second routes* — `k0_series` against the cosh quadrature, the three arrangements of the
//! gamma prefactor — are `pub(crate)` and so invisible from an integration test, which is the
//! right call for them and a real constraint here. What is reachable from outside is better
//! anyway: the **closed forms**, which are not more accurate approximations but exact answers,
//! and the **identities**, which tie two independently implemented families together.
//!
//! So every reference below is one of:
//!
//! - a closed form the kernel must reproduce (`I_x(a,1) = x^a`, `P(1,x) = −expm1(−x)`);
//! - a cross-family identity (`P(½,x) = erf(√x)`, the `I`/`K` Wronskian `= 1/x`).
//!
//! `condition_at` estimates `κ` by finite difference in **one** variable, so for the
//! two-argument kernels the shape is held fixed and `x` is varied. That is the right partial to
//! take: `x` is where the cancellation lives in every one of these.

use euclid_scalar::Real;
use euclid_stability::relation::{
    Exact, Relative, Ulps, associative, fixed_point, forward_stable, idempotent, insensitive_to,
    order_independent,
};
use euclid_stability::{Config, generate, property};

/// Safety factor on `κ·u`. `64` matches the round-trip property's choice and is what a kernel
/// built from a handful of correctly-rounded operations is entitled to.
const SAFETY: f64 = 64.0;

mod closed_forms {
    use super::*;

    property! {
        #![config(Config::default().cases(400))]

        /// `I_x(a, 1) = x^a`, with the budget set by the conditioning rather than by a literal.
        ///
        /// The closed form is exact, so this is the cleanest reference the beta family has: any
        /// disagreement is the kernel's. `b = 1` takes `BetaRegMethod::UnitUpper`, which *is*
        /// `x^a` — so the value is near-tautological and the **conditioning** is the point. It
        /// is the companion arm, `a = 1`, that the budget actually constrains.
        fn the_unit_shape_beta_is_forward_stable(
            x in generate::float::in_range(1e-3, 0.999),
            a in generate::float::in_range(0.5, 40.0)
        ) {
            forward_stable(
                "I_x(a,1) = x^a",
                x,
                |v| Real::beta_reg(v, a, 1.0),
                |v| v.powf(a),
                SAFETY,
            )?;
        }
    }

    property! {
        #![config(Config::default().cases(400))]

        /// `I_x(1, b) = 1 − (1−x)^b`, which is the arm with something to lose.
        ///
        /// `1 − (1−x)^b` cancels as `x → 0`: the power is near one and the subtraction keeps
        /// `x·b` of it. `κ` knows that, which is why the bound may be met here at an absolute
        /// error a flat tolerance would have to be widened to accept.
        fn the_unit_lower_beta_is_forward_stable(
            x in generate::float::in_range(1e-6, 0.999),
            b in generate::float::in_range(0.5, 40.0)
        ) {
            forward_stable(
                "I_x(1,b) = 1 - (1-x)^b",
                x,
                |v| Real::beta_reg(v, 1.0, b),
                |v| -(-v).ln_1p().mul_add(b, 0.0).exp_m1(),
                SAFETY,
            )?;
        }
    }

    property! {
        #![config(Config::default().cases(400))]

        /// `P(1, x) = −expm1(−x)`, exact at every argument, and `P(2, x) = 1 − e^{−x}(1+x)`
        /// above where that second form stops being usable.
        ///
        /// The shape-1 and shape-2 closed forms, which the kernel reaches through its series or
        /// its continued fraction depending on `x` — so this crosses a route boundary without
        /// naming it, and the budget follows the conditioning across it.
        ///
        /// # The shape-2 form is itself a cancelling difference, which this property found
        ///
        /// `1 − (1+x)e^{−x}` is `1 − 1` to leading order: the answer is `x²/2`, the terms are
        /// `O(1)`, so it keeps about `x²/2` of its digits. At `x = 1e-8` it returns
        /// `1.11e-16` where the true value is `5.0e-17` — and the first version of this property
        /// reported the *kernel* as not forward stable there, on a reference that was wrong by a
        /// factor of two. That is the standing trap exactly: a kernel that exists because the
        /// naive form cancels cannot be checked against the naive form.
        ///
        /// So the shape-2 arm starts at `0.25`, where the relative loss `2ε/x²` is `7e-15` and
        /// below the budget rather than above it. `P(1, ·)` needs no such bound: `−expm1(−x)` is
        /// the non-cancelling spelling, which is why it is the one that runs to `1e-8`.
        fn the_integer_shape_incomplete_gamma_is_forward_stable(
            x in generate::float::in_range(1e-8, 40.0)
        ) {
            forward_stable(
                "P(1,x) = -expm1(-x)",
                x,
                |v| Real::gamma_lr(v, 1.0),
                |v| -(-v).exp_m1(),
                SAFETY,
            )?;
            if x >= 0.25 {
                forward_stable(
                    "P(2,x) = 1 - e^-x (1+x)",
                    x,
                    |v| Real::gamma_lr(v, 2.0),
                    |v| -((-v).exp() * (1.0 + v) - 1.0),
                    SAFETY,
                )?;
            }
        }
    }
}

mod cross_family {
    use super::*;

    property! {
        #![config(Config::default().cases(400))]

        /// `P(½, x) = erf(√x)`, which ties the incomplete gamma to the error function.
        ///
        /// Two independently implemented families, so neither is the other's tautology: `erf`
        /// routes through `gamma_lr(½, ·)` only above its own linear threshold, and below that
        /// it is its own series. The budget is `κ(P(½,·))·u`, which near zero is large because
        /// `P(½,x) ~ 2√(x/π)` has a square-root singularity in its derivative — exactly the
        /// place a literal tolerance would have been set from the benign end and then widened.
        fn the_half_shape_gamma_agrees_with_erf_stably(
            x in generate::float::in_range(1e-10, 30.0)
        ) {
            forward_stable(
                "P(1/2,x) = erf(sqrt x)",
                x,
                |v| Real::gamma_lr(v, 0.5),
                |v| Real::erf(v.sqrt()),
                SAFETY,
            )?;
        }
    }

    property! {
        #![config(Config::default().cases(400))]

        /// The modified Wronskian `I₀ᵉ(x)K₁ᵉ(x) + I₁ᵉ(x)K₀ᵉ(x) = 1/x`, exactly.
        ///
        /// An identity-derived reference, and the strongest one the `K` family has reachable
        /// from outside the crate: `1/x` is exact, the four kernels are separate
        /// implementations, and the scaled forms make it checkable at any argument because the
        /// `e^{±x}` factors cancel pairwise.
        ///
        /// Its floor is the `I` family's, not `K`'s — `i0e`/`i1e` carry about `1.7e-14`
        /// relative, which flows straight into the products — so this is a `K` property only in
        /// the sense that a `K` regression would also break it. `forward_stable` reports `κ`
        /// beside the error, which is what lets a reader tell those apart.
        fn the_modified_wronskian_is_forward_stable(
            x in generate::float::in_range(1e-4, 1e4)
        ) {
            forward_stable(
                "i0e*k1e + i1e*k0e = 1/x",
                x,
                |v| Real::bessel_i0e(v) * Real::bessel_k1e(v)
                    + Real::bessel_i1e(v) * Real::bessel_k0e(v),
                f64::recip,
                SAFETY,
            )?;
        }
    }
}

/// The relations this suite had never used, applied where they say something.
///
/// Six of euclid-stability's eighteen had zero uses. They are applied here because each states a
/// fact about a kernel that no other property in the tree asserts — not for the sake of the
/// count. `distributes` is the one left out: over floats, `a(b+c) = ab+ac` is false by rounding
/// at almost every argument, so asserting it would mean choosing a tolerance that admits the
/// rounding, and that is the thing this file exists to avoid.
mod unused_relations {
    use super::*;

    property! {
        #![config(Config::default().cases(300))]

        /// `ln_add_exp` is associative, which is what makes a streaming log-sum-exp legitimate.
        ///
        /// `ln(e^a + e^b)` is a semigroup operation, so folding a sequence of log-weights in any
        /// nesting must give the same answer. Callers rely on this every time they accumulate
        /// log-likelihoods incrementally rather than in one pass, and nothing here asserted it.
        /// # The bound is not a constant, and finding that out is what this property was for
        ///
        /// `Ulps(4)` was the guess; then `Ulps(32)`; the generator kept finding worse. The
        /// reason is structural rather than a defect. One nesting passes through an intermediate
        /// log-weight of magnitude `300` — `op(−300, −299.9999994)` is `−299.30685` — and a
        /// double of that size has absolute resolution `ulp(300)/2 = 5.7e-14`. Exponentiating it
        /// turns that absolute error into *relative* error of the answer, so the two nestings can
        /// differ by `|log weight|·ε` however well each step is implemented.
        ///
        /// So associativity here is conditioned on the magnitude of the logs, and the comparator
        /// has to say so: `8·max|a,b,c|·ε`, which at `±300` is `5.3e-13` against a measured
        /// `2.0e-14`. A flat bound either excludes the large-magnitude inputs — the ones a
        /// streaming log-likelihood actually produces — or is loose enough at small magnitudes to
        /// assert nothing. This is the same reason the rest of this file exists, arriving at a
        /// relation that does not take a `κ`.
        fn log_add_exp_is_associative(
            a in generate::float::in_range(-300.0, 300.0),
            b in generate::float::in_range(-300.0, 300.0),
            c in generate::float::in_range(-300.0, 300.0)
        ) {
            let magnitude = a.abs().max(b.abs()).max(c.abs()).max(1.0);
            associative(
                "ln_add_exp",
                a,
                b,
                c,
                Real::ln_add_exp,
                &Relative(8.0 * magnitude * f64::EPSILON),
            )?;
        }
    }

    property! {
        #![config(Config::default().cases(300))]

        /// Compensated summation does not depend on the order it is handed its terms.
        ///
        /// The claim a compensated sum is *for*: a naive sum over a cancelling sequence depends
        /// on the order, and that is the defect `compensated::sum` exists to remove. Asserting
        /// it needs a permutation, which is why `order_independent` takes one — and a generated
        /// permutation is the only honest way to produce it.
        fn compensated_summation_is_order_independent(
            xs in generate::vec(generate::float::in_range(-1e8, 1e8), 2, 12),
            seed in generate::int(i64::MIN, i64::MAX)
        ) {
            let order = permutation(xs.len(), seed as u64);
            order_independent(
                "compensated::sum",
                &xs,
                &order,
                |slice: &[f64]| euclid_scalar::compensated::sum(slice),
                &Ulps(2),
            )?;
        }
    }

    property! {
        #![config(Config::default().cases(300))]

        /// `trunc` and `abs` are idempotent, exactly.
        ///
        /// The one place `Exact` is the right comparator: both land on a value already in their
        /// own image, so a second application must change nothing at all. A kernel that rounded
        /// instead of truncating would pass a tolerance here and fail this.
        fn the_rounding_kernels_are_idempotent(x in generate::float::any()) {
            if !x.is_finite() {
                return Ok(());
            }
            idempotent("trunc", x, f64::trunc, &Exact)?;
            idempotent("abs", x, f64::abs, &Exact)?;
            idempotent("floor", x, f64::floor, &Exact)?;
        }
    }

    property! {
        #![config(Config::default().cases(300))]

        /// `Γ(1) = 1` and `1/Γ(1) = 1`: one is a genuine fixed point of both.
        ///
        /// `fixed_point` asserts `f(x) = x`, and `Γ` has exactly one at `x = 1` among the
        /// arguments a caller is likely to pass. Worth pinning because it is where the
        /// reduced-argument route meets the Lanczos one, and because an off-by-one in the
        /// recurrence `Γ(x) = (x−1)Γ(x−1)` lands here first.
        fn gamma_fixes_one_exactly(_ignored in generate::int(0i64, 0i64)) {
            fixed_point("gamma", 1.0, Real::gamma, &Ulps(0))?;
            fixed_point("rgamma", 1.0, Real::rgamma, &Ulps(0))?;
            // And zero is a fixed point of every odd kernel, which is a different statement
            // from oddness: `f(−0) = −f(0)` holds for a kernel that returns `NaN` at zero.
            fixed_point("erf", 0.0, Real::erf, &Exact)?;
            fixed_point("erf_inv", 0.0, Real::erf_inv, &Exact)?;
            fixed_point("sinh", 0.0, f64::sinh, &Exact)?;
        }
    }

    property! {
        #![config(Config::default().cases(400))]

        /// `hypot` is insensitive to the signs of its arguments, and `ln_beta` to their order.
        ///
        /// `insensitive_to` is the relation this suite most needed and least had: it says a
        /// result does not move under a perturbation that should not matter. That is the shape
        /// of a real defect — a prefactor that quietly depends on how `a + b` rounded is
        /// *sensitive* to something it has no business seeing — and a property that varies the
        /// irrelevant thing is how you find it.
        fn the_sign_and_order_insensitive_kernels_do_not_move(
            a in generate::float::in_range(1e-3, 1e3),
            b in generate::float::in_range(1e-3, 1e3)
        ) {
            insensitive_to(
                "hypot under a sign flip",
                (a, b),
                |(l, r)| (-l, -r),
                |(l, r)| Real::hypot(l, r),
                &Exact,
            )?;
            insensitive_to(
                "ln_beta under argument order",
                (a, b),
                |(l, r)| (r, l),
                |(l, r)| Real::ln_beta(l, r),
                &Exact,
            )?;
            insensitive_to(
                "beta under argument order",
                (a, b),
                |(l, r)| (r, l),
                |(l, r)| Real::beta(l, r),
                &Relative(1e-15),
            )?;
        }
    }
}

/// A permutation of `0..len`, from a generated seed by Fisher-Yates over a SplitMix stream.
///
/// `generate::vec` cannot produce a permutation directly — its elements are drawn independently
/// — and `order_independent` needs one. Driving the shuffle from a generated integer keeps the
/// whole thing on the choice tape, so a failure shrinks and replays like any other.
fn permutation(len: usize, seed: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..len).collect();
    let mut state = seed | 1;
    for index in (1..len).rev() {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        order.swap(index, (z % (index as u64 + 1)) as usize);
    }
    order
}
