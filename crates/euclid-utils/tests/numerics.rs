//! The test-support numerics, property-tested — because every other suite's
//! assertions rest on them.
//!
//! `euclid-utils` is the crate the rest of the fleet's tests *measure with*:
//! `derivative` checks that a closed form matches a numerical slope,
//! `integrate` that a density integrates to one, `sweep` that a domain was
//! covered. A defect in any of them does not fail a test. It weakens one, which
//! is strictly worse — a suite that still passes while no longer checking what
//! it claims.
//!
//! So this is the one place in the fleet where testing the measuring instrument
//! is the point rather than a detour.
//!
//! ## These are exactness claims, not tolerance claims
//!
//! Both quadrature rules here are exact on a class of functions, and that is
//! what makes them testable without an oracle:
//!
//! | rule | exact for |
//! |---|---|
//! | `derivative`'s five-point stencil `(8·d₁ − d₂)/(12h)` | polynomials up to degree 4 |
//! | `integrate`'s composite Simpson | polynomials up to degree 3 |
//!
//! Those are algebraic facts about the coefficients, so a single mistyped weight
//! breaks them while leaving the `O(h⁴)` accuracy on a general function looking
//! roughly fine. A tolerance-based test against `sin` would not notice; an
//! exactness test against a cubic notices immediately.
//!
//! "Exact" here still means *to a few ULP of the terms involved*, not bit-equal.
//! The rule is exact in ℝ; in `f64` the stencil forms `8·d₁ − d₂` and divides by
//! `12h`, which rounds. The bound is taken relative to the magnitude of the
//! terms rather than of the answer, for the reason the planar suites record: a
//! derivative that is near zero where the function values are not is a
//! cancellation, and demanding a small *relative* error there would be
//! asserting arithmetic rather than calculus.
//!
//! ## Linearity is the sharpest property of the pair
//!
//! Both operators are linear, and linearity is checkable on *generated*
//! functions rather than named ones — which is what makes it strong. A stencil
//! with a transposed weight is still linear, so linearity alone is not enough;
//! combined with exactness on a polynomial basis it pins every coefficient.

use euclid_stability::{Config, Gen, check, classify, generate, property, target};
use euclid_utils::{derivative, integrate, log_sweep, rel, sweep};

fn settings(suite: &str) -> Config {
    Config::default()
        .cases(500)
        .regressions(format!("tests/regressions/{suite}.txt"))
}

/// Arguments spanning the range the helpers are used over, by decade.
///
/// Exponents are **binary**, so this is roughly `1e-3` to `1e3`. Zero and one
/// are included because `derivative`'s step is `h · max(|x|, 1)`, so `x = 0` is
/// the one argument where the step does not scale with the argument.
fn argument() -> Gen<f64> {
    generate::weighted(vec![
        (2, generate::just(0.0)),
        (2, generate::just(1.0)),
        (2, generate::just(-1.0)),
        (12, generate::float::magnitude(-10, 10)),
    ])
}

/// A small polynomial coefficient, kept modest so a quartic's terms stay well
/// inside `f64` across the argument range.
fn coefficient() -> Gen<f64> {
    generate::float::in_range(-4.0, 4.0)
}

/// Five coefficients: a generated quartic `c₀ + c₁x + c₂x² + c₃x³ + c₄x⁴`.
type Quartic = [f64; 5];

fn quartic() -> Gen<Quartic> {
    generate::pair(
        generate::triple(coefficient(), coefficient(), coefficient()),
        generate::pair(coefficient(), coefficient()),
    )
    .map(|((c0, c1, c2), (c3, c4))| [c0, c1, c2, c3, c4])
}

fn eval(c: &Quartic, x: f64) -> f64 {
    // Horner, so the evaluation itself is as well conditioned as it can be.
    c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * c[4])))
}

/// The analytic derivative of [`eval`].
fn eval_derivative(c: &Quartic, x: f64) -> f64 {
    c[1] + x * (2.0 * c[2] + x * (3.0 * c[3] + x * 4.0 * c[4]))
}

/// The analytic antiderivative of [`eval`], evaluated at `x`.
fn eval_integral(c: &Quartic, x: f64) -> f64 {
    x * (c[0] + x * (c[1] / 2.0 + x * (c[2] / 3.0 + x * (c[3] / 4.0 + x * c[4] / 5.0))))
}

/// The scale a stencil's rounding is proportional to: the size of the function
/// values it differenced, divided by the step.
///
/// Not the size of the answer. A cancelling derivative is small where the terms
/// are not, and a relative-to-answer bound would be demanding precision the
/// subtraction cannot deliver.
fn derivative_scale(c: &Quartic, x: f64) -> f64 {
    let h = 1e-5 * x.abs().max(1.0);
    let terms = [x - 2.0 * h, x - h, x, x + h, x + 2.0 * h]
        .iter()
        .map(|&t| eval(c, t).abs())
        .fold(0.0f64, f64::max);
    (terms / h).max(1.0)
}

// ===== derivative =========================================================

property! {
    #![config(settings("derivative"))]

    /// The five-point stencil is **exact on a quartic**.
    ///
    /// `(8·d₁ − d₂)/(12h)` annihilates every term up to `x⁴`, so this pins all
    /// three coefficients at once. A transposed weight — `(8·d₂ − d₁)`, or a
    /// divisor of `6h` — leaves the rule still linear and still roughly
    /// `O(h²)`-accurate on `sin`, and fails here on the first cubic.
    fn the_stencil_is_exact_on_a_quartic(c in quartic(), x in argument()) {
        classify!(if x == 0.0 { "at zero" } else { "away from zero" });
        let got = derivative(|t| eval(&c, t), x);
        let want = eval_derivative(&c, x);

        let slack = 1e-9 * derivative_scale(&c, x);
        let error = (got - want).abs();
        target(error / slack);

        check!(
            error <= slack,
            "d/dx of {c:?} at {x}: stencil gave {got}, analytic is {want} \
             (error {error:e}, slack {slack:e})"
        );
    }

    /// Differentiation is linear: `d(αf + βg) = α·d(f) + β·d(g)`.
    ///
    /// Checkable on generated functions rather than named ones, which is what
    /// makes it a strong claim — but not sufficient alone, since a stencil with
    /// a transposed weight is still linear. It pins the operator's *shape*
    /// while `the_stencil_is_exact_on_a_quartic` pins its coefficients.
    fn differentiation_is_linear(
        f in quartic(),
        g in quartic(),
        x in argument(),
        scale in coefficient(),
    ) {
        classify!(if scale == 0.0 { "degenerate scale" } else { "ordinary" });
        let combined = derivative(|t| scale * eval(&f, t) + eval(&g, t), x);
        let separate = scale * derivative(|t| eval(&f, t), x) + derivative(|t| eval(&g, t), x);

        let slack = 1e-9
            * (derivative_scale(&f, x) * scale.abs().max(1.0) + derivative_scale(&g, x));
        check!(
            (combined - separate).abs() <= slack,
            "linearity broke at {x}: combined {combined}, separate {separate}"
        );
    }

    /// The derivative of an even function is odd.
    ///
    /// A symmetry rather than a value, so it needs no analytic reference at all.
    /// It catches a stencil that is not centred — one that sampled `x` and
    /// `x + h` instead of straddling `x` — which both properties above would
    /// otherwise let through at a reduced accuracy that still looks plausible.
    fn the_derivative_of_an_even_function_is_odd(c in quartic(), x in argument()) {
        classify!(if x == 0.0 { "at zero" } else { "away from zero" });
        // Force evenness by using only the even-power coefficients.
        let even = [c[0], 0.0, c[2], 0.0, c[4]];
        let at_plus = derivative(|t| eval(&even, t), x);
        let at_minus = derivative(|t| eval(&even, t), -x);

        let slack = 1e-9 * derivative_scale(&even, x);
        check!(
            (at_plus + at_minus).abs() <= slack,
            "d/dx of an even function should be odd, but gave {at_plus} at {x} and \
             {at_minus} at {}",
            -x
        );
    }
}

// ===== integrate ==========================================================

/// Integration limits: a modest interval, so a quartic's antiderivative stays
/// well inside `f64`.
fn limits() -> Gen<(f64, f64)> {
    generate::pair(
        generate::float::in_range(-8.0, 8.0),
        generate::float::in_range(-8.0, 8.0),
    )
}

property! {
    #![config(settings("integrate"))]

    /// Composite Simpson is **exact on a cubic**.
    ///
    /// The defining property of the rule, and the one that pins its `1, 4, 2,
    /// 4, …, 1` weights and its `h/3`. Degree four is *not* claimed: Simpson
    /// integrates a quartic with an `O(h⁴)` error, so the cubic is the sharp
    /// boundary and asserting it is what distinguishes a correct rule from a
    /// nearly-correct one.
    fn simpson_is_exact_on_a_cubic(c in quartic(), ab in limits()) {
        let (a, b) = ab;
        // Drop the quartic term: this is the exactness class, not beyond it.
        let cubic = [c[0], c[1], c[2], c[3], 0.0];
        classify!(if (b - a).abs() < 1e-6 { "degenerate interval" } else { "ordinary" });

        let got = integrate(|t| eval(&cubic, t), a, b, 8);
        let want = eval_integral(&cubic, b) - eval_integral(&cubic, a);

        // Relative to the integrand's own scale over the interval, not to the
        // answer: an integral that cancels to near zero is not computable to a
        // tight relative bound.
        let magnitude = eval(&cubic, a).abs().max(eval(&cubic, b).abs()) * (b - a).abs();
        let slack = 1e-10 * magnitude.max(1.0);
        let error = (got - want).abs();
        target(error / slack);

        check!(
            error <= slack,
            "∫{cubic:?} over [{a}, {b}]: Simpson gave {got}, analytic is {want} \
             (error {error:e}, slack {slack:e})"
        );
    }

    /// Reversing the limits negates the integral.
    fn reversing_the_limits_negates_the_integral(c in quartic(), ab in limits()) {
        let (a, b) = ab;
        classify!(if a == b { "empty interval" } else { "ordinary" });
        let forward = integrate(|t| eval(&c, t), a, b, 16);
        let backward = integrate(|t| eval(&c, t), b, a, 16);
        let scale = forward.abs().max(backward.abs()).max(1.0);
        check!(
            (forward + backward).abs() <= 1e-12 * scale,
            "∫[{a},{b}] = {forward} but ∫[{b},{a}] = {backward}"
        );
    }

    /// Splitting the interval splits the integral.
    ///
    /// Additivity, and the metamorphic relation that needs no antiderivative at
    /// all. Both halves use the same panel count as the whole, so the two sides
    /// are genuinely different computations rather than a rearrangement of one.
    fn integration_is_additive_over_a_split(c in quartic(), ab in limits()) {
        let (a, b) = ab;
        let mid = a * 0.5 + b * 0.5;
        classify!(if (b - a).abs() < 1e-6 { "degenerate interval" } else { "ordinary" });

        // A cubic, so each piece is exact and the split is testable tightly.
        let cubic = [c[0], c[1], c[2], c[3], 0.0];
        let whole = integrate(|t| eval(&cubic, t), a, b, 8);
        let left = integrate(|t| eval(&cubic, t), a, mid, 8);
        let right = integrate(|t| eval(&cubic, t), mid, b, 8);

        let scale = whole.abs().max(left.abs() + right.abs()).max(1.0);
        check!(
            (whole - (left + right)).abs() <= 1e-10 * scale,
            "∫[{a},{b}] = {whole} but the halves sum to {}",
            left + right
        );
    }

    /// More panels never make a quartic's integral worse.
    ///
    /// The `O(h⁴)` convergence claim, stated as monotone improvement rather
    /// than as a rate — a rate would be asserting the constant too. Uses a
    /// quartic deliberately, since a cubic is already exact and has no
    /// convergence to observe.
    fn more_panels_converge_on_a_quartic(c in quartic(), ab in limits()) {
        let (a, b) = ab;
        if (b - a).abs() < 1e-3 {
            return Ok(());
        }
        classify!("ordinary");
        let want = eval_integral(&c, b) - eval_integral(&c, a);
        let err = |panels: usize| (integrate(|t| eval(&c, t), a, b, panels) - want).abs();

        let coarse = err(4);
        let fine = err(64);
        let scale = want.abs().max(1.0);
        // Refining may already be at the rounding floor, hence the absolute
        // term: what is excluded is refinement making things *worse*.
        check!(
            fine <= coarse + 1e-12 * scale,
            "refining made it worse: 4 panels gave error {coarse:e}, 64 gave {fine:e}"
        );
    }
}

// ===== sweeps =============================================================

property! {
    #![config(settings("sweeps"))]

    /// A sweep starts and ends **exactly** on its bounds.
    ///
    /// Exactness is the whole point of this one. A sweep exists to cover a
    /// domain, and a suite that sweeps to a boundary in order to test behaviour
    /// *at* that boundary misses it entirely if the last point is a ULP short —
    /// silently, since the test still passes on the point just inside.
    fn a_sweep_hits_both_of_its_bounds_exactly(bounds in limits(), n in generate::index(2, 64)) {
        let (lo, hi) = bounds;
        classify!(if lo == hi { "degenerate range" } else { "ordinary" });
        let points: Vec<f64> = sweep(lo, hi, n).collect();

        check!(points.len() == n, "sweep({lo}, {hi}, {n}) gave {} points", points.len());
        check!(
            points.first() == Some(&lo),
            "sweep({lo}, {hi}, {n}) starts at {:?}, not {lo}",
            points.first()
        );
        check!(
            points.last() == Some(&hi),
            "sweep({lo}, {hi}, {n}) ends at {:?}, not {hi}",
            points.last()
        );
    }

    /// A sweep is monotone in the direction of its bounds.
    fn a_sweep_is_monotone(bounds in limits(), n in generate::index(2, 64)) {
        let (lo, hi) = bounds;
        classify!(if hi >= lo { "ascending" } else { "descending" });
        let points: Vec<f64> = sweep(lo, hi, n).collect();
        for w in points.windows(2) {
            if hi >= lo {
                check!(w[1] >= w[0], "ascending sweep stepped back: {:?}", w);
            } else {
                check!(w[1] <= w[0], "descending sweep stepped forward: {:?}", w);
            }
        }
    }

    /// A log sweep hits both bounds and has a constant ratio between steps.
    ///
    /// The ratio is the defining property — a log sweep whose points were
    /// evenly spaced *linearly* would still hit its bounds and still be
    /// monotone, so neither of those catches the mistake this one does.
    fn a_log_sweep_has_a_constant_ratio(n in generate::index(3, 32)) {
        let (lo, hi) = (1e-6, 1e6);
        classify!("decades");
        let points: Vec<f64> = log_sweep(lo, hi, n).collect();
        check!(points.len() == n, "log_sweep gave {} points, not {n}", points.len());
        check!(
            rel(points[0], lo) < 1e-12,
            "log_sweep starts at {}, not {lo}",
            points[0]
        );
        check!(
            rel(points[n - 1], hi) < 1e-12,
            "log_sweep ends at {}, not {hi}",
            points[n - 1]
        );

        let first_ratio = points[1] / points[0];
        for w in points.windows(2) {
            check!(
                rel(w[1] / w[0], first_ratio) < 1e-9,
                "ratio drifted: {} against the first {first_ratio}",
                w[1] / w[0]
            );
        }
    }
}
