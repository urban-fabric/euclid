//! Symmetry properties.

use super::*;

// ===== Symmetry ===========================================================

property! {
    #![config(settings("symmetry"))]

    /// `sin`, `tan`, `sinh`, `tanh`, `asinh` and `erf` are all **odd**: `f(-x) = -f(x)`.
    ///
    /// Catches a kernel that takes `abs` too early, or that handles the sign by a branch which
    /// disagrees with itself across zero. One ulp, since each is a single evaluation.
    fn the_odd_functions_are_odd(x in generate::float::in_range(-20.0, 20.0)) {
        classify!("near zero", x.abs() < 1e-8);
        classify!("ordinary", x.abs() >= 1e-8);

        relation::odd("sin", x, f64::sin, &Ulps(1))?;
        relation::odd("tan", x, f64::tan, &Ulps(1))?;
        relation::odd("sinh", x, f64::sinh, &Ulps(1))?;
        relation::odd("tanh", x, f64::tanh, &Ulps(1))?;
        relation::odd("asinh", x, Real::asinh, &Ulps(1))?;
        relation::odd("erf", x, Real::erf, &Ulps(1))?;
    }

    /// `cos`, `cosh` and `abs` are **even**: `f(-x) = f(x)`.
    ///
    /// Exact, not approximate: an even function of a negated argument should be computed from
    /// the same reduced argument, so the results ought to be bit-identical.
    fn the_even_functions_are_even(x in generate::float::in_range(-20.0, 20.0)) {
        relation::even("cos", x, f64::cos, &Ulps(0))?;
        relation::even("cosh", x, f64::cosh, &Ulps(0))?;
        relation::even("abs", x, f64::abs, &Ulps(0))?;
    }

    /// `erfc(-x) = 2 - erfc(x)`, the reflection that follows from `erf` being odd.
    ///
    /// **Absolute**, and the reason is a conditioning result the suite found rather than one
    /// anyone wrote down first. A relative tolerance of `1e-14` fails at `x = -5`, where the
    /// true value is `1.5e-12`: the right-hand side computes it by subtracting two numbers
    /// either side of 2, so its absolute error is an ulp of 2 — about `4.4e-16` — and against a
    /// result of `1.5e-12` that is a relative error of `3e-4`.
    ///
    /// The kernels are fine. The *identity* is ill-conditioned in this direction, and its honest
    /// accuracy claim is absolute at the scale of the larger operand. Loosening the relative
    /// tolerance until it passed would have hidden exactly the fact worth knowing: do not
    /// compute a small `erfc` by reflecting a large one.
    fn erfc_reflects_about_one(x in generate::float::in_range(-5.0, 5.0)) {
        classify!("reflection cancels", x < -2.0);
        relation::agree(
            "erfc(-x) = 2 - erfc(x)",
            x,
            |v: f64| Real::erfc(-v),
            |v: f64| 2.0 - Real::erfc(v),
            &Absolute(2e-15),
        )?;
    }
}
