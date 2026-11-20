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

// ===== Parity of the special functions, which is exact ====================

/// A named kernel, for the parity tables below.
type Parity = (&'static str, fn(f64) -> f64);

property! {
    #![config(settings("special_parity").cases(500))]

    /// The special functions with a parity have it **to the bit**, not to a tolerance.
    ///
    /// # Why this is an exact assertion where the sibling above is not
    ///
    /// `the_odd_functions_are_odd` allows an ulp, because `sin`, `tanh` and friends are the
    /// platform's and reduce their argument independently on each side. Every kernel here is
    /// euclid's own, and each computes on `|x|` and reapplies the sign at the end — so the two
    /// sides share all of their arithmetic and can only differ in that final negation. Measured
    /// over 20,000 arguments spanning `1e-320` to `1`, every one of these is **0 ulp**
    /// asymmetric. An exact assertion is therefore the honest one, and it is far stronger: it
    /// pins the sign handling in every branch at once.
    ///
    /// That matters because the sign path is genuinely easy to get wrong and invisible to a
    /// one-sided test. `cargo mutants` previously found that deleting the negation in the `f32`
    /// `I₁ᵉ` left the whole suite green — the kernel could have been *even* and nothing noticed,
    /// because nothing passed it a negative argument.
    ///
    /// `atanh` is deliberately absent. It is the platform's `f64::atanh` rather than a euclid
    /// kernel — `Real::atanh(0.5)` and `0.5f64.atanh()` agree bit for bit — and it is **87 ulp**
    /// asymmetric at `x = 0.999`. Asserting parity on it would be asserting something about libm.
    fn the_special_functions_have_exact_parity(x in generate::float::magnitude(-320, 3)) {
        let x = x.abs();
        if x <= 0.0 || !x.is_finite() {
            return Ok(());
        }
        classify!("below one", x < 1.0);
        classify!("above one", x >= 1.0);

        // Odd: `f(-x)` must be the exact negation of `f(x)`.
        let odd: [Parity; 6] = [
            ("bessel_i1", Real::bessel_i1),
            ("bessel_i1e", Real::bessel_i1e),
            ("bessel_i1_over_i0", Real::bessel_i1_over_i0),
            ("bessel_j1", Real::bessel_j1),
            ("sin_int", Real::sin_int),
            ("cbrt", Real::cbrt),
        ];
        for (name, f) in odd {
            let (plus, minus) = (f(x), f(-x));
            if !(plus.is_finite() && minus.is_finite()) {
                continue;
            }
            check!(
                minus.to_bits() == (-plus).to_bits(),
                "{name} must be exactly odd: f({x:e}) = {plus:e} but f(-{x:e}) = {minus:e}"
            );
        }

        // Even: `f(-x)` must be bit-identical to `f(x)`.
        let even: [Parity; 6] = [
            ("bessel_i0", Real::bessel_i0),
            ("bessel_i0e", Real::bessel_i0e),
            ("ln_bessel_i0", Real::ln_bessel_i0),
            ("bessel_j0", Real::bessel_j0),
            ("sinc", Real::sinc),
            ("cos_m1", Real::cos_m1),
        ];
        for (name, f) in even {
            let (plus, minus) = (f(x), f(-x));
            if !(plus.is_finite() && minus.is_finite()) {
                continue;
            }
            check!(
                minus.to_bits() == plus.to_bits(),
                "{name} must be exactly even: f({x:e}) = {plus:e} but f(-{x:e}) = {minus:e}"
            );
        }

        // `erf_inv` is odd on its own domain, which is the unit interval rather than the axis.
        if x < 1.0 {
            let (plus, minus) = (Real::erf_inv(x), Real::erf_inv(-x));
            if plus.is_finite() && minus.is_finite() {
                check!(
                    minus.to_bits() == (-plus).to_bits(),
                    "erf_inv must be exactly odd: f({x:e}) = {plus:e} but f(-{x:e}) = {minus:e}"
                );
            }
        }
    }
}
