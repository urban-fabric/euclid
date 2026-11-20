//! The [`Real`](crate::Real) contract: the arithmetic and the special functions over it.
//!
//! The largest suite in the crate, and the one a carrier author is most likely to be checking.
//! Relations rather than reference values throughout — `exp(ln x) == x` to a tolerance, the
//! ordering laws, the sign and rounding agreements — so none of it depends on which algorithm
//! the carrier reached for.

use super::*;

/// The real-analysis laws, to within `tolerance`.
///
/// Relative comparison: `|got − want| ≤ tolerance · max(1, |want|)`, matching the convention in
/// `euclid-utils` — purely relative is wrong where a quantity passes through zero, purely
/// absolute is wrong where it is large.
///
/// Inverse pairs and identities, which is what makes these metamorphic rather than a table of
/// expected digits:
///
/// - `ln(exp(x)) = x` and `exp(ln(x)) = x`
/// - `sqrt(x)² = x` and `sqrt(x²) = |x|`
/// - `sin²(x) + cos²(x) = 1`
/// - `ln_1p(x) = ln(1 + x)` and `exp_m1(x) = exp(x) − 1`, which exist precisely to be more
///   accurate than the naive forms near zero, so agreeing away from zero is the check
/// - `atan(tan(x)) = x` on the principal branch
///
/// Exact laws, checked without the tolerance: `scalar` round-trips through `scalar_like`,
/// `abs` is idempotent, `min`/`max` are commutative and return an argument, and the rounding
/// family is idempotent.
///
/// # Sample requirements
///
/// Use values in `(0, π/2)`. The logarithm laws require positive inputs, and
/// `atan(tan(x))` requires inputs on the principal branch. Include `1` and a value
/// on either side to exercise series crossovers.
pub fn real_laws<T>(xs: &[T], tolerance: f64, what: &str)
where
    T: Real + PartialEq + core::fmt::Debug,
    T::Mask: PartialEq + core::fmt::Debug,
{
    let close = |got: &T, want: &T, law: &str, x: &T| {
        // Compare through the carrier so no host conversion is needed: |got − want| must not
        // exceed tol · max(1, |want|), expressed with `le` and checked as a mask.
        let scale = want.abs().max(T::one());
        let bound = scale * T::scalar(tolerance);
        let err = (got.clone() - want.clone()).abs();
        assert_eq!(
            err.le(&bound),
            <T::Mask as Element>::one(),
            "{what}: {law} at {x:?} — got {got:?}, want {want:?}"
        );
    };

    for x in xs {
        // --- exact laws ---
        assert_eq!(x.abs().abs(), x.abs(), "{what}: |·| is idempotent ({x:?})");
        assert_eq!(
            x.clone().min(x.clone()),
            *x,
            "{what}: min(x, x) = x ({x:?})"
        );
        assert_eq!(
            x.clone().max(x.clone()),
            *x,
            "{what}: max(x, x) = x ({x:?})"
        );
        for (law, r) in [
            ("floor", x.clone().floor()),
            ("ceil", x.clone().ceil()),
            ("trunc", x.clone().trunc()),
        ] {
            let twice = match law {
                "floor" => r.clone().floor(),
                "ceil" => r.clone().ceil(),
                _ => r.clone().trunc(),
            };
            assert_eq!(twice, r, "{what}: {law} is idempotent ({x:?})");
        }
        // lifting a literal and lifting it "like" an existing value must agree
        assert_eq!(
            T::scalar(2.0),
            x.scalar_like(2.0),
            "{what}: scalar and scalar_like must agree ({x:?})"
        );

        // --- inverse pairs, to tolerance ---
        close(&x.exp().ln(), x, "ln(exp(x)) = x", x);
        close(&x.ln().exp(), x, "exp(ln(x)) = x", x);
        close(&x.sqrt().square(), x, "sqrt(x)² = x", x);
        close(&x.square().sqrt(), &x.abs(), "sqrt(x²) = |x|", x);

        // --- identities ---
        let (s, c) = x.clone().sin_cos();
        close(
            &(s.square() + c.square()),
            &T::one(),
            "sin²(x) + cos²(x) = 1",
            x,
        );
        close(&x.clone().tan().atan(), x, "atan(tan(x)) = x", x);

        // --- the compensated forms must agree with the naive ones away from zero ---
        close(
            &x.ln_1p(),
            &(T::one() + x.clone()).ln(),
            "ln_1p = ln(1+x)",
            x,
        );
        close(&x.exp_m1(), &(x.exp() - T::one()), "exp_m1 = exp(x) − 1", x);

        for y in xs {
            // min and max are commutative and return one of their arguments
            assert_eq!(
                x.clone().min(y.clone()),
                y.clone().min(x.clone()),
                "{what}: min commutes ({x:?}, {y:?})"
            );
            let m = x.clone().min(y.clone());
            assert!(
                m == *x || m == *y,
                "{what}: min must return an argument, got {m:?} from ({x:?}, {y:?})"
            );
            // hypot against its definition
            close(
                &x.clone().hypot(y.clone()),
                &(x.square() + y.square()).sqrt(),
                "hypot(x, y) = sqrt(x² + y²)",
                x,
            );
        }
    }
}

/// [`real_laws`] over a carrier's own [`ConformanceSample`] and [`tolerance`].
///
/// [`tolerance`]: ConformanceSample::tolerance
pub fn real_laws_for<T>(what: &str)
where
    T: Real + ConformanceSample + PartialEq + core::fmt::Debug,
    T::Mask: PartialEq + core::fmt::Debug,
{
    real_laws(&T::sample(), T::tolerance(), what);
}
