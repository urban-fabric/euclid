//! The sampler bounds: that every draw lands in the support it claims.
//!
//! Range rather than distribution, deliberately. A bound is a hard contract a single draw can
//! violate, where a distributional claim needs a statistic and a tolerance; the shape of the
//! distribution is the property engine's business, not the contract's.

use super::*;

/// The parameter laws: a draw must respect the distribution parameters it was given.
///
/// These are the cases where the answer is forced regardless of the algorithm, so they hold for
/// any correct backend:
///
/// - `uniform(low, high)` lies within `[low, high]`
/// - `uniform(x, x)` is exactly `x` — a degenerate range has one answer
/// - `normal(mean, 0)` is exactly `mean` — zero variance has one answer
/// - `bernoulli(0)` is always false and `bernoulli(1)` always true
/// - `randint(low, high)` lies within `[low, high)`, and `randint(x, x+1)` is exactly `x`
///
/// Requires `PartialOrd` on the draw type, so this suite is for carriers with a host order. A
/// deferred carrier whose comparisons yield masks should check the same properties through its
/// own mask surface.
pub fn sampler_bound_laws<R>(seeds: &[u64], what: &str)
where
    R: RandomSampler,
    R::Real: PartialEq + PartialOrd + core::fmt::Debug,
    R::Bool: PartialEq + core::fmt::Debug,
    R::Int: PartialEq + PartialOrd + core::fmt::Debug,
{
    let zero = R::Real::zero();
    let one = R::Real::one();
    let two = R::Real::scalar(2.0);

    for &s in seeds {
        let k = R::key(s);

        // uniform stays inside its range, and collapses on a degenerate one
        let u = R::uniform(&k, SCALAR, zero.clone(), two.clone());
        assert!(
            u >= zero && u <= two,
            "{what}: uniform(0, 2) drew {u:?}, outside [0, 2]"
        );
        assert!(
            R::uniform(&k, SCALAR, one.clone(), one.clone()) == one,
            "{what}: uniform(x, x) must be exactly x"
        );

        // zero variance has exactly one answer
        assert!(
            R::normal(&k, SCALAR, one.clone(), zero.clone()) == one,
            "{what}: normal(mean, 0) must be exactly mean"
        );

        // the Bernoulli boundaries are forced
        assert!(
            R::bernoulli(&k, SCALAR, zero.clone()) == R::Bool::zero(),
            "{what}: bernoulli(0) must be false"
        );
        assert!(
            R::bernoulli(&k, SCALAR, one.clone()) == R::Bool::one(),
            "{what}: bernoulli(1) must be true"
        );

        // randint respects its half-open interval
        let lo = R::Int::zero();
        let hi = R::Int::from_i64(3);
        let n = R::randint(&k, SCALAR, lo.clone(), hi.clone());
        assert!(
            n >= lo && n < hi,
            "{what}: randint(0, 3) drew {n:?}, outside [0, 3)"
        );
        let single = R::Int::zero();
        let next = R::Int::one();
        assert!(
            R::randint(&k, SCALAR, single.clone(), next) == single,
            "{what}: randint(x, x+1) must be exactly x"
        );
    }
}
