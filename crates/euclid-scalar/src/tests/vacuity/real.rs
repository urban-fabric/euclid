//! Does the `Real` conformance suite notice a broken method?
//!
//! Part of the [`vacuity`](super) suite. Driven by [`Faulty<N>`](super::carrier::Faulty), which
//! perturbs method `N` and nothing else, so the sweep asks the suite about one method at a time.
//! The tolerance case is the sharpest: an error that only just exceeds the declared tolerance
//! has to be caught, or the tolerance is decoration.

use super::*;

/// The samples the laws are run on: finite, strictly positive, straddling one.
///
/// `real_laws`' own docstring states the requirement — "values must be **finite** and **strictly
/// positive** — the logarithm laws need `x > 0`, and the trig laws stay on the principal branch,
/// so keep values in `(0, π/2)`".
fn real_samples<const WHICH: u32>() -> Vec<Faulty<WHICH>> {
    [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
        .into_iter()
        .map(Faulty)
        .collect()
}

/// The tolerance the `Real` laws run at here: tight enough that a coarse perturbation cannot
/// hide inside it, loose enough for a faithful `f64` delegation to pass every inverse pair.
const REAL_TOLERANCE: f64 = 1e-12;

impl<const WHICH: u32> ConformanceSample for Faulty<WHICH> {
    fn sample() -> Vec<Self> {
        real_samples::<WHICH>()
    }
    fn tolerance() -> f64 {
        REAL_TOLERANCE
    }
}

impl<const WHICH: u32> GeneratedSample for Faulty<WHICH> {
    fn generator() -> euclid_stability::Gen<Self> {
        euclid_stability::generate::float::in_range(0.25, 1.5).map(Self)
    }
}

/// A faithful delegation satisfies every law, which is the premise the rejections rest on.
///
/// A suite that rejects *everything* is as useless as one that accepts everything, and it is the
/// easier mistake to make — so this is checked first and separately for all three entry points.
#[test]
fn the_real_laws_accept_a_faithful_carrier() {
    conformance::real_laws(&real_samples::<0>(), REAL_TOLERANCE, "Faulty<0>");
    conformance::real_laws_for::<Faulty<0>>("Faulty<0> via real_laws_for");
    gen_laws::real_laws_generated::<Faulty<0>>(REAL_TOLERANCE, "Faulty<0> via real_laws_generated");
}

/// Every law must reject a carrier whose one broken method that law depends on.
///
/// This is what the generated carrier exists for. Each entry names a method `real_laws`
/// exercises and the law that should notice it; if any were accepted, that law is absent or
/// vacuous.
///
/// # The perturbation has to break a fixed point, not just move a value
///
/// Worth recording, because the first attempt failed here and the failure was informative. With
/// an affine perturbation — `v ↦ 1.5v + 0.25` — the broken `floor` is **still idempotent**: its
/// image is `{0.25, 1.75}` and it maps that set to itself. Idempotence is the *only* law
/// `real_laws` states about `floor`, so the carrier passed, and the honest conclusion was not
/// that the test needed loosening but that the perturbation sat in the law's kernel. Negating as
/// well breaks the fixed point.
///
/// That is also a real limitation of the suite, now documented rather than discovered later: a
/// `floor` that is wrong but idempotent satisfies every law `real_laws` states.
#[test]
fn the_real_laws_reject_each_broken_method_they_depend_on() {
    macro_rules! reject {
        ($id:literal, $method:literal, $law:literal) => {{
            let outcome = rejects(|| {
                conformance::real_laws(
                    &real_samples::<$id>(),
                    REAL_TOLERANCE,
                    concat!("Faulty<", stringify!($id), ">"),
                )
            });
            assert!(
                outcome.is_some(),
                "real_laws accepted a carrier whose `{}` is wrong — the law `{}` is missing \
                 or vacuous",
                $method,
                $law
            );
        }};
    }

    reject!(
        14,
        "one",
        "the identity the tolerance scale and sin²+cos² both rest on"
    );
    reject!(16, "scalar_like", "scalar and scalar_like must agree");
    reject!(48, "exp", "ln(exp(x)) = x");
    reject!(49, "ln", "exp(ln(x)) = x");
    reject!(52, "sqrt", "sqrt(x)² = x");
    reject!(53, "abs", "|·| is idempotent");
    reject!(54, "square", "sqrt(x²) = |x|");
    reject!(70, "sin_cos", "sin²(x) + cos²(x) = 1");
    reject!(78, "floor", "floor is idempotent");
    reject!(79, "ceil", "ceil is idempotent");
    reject!(81, "trunc", "trunc is idempotent");
    reject!(88, "min", "min(x, x) = x");
    reject!(89, "max", "max(x, x) = x");
}

/// The `_for` and generated wrappers must reject too, since each is its own function with its
/// own empty-body mutant.
#[test]
fn the_real_wrappers_reject_a_broken_carrier() {
    rejects(|| conformance::real_laws_for::<Faulty<48>>("Faulty<48>"))
        .expect("real_laws_for must reject a carrier whose exp is wrong");
    rejects_engine(|| gen_laws::real_laws_generated::<Faulty<48>>(REAL_TOLERANCE, "Faulty<48>"))
        .expect("real_laws_generated must reject a carrier whose exp is wrong");
}

/// The *tolerance* mutants, which a coarse fault cannot reach.
///
/// `real_laws` bounds each inexact law by `scale · tolerance`, and mutating that `*` to `+` makes
/// the bound about `1` instead of about `1e-12` — vacuous, while every test above still passes.
/// The reason they pass is worth stating: a coarsely broken carrier mostly produces `NaN`, and
/// `NaN ≤ anything` is false, so it is rejected under either bound and says nothing about which
/// bound is in force.
///
/// `Faulty<1000 + k>` is wrong by `1e-9` relative in method `k` — three orders above the
/// tolerance and nine below one. So it must be rejected, and is accepted exactly when the bound
/// has been loosened.
#[test]
fn the_real_laws_reject_an_error_that_only_just_exceeds_the_tolerance() {
    for (identifier, method) in [(1048u32, "exp"), (1049, "ln"), (1052, "sqrt")] {
        let outcome = match identifier {
            1048 => rejects(|| {
                conformance::real_laws(&real_samples::<1048>(), REAL_TOLERANCE, "Faulty<1048>")
            }),
            1049 => rejects(|| {
                conformance::real_laws(&real_samples::<1049>(), REAL_TOLERANCE, "Faulty<1049>")
            }),
            _ => rejects(|| {
                conformance::real_laws(&real_samples::<1052>(), REAL_TOLERANCE, "Faulty<1052>")
            }),
        };
        assert!(
            outcome.is_some(),
            "real_laws accepted a `{method}` that is wrong by 1e-9 relative at a tolerance of \
             {REAL_TOLERANCE:e} — the bound is not `scale * tolerance`"
        );
    }
}

/// A carrier that is **lawful but unusual**, which is the only way to see the two
/// `delete match arm` mutants.
///
/// `real_laws` checks that each of `floor`, `ceil` and `trunc` is idempotent, dispatching on the
/// law's name. Delete the `"floor"` arm and the check falls through to `_ => trunc`, so it
/// becomes `trunc(floor(x)) == floor(x)` — which holds for every ordinary carrier, because
/// `floor(x)` is an integer and `trunc` is the identity on integers. The mutant is therefore
/// invisible to any carrier that rounds normally.
///
/// `Faulty<2000>` rounds `trunc` **down to an even integer**. That is idempotent, so every law
/// the suite states still holds and the real suite accepts it — while `trunc(floor(1.5))` is
/// `trunc(1)` is `0`, not `1`, so the altered suite rejects it. Assert acceptance of the
/// lawful carrier to catch a check that imposes a stronger rule than the contract.
#[test]
fn the_real_laws_accept_a_lawful_carrier_whose_trunc_is_unusual() {
    // The premise: this carrier's `trunc` really is both idempotent and not the identity on
    // `floor`'s output, or the test below would prove nothing.
    let unusual = Faulty::<2000>(1.5);
    let floored = crate::Real::floor(unusual);
    let truncated_floor = crate::Real::trunc(floored);
    assert_eq!(floored.0, 1.0, "floor is still the real floor");
    assert_eq!(
        truncated_floor.0, 0.0,
        "while trunc rounds down to even, so trunc(floor(1.5)) != floor(1.5)"
    );
    let twice = crate::Real::trunc(crate::Real::trunc(unusual));
    assert_eq!(
        twice.0,
        crate::Real::trunc(unusual).0,
        "and it is idempotent, which is the only law the suite states about it"
    );

    // The point: every law holds, so the suite must accept it. With either match arm deleted it
    // does not.
    conformance::real_laws(&real_samples::<2000>(), REAL_TOLERANCE, "Faulty<2000>");
}
