//! The fusion itself, over generated values: the three tiers must describe **the same number**.
//!
//! `HostReal` adds `euclid_algebra::OrderedField`, `Copy`, associated constants and host
//! predicates to the backend-neutral `Real`. This suite checks agreement between those tiers;
//! numerical kernel properties live separately under `tests/properties/`.
//!
//! `Real::is_finite` returns a carrier mask, while `HostReal::is_finite` returns a host `bool`.
//! They are separate methods with separate implementations. A carrier implementing both can
//! disagree, so their values must be checked independently of how the methods are named.
//!
//! So the properties below are **agreement** properties — the "two implementations agree"
//! relation, applied where two tiers overlap:
//!
//! - lifting: `HostReal::from_f64` against `euclid_scalar::Real::scalar`, which its default body
//!   claims to be a spelling of
//! - finiteness: the analysis tier's mask predicates against the concrete tier's `bool` ones,
//!   over input that is deliberately *not* finite
//! - selection: the comparison-free `min`/`max`/`clamp` against the host operations
//! - sign: the concrete `is_sign_negative` against the algebraic order, which is where the two
//!   genuinely diverge — see below
//!
//! ## `-0.0` is the one place the tiers do not agree, and that is correct
//!
//! `is_sign_negative` is a question about the *representation* and the order is a question
//! about the *value*, so they part company at negative zero: `(-0.0).is_sign_negative()` is
//! `true` while `-0.0 < 0.0` is `false`, and both answers are right. Asserting the tempting
//! form — that the predicate and the comparison agree — would be asserting something false
//! about floating point, so the property below excludes zero and states the zero case
//! separately. A generated sample finds this immediately; a hand-written one reaches for `-1.0`
//! and `2.5` and never does.

use euclid_algebra::HasZero;
use euclid_scalar::HostReal;
use euclid_stability::relation::Exact;
use euclid_stability::{Config, Gen, check, check_eq, classify, generate, property, relation};

fn settings(suite: &str) -> Config {
    Config::default()
        .cases(400)
        .regressions(format!("tests/regressions/{suite}.txt"))
}

/// Finite and non-finite alike. The finiteness predicates are the point of this suite, so a
/// generator that only produced finite values would make them pass vacuously.
fn any_value() -> Gen<f64> {
    generate::weighted(vec![
        (1, generate::just(0.0)),
        (1, generate::just(-0.0)),
        (3, generate::float::non_finite()),
        (3, generate::float::notable()),
        (12, generate::float::finite()),
    ])
}

/// Finite only, for the properties where a `NaN` would fail an equality by never equalling
/// itself — which says nothing about either tier.
fn finite_value() -> Gen<f64> {
    generate::float::finite()
}

fn label(x: f64) -> &'static str {
    if x.is_nan() {
        "nan"
    } else if x.is_infinite() {
        "infinite"
    } else if x == 0.0 {
        "zero"
    } else {
        "ordinary"
    }
}

property! {
    #![config(settings("scalar-lifting"))]

    /// `HostReal::from_f64` is a spelling of `euclid_scalar::Real::scalar`, at both widths.
    ///
    /// The default body says so, and a carrier is free to override it — so this checks the
    /// claim rather than the body. At `f32` it is also the statement that the lift narrows the
    /// same way through both names, which is where an override would most plausibly differ.
    fn lifting_agrees_across_the_two_tiers(x in finite_value()) {
        classify!(label(x));

        relation::agree(
            "from_f64 / scalar at f64",
            x,
            <f64 as HostReal>::from_f64,
            <f64 as euclid_scalar::Real>::scalar,
            &Exact,
        )?;
        relation::agree(
            "from_f64 / scalar at f32",
            x,
            <f32 as HostReal>::from_f64,
            <f32 as euclid_scalar::Real>::scalar,
            &Exact,
        )?;
    }
}

property! {
    #![config(
        settings("scalar-finiteness")
            .require("nan", 2.0)
            .require("infinite", 2.0)
            .require("zero", 1.0)
    )]

    /// The analysis tier's mask predicates and the concrete tier's `bool` predicates agree.
    ///
    /// The two answer the same question and return different types, and the crate's whole
    /// premise is that a reader can tell them apart by path. If they ever disagreed, every
    /// formula that moved between the tiers would change meaning silently — which is the
    /// failure euclid exists to prevent, occurring inside euclid.
    fn the_finiteness_predicates_agree_across_the_two_tiers(x in any_value()) {
        classify!(label(x));

        // `Mask` is `bool` for a host float, so the two are directly comparable here. A
        // deferred carrier would collapse its mask first; the claim is the same.
        check_eq!(
            <f64 as HostReal>::is_finite(x),
            euclid_scalar::Real::is_finite(&x),
            "the two is_finite disagree on {x:?}"
        );
        check_eq!(
            <f64 as HostReal>::is_nan(x),
            euclid_scalar::Real::is_nan(&x),
            "the two is_nan disagree on {x:?}"
        );

        // And the pair is exhaustive and exclusive: every value is finite, infinite or NaN.
        let finite = <f64 as HostReal>::is_finite(x);
        let nan = <f64 as HostReal>::is_nan(x);
        check!(
            !(finite && nan),
            "{x:?} is reported both finite and NaN"
        );
    }
}

property! {
    #![config(settings("scalar-selection"))]

    /// The comparison-free selectors agree with the host operations.
    ///
    /// `min`/`max`/`clamp` come from the analysis tier — they exist there precisely so a
    /// deferred carrier can supply them without a host `bool` — and a concrete carrier has a
    /// second, independent implementation available in `core`. Two implementations, one answer.
    fn the_selectors_agree_with_the_host_operations(a in finite_value(), b in finite_value()) {
        classify!("equal", a == b);
        classify!("a smaller", a < b);

        check_eq!(
            euclid_scalar::Real::min(a, b),
            f64::min(a, b),
            "min disagrees on ({a:?}, {b:?})"
        );
        check_eq!(
            euclid_scalar::Real::max(a, b),
            f64::max(a, b),
            "max disagrees on ({a:?}, {b:?})"
        );

        // `clamp` needs an ordered interval; build one from the pair rather than filtering.
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        let mid = (lo + hi) * <f64 as HostReal>::HALF;
        for x in [lo, hi, mid, lo - 1.0, hi + 1.0] {
            check_eq!(
                euclid_scalar::Real::clamp(x, lo, hi),
                f64::clamp(x, lo, hi),
                "clamp({x:?}, {lo:?}, {hi:?}) disagrees"
            );
        }
    }

    /// `is_sign_negative` agrees with the algebraic order **away from zero**.
    ///
    /// The exclusion is the content. At `±0.0` the sign bit and the order disagree by design —
    /// a signed zero is negative in its representation and equal to zero in its value — so the
    /// law holds on nonzero input and the zero case is asserted for what it actually is.
    fn the_sign_predicate_agrees_with_the_order_away_from_zero(x in finite_value()) {
        classify!("zero", x == 0.0);
        classify!("negative", x < 0.0);

        if x != 0.0 {
            check_eq!(
                <f64 as HostReal>::is_sign_negative(x),
                x < <f64 as HasZero>::ZERO,
                "the sign predicate and the order disagree on the nonzero {x:?}"
            );
        }

        // The documented divergence, stated rather than skipped: both zeros compare equal to
        // the additive identity, and only one of them has its sign bit set.
        check!(
            -0.0_f64 == <f64 as HasZero>::ZERO && 0.0_f64 == <f64 as HasZero>::ZERO,
            "both signed zeros must equal the additive identity"
        );
        check!(
            <f64 as HostReal>::is_sign_negative(-0.0) && !<f64 as HostReal>::is_sign_negative(0.0),
            "the sign bit must distinguish the two zeros the order cannot"
        );
    }
}
