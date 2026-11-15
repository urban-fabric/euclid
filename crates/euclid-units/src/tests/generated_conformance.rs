//! The quantity contracts over **generated** carriers.
//!
//! The companion to [`defaults`](super::defaults), which runs the same
//! `crate::conformance` laws over a fixed five-value sample. Nothing is
//! restated: the laws are the crate's own public functions, and this file
//! changes only where their slice comes from.
//!
//! ## What the generated sample reaches that five values do not
//!
//! The tier laws themselves are the kind euclid's docs argue are well served by
//! a fixed sample — they "either hold structurally or fail on the first awkward
//! triple". The reason to generate anyway is the **default method bodies**.
//!
//! `ScalarQuantity` and `LinearQuantity` supply `min`, `max`, `clamp`, `floor`,
//! `ceil`, `round`, `trunc`, `approx_eq`, `add`, `sub`, `neg` and `scale` as
//! defaults, so one body serves every carrier in every consuming crate —
//! `defaults`' own header calls each one "a single point of failure". Those
//! bodies branch on *orderings* and *signs*, and that is a combinatorial space
//! rather than a structural one:
//!
//! - `clamp(x, lo, hi)` has six orderings of its three arguments, and the fixed
//!   sample's five values in one fixed order reach a handful of them;
//! - `floor`, `ceil` and `trunc` agree on positives and differ on negatives, so
//!   a mostly-positive sample would pass a `trunc` written as `floor`;
//! - `round` has to choose a direction at exactly `±0.5`.
//!
//! The generator draws signs and half-integers deliberately for those three
//! reasons, and the coverage floors require that mixed-sign samples actually
//! appeared rather than hoping they did.
//!
//! `Len` and `Tmp` are reused rather than a fresh carrier being written: `Tmp`
//! withholds `LinearQuantity`, which is the asymmetry the tier split exists to
//! express, and a new carrier would only re-test what those two already pin.

use euclid_stability::{Config, Gen, classify, generate, property};

use crate::Measure;

use super::toy_units::{Len, Tmp};

fn settings(suite: &str) -> Config {
    Config::default()
        .cases(400)
        .regressions(format!("src/tests/regressions/{suite}.txt"))
}

/// A finite carrier value, weighted toward what the default bodies branch on.
fn raw() -> Gen<f64> {
    generate::weighted(vec![
        (2, generate::just(0.0)),
        (2, generate::just(0.5)),
        (2, generate::just(-0.5)),
        (2, generate::just(-1.5)),
        (2, generate::just(1.0)),
        (12, generate::float::magnitude(-10, 20)),
    ])
}

/// Three is the minimum that exercises a three-operand law like `clamp`; six
/// matches what the fixed suite uses.
fn lengths() -> Gen<Vec<Len>> {
    generate::vec(raw().map(Len::from_raw), 3, 6)
}

fn temperatures() -> Gen<Vec<Tmp>> {
    generate::vec(raw().map(Tmp::from_raw), 3, 6)
}

/// The tolerance the approximate comparisons in the scalar laws use.
fn tol_len() -> Len {
    Len::from_raw(1e-9)
}

fn signs(xs: &[f64]) -> &'static str {
    let any_neg = xs.iter().any(|v| *v < 0.0);
    let any_pos = xs.iter().any(|v| *v > 0.0);
    match (any_neg, any_pos) {
        (true, true) => "mixed signs",
        (true, false) => "all non-positive",
        (false, true) => "all non-negative",
        (false, false) => "all zero",
    }
}

property! {
    #![config(
        settings("units-linear")
            .require("mixed signs", 1.0)
            .require("all non-negative", 1.0)
    )]

    /// A linear carrier satisfies every law the crate states for one.
    ///
    /// `quantity_laws` is the aggregate a carrier author calls, so this is the
    /// same entry point a consumer would use — over generated samples rather
    /// than five fixed values.
    fn a_linear_carrier_satisfies_the_quantity_laws(xs in lengths()) {
        classify!(signs(&xs.iter().map(Measure::raw).collect::<Vec<_>>()));
        crate::conformance::quantity_laws(&xs, &tol_len(), "Len");
    }

    /// The measure round trip and the scalar helpers, separately.
    ///
    /// Run alongside the aggregate rather than instead of it: if
    /// `quantity_laws` ever stopped calling one of its parts, the aggregate
    /// would still pass while that part went unchecked, and these are what
    /// notice.
    fn a_linear_carrier_satisfies_the_measure_and_scalar_laws(xs in lengths()) {
        classify!(signs(&xs.iter().map(Measure::raw).collect::<Vec<_>>()));
        crate::conformance::measure_laws(&xs, "Len");
        crate::conformance::scalar_laws(&xs, &tol_len(), "Len");
    }
}

property! {
    #![config(
        settings("units-absolute")
            .require("mixed signs", 1.0)
            .require("all non-negative", 1.0)
    )]

    /// An absolute scalar satisfies everything but the linear laws.
    ///
    /// `Tmp` withholds `LinearQuantity` deliberately — adding two temperatures
    /// is meaningless — so it gets the measure and scalar laws only. That
    /// asymmetry is what the tier split exists to express, and running the
    /// scalar defaults through a carrier that stops short of linearity is how
    /// the split is shown to actually hold.
    fn an_absolute_carrier_satisfies_the_measure_and_scalar_laws(xs in temperatures()) {
        classify!(signs(&xs.iter().map(Measure::raw).collect::<Vec<_>>()));
        crate::conformance::measure_laws(&xs, "Tmp");
        crate::conformance::scalar_laws(&xs, &Tmp::from_raw(1e-9), "Tmp");
    }
}
