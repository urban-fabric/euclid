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

// ===== The exported generated laws, run against the toy carriers =================
//
// These call `conformance::generated` rather than re-deriving the sign/half-integer strategy.
// That strategy is public API now, so euclid has to be its first caller — and `Len`/`Tmp` reach
// it through `Measure::from_raw`, which is the same route a carrier author takes.
//
// `Len` and `Tmp` are reused rather than a third carrier being written: `Tmp` withholds
// `LinearQuantity`, which is the asymmetry the tier split exists to express, and running the
// scalar defaults through a carrier that stops short of linearity is how that split is shown to
// hold.

use crate::Measure;
use crate::conformance::generated;

use super::toy_units::{Len, Tmp};

/// A linear carrier satisfies the aggregate every carrier author calls.
#[test]
fn a_linear_carrier_satisfies_the_generated_quantity_laws() {
    generated::quantity_laws_generated::<Len>(&Len::from_raw(1e-9), "Len");
}

/// And the measure/scalar halves separately, so one dropping out of the aggregate is noticed.
#[test]
fn a_linear_carrier_satisfies_the_generated_measure_and_scalar_laws() {
    generated::measure_and_scalar_laws_generated::<Len>(&Len::from_raw(1e-9), "Len");
}

/// An absolute scalar satisfies everything but the linear laws.
///
/// `Tmp` withholds `LinearQuantity` deliberately — adding two temperatures is meaningless — so it
/// gets the measure and scalar laws only.
#[test]
fn an_absolute_carrier_satisfies_the_generated_measure_and_scalar_laws() {
    generated::measure_and_scalar_laws_generated::<Tmp>(&Tmp::from_raw(1e-9), "Tmp");
}
