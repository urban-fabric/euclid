//! The concrete scalar algorithms behind the trait surface — crate-internal throughout.
//!
//! Everything a [`crate::Real`] or [`crate::Natural`] method needs to *compute* an answer lives
//! here. Nothing in this module is public: the contract is the trait layer, and [`crate::impls`]
//! is the only caller, forwarding each method to the routine below that implements it. A consumer
//! never names a path into this tree.
//!
//! These docs are the module's own (`//!`), with no outer `///` on the `mod` declaration in
//! `lib.rs` — the same rule [`crate::conformance`] documents: the two forms are concatenated, and
//! an outer comment resolves intra-doc links in the *parent's* scope, so `[bessel]` below would
//! be looked up at the crate root and silently fail.
//!
//! This was called `special` until the contents outgrew the name. It holds three kinds of
//! thing, and only the first are special functions:
//!
//! | | modules | what they are |
//! |---|---|---|
//! | **Special functions** | [`airy`], [`bessel`], [`beta`], [`erf`], [`gamma`], [`integrals`], [`normal`], [`polygamma`], [`quantile`], [`transcendental`] | the named transcendentals of classical analysis, plus the inverse CDFs built on them |
//! | **Format primitives** | [`repr`] | `frexp`, `ldexp`, `next_after`, `ulp` — bit-level, exact, and not approximations of anything |
//! | **Host combinatorics** | [`host`] | integer `factorial`/`binomial` returning a float, for the `Natural` impls rather than the `Real` surface |
//!
//! # Precision
//!
//! Each family is written for `f64`. The `f32` peers live in [`f32`](mod@f32), which mostly
//! mirrors this layout — see its header for which functions are evaluated natively in single
//! precision and which deliberately narrow the `f64` result, and why that distinction is not a
//! shortcut.
//!
//! **The mirror has no exceptions.** `erf` was briefly folded the other way — both precisions in
//! one directory, with [`f32`](mod@f32) re-exporting the child — and it has been
//! folded back. One tree per precision is the layout: `kernels/<family>.rs` is the `f64`
//! implementation and `kernels/f32/<family>.rs` is its peer, for every family without exception,
//! so a reader never has to learn which shape a given family happens to use.
//!
//! What that layout deliberately does *not* duplicate is anything precision-generic. A step whose
//! arithmetic carries no format-dependent constant belongs in [`crate::math`], where it is written
//! once over `T: HostReal` and tested once for both formats; a *threshold* belongs beside the code
//! it governs, because it is a separate measurement in each format rather than one value shared.
//! That line — generic arithmetic up, measured constants down — is what keeps the two trees from
//! drifting without pretending they are the same code.
//!
//! # Where the tests are
//!
//! Split by what they can reach. A test that needs a private series, quadrature or crossover
//! constant is a `#[cfg(test)]` child of the module it tests — `bessel::tests` and
//! `airy::tests` are the fullest examples. Everything reachable through the trait lives in
//! the `crate::tests` tree instead, where it keeps passing when the algorithm here is replaced.

pub(crate) mod airy;
pub(crate) mod bessel;
pub(crate) mod beta;
pub(crate) mod erf;
pub(crate) mod gamma;
pub(crate) mod host;
pub(crate) mod integrals;
pub(crate) mod normal;
pub(crate) mod polygamma;
pub(crate) mod quantile;
pub(crate) mod repr;
pub(crate) mod transcendental;

pub(crate) mod f32;
