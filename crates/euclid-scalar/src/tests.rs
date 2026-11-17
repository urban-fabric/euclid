//! The crate's test suites, organized by the **question each one answers** rather than by the
//! function under test. A single function is therefore exercised from several directions: `erf`
//! is checked for accuracy here, for tail behavior in [`stability`], against `erfc` in
//! [`identities`], and against the incomplete gamma in [`relationships`].
//!
//! # Where a test belongs
//!
//! Tests live in one of two places, and the dividing line is **which surface they speak to**:
//!
//! - **Here**, when the test uses the public trait surface ([`Real`], [`Natural`], …). Anything
//!   comparing two functions to each other belongs here, because the trait is what both reach
//!   through.
//! - **Beside the implementation**, in a `#[cfg(test)]` child of the module it tests, when it
//!   needs the private series, quadrature or crossover constants — a `pub(super)` threshold is
//!   not visible from this tree. [`crate::kernels::bessel::tests`] is the fullest example.
//!
//! Prefer this tree when either would work: a test written against the trait keeps passing when
//! the algorithm behind it is replaced, which is the point of having a contract.
//!
//! # The suites
//!
//! | suite | question it answers |
//! |-------|--------------------|
//! | [`accuracy`] | how close is this to the true value, across the domain? |
//! | [`stability`] | does it survive the inputs where the naive formula fails? |
//! | [`identities`] | does it satisfy the mathematical relations it must? |
//! | [`relationships`] | do the functions agree *with each other*? |
//! | [`conventions`] | which convention does an ambiguous std-mirroring method follow? |
//! | [`edge_cases`] | what happens at `0`, `±∞`, `NaN`, and the representable limits? |
//! | [`poles`] | how does it behave where it is singular, zero, or out of range? |
//!
//! Four suites are scoped to a family or a representation rather than to a property, because
//! the thing under test is a cluster of functions that only make sense together:
//!
//! | suite | scope |
//! |-------|-------|
//! | [`normal_family`] | `ndtr`, `log_ndtr`, `ndtri`, `erfc_inv` and the relations binding them |
//! | [`quantiles`] | the gamma and beta inverses, including the extreme-shape regime |
//! | [`compensated`] | the error-free transforms, and the elementary functions built on them |
//! | [`representation`] | `frexp`, `ldexp`, `next_after`, `ulp`, `signbit` — the format itself |
//! | [`constants`] | are the lifted constants right, on both carriers? |
//!
//! One is scoped to the **contract** rather than to any function:
//!
//! | suite | question it answers |
//! |-------|--------------------|
//! | [`conformance_suites`] | do the reference carriers satisfy the public [`conformance`](crate::conformance) laws? |
//!
//! That one runs the suites this crate *ships* — the same functions a carrier author in another
//! repository calls — rather than testing anything only we can see. It is where a tolerance is
//! declared per carrier, since `f64` and `f32` cannot share one.
//!
//! And two are scoped to a *carrier* rather than to a function:
//!
//! | suite | scope |
//! |-------|-------|
//! | [`f32`] | is the separate single-precision implementation as good? |
//! | [`real_surface`], [`masks`], [`discrete`], [`control_flow`] | does a formula written *once* against a trait tier run on every carrier at that tier? |
//! | [`discrete_carriers`] | does *every* carrier of a discrete tier honour the contract, or only the one the suites instantiate? |
//!
//! The second row is the one that justifies the crate's existence: [`real_surface`] covers the
//! analysis tier, [`masks`] the comparison/selection tier, [`discrete`] the semiring and ring
//! tiers, and [`control_flow`] the conditional/loop contract. Each writes a formula with no
//! mention of a concrete number type and then runs it on several, which is the property a
//! consumer-defined carrier depends on.
//!
//! [`discrete_carriers`] asks the narrower question those miss. Running a formula on three
//! widths proves it *compiles* and agrees at the values sampled; it does not ask each width the
//! whole of its contract. Mutation testing found 103 survivors across `impls/natural_impl.rs`
//! and `impls/integer_impl.rs` for exactly that reason — the suites had settled on `u64`/`i64`
//! and the other four widths rode along — and one of them was a real defect.
//!
//! # House rules
//!
//! - **No file over 400 lines.** Split by sub-topic into a directory with a header explaining
//!   the split, as [`accuracy`] and [`poles`] do. The budget applies to files carrying tests;
//!   pure trait or impl surface ([`crate::traits::real`]) is exempt, having nothing to split along.
//! - **Tolerances are measured, then annotated.** Every bound says what the measured error was
//!   and why it is what it is. Never loosen one without understanding why the error grew — an
//!   unexplained loosening hides exactly the regression the bound existed to catch.
//! - **Prefer an identity to a transcribed digit.** A recurrence, complement or round-trip
//!   validates itself; a reference value can be mistyped. Where a value is unavoidable, derive
//!   it from a closed form or an independent implementation and say so in a comment.

use super::*;

/// A relative-tolerance comparison in a carrier's *own* real surface.
///
/// Shared by the carrier-conformance suites ([`discrete_carriers`], [`real_carriers`]), which
/// all have the same problem: the expectation is known as an `f64`, and it has to be checked
/// against whichever precision the carrier under test actually carries.
///
/// `|got − want| ≤ tol · max(1, |want|)`, the mixed criterion used throughout this crate —
/// purely relative is wrong where a quantity passes through zero, purely absolute is wrong where
/// it is large. `Mask = bool` is what pins this to the host carriers; an array carrier would
/// need the mask reduced first.
fn agrees<T>(got: &T, want: f64, tol: f64, what: &str, law: &str)
where
    T: Real<Mask = bool> + core::fmt::Debug,
{
    let expected = T::scalar(want);
    let scale = expected.abs();
    let unit = T::one();
    let denominator = if scale.gt(&unit) { scale } else { unit };
    let difference = (got.clone() - expected.clone()).abs();
    assert!(
        difference.le(&(T::scalar(tol) * denominator)),
        "{what}: {law} — got {got:?}, want {want}",
    );
}

mod accuracy;
mod compensated;
mod conformance_suites;
mod constants;
mod control_flow;
mod conventions;
mod discrete;
mod discrete_carriers;
mod edge_cases;
mod f32;
mod identities;
mod masks;
mod normal_family;
mod poles;
mod quantiles;
mod real_carriers;
mod real_surface;
mod relationships;
mod representation;
mod stability;
mod vacuity;
