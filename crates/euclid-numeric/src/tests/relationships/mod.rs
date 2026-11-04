//! **Do the functions agree with each other?** A consistency web that ties the special
//! functions together through the formulas that relate them.
//!
//! The other suites check each function against something external: a reference table, a
//! stability expectation, its own domain edges. This one checks functions against *each other*.
//! That buys something the others cannot: a relation like `ln_gamma' = digamma` or
//! `P(½, x) = erf(√x)` is satisfied by the true functions and essentially never by a wrong
//! one, so each formula constrains every function it mentions. With enough overlapping
//! relations, an error anywhere breaks at least one of them — and because the implementations
//! involved are *independent* (a continued fraction on one side, a polynomial on the other),
//! agreement is strong evidence rather than a tautology.
//!
//! The web has four kinds of edge:
//!
//! - **Derivative edges** — numerical differentiation ties a function to another function that
//!   is supposed to be its derivative (`softplus' = sigmoid`, `ln_gamma' = digamma`).
//! - **Integral edges** — quadrature ties a density to its CDF (`∫ gamma pdf = gamma_lr`),
//!   which also pins the normalizing constant, hence `ln_gamma`/`ln_beta`.
//! - **Closed-form edges** — a CDF at special parameters collapses to an unrelated function
//!   (`gamma_lr(x, ½) = erf(√x)`, `beta_reg(x, ½, ½) = (2/π) asin √x`).
//! - **Discrete↔continuous edges** — a discrete sum equals a continuous CDF (Poisson tail =
//!   `gamma_ur`, binomial CDF = `beta_reg`), which ties the combinatorics on
//!   [`Natural`](crate::Natural) to the incomplete functions.
//!
//! Split by edge kind; the shared densities live in [`support`].

mod closed_forms;
mod closure;
mod derivatives;
mod distributions;
mod functional_equations;
mod integrals;
mod support;
